import os
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk import WordNetLemmatizer
from pymorphy2 import MorphAnalyzer
import re
import json
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from scipy import sparse
import pickle
from transformers import AutoTokenizer, AutoModel
import torch
from torch.nn.functional import cosine_similarity
from abc import ABC, abstractmethod


class Preprocessing:
    """All the preprocessing of a string"""
    stops = stopwords.words("russian") + stopwords.words("english") + ['это', 'то', 'ты', 'мы', 'вы',
                                                                       'я', 'он', 'она', 'оно', 'они']
    lemm_rus = MorphAnalyzer()
    lemm_eng = WordNetLemmatizer()

    def __init__(self, text):
        self.text = text
        self.clean_text = self.preprocess(self.text)

    @staticmethod
    def has_cyrillic(text):
        """Checks if a string contains cyrillic letters"""
        return bool(re.search('[а-яА-Я]', text))

    def preprocess(self, text: str):
        """
        Tokenizes.
        Deletes punctuation&stop-words.
        Lemmatizes according to the language.
        Returns a clean string.
        """
        tokens = word_tokenize(text.lower())
        lemmas = []
        known_words = {}  # для ускорения работы pymorphy
        for w in tokens:
            if w.isalpha():  # вместе с пунктуацией убераю цифры и слова с пунктуацией
                if Preprocessing.has_cyrillic(w):
                    if w in known_words:
                        lemm = known_words[w]
                    else:
                        lemm = Preprocessing.lemm_rus.parse(w)[0].normal_form
                        known_words[w] = lemm
                else:
                    lemm = Preprocessing.lemm_eng.lemmatize(w)
                if lemm not in Preprocessing.stops:
                    lemmas.append(lemm)
        return ' '.join(lemmas)


class Corpus(ABC):
    """For initializing&creating a corpus"""

    def __init__(self, files_dir):
        """
        Either reads matrix, corpus and vectorizer from existing docs
        Or asks for their creation
        """
        self.files_dir = files_dir
        self.json_filepath = os.path.join(self.files_dir, 'data.jsonl')
        self.corpus_filepath = os.path.join(self.files_dir, 'corpus.txt')
        self.clean_corpus_filepath = os.path.join(self.files_dir, 'clean_corpus.txt')
        self.questions_filepath = os.path.join(self.files_dir, 'questions.txt')
        try:
            with open(self.corpus_filepath) as fc:
                self.corpus = np.array(fc.read().splitlines())
            with open(self.clean_corpus_filepath) as fcc:
                self.clean_corpus = fcc.read().splitlines()
            with open(self.questions_filepath) as fq:
                self.questions = fq.read().splitlines()
        except FileNotFoundError:
            print('This might take a while...')
            self.corpus, self.clean_corpus = self.get_corpus()

    def get_corpus(self):
        """Returns lists of original & preprocessed texts"""
        corpus = []
        clean_corpus = []
        quest_sents = []
        with open(self.json_filepath, 'r') as f:
            questions = list(f)
        i = 0
        for q in questions:
            if i == 50000:
                break
            q_json = json.loads(q)
            quest_sent = q_json['question']
            answer = ''
            max_value = 0
            for ans in q_json['answers']:
                a = ans['text']
                v = ans['author_rating']['value']
                if v != '' and int(v) >= max_value:
                    answer = a
                    max_value = int(v)
            if answer != '':
                prepr_answer = Preprocessing(answer)
                corpus.append(prepr_answer.text)
                clean_corpus.append(prepr_answer.clean_text)
                quest_sents.append(quest_sent)
                i += 1
        with open(self.corpus_filepath, 'w') as fc:
            fc.write('\n'.join(corpus))
        with open(self.clean_corpus_filepath, 'w') as fcc:
            fcc.write('\n'.join(clean_corpus))
        with open(self.questions_filepath, 'w') as fq:
            fq.write('\n'.join(quest_sents))
        return np.array(corpus), clean_corpus

    @abstractmethod
    def get_matrix(self):
        pass

    @abstractmethod
    def similarity(self):
        pass

    @abstractmethod
    def query_vect(self):
        pass


class BM25Matrix(Corpus):
    def __init__(self, files_dir):
        super().__init__(files_dir=files_dir)
        self.matrix_filepath = os.path.join(self.files_dir, 'matrix_bm25.npz')
        self.vectorizer_filepath = os.path.join(self.files_dir, 'count_vectorizer.pkl')
        try:
            self.matrix = sparse.load_npz(self.matrix_filepath)
            with open(self.vectorizer_filepath, 'rb') as f:
                self.count_vectorizer = pickle.load(f)
        except FileNotFoundError:
            print('This might take a while...')
            self.matrix, self.count_vectorizer = self.get_matrix()

    def get_matrix(self):
        """Initiates vectorizer and creates matrix"""
        k = 2
        b = 0.75

        # матрица tf + понадобится для индексации запроса
        count_vectorizer = CountVectorizer()
        count_vectorizer.fit(self.clean_corpus)
        tf = count_vectorizer.transform(self.clean_corpus)

        # расчет idf
        tfidf_vectorizer = TfidfVectorizer(use_idf=True)
        tfidf_vectorizer.fit(self.clean_corpus)
        idf = tfidf_vectorizer.idf_
        idf = np.expand_dims(idf, axis=0)

        # расчет количества слов в каждом документе - l(d)
        len_d = tf.sum(axis=1)
        # расчет среднего количества слов документов корпуса - avdl
        avdl = len_d.mean()

        # расчет знаменателя (первая часть)
        B_1 = (k * (1 - b + b * len_d / avdl))

        rows, cols = tf.nonzero()
        rows = rows.tolist()
        cols = cols.tolist()
        data = []
        for i, j in zip(rows, cols):
            a = tf[i, j] * idf[0, j] * (k + 1)  # расчет числителя
            b = tf[i, j] + B_1[i, 0]  # расчет знаменателя (вторая часть)
            data.append(a / b)

        # BM25
        matrix = sparse.csr_matrix((data, (rows, cols)))
        sparse.save_npz(self.matrix_filepath, matrix)
        with open(self.vectorizer_filepath, 'wb') as fwt:
            pickle.dump(count_vectorizer, fwt)
        return matrix, count_vectorizer

    def similarity(self, input_vec):
        """
        Computes Okapi BM25 similarity between one vector and matrix of documents.
        """
        docs_scores = self.matrix.dot(input_vec.T)
        return docs_scores

    @staticmethod
    def sort_indexes(matrix):
        """
        Sorts sparse matrix of shape (n, 1) according to the value
        Returns rows indexes in reversed order
        """
        rows = matrix.nonzero()[0]
        row_value = zip(rows, matrix.data)
        sorted_row_value = sorted(row_value, key=lambda v: v[1], reverse=True)
        sorted_indexes = [i[0] for i in sorted_row_value]
        return sorted_indexes

    def query_vect(self, query):
        """Preprocesses and vectorizes a query"""
        clean_q = Preprocessing(query).clean_text
        vec_q = self.count_vectorizer.transform([clean_q])
        return vec_q


class BERTMatrix(Corpus):
    def __init__(self, files_dir):
        super().__init__(files_dir=files_dir)
        self.matrix_filepath = os.path.join(self.files_dir, 'matrix_BERT.pt')
        # Load AutoModel from huggingface model repository
        self.tokenizer = AutoTokenizer.from_pretrained("sberbank-ai/sbert_large_mt_nlu_ru")
        self.model = AutoModel.from_pretrained("sberbank-ai/sbert_large_mt_nlu_ru")
        try:
            self.matrix = torch.load(self.matrix_filepath, map_location=torch.device('cpu'))
        except FileNotFoundError:
            print('This might take a while...')
            self.matrix = self.get_matrix()

    # Mean Pooling - Take attention mask into account for correct averaging
    @staticmethod
    def mean_pooling(model_output, attention_mask):
        token_embeddings = model_output[0] # First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask

    @staticmethod
    def vect_bert(sentences, tokenizer, model):
        # Sentences we want sentence embeddings for
        # Tokenize sentences
        encoded_input = tokenizer(sentences, padding=True, truncation=True, max_length=24, return_tensors='pt')
        # Compute token embeddings
        with torch.no_grad():
            model_output = model(**encoded_input)
        # Perform pooling. In this case, mean pooling
        matrix = BERTMatrix.mean_pooling(model_output, encoded_input['attention_mask'])
        return matrix

    def get_matrix(self):
        """Gets documents vectors from pretrained BERT-model"""
        matrix = self.vect_bert(self.corpus, self.tokenizer)
        torch.save(matrix, self.matrix_filepath)
        return matrix

    def similarity(self, input_vec):
        """Computes similarity between docs matrix and an input vector"""
        doc_scores = cosine_similarity(self.matrix, input_vec)
        return doc_scores

    @staticmethod
    def sort_indexes(vec):
        """Sorts torch matrix in descending order"""
        sorted_indexes = torch.argsort(vec, descending=True)
        return sorted_indexes

    def query_vect(self, query):
        """Gets query vector from pretrained BERT-model"""
        vec_q = self.vect_bert(query, self.tokenizer, self.model)
        return vec_q
