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

class Corpus:
    """For initializing&creating a corpus"""

    def __init__(self, files_dir):
        """
        Either reads matrix, corpus and vectorizer from existing docs
        Or asks for their creation
        """
        self.files_dir = files_dir
        self.json_filepath = os.path.join(self.files_dir, 'data.jsonl')
        self.matrix_filepath = os.path.join(self.files_dir, 'matrix_bm25.npz')
        self.corpus_filepath = os.path.join(self.files_dir, 'corpus.txt')
        self.vectorizer_filepath = os.path.join(self.files_dir, 'count_vectorizer.pkl')
        try:
            self.matrix = sparse.load_npz(self.matrix_filepath)
            with open(self.corpus_filepath) as f:
                self.corpus = f.read().splitlines()
            with open(self.vectorizer_filepath, 'rb') as f:
                self.count_vectorizer = pickle.load(f)
        except FileNotFoundError:
            print('This might take a while...')
            self.corpus, self.clean_corpus = self.get_corpus()
            self.matrix, self.count_vectorizer = self.get_matrix()

    def get_corpus(self):
        """Returns lists of original & preprocessed texts"""
        corpus = []
        clean_corpus = []
        with open(self.json_filepath, 'r') as f:
            questions = list(f)
        i = 0
        for q in questions:
            if i == 50000:
                break
            q_json = json.loads(q)
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
                i += 1
        with open(self.corpus_filepath, 'w') as f:
            f.write('\n'.join(corpus))
        return corpus, clean_corpus

    def get_matrix(self):
        """
        Initiates vectorizer and creates matrix
        """
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
