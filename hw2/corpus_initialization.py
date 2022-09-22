import os
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk import WordNetLemmatizer
from pymorphy2 import MorphAnalyzer
import re
from joblib import dump
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
        for w in tokens:
            if w.isalpha():  # вместе с пунктуацией убераю цифры и слова с пунктуацией
                if Preprocessing.has_cyrillic(w):
                    lemm = Preprocessing.lemm_rus.parse(w)[0].normal_form
                else:
                    lemm = Preprocessing.lemm_eng.lemmatize(w)
                if lemm not in Preprocessing.stops:
                    lemmas.append(lemm)
        return ' '.join(lemmas)


class Corpus:
    """For initializing&creating a corpus"""
    def __init__(self, corpus_dir):
        """
        Either reads matrix, doc_names and vectorizer from existing docs
        Or asks for their creation
        """
        self.corpus_dir = corpus_dir
        self.filepath = os.path.join(self.corpus_dir, 'matrix_tfidf.txt')
        self.doc_names_file = os.path.join(self.corpus_dir, 'doc_names.txt')
        self.vectorizer_file = os.path.join(self.corpus_dir, 'tfidf_vectorizer.pkl')
        try:
            self.matrix = np.genfromtxt(self.filepath, delimiter=",")
            with open(self.doc_names_file) as f:
                self.doc_names = f.read().splitlines()
            with open(self.vectorizer_file, 'rb') as f:
                self.vectorizer = pickle.load(f)
        except FileNotFoundError:
            self.matrix, self.doc_names, self.vectorizer = self.get_matrix()

    def get_matrix(self):
        """
        Creates matrix, doc_names and vectorizer according given filepath to the corpus
        """
        f = FilePaths(self.corpus_dir)
        clean_texts = self.get_corpus(f.filepaths)
        vectorizer = TfidfVectorizer()
        matrix = vectorizer.fit_transform(clean_texts)
        matrix = matrix.toarray()
        with open(self.vectorizer_file, 'wb') as fwt:
            pickle.dump(vectorizer, fwt)
        np.savetxt(self.filepath, matrix, delimiter=',')
        with open(self.doc_names_file, 'w') as fw:
            fw.write('\n'.join(f.doc_names))
        return matrix, f.doc_names, vectorizer

    def get_corpus(self, filepaths: list) -> list:
        """Returns corpus text with given list of filepaths to every episode"""
        corpus = []
        for file in filepaths:
            with open(file, 'r') as f:
                text = f.read()
            corpus.append(Preprocessing(text).clean_text)
        return corpus


class FilePaths:
    """Returns a list of filepaths to every episode and a list of documents names"""
    def __init__(self, corpus_dir):
        self.corpus_dir = corpus_dir
        self.filepaths, self.doc_names = self.get_filepaths()

    def get_filepaths(self):
        series_filepaths = []
        doc_names = []
        for root, dirs, files in os.walk(self.corpus_dir):
            for name in files:
                if name.endswith('.txt'): # исключение системных директорий и файлов
                    series_filepaths.append(os.path.join(root, name))
                    doc_names.append(name[:-4])
        assert len(series_filepaths) == 165
        return series_filepaths, doc_names
