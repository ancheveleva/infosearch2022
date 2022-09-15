import os
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk import WordNetLemmatizer
from pymorphy2 import MorphAnalyzer
import re
import scipy.sparse as sp
import math


stops = stopwords.words("russian") + stopwords.words("english") + ['это', 'то', 'ты', 'мы', 'вы', 
                                                                   'я', 'он', 'она', 'оно', 'они']
lemm_rus = MorphAnalyzer()
lemm_eng = WordNetLemmatizer()
vectorizer = CountVectorizer(analyzer='word')


def get_filepaths():
    curr_dir = os.getcwd()
    friends_folder = os.path.join(curr_dir, 'friends-data')
    series_filepaths = []
    for root, dirs, files in os.walk(friends_folder):
        for name in files:
            if name.endswith('.txt'): # исключение системных директорий и файлов
                series_filepaths.append(os.path.join(root, name))
    assert len(series_filepaths) == 165
    return series_filepaths


def has_cyrillic(text):
    return bool(re.search('[а-яА-Я]', text))


def preprocess(text: str):
    tokens = word_tokenize(text.lower())
    lemmas = []
    for w in tokens:
        if w.isalpha(): # вместе с пунктуацией убераю цифры и слова с пунктуацией
            if has_cyrillic(w):
                lemm = lemm_rus.parse(w)[0].normal_form
            else:
                lemm = lemm_eng.lemmatize(w)
            if lemm not in stops:
                lemmas.append(lemm)
    return ' '.join(lemmas)


def get_corpus(filepaths: list) -> list:
    corpus = []
    for file in filepaths:
        with open(file, 'r') as f:
            text = f.read()
        corpus.append(preprocess(text))
    return corpus


def get_index(corpus: list, 
              mode: {'dict', 'matrix'}):
    matrix = vectorizer.fit_transform(corpus)
    matrix = matrix.toarray() # потому что я не разобралась как работает sparse матрица с поиском и where
    if mode == 'matrix':
        return matrix
    else:
        dictionary = dict()
        unique_words = vectorizer.get_feature_names()
        for i, uw in enumerate(unique_words):
            dictionary[uw] = [np.where(matrix[:, i] > 0)[0].tolist(), 
                              np.sum(matrix[:, i], axis=0)                             
                             ]
        return dictionary
    
    
def most_freq(index: {dict, np.array}, mode: {'dict', 'matrix'}) -> str:
    if mode == 'matrix':
        i = np.argmax(np.sum(index, axis=0))
        count = np.amax(np.sum(index, axis=0))
        return vectorizer.get_feature_names()[i], count
    else:
        word = ''
        max_count = 0
        for w, docs in index.items():
            new_count = docs[1]
            if new_count >= max_count:
                max_count = new_count
                word = w
        return word, max_count
    
    
def most_rare(index: {dict, np.array}, mode: {'dict', 'matrix'}) -> str:
    if mode == 'matrix':
        i = np.argmin(np.sum(index, axis=0))
        count = np.amin(np.sum(index, axis=0))
        return vectorizer.get_feature_names()[i], count
    else:
        word = ''
        min_count = 1000
        for w, docs in index.items():
            new_count = docs[1]
            if new_count <= min_count:
                min_count = new_count
                word = w
        return word, min_count
    
    
def in_all_docs(index: {dict, np.array}, mode: {'dict', 'matrix'}) -> str:
    if mode == 'matrix':
        p = np.prod(index/100, axis=0) # деление потому что иначе np слишком большой int превращает в 0
        i_list = np.where(p != 0)[0].tolist() # но теперь тут флоат и их не сравнить(((
        return [vectorizer.get_feature_names()[i] for i in i_list]
    else:
        everywhere = []
        for w, docs in index.items():
            if len(docs[0]) == 165:
                everywhere.append(w)
        return everywhere
    
    
def main():
    friends_corpus = get_corpus(get_filepaths())
    
    friends_matrix = get_index(friends_corpus, mode='matrix')
    friends_dict = get_index(friends_corpus, mode='dict')
    
    # a
    mf_m, cf_m = most_freq(friends_matrix, mode='matrix')
    mf_d, cf_d = most_freq(friends_dict, mode='dict')
    print(f"Самое частотное слово (по матрице): {mf_m}, {cf_m} вхождений")
    print(f"Самое частотное слово (по словарю): {mf_d}, {cf_d} вхождений")    
    
    # b
    mr_m, cr_m = most_rare(friends_matrix, mode='matrix')
    mr_d, cr_d = most_rare(friends_dict, mode='dict')
    print(f"Самое редкое слово (по матрице): {mr_m}, {cr_m} вхождений")
    print(f"Самое редкое слово (по словарю): {mr_d}, {cr_d} вхождений")
    
    # c
    all_m = in_all_docs(friends_matrix, mode='matrix')
    all_d = in_all_docs(friends_dict, mode='dict')
    #assert set(all_m) == set(all_d)
    print("Слова из всех документов (по матрице):", all_m) # не получилось, проблему см выше
    print("Слова из всех документов (по словарю):", all_d)
    
    # d
    

if __name__ == '__main__':
	main()