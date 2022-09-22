from argparse import ArgumentParser
from corpus_initialization import Corpus, Preprocessing
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


def similarity(input_vec, corpus_matrix):
    """
    Computes cosine similarity between one vector and matrix of documents.
    Returns flattened and reversed vector of values for each document.
    """
    doc_scores = cosine_similarity(corpus_matrix, input_vec)
    return -doc_scores.flatten()


def main(corpus_dir, query_file):
    """
    For every query vectorizes it, computes cosine similarity
    and return a list of the documents in the relevant order
    """
    corpus = Corpus(corpus_dir)
    corpus_matrix = corpus.matrix
    doc_names = np.array(corpus.doc_names)
    tfidf_vectorizer = corpus.vectorizer

    with open(query_file) as f:
        queries = f.read().splitlines()

    for q in queries:
        print("Your query:", q)
        clean_q = Preprocessing(q).clean_text
        vec_q = tfidf_vectorizer.transform([clean_q])
        docs_relevance = similarity(vec_q, corpus_matrix)
        indexes = np.argsort(docs_relevance)
        print("Documents in relevant order:")
        print(*np.take_along_axis(doc_names, indexes, axis=0), sep='\n')


if __name__ == '__main__':
    argparser = ArgumentParser()
    argparser.add_argument('corpus', type=str, default='./friends-data',
                           help='provide a route to your corpus directory "friends-data" (default: ./friends-data)'
                           )
    argparser.add_argument('query_file', type=str, default='./queries.txt',
                           help='provide a route to file with your queries (default: ./queries.txt)'
                           )
    args = argparser.parse_args()
    main(corpus_dir=args.corpus, query_file=args.query_file)
