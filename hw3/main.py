from argparse import ArgumentParser
from corpus_initialization import Corpus, Preprocessing
import numpy as np


def similarity(input_vec, corpus_matrix):
    """
    Computes Okapi BM25 similarity between one vector and matrix of documents.
    """
    docs_scores = corpus_matrix.dot(input_vec.T)
    return docs_scores


def sort_sparse(matrix):
    """
    Sorts sparse matrix of shape (n, 1) according to the value
    Returns rows indexes in reversed order
    """
    rows = matrix.nonzero()[0]
    row_value = zip(rows, matrix.data)
    sorted_row_value = sorted(row_value, key=lambda v: v[1], reverse=True)
    sorted_indexes = [i[0] for i in sorted_row_value]
    return sorted_indexes


def main(corpus_dir, query_file):
    """
    For every query vectorizes it, computes cosine similarity
    and return a list of the documents in the relevant order
    """
    corpus = Corpus(corpus_dir)
    corpus_matrix = corpus.matrix
    doc_names = np.array(corpus.corpus)
    count_vectorizer = corpus.count_vectorizer

    with open(query_file) as f:
        queries = f.read().splitlines()

    for q in queries:
        print("Your query:", q)
        clean_q = Preprocessing(q).clean_text
        vec_q = count_vectorizer.transform([clean_q])
        docs_relevance = similarity(vec_q, corpus_matrix)
        indexes = sort_sparse(docs_relevance)
        print("Documents in relevant order:")
        print(*doc_names[indexes], sep='\n')


if __name__ == '__main__':
    argparser = ArgumentParser()
    argparser.add_argument('corpus', type=str, default='./corpus',
                           help='provide a route to your the directory with your corpus file (default: ./corpus)'
                           )
    argparser.add_argument('query_file', type=str, default='./queries.txt',
                           help='provide a route to file with your queries (default: ./queries.txt)'
                           )
    args = argparser.parse_args()
    main(corpus_dir=args.corpus, query_file=args.query_file)
