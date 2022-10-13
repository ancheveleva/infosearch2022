from argparse import ArgumentParser
from matrix_models import BERTMatrix, BM25Matrix


def main(corpus_dir, query_file, model_type):
    """
    For every query vectorizes it, computes cosine similarity
    and return a list of the documents in the relevant order
    """
    if model_type == 'BERT':
        corpus = BERTMatrix(corpus_dir)
    else:
        corpus = BM25Matrix(corpus_dir)

    with open(query_file) as f:
        queries = f.read().splitlines()
    for q in queries:
        print("Your query:", q)
        vec_q = corpus.query_vect(q)
        docs_relevance = corpus.similarity(vec_q)
        indexes = corpus.sort_indexes(docs_relevance)
        print("Documents in relevant order:")
        print(*corpus.corpus[indexes][:3], sep='\n')
        print('\n')


if __name__ == '__main__':
    argparser = ArgumentParser()
    argparser.add_argument('corpus', type=str, default='./corpus',
                           help='provide a route to your the directory with your corpus file (default: ./corpus)'
                           )
    argparser.add_argument('query_file', type=str, default='./queries.txt',
                           help='provide a route to file with your queries (default: ./queries.txt)'
                           )
    argparser.add_argument('model_type', type=str, default='BM25',
                           help='provide a type of vectorizer (BERT or BM25) (default: BM25)'
                           )
    args = argparser.parse_args()
    main(corpus_dir=args.corpus, query_file=args.query_file, model_type=args.model_type)
