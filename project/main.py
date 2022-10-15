from matrix_models import TFIDFMatrix, BERTMatrix, BM25Matrix


def main(corpus_dir, query, model_type):
    """
    For every query vectorizes it, computes cosine similarity
    and return a list of the documents in the relevant order
    """
    if model_type == 'TF-IDF':
        corpus = TFIDFMatrix(corpus_dir)
    elif model_type == 'BERT':
        corpus = BERTMatrix(corpus_dir)
    else:
        corpus = BM25Matrix(corpus_dir)

    vec_q = corpus.query_vect(query)
    docs_relevance = corpus.similarity(vec_q)
    indexes = corpus.sort_indexes(docs_relevance)
    return corpus.corpus[indexes][:10]
