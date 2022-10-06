# HW3: Ranking BM25

For runnig a search through the corpus of LOVE 💕 from `hw3` folder in a command line
```
python3 main.py path_to_your_corpus_directory path_to_your_queries_file
```

## Corpus directory
* This is the directory with your `data.jsonl` file
* Please leave the file name as `data.jsonl`
* During the first launch in the corpus folder special files will be created: `count_vectorizer.pkl`, `matrix_bm25.npz`, `corpus.txt`. **Please do not delete or replace them for the speed-up of the next launches.**

## Queries file
* Queries file should be `.txt` format
* Each query should be written on a separate line