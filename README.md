# Information retrieval project
This project focuses on using different statistical and more advanced models to retrieve the most relevant passages (documents) for user queries. 

I implement and evaluate three contemporary models: logistic regression with word embeddings, LambdaMART, and a transformer-based Large Language Model (BERT), comparing their performance against the traditional Best Match 25 model (BM25) on a validation dataset.  

## External files used (not available in repo)
passage-collection.txt: contains the collection of passages used for text analysis.

candidate-passages-top1000.tsv:  a TSV file containing a collection of passages with columns: qid (query ID), pid (passage ID), query, passage.

test-queries.tsv: a TSV file with test queries.

## text_processing_and_statistics.py
Process text to remove stopwords, calculate word frequencies, build vocabulary and visualize Zipf's law.

## build_inverted_index.py 
Based on previously built vocabulary, build and save an inverted index that contains the amount of times each term appears in each passage.

## tdidf_bm25_scoring.py
Implements text retrieval using TF-IDF (term frequency and inverse document frequency) and (Best Match 25) BM25 scoring models.

Reads data, processes queries and passages, calculates scores, and saves the results to CSV files.

## likelihood_language_models_ranking.py
Implements query likelihood language models with Laplace smoothing, Lidstone correction, and Dirichlet smoothing.

Ranks passages based on queries, calculates probabilities, and saves the results to CSV files.


