"""
Performs text retrieval using TF-IDF and BM25 scoring models.
Reads data, processes queries and passages, calculates scores, and saves the results to CSV files.
"""

import json
import pandas as pd
import numpy as np
from text_processing_and_statistics import process_text, text_stats
from termcolor import colored
from time import perf_counter
from tqdm import tqdm


############################################################################################################
################################### Read in data ##############################################
###############################################################################
with open('inverted_index.json', 'r') as f:
    inverted_index = json.load(f) # tf
candidate_passages = pd.read_csv('candidate-passages-top1000.tsv', sep='\t', header=None)
test_queries = pd.read_csv('test-queries.tsv', sep='\t', header=None)


"""
TF-IDF
Vector representation of documents and queries.
"""

# Prepare lists of unique passage IDs and query IDs
all_pid = candidate_passages[1].unique().tolist()
all_qid = test_queries[0].tolist()

# Calculate nt and idf for each term
print(colored('Calculating nt and idf..', 'green', attrs=['reverse', 'blink']))
N = len(all_pid)
logN = np.log(N)
# Initialize an empty dictionary for nt and idf values
nt_idf_dict = {}
# Calculate nt and idf for each term
for term, docs in tqdm(inverted_index.items()):
    nt = len(docs)  # number of documents containing the term
    idf = logN - np.log(nt)  # optimized calculation
    nt_idf_dict[term] = [nt, idf]
print(colored('Success\n', 'green', attrs=['reverse', 'blink']))

# Function to create passage vector
def p_vectorrep(pid, inverted_index, nt_idf_dict, query_vocab):
    pvector = np.zeros(len(query_vocab))  # Initialize pvector with the size of query_vocab
    for i, term in enumerate(query_vocab):
        if term in inverted_index:
            idf = nt_idf_dict[term][1]  # idf value for the term
            tf = inverted_index[term].get(pid, 0)  # tf for the term in the document, default to 0 if not present
            pvector[i] = tf * idf  # Calculate tf-idf and assign to the correct position in the vector
    return pvector
    
# Function to create query vector
def q_vectorrep(query_vocab, nt_idf_dict):
    q_vector = np.zeros(len(query_vocab))
    for i, term in enumerate(query_vocab):
        if term in nt_idf_dict:  # Check if term is in nt_idf_dict
            nt = nt_idf_dict[term][0]  # number of documents that contain the term
            idf = nt_idf_dict[term][1]
            tf = query_vocab.get(term, 0)
            q_vector[i] = tf * idf
    return q_vector

"""
BM25
"""

# BM25 parameters
k1 = 1.2
k2 = 100
b = 0.75

# Calculate document lengths for BM25
print(colored('Calculating document lengths..', 'green', attrs=['reverse', 'blink']))
doc_lengths_dict = {}
for index, row in tqdm(candidate_passages.iterrows()):
    pid = str(row[1])
    passage = row[3]
    length_passage = text_stats(passage, remove_stopwords=True)[2]
    doc_lengths_dict[pid] = length_passage
avg_dl = np.mean(list(doc_lengths_dict.values()))
print(colored('Success\n', 'green', attrs=['reverse', 'blink']))


# Group candidate passages by query ID
candidate_pids_per_qid = candidate_passages.groupby(0)[1].apply(lambda x: [str(pid) for pid in x]).to_dict()
results_cosine = {qid: {} for qid in all_qid} # Initialize a dictionary to store the results
results_bm25 = {qid: {} for qid in all_qid} # Initialize a dictionary to store the results

# Calculate scores for each query
print(colored('Calculating scores..', 'green', attrs=['reverse', 'blink']))
for qid in tqdm(all_qid):
    query = test_queries[test_queries[0] == qid][1].values[0]
    query_vocab = process_text(query)# tokenize query
    candidate_pids = candidate_pids_per_qid[qid]
    q_vector = q_vectorrep(query_vocab, nt_idf_dict)
    for pid in candidate_pids:
        p_vector = p_vectorrep(pid, inverted_index, nt_idf_dict, query_vocab)
        if np.linalg.norm(q_vector) != 0 and np.linalg.norm(p_vector) != 0:
            score = np.dot(q_vector, p_vector) / (np.linalg.norm(q_vector) * np.linalg.norm(p_vector))
        else:
            score = 0
        results_cosine[qid][pid] = score
        bm25 = 0
        for term, tf_q in query_vocab.items():
            if term in inverted_index and pid in inverted_index[term].keys():
                doc_length = doc_lengths_dict[pid]
                tf_d = inverted_index[term][pid]
                nt = nt_idf_dict[term][0]
                bm25 = bm25 + np.log((N-nt + 0.5)/(nt+0.5))*(k1+1)*tf_d/(k1*((1-b)+b*(doc_length/avg_dl))+tf_d)*(k2+1)*tf_q/(k2+tf_q)
        results_bm25[qid][pid] = bm25
print(colored('Success\n', 'green', attrs=['reverse', 'blink']))


############################################################################################################
############################### Format output ##############################################################
############################################################################################################


#########################################TFIDF########################################################
# Initialise empty results dataframe
final = pd.DataFrame(columns=["qid", "pid", "score"])
row_count = 0

# Loop through results and append to dataframe
print(colored('Saving cosine scores..', 'green', attrs=['reverse', 'blink']))
for key in tqdm(results_cosine):
    
    # Sort the results by score, and select top 100
    df = pd.Series(dict(sorted(results_cosine[key].items(), key=lambda item: item[1], reverse=True))).reset_index(name="score")[0:100]
    # Build dataframe out of the results using row_count as the index
    to_concat = pd.concat([pd.DataFrame(np.repeat(key, len(df))), pd.DataFrame(df["index"].astype("str")), pd.DataFrame(df["score"])], axis=1)
    to_concat = to_concat.rename(columns={0: 'qid', 'index': 'pid'})
    # Append to final dataframe
    final = pd.concat([final, to_concat])
    # Update row count
    row_count += len(df)
print(colored('Success\n', 'green', attrs=['reverse', 'blink']))

# Expected length of the DataFrame
expected_length = 19290

# Check the length of the DataFrame
assert len(final) == expected_length, f"Error: The length of the DataFrame is {len(final)}, but it should be {expected_length}"

# If the assertion passes, save the DataFrame
final.to_csv("tfidf.csv", header=None, index=False)

##############################BM25############################################

# Initialise empty results dataframe
final = pd.DataFrame(columns=["qid", "pid", "score"])
row_count = 0

# Loop through results and append to dataframe
print(colored('Saving bm25 scores..', 'green', attrs=['reverse', 'blink']))
for key in tqdm(results_bm25):
    
    # Sort the results by score, and select top 100
    df = pd.Series(dict(sorted(results_bm25[key].items(), key=lambda item: item[1], reverse=True))).reset_index(name="score")[0:100]
    # Build dataframe out of the results using row_count as the index
    to_concat = pd.concat([pd.DataFrame(np.repeat(key, len(df))), pd.DataFrame(df["index"].astype("str")), pd.DataFrame(df["score"])], axis=1)
    to_concat = to_concat.rename(columns={0: 'qid', 'index': 'pid'})
    # Append to final dataframe
    final = pd.concat([final, to_concat])
    # Update row count
    row_count += len(df)
print(colored('Success\n', 'green', attrs=['reverse', 'blink']))

# Expected length of the DataFrame
expected_length = 19290

# Check the length of the DataFrame
assert len(final) == expected_length, f"Error: The length of the DataFrame is {len(final)}, but it should be {expected_length}"

# If the assertion passes, save the DataFrame
final.to_csv("bm25.csv", header=None, index=False)
