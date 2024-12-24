"""
Builds an inverted index from a collection of passages.
It reads a previously generated vocabulary list and candidate passages, processes each passage,
and saves the inverted index as a JSON file.
"""
# candidate-passages-top1000.tsv 
    # Collection of passages in tsv file: qid - pid - query - passage
    # Contains unique instances of column pairs pid (passage ID) and passage

import pandas as pd
from collections import defaultdict
from tqdm import tqdm
from termcolor import colored
from text_processing_and_statistics import process_text
import json

# Read candidate passages from TSV file and vocabulary
candidate_passages = pd.read_csv('candidate-passages-top1000.tsv', sep='\t', header=None)
with open('vocab.txt', 'r', encoding ='utf-8') as f:
    vocab = f.read().splitlines()


print(colored('Building index..', 'green', attrs=['reverse', 'blink']))

# Create nested dictionary with terms as keys that map to another dictionary where documents are keys.
inverted_index = defaultdict(lambda: defaultdict(int))

# For each passage, record the amount of time terms appear. 
done_pid = []
for index, row in tqdm(candidate_passages.iterrows()):
    if row[1] not in done_pid:
        vocab = process_text(row[3])
        pid = row[1]
        done_pid.append(pid)
        for term, count in vocab.items():
            inverted_index[term][pid] = count
print(colored('Success\n', 'green', attrs=['reverse', 'blink']))

# Save the inverted index to a JSON file
print(colored('Saving file..', 'green', attrs=['reverse', 'blink']))
with open('inverted_index.json', 'w') as f:
    json.dump(inverted_index, f)
print(colored('Success\n', 'green', attrs=['reverse', 'blink']))
