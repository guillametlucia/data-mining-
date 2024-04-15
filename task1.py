import pandas as pd
import numpy as np
import re
from collections import Counter
import matplotlib.pyplot as plt
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
import pickle

# Read passage collection data into list.
with open('passage-collection.txt', 'r', encoding ='utf-8') as f:
    passages = f.read()


def text_stats(text, remove_stopwords=False):
    # INPUT: string of text 
    # RETURNS: 
        # vocab: dictionary containing all the vocabulary terms and how many times they occur.
        # vocab_size: number of unique words in the text
        # total: total number of words in the text
        # norm_freq: normalized frequency of each word in the text. np array
    text = text.lower() # make all lower case
    text = re.sub(r'[^\w\s]', ' ', text)
    all_words = text.split()
    if remove_stopwords == True:
        sw_nltk = set(stopwords.words('english'))
        all_words = [word for word in all_words if word not in sw_nltk and len(word) > 1]
    vocab = Counter(all_words)
    vocab_size = len(vocab)
    counts = np.array(list(vocab.values()))
    total = counts.sum()
    norm_freq = counts / total  
    return(vocab, vocab_size, total, norm_freq)
def process_text(text):
    text = text.lower() # make all lower case
    text = re.sub(r'[^\w\s]', ' ', text)
    all_words = text.split()
    sw_nltk = set(stopwords.words('english'))
    all_words = [word for word in all_words if word not in sw_nltk and len(word) > 1]
    vocab = Counter(all_words)
    return  vocab


vocab, vocab_size, total, norm_freq = text_stats(passages, remove_stopwords=True)
vocab_i, vocab_size_i, total_i, norm_freq_i = text_stats(passages, remove_stopwords=False)

with open('vocab.txt', 'w', encoding='utf-8') as f:
    for key, value in vocab.items():
        f.write(f"{key}: {value}\n")


#Rank based on highest frequency        
sorted_indices = np.argsort(norm_freq)[::-1]  # Get indices to sort the array in descending order
sorted_frequencies = norm_freq[sorted_indices]
ranks = np.arange(1, vocab_size + 1)
# Zipf's law
s= 1 # distribution parameter zipf law 
denominator = np.sum(ranks.astype(float) ** -s)
theoretical_freq = ranks.astype(float) **-s/denominator

#Rank based on highest frequency        
sorted_indices_i = np.argsort(norm_freq_i)[::-1]  # Get indices to sort the array in descending order
sorted_frequencies_i = norm_freq_i[sorted_indices_i]
ranks_i = np.arange(1, vocab_size_i + 1)
# Zipf's law
s= 1 # distribution parameter zipf law 
denominator_i = np.sum(ranks_i.astype(float) ** -s)
theoretical_freq_i = ranks_i.astype(float) **-s/denominator_i

# Plot normalized frequency against rank and log-log plot. 
def create_plot(ranks, frequencies, theoretical_freq, filename, log_scale=False):
    plt.figure(figsize=(10, 6))
    plt.plot(ranks, frequencies, label='Empirical', linestyle='--')
    plt.plot(ranks, theoretical_freq, label='Theoretical', linestyle='-')
    plt.ylabel('Normalized frequency' if not log_scale else 'Term normalized frequency (log)')
    plt.xlabel('Frequency ranking' if not log_scale else 'Term frequency ranking (log)')
    title = 'Normalized frequency against frequency ranking' if not log_scale else 'Log-log plot: normalized frequency against frequency ranking'
    plt.title(title)
    plt.legend()
    if log_scale:
        plt.xscale('log')
        plt.yscale('log')
        plt.grid(True, which="both", ls="--", linewidth=0.5)
    plt.savefig(filename, format='pdf')
    plt.close()

create_plot(ranks_i, sorted_frequencies_i, theoretical_freq_i, 'figure1both_all.pdf')
create_plot(ranks_i, sorted_frequencies_i, theoretical_freq_i, 'loglog_all.pdf', log_scale=True)
create_plot(ranks, sorted_frequencies, theoretical_freq, 'figure1both.pdf')
create_plot(ranks, sorted_frequencies, theoretical_freq, 'loglog.pdf', log_scale=True)
