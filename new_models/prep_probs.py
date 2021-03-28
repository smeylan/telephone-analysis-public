## Used to convert precomputed probabilities from BERT, GPT-2, and GPT-2 Medium models
## to the correct log10(Probability score) format.

import pandas as pd
import os
import pickle
import numpy as np

from os.path import join, exists

def load_word_scores(model_name, results_folder):
    """
    These are log10(Probability)
    """
    
    def probability_to_negative_surprisal(this_df):
        # 3/27: https://www.geeksforgeeks.org/log-and-natural-logarithmic-value-of-a-column-in-pandas-python/
        col_name = 'prob'
        this_df[col_name] = np.log10(this_df[col_name]) # How to do this?
        return this_df
        
    
    score_label = f'{model_name}_probability'
    raw_scores_path = join(results_folder, f'{model_name}_predictions.txt')
    
    # 3/27: https://stackoverflow.com/questions/27745500/how-to-save-a-list-to-a-file-and-read-it-as-a-list-type
    with open(raw_scores_path, 'rb') as f:
        raw_scores = pickle.load(f)
    
    neg_surprisals = list(map(probability_to_negative_surprisal, raw_scores))
    return neg_surprisals

def load_sentence_scores(model_name, results_folder):
    """
    These are log10(Probability)
    """
    
    def sum_scores(df):
        return np.sum(df['prob'])
        
    word_neg_surprisals = load_word_scores(model_name, results_folder)
    sentence_scores = list(map(sum_scores, word_neg_surprisals))
    
    return sentence_scores