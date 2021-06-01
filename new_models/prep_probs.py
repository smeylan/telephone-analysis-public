## Used to convert precomputed probabilities from BERT, BART, GPT-2, and GPT-2 Medium models
## to the correct log10(Probability score) format.

import pandas as pd
import os
import pickle
import numpy as np

import glob

from os.path import join, exists


def probability_to_negative_surprisal(this_df):
    # 3/27: https://www.geeksforgeeks.org/log-and-natural-logarithmic-value-of-a-column-in-pandas-python/
    col_name = 'prob'
    this_df = this_df.copy()
    this_df[col_name] = np.log10(this_df[col_name])
    return this_df
    
    
def load_word_scores(model_name, results_folder, give_probs = False):
    """
    These will return the log10(Probability) scores used as inputs to Aggregate notebook.
    """
    score_label = f'{model_name}_probability'
    raw_scores_path = join(results_folder, f'{model_name}_predictions.txt')
    
    # 3/27: https://stackoverflow.com/questions/27745500/how-to-save-a-list-to-a-file-and-read-it-as-a-list-type
    with open(raw_scores_path, 'rb') as f:
        raw_scores = pickle.load(f)
    
    # Always omit EOS from the calculations, for both word and sentence scoring.
    filtered_raw_scores = list(map(filter_eos_punct_df, raw_scores))
    
    if give_probs: # Don't convert to negative surprisals
        return filtered_raw_scores
    
    neg_surprisals = list(map(probability_to_negative_surprisal, filtered_raw_scores))
    return neg_surprisals

def filter_eos_punct_df(raw_df):
    
    # Omit the EOS probabilities in sentence calculation
    # 3/27 From line 805 reference
    #   https://github.com/smeylan/telephone-analysis-public/blob/master/telephone_analysis.py

    # Note: EOS for GPT-2 is <|endoftext|>, for BERT is [SEP], for BART is </s>
    # For other models in the previous version, '</S>'.
    
    df = raw_df.copy() 
    for eos in {'<|endoftext|>', '[SEP]', '</s>', 'Ġ.', '.', '</S>'}: 
        df = df[df.word != eos]
    
    return df

def load_sentence_scores(model_name, results_folder):
    """
    These are log10(Probability)
    """

    def sum_scores(df):
        new_df = filter_eos_punct_df(df)
        return np.sum(new_df['prob'])
        
    word_neg_surprisals = load_word_scores(model_name, results_folder)
    sentence_scores = list(map(sum_scores, word_neg_surprisals))
    
    return sentence_scores

def load_postprocessed_logistic_prep_scores(data_prep_folder, model_name = ''):
    
    """
    Loads the NaN-aligned scores postprocessed in the Data Prep notebook.
    """
    
    model_names = [filename.split('logistic/')[1].split('_predictions.txt')[0]
               for filename in glob.glob(data_prep_folder+'/*')] if not model_name else [model_name]

    lm = {}

    for lm_name in model_names:
        raw_scores_path = join(data_prep_folder, f"{lm_name}_predictions.txt")
        # 3/27: https://stackoverflow.com/questions/27745500/how-to-save-a-list-to-a-file-and-read-it-as-a-list-type
        with open(raw_scores_path, 'rb') as f:
            raw_scores = pickle.load(f)
            lm[lm_name] = raw_scores
            
    return lm

def load_word_changes(folder, model_name = ''):
    """
    Loads the NaN-aligned scores postprocessed in the Data Prep notebook.
    """
    
    model_names = ['gpt2_normal', 'gpt2_medium', 'bert', 'bart']
    lm = {}

    for lm_name in model_names:
        raw_scores_path = join(folder, f"word_change_probs_{lm_name}.csv")
        lm[lm_name] = pd.read_csv(raw_scores_path)
            
    return lm

    