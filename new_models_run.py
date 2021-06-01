## For running the BERT, GPT-2, and GPT-2 medium models.
## This will save the probability outputs of the models.

import pandas as pd
import os
from os.path import join, exists

import pickle

import new_models
from new_models import model_score_funcs
import load_runs


def get_inputs(prefix_only = False):
    """
    prefix_only mode is strictly for debugging purposes.
    """
    
    sentences = load_runs.load_runs()['user_candidate_transcription']
    if prefix_only:
        sentences = sentences[:2]
    
    print(f'Getting inputs of length: {len(sentences)}')
    return sentences

def pickle_word_predictions(score_outputs, model_name, save_folder):
    
    # 3/27: https://stackoverflow.com/questions/27745500/how-to-save-a-list-to-a-file-and-read-it-as-a-list-type
    save_path = join(save_folder, f"{model_name}_predictions.txt")
    with open(save_path, 'wb') as f:
        pickle.dump(score_outputs, f)
        
    print(f'Saved per-word scores to {save_path}.')
       
    return score_outputs
    
def make_gpt2_word_scores(save_folder_path, prefix_only = False, verbose = True):
    
    inputs = get_inputs(prefix_only)
    prob_outputs = model_score_funcs.get_gpt2_scores(inputs, verbose = verbose)
    
    pickle_word_predictions(prob_outputs, 'gpt2_normal', save_folder_path)
    
    return prob_outputs
                            
        
def make_gpt2_medium_word_scores(save_folder_path, prefix_only = False, verbose = True):
    
    inputs = get_inputs(prefix_only)
    prob_outputs = model_score_funcs.get_gpt2_scores(inputs, '-medium', verbose = verbose)
    
    pickle_word_predictions(prob_outputs, 'gpt2_medium', save_folder_path)
    return prob_outputs
    
    
def make_bert_word_scores(save_folder_path, prefix_only = False, verbose = True):
    
    inputs = get_inputs(prefix_only)
    prob_outputs = model_score_funcs.get_bert_scores(inputs, verbose = verbose)
    
    pickle_word_predictions(prob_outputs, 'bert', save_folder_path)
    return prob_outputs

def make_bart_word_scores(save_folder_path, prefix_only = False, verbose = True):
    
    inputs = get_inputs(prefix_only)
    prob_outputs = model_score_funcs.get_bart_scores(inputs, verbose = verbose)
    
    pickle_word_predictions(prob_outputs, 'bart', save_folder_path)
    return prob_outputs
        
    
if __name__ == '__main__':
    
    # Please note that these models output probabilities in both the word and sentence case,
    #    NOT log10(probability) scores.
    
    RESULTS_FOLDER = './intermediate_results/new_models_probs'

    if not os.path.exists(RESULTS_FOLDER):
        os.makedirs(RESULTS_FOLDER)
        
    make_gpt2_word_scores(RESULTS_FOLDER, prefix_only = True)
    make_gpt2_medium_word_scores(RESULTS_FOLDER, prefix_only = True)
    make_bert_word_scores(RESULTS_FOLDER, prefix_only = True)
    make_bart_word_scores(RESULTS_FOLDER, prefix_only = True)
    
    print('Completed predictions')
    
    
    
        
 
    
    