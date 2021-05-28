import pandas as pd
import os

from os.path import join, exists

from new_models import model_score_utils

import transformers
from transformers import GPT2LMHeadModel, GPT2Tokenizer


def find_true_token_position(sentence, word, position, tokenizer):
    
    """
    Note: This does NOT consider the CLS. That adjustment is made separately.
    """

    tokenized = tokenizer.tokenize(sentence)
    if isinstance(tokenizer, GPT2Tokenizer):
        tokenized = [this_str.strip('Ġ') for this_str in tokenized]
        
    if tokenized[position] != word:
        # 5/3 : https://www.geeksforgeeks.org/python-list-index/
        try:
            return tokenized.index(word, position) # The true position in the tokenized list of the changed word.
        except ValueError:
            return -2 # Couldn't find the true token. It was possibly broken in tokenization.
            # Use -2 to avoid this becoming a valid index with the CLS shift (i.e. 0)
        
    return position # No tokenization issues, did not shift the true position of the word.


def process_single_substitution(sentence, word, position, model, tokenizer, prefix_func):
    
    position = find_true_token_position(sentence, word, int(position), tokenizer) 
    # Note: above position is NOT with CLS added. This is accounted for in process_single_substitution. 
   
    if position == -2: return None
    # The desired token was probably fragmented by the tokenizer.
    # That is, the changed word couldn't be found in whole form after tokenization.
    
    position = position + 1
    # This is because CLS is not accounted for in the original index. This is for correctness of the prefixes.
    
    this_token_prefix, orig_tokens = prefix_func(sentence, tokenizer, [position])
   
    this_ground_truth_idx = orig_tokens[position]
    print(tokenizer.convert_tokens_to_ids([this_ground_truth_idx]))

    logit_position = position if not isinstance(model, GPT2LMHeadModel) else position - 1
    
    # GPT-2 stores the prediction for word i+1 at word i, so need to decrease the prediction position by 1.
    word_prob = model_score_utils.get_model_probabilities(this_token_prefix, model, this_ground_truth_idx, logit_position)
    
    return word_prob
    
    
def process_substitution_entry(df_entry, model, tokenizer, prefix_func):
    
    model_args = (model, tokenizer, prefix_func)
    
    orig_prob = process_single_substitution(df_entry['sentence'], df_entry['sWord'], df_entry['sCounter'], *model_args)
    edited_prob = process_single_substitution(df_entry['response'], df_entry['rWord'],  df_entry['rCounter'], *model_args)
    
    return (orig_prob, edited_prob)
    
       
def analyze_substitutions(sub_df, model, tokenizer, prefix_func, prefix = False):
    
    # sub_df = the result of isolating substitutions from editTables as a pandas DataFrame.
    # Please see Prep notebook for details.
    
    # Need to process each entry
    
    orig_prob_list = []; edited_prob_list = []
    
    sub_df = sub_df.head() if prefix else sub_df
    
    # 4/24: https://stackoverflow.com/questions/16476924/how-to-iterate-over-rows-in-a-dataframe-in-pandas
    for i, entry in sub_df.iterrows():
        if i % 100 == 0: print(f'Entry {i} / {len(sub_df)}')
        entry = sub_df.iloc[i]
        orig_prob, edited_prob = process_substitution_entry(entry, model, tokenizer, prefix_func)
        orig_prob_list.append(orig_prob)
        edited_prob_list.append(edited_prob)
        
    prob_df = pd.DataFrame.from_records({'orig_prob' : orig_prob_list, 'edited_prob' : edited_prob_list})
    
    has_prob_df = sub_df.assign(orig_prob = orig_prob_list) 
    has_prob_df = has_prob_df.assign(edited_prob = edited_prob_list)
    
    return has_prob_df
    
    
    


