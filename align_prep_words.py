import pandas as pd
import pdb

def filter_symbols(this_str):
    
    symbols_filter = {'Ġ', 'ġ', '#'}
    for sym in symbols_filter:
        this_str = this_str.replace(sym, '')
    return this_str


def find_next_word_loc(curr_idx, tokens, target_word):
    
    # Note that this works because GPT-2 special-G symbol, as well as BERT #, are filtered beforehand.
    
    build_word = ''
    
    for idx in range(curr_idx, len(tokens)):
    
        build_word += tokens[idx].lower()
              
        if build_word == target_word:
            return idx + 1 # the location of the next word
    
    assert False, 'Find next word loc reached end without finding next word to process.'
     
    
def process_nan_single_df(this_df, this_tokenizer, entire_word_tokens):
    
    to_collapse_tokens = [filter_symbols(word).lower() for word in this_df.word]
    
    entire_idx = 0; to_collapse_idx = 0;
    collapsed_probs = []
    
    while entire_idx < len(entire_word_tokens): 
        
        if to_collapse_idx == len(to_collapse_tokens): break # Means that the last word was a failure.
        
        if entire_word_tokens[entire_idx] != to_collapse_tokens[to_collapse_idx]: # This will be shifted. How to manage? 
            
            collapsed_probs.append(None)
            to_collapse_idx = find_next_word_loc(to_collapse_idx, to_collapse_tokens, entire_word_tokens[entire_idx])
            
        else: 
            try:
                collapsed_probs.append(this_df.iloc[to_collapse_idx]['prob'])
            except:
                pdb.set_trace()
            to_collapse_idx += 1
        entire_idx += 1
    
    new_df = pd.DataFrame.from_records({'word': entire_word_tokens, 'prob' : collapsed_probs})
    return new_df
 
def align_model_word_dfs(this_raw_df_list, this_tokenizer, entire_word_reference):
        
    # entire word reference needs to not be a stacked list of words, but a nested list of words.
    
    list_new_dfs = []
    for idx, (this_df, this_sentence) in enumerate(zip(this_raw_df_list, entire_word_reference)):
        if idx % 100 == 0: print(f'Index: {idx}')
        list_new_dfs.append(process_nan_single_df(this_df, this_tokenizer, this_sentence))

    return list_new_dfs