import os
from os.path import join, exists
import pandas as pd

from new_models import model_score_funcs, sub_analysis


def score_model(model_name, sub_df, save_data = True, prefix = False): 
    
    model_spec_dict = {
        'gpt2_normal': model_score_funcs.get_gpt2_modules(),
        'gpt2_medium': model_score_funcs.get_gpt2_modules('-medium'), # Need to check how to specify the model itself. 
        'bert': model_score_funcs.get_bert_modules(),
        'bart': model_score_funcs.get_bart_modules(),
    }
    
    assert model_name in model_spec_dict.keys(), f'Invalid model name. Options are: {list(model_spec_dict.keys())}'
    
    model, tokenizer, prefix_func = model_spec_dict[model_name]
    sub_prob_df = sub_analysis.analyze_substitutions(sub_df, model, tokenizer, prefix_func, prefix = prefix)
    
    # This is repeatedly updating the data in the new csv as new models are run.
    if save_data:
        save_path = join(WORD_CHANGES_FOLDER, 'word_change_probs.csv')
        sub_prob_df.to_csv(save_path)
        print(f'Writing probability dataframe to: {save_path}, for model results of {model_name}')
                                     
    return sub_prob_df
                       

if __name__ == '__main__':


    WORD_CHANGES_FOLDER = './intermediate_results/word_changes'
    sub_df = pd.read_csv(join(WORD_CHANGES_FOLDER, 'edit_substitutions.csv'))
    
    for model_name in ['bert', 'gpt2_normal', 'gpt2_medium', 'bart']:
        print(f'Processing: {model_name}')
        sub_df = score_model(model_name, sub_df, prefix = False)
    
    