
import numpy as np
import pandas as pd

# 5/31: https://stackoverflow.com/questions/25050141/how-to-filter-in-nan-pandas
def filter_nan_df(df):
    df = df[~(df['orig_prob'].isnull() | df['edited_prob'].isnull())]
    return df

def to_surprisal(prob_series):
    return -np.log10(prob_series)

# 6/1 : https://seaborn.pydata.org/generated/seaborn.violinplot.html

def prep_single_model_df_violin(model_name, df, is_surprisal = False):
    
    """
    Convert each of the models to a yes/no dataset
        concat them all with a "modelname" attribute.
    """
    
    def yes_no_gen(label, df):
        return [label for _ in range(len(df))]

    
    orig_probs = to_surprisal(df['orig_prob']) if is_surprisal else df['orig_prob']
    edited_probs = to_surprisal(df['edited_prob']) if is_surprisal else df['edited_prob']
    
    all_probs = pd.concat([orig_probs, edited_probs])
    yes_no_labels = yes_no_gen("Yes", df) + yes_no_gen("No", df)
    
    this_model_probs_df = pd.DataFrame.from_records({'score': all_probs,
                                                     'is_original_prob' : yes_no_labels,
                                                     'model_name' : [model_name for _ in range(len(yes_no_labels))]
                                                    })
    
    return this_model_probs_df

def gen_violin_plots(all_changes_probs, use_surprisal):
    """
    all_changes_probs = The output of prep_probs.load_word_changes
        This is the DataFrame with the original/edited probabilities 
        
        if use_negative_surprisal is False, word probabilities are used instead
    """
    
    return pd.concat([
        prep_single_model_df_violin(model_name, filter_nan_df(raw_df), 
                                   is_surprisal = use_surprisal)
        for model_name, raw_df in all_changes_probs.items()
    ])