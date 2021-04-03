import torch
import pandas as pd

# 3/5 Below function: Taken from Dr. Meylan's telephone_analysis.py

def valid_mode(mode):
    
    return mode in {'single_word', 'sentence'}


def prepSentence(x):
    """
    Big LM expects initial capitalization and sentences end with a period
    """
    
    # Below: using .capitalize() will mess up BERT interpretation of [MASK] string.
    # The replace below fixes this problem.
    
    raw_formatted = x.capitalize()
    assert raw_formatted.count('[mask]') <= 1, '[mask] appears more than once in the BERT masking.'
    remasked = raw_formatted.replace('[mask]', '[MASK]')
    
    return(' '.join([remasked,'.']))

def word_probs_to_df(selected_word_probs, ground_truth_word_idxs, tokenizer):
    """
    Converts a tensor of position prediction probabilities
        to the df-form required by Logistic Regression notebooks.
    Inputs:
        selected_word_probs: of the ground truth words. The result of sentence scoring, before sentence-wise sum.
            shape: (positions to predict,)
        ground_truth_word_idxs: the tokenized form of the ground truth words
            shape: (positions to predict,)
    Outputs:
        a List of df, such that each df has columns: "prob", "word"
            corresponding to a single sentence run.
        NOTE: Unlike the other functions in this repository, the EOS and "." removal
            happens in the position extraction, not here.
    """
    
    # Note, this is per sentence.
    words = tokenizer.convert_ids_to_tokens(ground_truth_word_idxs)
    probs = selected_word_probs.numpy()
    
    # 3/21/21: https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.from_dict.html#pandas.DataFrame.from_dict
    df = pd.DataFrame.from_dict({'word' : words, 'prob' : probs})
    
    # 3/21/21: Note that for GPT-2 there will be a modified "G" before each word that has a space before it.
    # https://github.com/openai/gpt-2/issues/80
    
    return df

def get_next_word_probs(next_word_probs, target_word_tokens):
    
    """
    Inputs:
        next_word_probs, a (positions to predict, vocab size) Tensor
            each row corresponds to the position to predict's softmax. 
        target_word_tokens, a (positions to predict,) Tensor
            each value is the ground truth word's token
    Outputs:
        probs, a (positions to predict,) Tensor of the probability
            for each prefix with ground truth completion
    """
    
    target_word_tokens = target_word_tokens.view(-1, 1).long()
    
    # Gather/indexing from 2/27: https://github.com/huggingface/transformers/issues/3021
    
    probs = torch.gather(next_word_probs, dim = 1, index = target_word_tokens).squeeze()
    
    return probs