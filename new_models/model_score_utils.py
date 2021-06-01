import torch
import torch.nn.functional as F

import pandas as pd

from transformers import GPT2Tokenizer, BertForMaskedLM, GPT2LMHeadModel

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
    
def get_model_probabilities(this_input, model, this_ground_truth_idx, prediction_position, verifying = False):
    
    this_input = torch.Tensor(this_input)
    with torch.no_grad():
        raw_logits = model(this_input.long().unsqueeze(0) if len(this_input.shape) == 1 else this_input.long())
        logits = raw_logits['logits'] 
        
        all_probs = F.softmax(logits, -1)
        probs = all_probs[0, prediction_position, :]
        # Processed per example, not per batch.
        ground_truth_prob = probs[this_ground_truth_idx].item()

    return ground_truth_prob if not verifying else (ground_truth_prob, probs, all_probs)

def score_input(sentence, model, tokenizer, prefix_func):
    
    sentence = prepSentence(sentence)
    this_inputs, raw_tokens = prefix_func(sentence, tokenizer)
    
    # The order of the masking for all of these is [1, the last token]
    this_probs = []
    
    ground_truth_tokens = raw_tokens[1:]
    
    for idx, (this_input, ground_truth_token) in enumerate(zip(this_inputs, ground_truth_tokens)): # Need to enumerate the next prediction locations.
        
        token_idx = idx + 1 if not isinstance(model, GPT2LMHeadModel) else idx
        
        # The reason for the above is that GPT-2 stores at idx, the softmax for the word predicted at idx + 1 (not in the inputs presented)
        # See here 4/3: https://huggingface.co/transformers/quickstart.html  
        # In contrast, BERT stores at idx, the softmax for the word predicted and masked at idx.
 
        this_prob = get_model_probabilities(this_input, model, ground_truth_token,
                                            prediction_position = token_idx)
        # token_idx refers to the position-batched softmax position.
        this_probs.append(this_prob)
      
    this_tokens = tokenizer.convert_ids_to_tokens(ground_truth_tokens)
    
    return this_tokens, this_probs



def score_inputs(sentences, model, tokenizer, prefix_func, verbose = True):
    
    all_score_dfs = []
    for idx, sentence in enumerate(sentences):
        if verbose and idx % 50 == 0: print(f'Processing index: {idx}')
            
        # 3/21/21: https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.from_dict.html#pandas.DataFrame.from_dict
        this_tokens, this_scores = score_input(sentence, model, tokenizer, prefix_func)
        
        this_score_df = pd.DataFrame.from_dict({'word' : this_tokens, 'prob' : this_scores})
        
        all_score_dfs.append(this_score_df)
        
    return all_score_dfs
        
        