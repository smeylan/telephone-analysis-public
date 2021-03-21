import torch

# 3/5 Below function: Taken from Dr. Meylan's telephone_analysis.py
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