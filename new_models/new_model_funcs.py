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


def get_next_word_surprisal(next_word_probs, target_word_tokens):
    
    # Gather/indexing from 2/27: https://github.com/huggingface/transformers/issues/3021
    
    softmax_targets = torch.gather(next_word_probs, dim = 1, index = target_word_tokens).squeeze()
    surprisals = -torch.log(softmax_targets)
    
    return surprisals