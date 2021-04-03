## For functions shared between BART and BERT.
import torch
import pandas as pd

def select_prediction_position(this_probs, extract_word_pos):
    """
    Selects the softmax corresponding to the word position prediction of interest.
    Inputs:
        this_probs, a (prediction position or batch, language model prediction positions, vocabulary size) Tensor
        extract_word_pos, the position in the original sequence to extract from (the location of the masked word)
    Outputs:
        softmax_targets,
            (Returns an ascending dim = 1 slice for all of the positions
                at which a prediction of interest is being made, for [0, num_repeats))
        shape: (batch, vocab size)
    Note that the omission of positions is taken care of in the original generation of the "batch".
    """
    
    # Gather/indexing from 2/27: https://github.com/huggingface/transformers/issues/3021
    extract_word_pos = extract_word_pos.view(-1, 1).repeat(1, this_probs.shape[-1]).unsqueeze(1).long()
    
    softmax_targets = torch.gather(this_probs, dim = 1, index = extract_word_pos).squeeze()
    
    return softmax_targets


def report_mask_words(scores, sentence, tokenizer):
    """
    raw_scores = a (vocabulary,) tensor of selected softmax values for a pre-selected position.
    mask_idx, the position to select for analysis.
    
    sentence = the prefix to do the prediction on
    tokenizer = BERT tokenizer
    """
    
    # It should intake the raw scores itself.
    score_vals, word_idxs = torch.sort(scores, descending = True)
    
    words = tokenizer.convert_ids_to_tokens(word_idxs)

    print(f"Reporting most likely tokens to complete '{sentence}' in descending order")

    num_report = 20

    score_df = pd.DataFrame.from_dict({
      'Word': words,
      'Score value': list(map(lambda x : round(x, 5), score_vals.numpy().tolist()))
      })

    return score_df[:num_report], score_vals
