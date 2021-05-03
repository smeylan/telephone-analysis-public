# For both prefix generation scenarios, omit index = 0, that is, [CLS]

def get_gpt2_prefixes(sentence, tokenizer, positions = []):
    
    # Note: GPT-2 does not add these tokens for you. However BART Tokenizer is considered an instance of GPT2 Tokenizer,
    #   and does add the bos/eos for you -- meaning there will be double tokens if this if statement is run on BART.
    # If you do bos SPACE sentence SPACE eos, then it will add an extra white space token before the first word and the last word.
    
    sentence = f'{tokenizer.bos_token}{sentence}{tokenizer.eos_token}'
    sentence_tokens = tokenizer.encode(sentence)
    
    all_prefixes = []
    num_tokens = len(sentence_tokens)
    
    # Don't mask positions, just give all prefixes. This is default behavior for general scoring.
    if not positions: positions = range(1, num_tokens)
    
    for length in positions:
        # Omits the empty prefix that predicts CLS. (start at 1)
        # Omits predicting beyond the full input (i.e. predicting the token that comes after EOS)
        all_prefixes.append(sentence_tokens[:length])
        
    return all_prefixes, sentence_tokens


def get_bertlike_mask_func(tokenizer):
    
    # 4/2: Below string to token int conversion from Dr. Meylan
    mask_id = tokenizer.convert_tokens_to_ids([tokenizer.mask_token])[0]
        
    def _bertlike_masks_func(sentence, tokenizer, positions = []):
        
        sentence_tokens = tokenizer.encode(sentence)
        num_tokens = len(sentence_tokens)
        
        # Don't mask positions, just give all prefixes. This is default behavior for general scoring.
        if not positions: positions = range(1, num_tokens)
            
        tokens = []
        
        for masked_idx in positions:
            this_repeat = sentence_tokens[:]
            this_repeat[masked_idx] = mask_id
            
            tokens.append(this_repeat)

        return tokens, sentence_tokens

    return _bertlike_masks_func
