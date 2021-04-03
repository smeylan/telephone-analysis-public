import gpt2_scores
import numpy as np
import torch

import transformers
from transformers import GPT2LMHeadModel, GPT2Tokenizer

def test_get_sentence_prefixes():
    
    sentence = "how are you"
    prefixes, next_words = gpt2_scores.get_sentence_prefixes(sentence, tokenizer, mode = 'sentence')
    
    translated_prefixes = []
    for prefix in prefixes:
        translated_prefixes.append(list(map(tokenizer.decode, prefix)))
    translated_next_words = list(map(tokenizer.decode, next_words))
    
    expected_prefixes = [['<|endoftext|>'],
                         ['<|endoftext|>', 'How'],
                         ['<|endoftext|>', 'How', ' are'],
                        ]
    expected_next_words = ['How', ' are', ' you']
    # Don't predict on/include in prefix the final word, because want to omit influence of added punctuation.
    
    assert expected_prefixes == translated_prefixes
    assert expected_next_words == translated_next_words
     
def test_get_next_word_surprisal():
    
    """
    Tests for expected behavior in idx extraction.
    """
    
    def get_examine_idxs(this_examine_words):
        return list(map(lambda word: tokenizer.encode(word)[0], this_examine_words))
    
    sentence, _ = gpt2_scores.get_sentence_prefixes("how are you", tokenizer)
    
    this_probs = gpt2_scores.get_next_word_probs(sentence, tokenizer, model)
    
    examine_idxs = get_examine_idxs(["How", "you", tokenizer.eos_token])

    expected_surprisals = -1 * np.log([ this_probs[idx][token].item() for idx, token in enumerate(examine_idxs)])

    actual = gpt2_scores.get_next_word_surprisal(this_probs, examine_idxs, tokenizer).numpy()
    
    assert np.sum(expected_surprisals - actual) < 1e-6
    # That is, virtually equal (except for float error)
    
    
def test_pad_sentences():
    """
    This works on non-punctuation/capitalize.
    """
    
    # 2/26: Tokenizer code below line https://huggingface.co/transformers/model_doc/gpt2.html#gpt2lmheadmodel
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    a_no_space_val = tokenizer.encode("A")[0]
    a_val = tokenizer.encode(" a")[0]
    b_val = tokenizer.encode(" b")[0]
    eos_val = tokenizer.encode(tokenizer.eos_token)[0]
    
    sentence = "a a b"
    
    # Ignore the next words for this test.
    sentences, _ = gpt2_scores.get_sentence_prefixes(sentence, tokenizer)
    new_tokens, attentions, real_lengths = gpt2_scores.pad_sentences(sentences, tokenizer)
    
    expected_tokens = torch.Tensor([
        [eos_val, eos_val, eos_val],
        [eos_val, a_no_space_val, eos_val],
        [eos_val, a_no_space_val, a_val],
    ])
    
    expected_attentions = torch.Tensor([
        [1, 0, 0],
        [1, 1, 0],
        [1, 1, 1],
    ])
    
    # In reality, this represents the index of the last word, not the length of the input.
    # 3/4: https://discuss.pytorch.org/t/how-to-tile-a-tensor/13853
    # Above: reminder on how to tile.
    
    expected_real_lengths = torch.Tensor([
        [0], # You will need to tile this?
        [1],
        [2],
    ]).unsqueeze(1).repeat(1, 1, 50257)
    
    # Unsqueeze because it will be used for selecting the last word out of the sequence word dimension.
    
    assert torch.all(new_tokens == expected_tokens)
    assert torch.all(real_lengths == expected_real_lengths)
    assert torch.all(attentions == expected_attentions) # Why do attentions have to be expanded in such a way?
    
    print('Pad sentences test passed.')
    
# Below was generally built with guidance/code from the following sources:

# 2/26 https://huggingface.co/transformers/quickstart.html
# 2/27 information on variable-length batches: https://github.com/huggingface/transformers/issues/2001
# 2/27 Debugging help on trying to predict on end tokens by accident
#    https://github.com/huggingface/transformers/issues/3021

def check_give_top_words():
    """
    For sanity check purposes, for next_word_probs.
        Will need to manually run this and check whether the predicted next word is reasonable.
        
    Note that this is slightly different than the first exercise.
    The model will attempt to predict the last word in the full sentence.
    """
    raw_sentence = "it's time to go to the front"
    #raw_sentence = "the President of the United States"
    
    prefixes, _ = gpt2_scores.get_sentence_prefixes(raw_sentence, tokenizer)
    
    next_word_probs = gpt2_scores.get_next_word_probs(prefixes, tokenizer, model)
        
    logits_sorted, words_sorted = torch.sort(next_word_probs, dim= -1, descending = True)

    top_poss_words = words_sorted[:, :10]

    raw_word_translations = np.array([tokenizer.decode(elem)
                                           for elem in top_poss_words.flatten()])
    top_poss_translations = np.reshape(raw_word_translations, top_poss_words.shape)

    report = "Top next words for the following sentence:\n"
    #report += f"{raw_sentence}\n"
    report += str(top_poss_translations[-1])
    
    print(report)

# 2/26: https://huggingface.co/transformers/model_doc/gpt2.html#gpt2lmheadmodel
# GPT2LMHeadModel returns unnormalized probabilities over the next word -- requires softmax.

# or, gpt-2{medium, large, xl}
# 2/26: options from here https://huggingface.co/transformers/pretrained_models.html

model = GPT2LMHeadModel.from_pretrained('gpt2')
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    
    
if __name__ == '__main__':
    

    tests = [
        test_get_sentence_prefixes,
        test_get_next_word_surprisal,
        test_pad_sentences,
    ]

    for test in tests:
        test() 

    # Checks require manual inspection and are less formal.

    checks = [
        check_give_top_words,
    ]

    for check in checks:
        check()