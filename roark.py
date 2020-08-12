import pandas as pd 
import re
import numpy as np 
import os
import tempfile
from itertools import zip_longest
from subprocess import Popen
from io import StringIO
from joblib import Parallel, delayed

def baseEtoBase10(X):    
    probs = np.exp(1) **  X
    return(np.log10(probs))

def base2toBase10(X):
    probs = 2. ** X
    return(np.log10(probs))


# [X] use temp files, absolute paths
# [X] alternate between the simple and k-best with a flag that generates the command
# [ ] Parallize by sharding and calling a bunch of subprocesses 

def parse(sentences, roark_base_path, numWorkers, mode):    
    valid_modes =  ('k_best', 'full_entropy', 'single_word', 'sentence')
    if mode not in valid_modes:
        raise ValueError('mode must be one of the following: '+str(' '.join(valid_modes)))
    if mode == 'full_entropy':
        print('Note that full_entropy mode will take several minutes (!!!!) per sentence')

    print('Input contains '+str(len(sentences))+' sentences')        
    unique_sentences = np.array(np.unique(sentences))
    print('Input contains '+str(len(unique_sentences))+' unique sentences')    

    numPerFile = np.ceil(len(unique_sentences) / float(numWorkers))
    fileIndices = np.floor(range(len(unique_sentences)) / numPerFile).astype(int)
    # remove the round robin assignment

    #00000,11111,22222,
    
    #np.array(range(len(unique_sentences))) % numWorkers
    # round robin allocation to subfiles-- when we concatenate these, then there is an issue with the 

    workerInputPaths = []
    for fileIndex in np.unique(fileIndices):
        workerInputPath = os.path.join(roark_base_path,'temp', str(fileIndex)+'.input')
        with open(workerInputPath, 'w') as f:
            f.write('\n'.join(unique_sentences[fileIndices == fileIndex]))
        workerInputPaths.append(workerInputPath)
    
    # in parallel call tdparse on each input text file
    def tdparse(inputFile, mode):
        outputfile = inputFile.replace('.input','.output')

        mode_switches = {
            'k_best' : '-k 50',
            'single_word' : '-p',
            'full_entropy' : '-a',
            'sentence' : '-p -k 50' # to marginalize the probability over trees, going to look at the top 50
        }

        command = os.path.join(roark_base_path, 'bin/tdparse')+' -v  '+ mode_switches[mode] +' -F '+outputfile+' '+os.path.join(roark_base_path, 'parse.model.wsj.slc.p05') + ' ' + inputFile
            
        #print(command)
        return(command)

    commands = [tdparse(x, mode) for x in workerInputPaths]
    processes = [Popen(x, shell=True) for x in commands]    
    # wait for all commands to complete
    exitcodes = [p.wait() for p in processes] # if this yields "Failed: 0" that means the parsing worked    
    print('Finished external parsing')    

    # first return a dictionary indexed by the unique input sentence
    results_dict = process_shards(workerInputPaths, numWorkers, mode)    
    
    #!!! try poooping out the results_dict and using that in a function that takes the sentence as input
    #!!! is it possible that other models are running into this issue?
    if mode == 'sentence':
        return(results_dict)    

    rlist = [results_dict[x] for x in sentences]
    return(rlist)
    

def process_shards(workerInputPaths, numWorkers, mode):
    # debugging version:
    shard_results = [process_shard(workerInputPath, mode) for workerInputPath in workerInputPaths]
    # parallel version:
    #shard_results = Parallel(n_jobs=numWorkers)(delayed(process_shard)(workerInputPath, mode) for workerInputPath in workerInputPaths)

    # flatten the shards into a single dictionary
    rdict = {}
    for shard_result in shard_results:
        rdict.update(shard_result)
    return(rdict)

def process_shard(workerInputPath, mode):
    # note the this is for pocessing a single input/output shard
    workerOutputPath = workerInputPath.replace('.input','.output')

    with open(workerInputPath,'r') as f:
        input_sentences = f.read().splitlines()

    with open(workerOutputPath,'r') as f:
        output_scores = f.read().splitlines()        
    
    if mode == 'sentence':        
        # # each line is a tree + score for a sentence
        # # 1 or more lines ccorrespond to a sentence
        
        # parse_lines = []
        # for  i in range(len(output_scores)):                
        #     df = process_sentence_results(output_scores[i], mode)        
        #     parse_lines.append(df)
    
         
        # parse_df = pd.concat(parse_lines)        

        # # restore sentence indices
        # sentence_index_counter = 0
        # last_index = 0
        # sentence_indices = []
        # for i in parse_df.Rank:
        #     if i < last_index:
        #         sentence_index_counter += 1
        #     sentence_indices.append(sentence_index_counter)
        #     last_index = i
        # parse_df['sentence_index'] = sentence_indices

        # # flip the signs
        # parse_df['Negative Log Probability'] = -1. * parse_df['Negative Log Probability']
        # parse_df['Syntactic Contributions to Probability'] = -1. * parse_df['Syntactic Contributions to Probability']
        # parse_df['Lexical Contributions to Probability'] = -1. * parse_df['Lexical Contributions to Probability']    
        
        # # inpsect parse_df to make sure that its values look correct. DO this in a stanadlone pdb?

        # shard_dict = {}    
        # if len(np.unique(parse_df.Rank)) == 1: #only 1 ranked item per each in the returned dataframe
        #     for i in range(parse_df.shape[0]):
        #         shard_dict[input_sentences[i]] = parse_df.loc[parse_df.sentence_index == i,:]    
        # else:        
        #     # Sum over trees
        #     aggregated_parses = parse_df.groupby(['sentence_index'])['Negative Log Probability', 'Syntactic Contributions to Probability','Lexical Contributions to Probability'].aggregate(lambda x: sumLogProbs(x)).reset_index()
        #     for column in ['Negative Log Probability', 'Syntactic Contributions to Probability','Lexical Contributions to Probability']:
        #         aggregated_parses[column] = baseEtoBase10(aggregated_parses[column])
            
        #     for i in np.unique(sentence_indices):
        #         shard_dict[input_sentences[i]] = aggregated_parses.loc[aggregated_parses.sentence_index == i,:]    
        
        # # the shard_dict returns a single line for each sentence key
        # return(shard_dict)

        #Note that in single word we get surprisal estimates from the word-by-word prefix probabilities, which reflect all trees in the beam                
        
        # use pfix header to identify the split between sentences
        by_sentence= '\n'.join(output_scores).split('pfix header')
        output_scores = by_sentence[1:len(by_sentence)] #0th is an empty string

        assert(len(input_sentences) == len(output_scores))

        probabilities = []
        for  i in range(len(output_scores)):              
            df = process_sentence_results(output_scores[i], mode)

            total_surprisal_at_last_word = df.iloc[[df.shape[0]-2]][['prefix.1']]            
            probabilities.append(-1. * baseEtoBase10(total_surprisal_at_last_word))        

        assert(len(input_sentences) == len(probabilities))        
        shard_dict = {}    
        for i in range(len(probabilities)):
            shard_dict[input_sentences[i]] = probabilities[i]
        
        return(shard_dict)

    
    elif mode == 'single_word':  


        #Note that in single word we get surprisal estimates from the word-by-word prefix probabilities, which reflect all trees in the beam                
        
        # use pfix header to identify the split between sentences
        by_sentence= '\n'.join(output_scores).split('pfix header')
        output_scores = by_sentence[1:len(by_sentence)] #0th doesn't have anything in it
        assert(len(input_sentences) == len(output_scores))

        parse_lines = []
        for  i in range(len(output_scores)):                
            df = process_sentence_results(output_scores[i], mode)        
            parse_lines.append(df)
    
        assert(len(input_sentences) == len(parse_lines))
        shard_dict = {}    
        for i in range(len(parse_lines)):
            shard_dict[input_sentences[i]] = parse_lines[i]

        # the shard dict returns a data frame for each sentence key 
        return(shard_dict)        
        

def sumLogProbs(x):     
    return(np.log(np.sum(np.exp(1) **  x)))  #scores from Roark are base e  

def process_sentence_results(i, mode):
    '''return a dataframe for every sentence. May be a single row if mode is sentenc; several rows if the mode is single_word'''
    if mode == 'sentence':
        mode = 'single_word'
    if mode == 'sentence':        
        d = {
            'Sentence': [],
            'Tree Structure': [],
            'Rank': [],
            'Number of Candidates': [],
            'Tree Score': [],
            'Negative Log Probability': [],
            'Syntactic Contributions to Probability': [],
            'Lexical Contributions to Probability': []
        }        
        stats = ' '.join(i.split()).split('(')[0].split(' ')
        d['Sentence'].append(" ".join(re.findall('[a-z]\w+', i)))
        stat_keys = ['Rank', 'Number of Candidates', 'Tree Score', 'Negative Log Probability', 'Syntactic Contributions to Probability', 'Lexical Contributions to Probability']
        parens = ' '.join(i.split())
        d['Tree Structure'].append(' '.join(parens.split(' ')[6:]))
        for j in np.arange(len(stats) - 1):
            d[stat_keys[j]].append(float(stats[j]))
    
        return pd.DataFrame(d)
    elif mode == 'single_word':
        lines = np.array(i.split('\n'))
        colnames = ' '.join(lines[0].split()).strip().split(' ')

        regexp = re.compile(r'pfix\-|pfix\:')
        pfix_indices = np.argwhere([regexp.search(line) for line in lines]).flatten()    

        y = [' '.join(x.split()) for x in lines[pfix_indices]]
        pfix_df = pd.read_table(StringIO('\n'.join(y)), sep=' ', header=None, names = ['prefix','word']+colnames)
    

        for column in ['srprsl','SynSp','LexSp']:
            pfix_df[column] = [baseEtoBase10(x) for x in pfix_df[column]]

        return(pfix_df)
    else:
        raise NotImplementedError

