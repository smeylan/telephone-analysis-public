import hashlib
import binascii
import glob
import re
import os
import sys
import pandas as pd 
import numpy as np 
from dateutil.parser import parse
import pyxdameraulevenshtein
import Levenshtein
import time
import networkx as nx
import glob
import shutil
import subprocess

# LOG PROCESSING FUNCTIONS
def load_log(log_dir):    
    '''load Apache logs, including ones that have turned over'''
    log_indices = range(1,10)[::-1]
    logs_to_check =  [os.path.join(log_dir,'error.log.'+str(i)) for i in log_indices]
    logs_to_check.append(os.path.join(log_dir,'error.log')) 
    #print(logs_to_check)

    log = []
    for log_to_check in logs_to_check:
        if os.path.exists(log_to_check):
            with open(log_to_check) as inputfile:
                for line in inputfile:
                    log.append(line.strip())
    return(log)        

def processTranscriptionTime(line):
    pieces = re.split(r"[\[|\]]+", line)    
    pieces = [x for x in pieces if x not in ('',' ')]
    try:
        elapsed = float(pieces[-1].replace(' transcribed in ','').replace('s.',''))
    except:
        import pdb 
        pdb.set_trace()
    return({'time': int(parse(pieces[0]).strftime("%s")), 'transcription_time':elapsed})

def processNumberOfConnections(line):
    pieces = re.split(r"[\[|\]]+", line)
    pieces = [x for x in pieces if x not in ('',' ')]
    try:
        elapsed = float(pieces[-1].replace(' There are currently ','').replace('active users (limit 30)',''))
    except:
        import pdb 
        pdb.set_trace()     
    return({'time': int(parse(pieces[0]).strftime("%s")), 'transcription_time':elapsed})    

def filter_logs(filter_string, log):
    filtered_log = [x for x in log if filter_string in x]
    return(filtered_log)

# User Extraction and Checking
def check_files(user, user_recording_dir, fb):
    glob_return = glob.glob(os.path.join(user_recording_dir,user,'*_none.wav'))
    completed = set([int(os.path.basename(x).replace('_none.wav','')) for x in glob_return])    

    # subtract the flagged ones
    if user in fb['flags']:         
        flagged = fb['flags'][user]
        if type(flagged) is dict:
            upstream_flagged = [flagged[x]['stimulus_id'] for x in flagged.keys() if flagged[x]['flag_type']=='stimulus']
        elif type(flagged) is list:
            upstream_flagged = [x['stimulus_id'] for x in [y for y in flagged if y is not None]  if x['flag_type']=='stimulus']

        #print(upstream_flagged)    
        expected = set(range(51)) - set(upstream_flagged) 
    else:
        expected = set(range(51))   

    missing = list(expected - completed)
    missing.sort()
    if len(missing) == 0:
        missing_string = 'All present'
    elif len(missing) > 5:
        missing_string = 'More than 5 audio files missing'
    else:
        missing_string  = ' '.join([str(x) for x in missing])
    
    #get the size of the missing files
    size_on_disk = sum([os.path.getsize(f) for f in glob_return]) / (1024. * 1024.)    
    return({'user':user, 'missing':missing_string, 'size':size_on_disk, 'num_missing':len(missing)})

def getUserTable(fb, user_recording_dir, amt_results_path):
    recordings_per_user = []
    for user in fb['transcriptions'].keys(): 
        if user != 'Dummy':
            check_files_results = check_files(user, user_recording_dir, fb)
            if fb['participants'][user]['workerID_started'] != -1:
                hashed_workerID = fb['participants'][user]['workerID_started']
            else:
                hashed_workerID = ''    

            recordings_per_user.append({'user':user, 
                                        'fb_num_recordings':len(fb['transcriptions'][user]),
                                       'status':fb['participants'][user]['status'],
                                       'missing': check_files_results['missing'],
                                       'condition':fb['participants'][user]['condition']['noise_condition'],
                                        'size':check_files_results['size'],
                                        'num_missing': check_files_results['num_missing'],
                                        'hashed_workerID':hashed_workerID
                                       })    
    
    user_table = pd.DataFrame(recordings_per_user)
    #print(user_table.hashed_workerID)

    # merge against the results from Mechanical Turk
    if os.path.exists(amt_results_path):
        amt_table = loadAMTresults(amt_results_path)
        #print(amt_table.hashed_workerID)
        merged_table = user_table.merge(amt_table, how='outer').sort_values(by='fb_num_recordings') 
        merged_table = merged_table.drop(['WorkerId'] , axis=1) # disable this to keep WorkerID around for debug purposes
        return(merged_table)            
    else:
        return(user_table)    

def hash(string_to_hash):
    # note that this works in Python 3 but not python 2
    hash_results = binascii.hexlify(hashlib.pbkdf2_hmac('sha256', string_to_hash.encode("ascii"), config.TELEPHONE_SALT.encode("ascii"), 100000)).decode("ascii")
    return(hash_results)

def loadAMTresults(file):
    amt_results = pd.read_csv(file)
    amt_results['hashed_workerID'] = [hash(x) for x in amt_results.WorkerId] 
    return(amt_results)


# Evaluating Transcriptions
def getTranscriptionTable(fb):
    transcriptions_for_eval = []
    numberTranscriptionsVisited = 0
    numberOfNullNodes = 0
    add_failure = []
    for user in fb['transcriptions'].keys():
        i = -1
        t = -1
        practiceOnly = True
        if user != 'Dummy':         
            #handle users with list and with dictionary transcriptions; in future versions will collapse to one kind of storage
            if type(fb['transcriptions'][user]) is dict:
                for key in fb['transcriptions'][user].keys():                               
                    numberTranscriptionsVisited += 1
                    transcription = fb['transcriptions'][user][key]          
                    added_this_sentence = False   
                    for transcription_type in ('vs_user','vs_gold','vs_length'):                    
                        if transcription_type in transcription:
                            entry = {'user':user,
                                'stimulus':int(key),                 
                                'transcription_type':transcription_type,    
                                'candidate_transcription':transcription[transcription_type]['candidate_transcription'],
                                'comparison_transcription':transcription[transcription_type]['comparison_transcription']
                            }
                            if transcription_type in ('vs_user','vs_gold'):
                                entry['dist'] = transcription[transcription_type]['dist']
                            elif transcription_type == 'vs_length':
                                entry['length_accept'] = int(transcription[transcription_type]['accept'])
                            else:
                                raise ValueError('transcription_type unrecognized')
                            if 'timing' in transcription:
                                entry['upload_time'] = transcription['timing']['upload']
                                entry['check_time'] = transcription['timing']['checkTranscription']
                            transcriptions_for_eval.append(entry)
                            added_this_sentence = True
                    if not added_this_sentence:
                        flagFailure = False    
                        if user in fb['flags']:
                            if str(key) in fb['flags'][user]:
                                flagFailure = True
                        add_failure.append({'user': user,'key':key, 'type': 'dict', 'flagged':flagFailure })

                    # do something about length here
            elif type(fb['transcriptions'][user]) is list:
                key = -1
                for transcription in fb['transcriptions'][user]:
                    key += 1
                    numberTranscriptionsVisited += 1
                    added_this_sentence = False
                    if transcription is not None:                                    
                        for transcription_type in ('vs_user','vs_gold', 'vs_length'):                    
                            if transcription_type in transcription:
                                entry = {'user':user,
                                    'stimulus':int(key),                 
                                    'transcription_type':transcription_type,    
                                    'candidate_transcription':transcription[transcription_type]['candidate_transcription'],
                                    'comparison_transcription':transcription[transcription_type]['comparison_transcription']
                                }
                                if transcription_type in ('vs_user','vs_gold'):
                                    entry['dist'] = transcription[transcription_type]['dist']
                                elif transcription_type == 'vs_length':
                                    entry['length_accept'] = int(transcription[transcription_type]['accept'])
                                else:
                                    raise ValueError('transcription_type unrecognized')
                                if 'timing' in transcription:
                                    entry['upload_time'] = transcription['timing']['upload']
                                    entry['check_time'] = transcription['timing']['checkTranscription']
                                transcriptions_for_eval.append(entry) 
                                added_this_sentence = True
                        
                        if not added_this_sentence:
                            flagFailure = False    
                            if user in fb['flags']:
                                if str(key) in fb['flags'][user]:
                                    flagFailure = True
                            add_failure.append({'user': user,'key':key, 'type': 'dict', 'flagged':flagFailure })
                    else:
                        numberOfNullNodes += 1
    
    print('Visited '+str(numberTranscriptionsVisited)+' transcriptions')    
    print('Null nodes '+str(numberOfNullNodes))

    rdf = pd.DataFrame(transcriptions_for_eval) 
    rdf['user_short'] = [x[0:8] for x in rdf['user']]
    add_failure_df = pd.DataFrame(add_failure)

    return(rdf, add_failure_df)             

def merge_two_dicts(x, y):
    z = x.copy()   # start with x's keys and values
    z.update(y)    # modifies z with y's keys and values & returns None
    return z

def getEditTable(transcription_df_s):
    edit_store = []
    for x in transcription_df_s.to_dict('records'):
        edits = wordDistance(x['gold_candidate_transcription'],x['user_candidate_transcription'])
        #edits can either be a dictionary or a list of dictionaries
        if edits['num_edits'] == 0:
            # no edits of interest to keep around   
            pass
        else:               
            for edit_op in edits['edit_ops']:
                edit_store.append(merge_two_dicts(edit_op, x))
    return(pd.DataFrame(edit_store))


def levenshtein(sentence1, sentence2):
    dist = pyxdameraulevenshtein.normalized_damerau_levenshtein_distance(sentence1, sentence2)
    return(dist)

def wordDistance(sentence1, sentence2):
    '''get the damerau levenshtein distance between sentences, in terms of words'''
    symbolset = list("abcdefghijklmnopqrstuvwxyz")
    symbolset = symbolset + [x.upper() for x in symbolset]
    symbolset = symbolset + list("1234567890")
    s1 = sentence1.lower().split(' ')
    s2 = sentence2.lower().split(' ')
    vocab = list(set(s1).union(set(s2)))
    symbol_to_word = dict(zip([symbolset[x] for x in range(len(vocab))], vocab))
    word_to_symbol = dict(zip(symbol_to_word.values(), symbol_to_word.keys()))
    s1_translated = ''.join([word_to_symbol[x] for x in s1])
    s2_translated = ''.join([word_to_symbol[x] for x in s2])
    ls1 = list(s1_translated)
    ls2 = list(s2_translated)

    #dist = pyxdameraulevenshtein.damerau_levenshtein_distance(s1_translated, s2_translated)
    editops = Levenshtein.editops(s1_translated, s2_translated)
    translated_editops = []
    for editop in editops:
        if editop[0] == 'replace':
            translated_editops.append({
                'operation': editop[0],
                'in_input': symbol_to_word[ls1[editop[1]]],
                'in_output': symbol_to_word[ls2[editop[2]]]
            }) 
        elif editop[0] == 'insert':
            translated_editops.append({
                'operation': editop[0],
                'in_input': '',
                'in_output': symbol_to_word[ls2[editop[2]]]
            })
        elif editop[0] == 'delete':
            translated_editops.append({
                'operation': editop[0],
                'in_input': symbol_to_word[ls1[editop[1]]],
                'in_output': ''
            })
        else:
            raise NotImplementedError        

    return({'num_edits':len(translated_editops),
        'edit_ops':translated_editops,
        'normalized_dist':len(translated_editops) / float(len(list(s1)))})


def augmentHistory(history, fb):
    augmented_history = history.copy()
    for user in fb['flags'].keys():
        user_flags = fb['flags'][user]
        if type(user_flags) is dict:
            for stimulus_id in user_flags.keys():
                if int(stimulus_id) < 40:
                    flag = user_flags[stimulus_id]
                    if flag['flag_type'] == 'stimulus':
                        relevant_item = augmented_history[flag['chain'],flag['stimulus_id']] \
                        [flag['upstream_pointer']]
                        relevant_item['reason'] = flag['reason']
                        relevant_item['flagged_by'] = flag['subject_id']
        elif type(user_flags) is list:
             for flag in user_flags:
                if flag is not None:
                    if int(flag['stimulus_id']) < 40:
                        if flag['flag_type'] == 'stimulus':
                            relevant_item = augmented_history[flag['chain'],flag['stimulus_id']] \
                            [flag['upstream_pointer']]
                            relevant_item['reason'] = flag['reason']
                            relevant_item['flagged_by'] = flag['subject_id']                                                

    for i in range(augmented_history.shape[0]):
        for j in range(augmented_history.shape[1]):
            for k in range(len(augmented_history[i,j])):
                augmented_history[i,j][k]['chain'] = i
                augmented_history[i,j][k]['stimulus_id'] = j                                        
    return(augmented_history)               


def applyToThreads(thread_table, function, **keyword_parameters):
    #results = np.zeros([thread_table.shape[0], thread_table.shape[1]])
    results = np.empty([thread_table.shape[0], thread_table.shape[1]], dtype=object)
    for i in range(thread_table.shape[0]):
        for j in range(thread_table.shape[1]):            
            results[i,j] = function(thread_table[i,j], chain=i, stimulus_id=j, **keyword_parameters)
    return(results)


def checkConsistencyForChains(augmented_history):
    all_consistent = applyToThreads(augmented_history, checkConsistency)
    return(np.all(all_consistent))


# def checkConsistencyForChains(augmented_history):
#     all_consistent = []
#     for i in range(augmented_history.shape[0]):
#         for j in range(augmented_history.shape[1]):
#             all_consistent += checkConsistency(augmented_history[i,j])
#     return(np.all(all_consistent))


def getLastOkayForChains(augmented_history):
    lastOkay = applyToThreads(augmented_history, getLastOkay)
    return(lastOkay)    

def getLastOkay(history_thread, **keyword_parameters):
    status_array = np.array([x['status'] for x in history_thread])
    pointers = np.array([x['current_pointer'] for x in history_thread])
    lastOkay = int(pointers[len(history_thread) -1 - (np.argmax(status_array[::-1] == 'ok'))])
    return(lastOkay)


def checkConsistency(history_thread, **keyword_parameters):
    if type(history_thread) is dict:
        pass
    elif type(history_thread) is list:    
        pointers = np.array([x['current_pointer'] for x in history_thread])
        status_array = np.array([x['status'] for x in history_thread])
        upstream = np.array([x['upstream_pointer'] for x in history_thread])
        
        flaggedIndices = np.where(status_array == 'flagged')[0] # where is status flagged
        consistent = []

        if len(flaggedIndices) > 0:
            for flaggedIndex in flaggedIndices:
                if flaggedIndex < len(pointers) -1:
                    #print(status_array[0:flaggedIndex[0]])
                    try:
                        lastUnflagged = flaggedIndex - 1 - np.argmax(status_array[0:flaggedIndex][::-1] == 'ok')
                    except:
                        import pdb
                        pdb.set_trace()
                    #print(upstream[flaggedIndex + 1])    
                    #print(pointers[lastUnflagged])
                    consistent.append(upstream[flaggedIndex + 1] == pointers[lastUnflagged])
                
    return(np.all(consistent))


def checkAudioPathsExistForChain(switchboard, user_recording_dir, stimulus_type):
    audioPathsExist = applyToThreads(switchboard, checkAudioPathExists, user_recording_dir=user_recording_dir, stimulus_type=stimulus_type)
    return(audioPathsExist)    

def checkAudioPathExists(switchboard_thread, chain, stimulus_id, user_recording_dir, stimulus_type):
    filepath = os.path.join(user_recording_dir, 
        str(switchboard_thread['subject_id']),
        str(stimulus_id) + '_' + stimulus_type+'.wav')
    return(os.path.exists(filepath))

def getTrialsTable(fb):
    trials = []
    for user in fb['participants'].keys():
        try:
            if user != 'Dummy':
                assignment = fb['participants'][user]['assignment']
                if assignment is None:
                    print('assignment is none for user: '+user)
                if type(assignment) is dict:
                    for trial in assignment:
                        item = fb['participants'][user]['assignment'][trial]
                        if item is not None:
                            item['user'] = user
                            trials.append(item)
                elif type(assignment) is list:        
                    for trial in assignment:                    
                        item = trial
                        if item is not None:
                            item['user'] = user
                            trials.append(item)
                else:
                    import pdb
                    pdb.set_trace()
        except Exception as e:
            print(e)
            print(user)
            print(trial)
    trials = [x for x in trials if x is not None]
    trials_df = pd.DataFrame(trials)
    return(trials_df)

def getFlagTable(fb):
    flags = []
    for user in fb['flags'].keys():
        user_flags = fb['flags'][user]
        try:
            if type(user_flags) is dict:
                for stimulus_id in user_flags.keys():
                    item = user_flags[stimulus_id]
                    item['user'] = user
                    flags.append(item)
            elif type(user_flags) is list:
                 for flag in user_flags:
                    if flag is not None:
                        item = flag
                        item['user'] = user
                        flags.append(item)
        except Exception as e:
            print(user)
            print(stimulus_id)
            print(e)

    flags_df = pd.DataFrame(flags)
    flags_df['stimulus_id'] = [int(x) for x in flags_df['stimulus_id']]
    return(flags_df)


def extractSequence(t_okay, goldTranscriptions, stimulus_id, chainId):
    '''Get the highest approved one and reproduce the sequence needed to produce it'''
    print('Extracting chain for Stimulus '+str(stimulus_id)+', chain '+str(chainId))
    chain_list = []

    #find candidates by finding the highest achieved index
    highest_index = np.max(t_okay.loc[pd.isnull(t_okay.flag_type)].upstream_pointer)    
    final_candidates = t_okay.loc[t_okay.upstream_pointer == highest_index]
    # not sure why there would be multiple final candidates

    i = -1
    success = False
    
    while not success:
        i += 1
        if i > 10:
            raise ValueError('No candidates work')
        else:
            print('Trying candidate '+str(i))            
        try:
            final_candidate = final_candidates.iloc[i]    
            upstream_subject_id = final_candidate.upstream_subject_id
            chain_list.append(pd.DataFrame(final_candidates.iloc[[i]]))

            while str(upstream_subject_id) != "0":        
                print('Seeking ID: '+ upstream_subject_id)
                print(np.argwhere(t_okay.user == upstream_subject_id))                                 
                #c2bbae951ae84dc48207b1935389d8de is in t_okay
                #has a flag_type of "stimulus" but is still treated as a link in the chain -- not sure what this means   

                previous_user = t_okay.iloc[np.argwhere(t_okay.user == str(upstream_subject_id))[0]]                            

                if previous_user.shape[0] > 1:
                    raise ValueError('More than 1 upstream user')
                    print(previous_user)
                elif previous_user.shape[0] == 0:   
                    raise ValueError('No upstream users')
                else:    
                    chain_list.append(previous_user)            
                    upstream_subject_id = previous_user.iloc[0]['upstream_subject_id']
                    print("New ID to seek: "+upstream_subject_id)

            chain_trials = pd.concat(chain_list[::-1]) #40 * 25, though chain_list has 17 items

            initial_sentence = goldTranscriptions[stimulus_id]
            initial_sentence_df = pd.DataFrame({'user_candidate_transcription':initial_sentence, 
                'stimulus_id':stimulus_id,
                'chain':chainId,
                'upstream_pointer': -1,
                'user':0
            }, index=[0])

            chain = pd.concat([initial_sentence_df, chain_trials])

            # reverse the chain because we built it from the back to front             
            success = True
        except:
            success = False  

    # if ((chainId == 1) and (stimulus_id == 14)):
    #     import pdb
    #     pdb.set_trace()

    return(chain)        

# def checkForInherited(t_okay):
#     inherited = []
#     num_disinherited = 0
#     for i in range(len(t_okay['user'])):
#         this_user = t_okay.iloc[i]['user']
#         downstream = t_okay.iloc[i+1:len(t_okay['upstream_subject_id'])]['upstream_subject_id']
#         if downstream.str.contains(this_user).any():
#             inherited.append(True)
#             #print('found')
#         else:
#             inherited.append(False)
#             num_disinherited += 1
#             #print('not found')
#     inherited[-1] = True
#     num_disinherited -=1 #final item is always disinherited otherwise
#     t_okay['inherited'] = inherited
#     return(t_okay.loc[t_okay.inherited], num_disinherited)

def extractThreadFromTranscriptions(dummy, stimulus_id, chain, transcription_df_merged
    , goldTranscriptions):
    #print('Extracting chain for stimulus '+str(stimulus_id)+' for chain '+str(chain))
    #fields = ['upstream_subject_id','user','gold_candidate_transcription','user_candidate_transcription','user_comparison_transcription','gold_dist','user_dist','upstream_pointer','user_numTranscripts','flag_type','chain']
    t_set = transcription_df_merged.loc[(transcription_df_merged.stimulus_id==stimulus_id) & (transcription_df_merged.chain == chain)].sort_values(by='upstream_pointer')#[fields]



    # remove flagged items

    #if chain == 0 and stimulus_id == 4:
    #    import pdb
    #    pdb.set_trace()

    #t_okay = t_set.loc[pd.isnull(t_set.flag_type) & ~(pd.isnull(t_set.gold_candidate_transcription)) & (t_set.gold_dist <= .58)]
    t_okay2 = t_set.loc[~(pd.isnull(t_set.gold_candidate_transcription)) & (t_set.gold_dist <= .58)]

    if t_okay2.shape[0] > 0:
        rv = extractSequence(t_okay2, goldTranscriptions, stimulus_id, chain)
        return(rv)    
    else: 
        return(None)    

def numParticipants(x, **keyword_arguments):
    if x is not None:
        return(x.shape[0])
    else:
        return(0)

def extractChain(x, **keyword_arguments):
    if x is not None:
        return(x)
    else:
        return(None)        


def getLengthInWords(string, replacements):
    sentence_string = string.strip().split(' ')
    new_sentence_string = []
    for word in sentence_string:

        if word in replacements:
            new_sentence_string += replacements[word]
        else:
            new_sentence_string += [word]
    return(len(new_sentence_string))

def getLogProb(string, replacements, bnc_lm, unigram=False, mode='sentence'):
    return(getSRILMprob(string, replacements, bnc_lm, unigram))

def getSRILMprob(string, replacements, bnc_lm, unigram=False, mode='sentence'):    
    sentence_string = string.strip().split(' ')
    
    new_sentence_string = []
    for word in sentence_string:
        if word in replacements:
            new_sentence_string += replacements[word]
        else:
            new_sentence_string += [word]
               
    total_logprob = bnc_lm.total_logprob_strings(new_sentence_string) # this assumes the highest order

    augmented_sentence_string = np.array(['<s>']+ sentence_string + ['</s>'])
    
    word_probs = []            
    for i in range(1,len(augmented_sentence_string)):
        if unigram:
            context = []
        else:
            context = augmented_sentence_string[0:i][::-1]
        prob = bnc_lm.logprob_strings(augmented_sentence_string[i], context)
        word_probs.append({'index':i-1, 'word':augmented_sentence_string[i], 'prob':prob})
 
    by_word_probs = pd.DataFrame(word_probs)    

    #if bnc_lm.total_logprob_strings(sentence_string) != np.sum(by_word_probs.prob[0:by_word_probs.shape[0]-1]):
    #    import pdb
    #    pdb.set_trace()
    #    # this fails in a small number of situations
    
    if mode == 'sentence':
        # exclude the final /s in computing sentence probability
        return(np.sum(by_word_probs.prob[0:(by_word_probs.shape[0]-1)]))    
    elif mode == 'single_word':
        return(by_word_probs)
    else:
        raise NotImplementedError

def getInitialProbs(chainSentences, variable):
    print('Getting initial probabilities for '+variable)
    #if variable ==  'WSJ_Roark_Negative.Log.Probability':
    #    import pdb
    #    pdb.set_trace()
    initial_probs = chainSentences[chainSentences.upstream_pointer == -1].groupby(
    'stimulus_id')[variable].agg(np.mean).reset_index()

    #detect coordination problems
    numUniqueInitialProbEsitmates = chainSentences[chainSentences.upstream_pointer == -1].groupby('stimulus_id')[variable].agg(lambda x: len(np.unique(x))).reset_index()
    numUniqueInitialProbEsitmates.columns = ['stimulus_id','count']

    if numUniqueInitialProbEsitmates[numUniqueInitialProbEsitmates['count'] != 1].shape[0] > 0:
        print('Multiple values for initial sentence, indicating a coordination problem for the language model')
        import pdb
        pdb.set_trace()

    #!!! this indicates an indexing issue
    #chainSentences[(chainSentences.upstream_pointer == -1) & (chainSentences.stimulus_id == 0)]
    # this indicates that there is a merging problem: several unrelated estimates

    initial_probs.columns = ['stimulus_id', 'initial_'+variable]
    initial_probs = initial_probs.sort_values(by='initial_'+variable)
    initial_probs['initial_'+variable+'_rank'] = range(initial_probs.shape[0])
    initial_probs['initial_'+variable+'_quartile'] = np.floor(initial_probs['initial_'+variable+'_rank'] / 10)
    return(initial_probs)

def get_shortform_transcription_table(df, user_table):
    # slice the transcription_df so that both transcriptions are on the same row
    vs_gold_df = df.loc[df.transcription_type == 'vs_gold']
    old_gold_columns = vs_gold_df.columns
    new_gold_columns = ['gold_candidate_transcription', 'check_time', 'gold_comparison_transcription', 
        'gold_dist', 'gold_length_accept', 'stimulus', 'transcription_type', 'upload_time', 'user', 'user_short']
    vs_gold_df.columns = new_gold_columns
    vs_gold_df = vs_gold_df.drop(['transcription_type','gold_length_accept'], axis=1)
    
    
    vs_user_df = df.loc[df.transcription_type == 'vs_user']
    vs_user_df.columns = ['user_candidate_transcription', 'check_time', 'user_comparison_transcription', 
'user_dist', 'user_length_accept', 'stimulus', 'transcription_type', 'upload_time', 'user', 'user_short']
    vs_user_df = vs_user_df.drop(['user_length_accept','transcription_type'], axis=1)

    vs_length_df = df.loc[df.transcription_type == 'vs_length'] 
    vs_length_df.columns = ['length_candidate_transcription', 'check_time', 'length_comparison_transcription', 
'dist', 'length_accept', 'stimulus', 'transcription_type', 'upload_time', 'user', 'user_short']
    vs_length_df = vs_length_df.drop(['dist','length_candidate_transcription','length_comparison_transcription','transcription_type'], axis=1)

    rdf = vs_user_df.merge(vs_gold_df, how='outer').merge(vs_length_df, how='outer').merge(user_table[['user','condition']])
    print('Shape after merging')
    print(rdf.shape)

    word_distances = [np.nan if np.isnan(x['length_accept']) else wordDistance(x['gold_candidate_transcription'],x['user_candidate_transcription']) for x in rdf.to_dict('records')]
    rdf['word_distances'] = [x['num_edits'] if type(x) is dict  else np.nan for x in word_distances]    

    character_levdau = [np.nan if np.isnan(x['length_accept']) else levenshtein(x['gold_candidate_transcription'],x['user_candidate_transcription']) for x in rdf.to_dict('records')]
    rdf['character_levdau'] = character_levdau

    rdf['stimulus_id']  = rdf.stimulus
    return(rdf)

def plotChain(df):
    df  = df.sort_values(by='upstream_pointer')
    flagged_set = set(df[df.flag_type == 'stimulus'].upstream_subject_id.tolist())

    user_set = df.user.tolist() + ['0']

    hash_to_index = dict(zip(user_set, range(len(user_set))))
    index_to_hash = dict(zip(hash_to_index.values(), hash_to_index.keys()))


    G=nx.DiGraph(directed = True)
    vertices = index_to_hash.keys()
    assert(len(vertices) == df.shape[0]+1)
    G.add_nodes_from(vertices)
    color_map = []

    for x in df[['upstream_subject_id','user','flag_type']].to_dict('records'):
        if x['upstream_subject_id'] not in user_set:
            print(x['upstream_subject_id'])
            raise ValueError('Upstream missing!')

        if x['flag_type'] == 'self':
            # self flag
            #G.add_edges_from([(int(hash_to_index[x['upstream_subject_id']]), int(hash_to_index[x['user']]))]) 
            G.add_edges_from([(int(hash_to_index[x['user']]), int(hash_to_index[x['upstream_subject_id']]))]) 
            color_map.append('purple')
        elif x['flag_type'] == 'stimulus':    
            # upstream flag
            #G.add_edges_from([(int(hash_to_index[x['upstream_subject_id']]), int(hash_to_index[x['user']]))]) 
            G.add_edges_from([(int(hash_to_index[x['user']]), int(hash_to_index[x['upstream_subject_id']]))]) 
            color_map.append('red')
        else:
            G.add_edges_from([(int(hash_to_index[x['upstream_subject_id']]), int(hash_to_index[x['user']]))]) 
            if x['user'] in flagged_set:
                color_map.append('orange')
            else:
                color_map.append('green')


    pos = nx.spring_layout(G, iterations=2000, k=.05)
    # hierarchical plots are all hard to get working with graphviz
    #pos = nx.nx_pydot.graphviz_layout(G)
    #from networkx.drawing.nx_agraph import graphviz_layout
    #pos = graphviz_layout(G)
    #pos = write_dot(G)
    
    # [X] size of the graph
    # [O] arrow style: need to upgrade; hard on Chompsky
    # [X] what are all of the singletons 
    # [O] better graph positioning  -- too much work
    # [X] colorize
    # [O] index by submission time? don't have this information
    return(hash_to_index, index_to_hash, G, pos, color_map)

# big LM (Google): automate the process of getting language model results

def getBigLMscores(sentences, tempDir, cacheDir, colname='BigLM_probability',
                   lm_1b_dir='/home/stephan/python/lm_1b', mode ='sentence'):
    
    unique_sentences = np.unique(sentences)

    def prepSentence(x): # Big LM expects initial capitalization and sentences end with a period
        return(' '.join([x.capitalize(),'.']))

    bigLM_input = pd.DataFrame([prepSentence(x) for x in unique_sentences])

    bigLM_input['index'] = range(bigLM_input.shape[0])
    bigLM_input.columns = ['sentence', 'index']
    
    if not os.path.exists(tempDir):
        os.mkdir(tempDir)
        
    if not os.path.exists(cacheDir):
        os.mkdir(cacheDir)
            
    # list of sentences already evalauted
    cache_files = glob.glob(os.path.join(cacheDir,'*.out'))
    basenames = [os.path.basename(x) for x in cache_files]
    
    files_to_cache = []
    for record in bigLM_input.to_dict('records'):        
        if record['sentence'] in basenames:
            # copy the outfile from the cachedir to the tempdir
            cached_result = os.path.join(cacheDir,record['sentence']+'.out')
            output = os.path.join(tempDir,record['sentence']+'.out')
            shutil.copyfile(cached_result, output)            
        else:
            # write it out in preparation for evaluation        
            with open(os.path.join(tempDir, record['sentence'] + '.txt'), 'w') as f:
                f.write(record['sentence'])
            # flag the result as something we need to flag later
            files_to_cache.append(record['sentence'] + '.out')

    # invoke the language modeling code
    big_lm_command = [] 
    big_lm_command.append('cd '+lm_1b_dir+' &&')
    big_lm_command.append('source bin/activate &&')
    big_lm_command.append('bazel-bin/lm_1b/lm_1b_eval --mode eval_sentences')
    big_lm_command.append('--pbtxt data/graph-2016-09-10.pbtxt')
    big_lm_command.append('--vocab_file data/vocab-2016-09-10.txt')
    run_from_path = os.getcwd()
    big_lm_command.append('--eval_dir '+os.path.join(run_from_path, tempDir))
    big_lm_command.append("--ckpt 'data/ckpt-*'")
    big_lm_command_string = ' '.join(big_lm_command)
    print('Big LM call:')
    print(big_lm_command_string)    
    output = subprocess.call(big_lm_command_string, shell=True)
    print(output)
    
    #copy over any new files in the results to the cached directory
    for file_to_cache in files_to_cache:
        output = os.path.join(tempDir,file_to_cache)
        cached_result = os.path.join(cacheDir,file_to_cache)           
        shutil.copyfile(output, cached_result)
    
    # collect the scores from tempDir    
    result_files = glob.glob(os.path.join(tempDir, '*.out'))    
    full_results_dict = {}
    prob_dict = {}
    for result_file in result_files:
        keyName = os.path.basename(result_file).replace('.out','')
        df = pd.read_csv(result_file) 
        if mode == 'single_word':
            df = df[df.word != '.'] #exclude punctuation
        elif mode == 'sentence':
            df = df[df.word != '.'] #exclude punctuation
            df = df[df.word != '</S>'] #exclude end of sentence

        df.prob = -1. * df.prob 
        full_results_dict[keyName] = df
        prob_dict[keyName] = np.sum(df.prob)  

    # each of these resulte files has all of the words in it    
    if mode == 'sentence':
        return([prob_dict[prepSentence(x)] for x in sentences])
    elif mode == 'single_word':
        return([full_results_dict[prepSentence(x)] for x in sentences])
    else:
        raise NotImplementedError

def loadRun(runpath):
    basename = os.path.basename(runpath).replace('.csv','')
    df = pd.read_csv(runpath)
    if u'Unnamed: 0' in df.columns:
        df = df.drop([u'Unnamed: 0'], axis=1)
    df['run'] = basename
    return(df)

def loadAndCombineExptRuns(runPath, runnames, minItemsInChain=0):
    dfs = []
    for runname in runnames:
        dfs.append(loadRun(os.path.join(runPath, runname)))
    # !!! combine and index the runname + chain
    all_runs = pd.concat(dfs)
    all_runs['user_candidate_transcription'] = [x.lower() for x in all_runs['user_candidate_transcription']]
    all_runs['unique_chain_identifier'] = all_runs.run.map(str) + '_'+ all_runs.chain.astype('str')
    all_runs['global_chain'] = all_runs.unique_chain_identifier.astype('category').cat.codes

    # get the number of items in each chain
    chain_counts = all_runs.global_chain.value_counts().reset_index()
    retained_chains = chain_counts[chain_counts.global_chain > minItemsInChain]['index']
    print(retained_chains)
    all_runs = all_runs[all_runs.global_chain.isin(retained_chains)]    
    
    return(all_runs)

def loadInfiniteTelephoneRuns(runPath):
    '''Load the results of an infinite telephone run. This should produce a dataframe with a subset of columns from loadAndCombineExptRuns'''
    
    transcription_paths = glob.glob(os.path.join(runPath,'*.txt'))
    inf_tel_store =  [] 
    for transcription_path in transcription_paths:
        basename = os.path.basename(transcription_path)
        generation, stimulus_id = basename.replace('.txt','').split('_')
        generation = int(generation)
        stimulus_id = int(stimulus_id)

        with open(transcription_path, 'r') as tpf:
            #print('Reading '+transcription_path)
            transcription = tpf.readlines()[0].replace('\n','').lower().strip()
            inf_tel_store.append({
                'user_candidate_transcription': transcription,
                'global_chain': 0,
                'upstream_pointer': generation-1,
                'stimulus': stimulus_id,
                'stimulus_id' : stimulus_id
            })
    
    inf_tel_df = pd.DataFrame(inf_tel_store)
    in_chains = inf_tel_df[inf_tel_df.stimulus < 40] 

    return(in_chains)

def getKenLMProb(sentence, m, mode='sentence', eos=False):
    s_prob = list(m.full_scores(sentence))
    #turn this into a dataframe
    kenDF = pd.DataFrame(s_prob)
    kenDF.columns = ['prob','preceding','unk']
    try:        
        kenDF['words'] = sentence.strip().split(' ')+['</S>']
        
    except:
        import pdb
        pdb.set_trace()
    if mode =='single_word':
        return(kenDF)
    elif mode == 'sentence':        
        if eos:            
            return(np.sum(kenDF.prob))
        else:
            return(np.sum(kenDF.iloc[0:(kenDF.shape[0]-1)].prob))
    else:
        raise NotImplementedError

def getCharniakJohnsonBeamProb(sentence, rrp):
    # this is the probability of this  sentence under the top 50 most probable parses of this sentence
    nbest_list = rrp.parse(sentence)
    #import pdb
    #pdb.set_trace()
    #return(np.log10(np.sum([np.exp(x.reranker_score) for x in nbest_list])))
    return(np.log10(np.sum([10 ** x.reranker_score for x in nbest_list])))


def getMikolovWSJprobs(sentences, rnnpath, telephoneAnalysisPath):
    to_score = pd.DataFrame(sentences)
    to_score['input_index'] = range(to_score.shape[0])
    to_score['unique_sentence_index'] = to_score.user_candidate_transcription.astype('category').cat.codes.astype('int16')
    
    # write out an indexed version of the sentences
    model_inputs = to_score.drop_duplicates(subset=['unique_sentence_index']) # offending line
    model_inputs = model_inputs.sort_values(by=['unique_sentence_index'])
    if not os.path.exists(os.path.join(telephoneAnalysisPath, 'mikolov')):
        os.makedirs(os.path.join(telephoneAnalysisPath, 'mikolov'))
    model_input_path = os.path.join(telephoneAnalysisPath, 'mikolov','mikolov_input.txt')
    scores_path = os.path.join(telephoneAnalysisPath, 'mikolov','mikolov_scores.txt')
    model_inputs['to_write'] = model_inputs.unique_sentence_index.map(str) + ' '+  model_inputs.user_candidate_transcription    
    model_inputs['to_write'].to_csv(model_input_path, header = False, index=False)

    # call a subprocess to get the scores
    RNN_command_string  = "cd "+os.path.join(rnnpath,'rnnlm-0.2b && ./rnnlm')+' -rnnlm '+os.path.join(rnnpath,'rnnlm-0.2b','model') + ' -test '+model_input_path+' -nbest -debug 0  > '+scores_path
    print(RNN_command_string)        
    output = subprocess.call(RNN_command_string, shell=True)
    print(output)

    # read in the scores
    scores = pd.read_csv(scores_path, header=None) # same order as model_inputs
    scores.columns = ['score']
    
    assert(scores.shape[0] == model_inputs.shape[0])    

    # add back to the model_inputs for indexing
    scores = scores.reindex()
    model_inputs = model_inputs.reindex()
    model_inputs['score'] = scores['score'].tolist()  # if not cast to a list Pandas tries to do some worrisome indexing  

    #merge back the scores with to_score
    # unique_sentence_index is getting corrupted for some reason
    rdf = to_score.merge(model_inputs[['unique_sentence_index','score']], on ='unique_sentence_index')
    rdf = rdf.sort_values(by='input_index')    

    assert(len(rdf['score']) == len(sentences))
    return(rdf['score'].tolist())
