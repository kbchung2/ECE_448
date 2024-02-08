'''
This is the module you'll submit to the autograder.

There are several function definitions, here, that raise RuntimeErrors.  You should replace
each "raise RuntimeError" line with a line that performs the function specified in the
function's docstring.

For implementation of this MP, You may use numpy (though it's not needed). You may not 
use other non-standard modules (including nltk). Some modules that might be helpful are 
already imported for you.
'''

import math
from collections import defaultdict, Counter
from math import log
import numpy as np

# define your epsilon for laplace smoothing here

epsilon = 1e-10 # is this a reasonable value?

def baseline(test, train):
    '''
    Implementation for the baseline tagger.
    input:  test data (list of sentences, no tags on the words, use utils.strip_tags to remove tags from data)
            training data (list of sentences, with tags on the words)
    output: list of sentences, each sentence is a list of (word,tag) pairs.
            E.g., [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
    '''
    highest_tag_count = 0
    most_frequent_tag_to_highest_tag_count = Counter()
    wordToTagCount = {}
    output = []
    for sentence in train:
        for word,tag in sentence:
                if word not in wordToTagCount:
                        wordToTagCount[word] = Counter()
                wordToTagCount[word][tag] += 1
                most_frequent_tag_to_highest_tag_count[tag] += 1
                if wordToTagCount[word][tag] > highest_tag_count:
                        highest_tag_count = wordToTagCount[word][tag]
#     print(wordToTagCount)
    for sentence in test:
            sentence_list = []
            for word in sentence:
                    if word not in wordToTagCount:
                            sentence_list.append((word, most_frequent_tag_to_highest_tag_count.most_common(1)[0][0]))
                    else:
                            frequent_tag = wordToTagCount[word].most_common(1)[0][0]
                            sentence_list.append((word,frequent_tag))
            output.append(sentence_list)
    return output


def viterbi(test, train):
    '''
    Implementation for the viterbi tagger.
    input:  test data (list of sentences, no tags on the words)
            training data (list of sentences, with tags on the words)
    output: list of sentences with tags on the words
            E.g., [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
    '''
    #The laplace smoothing formula is (# of that particular token + k) / (total # tokens + k * (# types + 1) )
   
    
    output = []
    first_tag_occurences = Counter()
    tag_occurences = Counter() # count of tags 
    tag_pair_occurences =Counter() # count of tag pairs
    tag_word_occurences = Counter() # Count of tag word pairs
        
    initial_tag_prob = {}
    tag_pair_prob = {}
    tag_word_prob  = {}
    
    # go through train. Update tag count, tag_pair count, tag_word count.
    
    for sentence in train:
            for idx in range(len(sentence) ):
                    word,tag = sentence[idx]
                    tag_occurences[tag] += 1 # occurence of this tag
                    if idx == 1:
                        first_tag_occurences[tag] += 1
                    if idx > 1:
                        first_tag = sentence[idx - 1][1]
                        second_tag = sentence[idx][1]
                        tag_pair_occurences[(first_tag,second_tag)] += 1 # occurence of the tag pairs
                    tag_word_occurences[(word,tag)] += 1
       
            
#     print(tag_occurences)
#     print(tag_pair_occurences)
#     print(tag_word_occurences, "\n\n\n")
    
    # get number of unique tokens and sum of all tokens, to help
    
    num_unique_tags= len(list(tag_occurences.keys()))
    num_tags = np.sum(list(tag_occurences.values())) 
    
    num_unique_initial_tags = len(list(first_tag_occurences.keys() ))
    num_initial_tags = np.sum(list(first_tag_occurences.values()   ))
#     num_unique_tag_pairs = len(list(tag_pair_occurences.keys()))
#     num_tag_pairs = np.sum(list(tag_pair_occurences.values()) ) 
    
#     num_unique_tag_words=0
#     num_tag_words = 0
    
    # get number of unique tokens and sum of all tokens, to help
    
#     for tag in tag_word_occurences:
#             num_unique_tag_words += len(list(tag_word_occurences[tag].keys() )  )
#             num_tag_words += np.sum(list(tag_word_occurences[tag].values() ) )
    
    tag_word_sums = Counter()
    unique_tag_word_sums = Counter()
    tag_pair_sums = Counter()
    unique_tag_pair_sums = Counter()
    
    # Apply Laplace smoothing
#     print(tag_occurences)
            
    for tag in tag_occurences:
            initial_tag_prob[tag] = np.log( (first_tag_occurences[tag] + epsilon) / ( len(train) + epsilon * (num_unique_initial_tags + 1) ) )
            tag_word_prob[("OOV", tag)] = np.log( epsilon / (len(train) + epsilon * (num_unique_tags + 1)   )    )
            
            
    
    for tag_i in tag_occurences: 
            for tag_j in tag_occurences:
                    if (tag_i, tag_j) not in tag_pair_occurences:
                        tag_pair_prob[(tag_i,tag_j)] = np.log( epsilon / (  tag_pair_occurences[(tag_i,tag_j)] + epsilon * (num_unique_tags + 1)   )  )
                    else:
                        tag_pair_prob[(tag_i,tag_j)] = np.log(  (tag_pair_occurences[(tag_i,tag_j)] + epsilon) / (tag_occurences[tag_i] + epsilon * (num_unique_tags + 1)  )   )


    for (word,tag) in tag_word_occurences: # need to use smoothing here
        tag_word_prob[(word,tag)] = (tag_word_occurences[(word,tag)] + epsilon) / (tag_occurences[tag] + epsilon * ( num_unique_tags + 1))
        tag_word_prob[(word,tag)] = np.log(tag_word_prob[(word,tag)])
                     
        # construct trellis

    print("Made it past the probability construction")


    for sentence in test: 
               
        # step 1: Initialize 
        sentence_list = []
        v = []
        psi = []
        first_dict = {}
        first_psidict = {} 
        
        # Idea: What if I just made it simply P(tag) without adding P(word |  tag) ? REason is because every sentence starts with START
        for tag in tag_occurences: # for all states
                if tag == "START":
                        first_dict[tag] = tag_word_prob[("START", tag)]
                else:
                        first_dict[tag] = -np.inf
                first_psidict[tag] = "N/A"
                        
        # print(first_dict)
        v.append(first_dict)
        psi.append(first_dict)
        # print(initial_tag_prob)
        idx_counter = 1
        #step 2: iterate
        for idx,word in enumerate(sentence): # For 2 <= t <= d
                if idx == 0:
                        continue
                vdict = {}
                psidict = {}
        
                for tagj in tag_occurences: # for all states j 
                        if idx == 1:
                                vdict[tagj] = initial_tag_prob[tagj]
                                if (word,tagj) in tag_word_prob:
                                        vdict[tagj] += tag_word_prob[(word,tagj)]
                                else:
                                        vdict[tagj] += tag_word_prob[("OOV", tagj)]
                                psidict[tagj] = "START"
                                
                        else:
                                max_prob = -np.inf
                                max_tag = "N/A"
                                for tagi in tag_occurences:
                                
                                        tp_prob = -np.inf
                                        tw_prob = -np.inf
                                        tp_prob = tag_pair_prob[(tagi,tagj)]
                                        if (word,tagj) not in tag_word_prob:
                                                tw_prob = tag_word_prob[("OOV", tagj)]  
                                        else:
                                                tw_prob = tag_word_prob[(word,tagj)]
                                        
                                        cur_prob =  v[idx_counter-1][tagi] + tw_prob + tp_prob
                                        if cur_prob > max_prob:
                                        
                                                max_prob = cur_prob #v_t(j) = max
                                                max_tag = tagi #psi_t(j) = argmax
                                
                                vdict[tagj] = max_prob
                                psidict[tagj] = max_tag
                
                v.append(vdict)
                psi.append(psidict)
                idx_counter += 1
        # print(vdict)
        # print(psidict)
        #step 3: terminate
        last_tag = ""
        max_prob = -np.inf
        for tag in tag_occurences:
                if v[len(v) - 1][tag] > max_prob:
                        max_prob = v[len(v) - 1][tag]
                        last_tag = tag
        # print(last_tag)

        
        # print("Print psi[t]")
        # print(tag_pair_prob["END"])
        # step 4: back trace
        tags = [last_tag]
        cur_tag = last_tag
        for t in reversed(range(1, len(psi ) )  ):
                # print("t= ", t, " ,", psi[t])
                # print(cur_tag)
                
                
                        
                cur_tag = psi[t][cur_tag]
                tags.insert(0,cur_tag)
        
        # step 5: insert tags
        tag_counter = 0
        
        for word in sentence:
                sentence_list.append((word, tags[tag_counter]))
                tag_counter += 1
                # print(tag_counter)
                
        output.append(sentence_list)
                                             

    return output

def viterbi_ec(test, train):
    '''
    Implementation for the improved viterbi tagger.
    input:  test data (list of sentences, no tags on the words). E.g.,  [[word1, word2], [word3, word4]]
            training data (list of sentences, with tags on the words). E.g.,  [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
    output: list of sentences, each sentence is a list of (word,tag) pairs.
            E.g., [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
    '''



