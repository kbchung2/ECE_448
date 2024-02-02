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

epsilon = 0.05 # is this a reasonable value?

def baseline(test, train):
    '''
    Implementation for the baseline tagger.
    input:  test data (list of sentences, no tags on the words, use utils.strip_tags to remove tags from data)
            training data (list of sentences, with tags on the words)
    output: list of sentences, each sentence is a list of (word,tag) pairs.
            E.g., [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
    '''
    most_frequent_tag = ""
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
                        most_frequent_tag = tag
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
    output = []
    return output

def viterbi_ec(test, train):
    '''
    Implementation for the improved viterbi tagger.
    input:  test data (list of sentences, no tags on the words). E.g.,  [[word1, word2], [word3, word4]]
            training data (list of sentences, with tags on the words). E.g.,  [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
    output: list of sentences, each sentence is a list of (word,tag) pairs.
            E.g., [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
    '''



