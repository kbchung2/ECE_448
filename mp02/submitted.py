'''
This is the module you'll submit to the autograder.

There are several function definitions, here, that raise RuntimeErrors.  You should replace
each "raise RuntimeError" line with a line that performs the function specified in the
function's docstring.
'''
'''
Note:
For grading purpose, all bigrams are represented as word1*-*-*-*word2

Although you may use tuple representations of bigrams within your computation, 
the key of the dictionary itself must be word1*-*-*-*word2 at the end of the computation.
'''

import numpy as np
from collections import Counter

stopwords = set(["a","about","above","after","again","against",
                 "all","am","an","and","any","are","aren","'t","as",
                 "at","be","because","been","before","being","below","between",
                 "both","but","by","can","cannot","could","couldn","did","didn",
                 "do","does","doesn","doing","don","down","during","each","few",
                 "for","from","further","had","hadn","has","hasn","have","haven"
                 ,"having","he","he","'d","he","'ll","he","'s","her","here","here",
                 "hers","herself","him","himself","his","how","how","i","'m","'ve",
                 "if","in","into","is","isn","it","its","itself","let","'s","me",
                 "more","most","mustn","my","myself","no","nor","not","of","off",
                 "on","once","only","or","other","ought","our","ours","ourselves",
                 "out","over","own","same","shan","she","she","'d","she","ll","she",
                 "should","shouldn","so","some","such","than","that","that","the","their",
                 "theirs","them","themselves","then","there","there","these","they","they",
                 "they","they","'re","they","this","those","through","to","too","under",
                 "until","up","very","was","wasn","we","we","we","we","we","'ve","were",
                 "weren","what","what","when","when","where","where","which","while","who",
                 "who","whom","why","why","with","won","would","wouldn","you","your","yours","yourself","yourselves"])

def create_frequency_table(train):
    '''
    Parameters:
    train (dict of list of lists) 
        - train[y][i][k] = k'th token of i'th text of class y

    Output:
    frequency (dict of Counters): 
        - frequency[y][x] = number of occurrences of bigram x in texts of class y,
          where x is in the format 'word1*-*-*-*word2'
    '''
    # print(train)
    dictionary = (dict(train))
    frequency = {}
    for class_class in train.keys():
        frequency[class_class] = Counter()
        texts = train[class_class] # this is the list of lists (just a collection of texts) (each list represents a text)
        for text in texts:
            for bigram_token_idx in range(len(text) - 1):
                first_word = text[bigram_token_idx]
                second_word = text[bigram_token_idx + 1]
                frequency[class_class][first_word + "*-*-*-*"  + second_word] += 1
    return frequency
        
        

def remove_stopwords(frequency):
    '''
    Parameters:
    frequency (dict of Counters): 
        - frequency[y][x] = number of occurrences of bigram x in texts of class y,
          where x is in the format 'word1*-*-*-*word2'
    stopwords (set of str):
        - Set of stopwords to be excluded

    Output:
    nonstop (dict of Counters): 
        - nonstop[y][x] = frequency of bigram x in texts of class y,
          but only if neither token in x is a stopword. x is in the format 'word1*-*-*-*word2'
    '''
    nonstop = {}
    for class_class in frequency.keys():
        cur_counter = frequency[class_class] # the Counter()
        nonstop[class_class] = Counter()
        for bigram in list(cur_counter):
            word_one,word_two = bigram.split("*-*-*-*")
            
            if word_one in stopwords and word_two in stopwords:
                continue
            
            # print(bigram, cur_counter[bigram])

            nonstop[class_class][word_one + "*-*-*-*" + word_two] = cur_counter[bigram]
    return nonstop

def laplace_smoothing(nonstop, smoothness):
    '''
    Parameters:
    nonstop (dict of Counters) 
        - nonstop[y][x] = frequency of bigram x in y, where x is in the format 'word1*-*-*-*word2'
          and neither word1 nor word2 is a stopword
    smoothness (float)
        - smoothness = Laplace smoothing hyperparameter

    Output:
    likelihood (dict of dicts) 
        - likelihood[y][x] = Laplace-smoothed likelihood of bigram x given y,
          where x is in the format 'word1*-*-*-*word2'
        - likelihood[y]['OOV'] = likelihood of an out-of-vocabulary bigram given y


    Important: 
    Be careful that your vocabulary only counts bigrams that occurred at least once
    in the training data for class y.
    '''
    k = smoothness
    c = Counter()
    
    likelihood = {}
    
    # print(len(list(nonstop.keys()) ) )
    # print(nonstop)
    # print(  )
    
    for class_class in nonstop: #class y
        likelihood[class_class] = {}
        num_unique_bigrams = len(list(nonstop[class_class].keys()   ))
        num_tokens = sum(nonstop[class_class].values()) 
        for bigram in list(nonstop[class_class].keys()):
            likelihood[class_class][bigram] = (nonstop[class_class][bigram] + k ) / (num_tokens+ k * (num_unique_bigrams + 1))
        likelihood[class_class]["OOV"] = k /  (num_tokens + k * (num_unique_bigrams + 1))
    return likelihood
def naive_bayes(texts, likelihood, prior):
    '''
    Parameters:
    texts (list of lists) -
        - texts[i][k] = k'th token of i'th text
    likelihood (dict of dicts) 
        - likelihood[y][x] = Laplace-smoothed likelihood of bigram x given y,
          where x is in the format 'word1*-*-*-*word2'
    prior (float)
        - prior = the prior probability of the class called "pos"

    Output:
    hypotheses (list)
        - hypotheses[i] = class label for the i'th text
    '''
    hypotheses = []
    # print(len(texts))
    # print(prior)
    for text in texts:
        prob_pos = np.log(prior)
        prob_neg = np.log(1 - prior)
        
        for idx in range(len(text) - 1):
            if text[idx] in stopwords and text[idx+1] in stopwords:
                continue
            bigram = text[idx] + "*-*-*-*" + text[idx + 1]
            
            if bigram not in likelihood["pos"]:
                prob_pos += np.log(likelihood["pos"]["OOV"])
            else:
                prob_pos += np.log(likelihood["pos"][bigram])
            if bigram not in likelihood["neg"]:
                prob_neg += np.log(likelihood["neg"]["OOV"])
            else:
                prob_neg += np.log(likelihood["neg"][bigram])
            
        if prob_pos > prob_neg:
            hypotheses.append("pos")
        else:
            hypotheses.append("neg")  
    return hypotheses



def optimize_hyperparameters(texts, labels, nonstop, priors, smoothnesses):
    '''
    Parameters:
    texts (list of lists) - dev set texts
        - texts[i][k] = k'th token of i'th text
    labels (list) - dev set labels
        - labels[i] = class label of i'th text
    nonstop (dict of Counters) 
        - nonstop[y][x] = frequency of word x in class y, x not stopword
    priors (list)
        - a list of different possible values of the prior
    smoothnesses (list)
        - a list of different possible values of the smoothness

    Output:
    accuracies (numpy array, shape = len(priors) x len(smoothnesses))
        - accuracies[m,n] = dev set accuracy achieved using the
          m'th candidate prior and the n'th candidate smoothness
    '''
    accuracies = np.zeros((len(priors),len(smoothnesses)))
    for pr_idx, prior in enumerate(priors):
        sum_total = len(labels)
       
        for sm_idx, k in enumerate(smoothnesses):
            sum_correct = 0
            likelihood = laplace_smoothing(nonstop,k)
            classified = naive_bayes(texts,likelihood,prior)
            
            for lidx, label in enumerate(labels):
                if label == classified[lidx]:
                    sum_correct += 1
            accuracies[pr_idx,sm_idx] = sum_correct / sum_total
    return accuracies
                          