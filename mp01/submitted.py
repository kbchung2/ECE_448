'''
This is the module you'll submit to the autograder.

There are several function definitions, here, that raise RuntimeErrors.  You should replace
each "raise RuntimeError" line with a line that performs the function specified in the
function's docstring.
'''

import numpy as np
from collections import Counter

def marginal_distribution_of_word_counts(texts, word0):
    '''
    Parameters:
    texts (list of lists) - a list of texts; each text is a list of words
    word0 (str) - the word that you want to count

    Output:
    Pmarginal (numpy array of length cX0) - Pmarginal[x0] = P(X0=x0), where
      X0 is the number of times that word0 occurs in a document
      cX0-1 is the largest value of X0 observed in the provided texts
    '''
    num_of_texts = len(texts)
    text_counts = np.zeros(num_of_texts,int)
    for idx, text in enumerate(texts):
      for word in text:
        if word == word0:
          text_counts[idx] += 1
    Pmarginal = np.zeros(np.max(text_counts) + 1 )
    for num in text_counts:
      Pmarginal[num] += 1
    Pmarginal /= num_of_texts
    return Pmarginal
    
def conditional_distribution_of_word_counts(texts, word0, word1):
    '''
    Parameters:
    texts (list of lists) - a list of texts; each text is a list of words
    word0 (str) - the first word that you want to count
    word1 (str) - the second word that you want to count

    Outputs: 
    Pcond (numpy array, shape=(cX0,cX1)) - Pcond[x0,x1] = P(X1=x1|X0=x0), where
      X0 is the number of times that word0 occurs in a document
      cX0-1 is the largest value of X0 observed in the provided texts
      X1 is the number of times that word1 occurs in a document
      cX1-1 is the largest value of X0 observed in the provided texts
      CAUTION: If P(X0=x0) is zero, then P(X1=x1|X0=x0) should be np.nan.
    '''
    num_of_texts = len(texts)
    Pcond  = np.zeros( (len(marginal_distribution_of_word_counts(texts,word0)) ,len(marginal_distribution_of_word_counts(texts,word1))  )  )

    
    for idx,text in enumerate(texts):
      count_word0 = 0
      count_word1 = 0
      for word in text:
        if word == word0:
          count_word0 += 1
        if word == word1:
          count_word1+= 1
      Pcond[count_word0,count_word1] += 1
  
    
    for i in range(Pcond.shape[0]):
      if np.sum(Pcond[i] ) != 0:
        Pcond[i] /= np.sum(Pcond[i])
      else:
        Pcond[i] = np.nan
        
      
    return Pcond

def joint_distribution_of_word_counts(Pmarginal, Pcond):
    '''
    Parameters:
    Pmarginal (numpy array of length cX0) - Pmarginal[x0] = P(X0=x0), where
    Pcond (numpy array, shape=(cX0,cX1)) - Pcond[x0,x1] = P(X1=x1|X0=x0)

    Output:
    Pjoint (numpy array, shape=(cX0,cX1)) - Pjoint[x0,x1] = P(X0=x0, X1=x1)
      X0 is the number of times that word0 occurs in a given text,
      X1 is the number of times that word1 occurs in the same text.
      CAUTION: if P(X0=x0) then P(X0=x0,X1=x1)=0, even if P(X1=x1|X0=x0)=np.nan.
    '''
    
    
    Pjoint = np.zeros(Pcond.shape)
    max_times_word0,max_times_word1 = Pcond.shape
    for idx_zero in range(max_times_word0):
      for idx_one in range(max_times_word1):
        if Pmarginal[idx_zero] == 0:
          Pjoint[idx_zero] = np.zeros(max_times_word1)
          break
        else:
          Pjoint[idx_zero,idx_one] = Pcond[idx_zero,idx_one] * Pmarginal[idx_zero]
        
    return Pjoint

def mean_vector(Pjoint):
    '''
    Parameters:
    Pjoint (numpy array, shape=(cX0,cX1)) - Pjoint[x0,x1] = P(X0=x0, X1=x1)
    Outputs:
    mu (numpy array, length 2) - the mean of the vector [X0, X1]
    '''
    mu = np.array([ 0,0 ],float)
    for idx_zero in range(Pjoint.shape[0]):
      for idx_one in range(Pjoint.shape[1]):
        mu += np.array([idx_zero,idx_one],float) * Pjoint[idx_zero,idx_one]
    
    return mu

def covariance_matrix(Pjoint, mu):
    '''
    Parameters:
    Pjoint (numpy array, shape=(cX0,cX1)) - Pjoint[x0,x1] = P(X0=x0, X1=x1)
    mu (numpy array, length 2) - the mean of the vector [X0, X1]
    
    Outputs:
    Sigma (numpy array, shape=(2,2)) - matrix of variance and covariances of [X0,X1]
    '''
    Sigma = np.zeros((2,2))
    
    
    
    for idx_zero in range(Pjoint.shape[0]):
      prob_x0 = np.sum(Pjoint[idx_zero])

      Sigma[0,0] += prob_x0 * (idx_zero - mu[0]) ** 2

      for idx_one in range(Pjoint.shape[1]):
        Sigma[0,1] += Pjoint[idx_zero,idx_one] * (idx_zero - mu[0]) * (idx_one-mu[1])
        Sigma[1,0] += Pjoint[idx_zero,idx_one] * (idx_zero - mu[0]) * (idx_one-mu[1])
        
    for idx_one in range(Pjoint.shape[1]):    
      prob_x1 = np.sum(Pjoint[:,idx_one])
      Sigma[1,1] += prob_x1 * (idx_one - mu[1] ) ** 2

    
    return Sigma

def distribution_of_a_function(Pjoint, f):
    '''
    Parameters:
    Pjoint (numpy array, shape=(cX0,cX1)) - Pjoint[x0,x1] = P(X0=x0, X1=x1)
    f (function) - f should be a function that takes two
       real-valued inputs, x0 and x1.  The output, z=f(x0,x1),
       may be any hashable value (number, string, or even a tuple).

    Output:
    Pfunc (Counter) - Pfunc[z] = P(Z=z)
       Pfunc should be a collections.defaultdict or collections.Counter, 
       so that previously unobserved values of z have a default setting
       of Pfunc[z]=0.
    '''
    Pfunc = Counter()
    for idx_zero in range(Pjoint.shape[0]):
      for idx_one in range(Pjoint.shape[1]):
        Pfunc[f(idx_zero,idx_one)] += Pjoint[idx_zero,idx_one]
      
    return Pfunc
    
