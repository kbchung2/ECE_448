# perceptron.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Justin Lizama (jlizama2@illinois.edu) on 10/27/2018
# Extended by Daniel Gonzales (dsgonza2@illinois.edu) on 3/11/2020



import numpy as np

def trainPerceptron(train_set, train_labels,  max_iter):
    
    #Write code for Mp4
    learning_rate = 0.01 #n
    num_labels = len(np.unique(train_labels) ) #len(train_labels) # v
    num_features = train_set.shape[1] # d
    print(num_labels, ", " , num_features)
    W = np.zeros( (num_labels, num_features)) # row vectors represent weight vectors
    b = np.zeros(num_labels) # each element represents individual biases
    
    features = train_set # contains [ [x1] 
    #                                 [x2]       
    #                                 [x3]
    #                                 [x4]
    #                                   ... ]
   
    for iter in range(max_iter):
        W[0] += learning_rate * features[0] 
        b[0] += learning_rate * features[0,0]
        for idx, x_vector in enumerate(features):
            if idx == 0: 
                continue
            product_vector = W @ x_vector + b
            yhat = np.argmax(product_vector)
            W[yhat] -= learning_rate * x_vector
            correct_label = int(train_labels[idx])
            W[  correct_label] += learning_rate * x_vector
            b[yhat] -= learning_rate
            b[correct_label] += learning_rate





    return W, b

def classifyPerceptron(train_set, train_labels, dev_set, max_iter):
    #Write code for Mp4
    # print(train_set)
    # print(train_labels)
    print(train_set.shape)
    print(len(train_labels))
    
    
    W,b = trainPerceptron(train_set,train_labels,max_iter)
    print(W)
    print(b)
    
    classified = []
    for vector in dev_set:
        if np.sum(W @ vector + b) < 0:
            classified.append(0)
        else:
            classified.append(1)
    return classified



