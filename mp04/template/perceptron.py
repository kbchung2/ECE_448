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
    W = np.zeros(num_features)  # w vector = w1 - w2
    b = 0  # each element represents individual biases
    
    features = train_set # contains [ [x1] 
    #                                 [x2]       
    #                                 [x3]
    #                                 [x4]
    #                                   ... ]
    W += learning_rate * features[0] 
    b += learning_rate

    for iter in range(max_iter):
        for idx, x_vector in enumerate(features):
            product = np.dot(W, x_vector) + b

            correct_label = int(train_labels[idx]) # y
            predicted_label = 1 if product > 0 else 0 

            if predicted_label != correct_label:
                if predicted_label == 1:
                    W -= learning_rate * x_vector
                    b -= learning_rate
                elif predicted_label == 0:
                    W += learning_rate * x_vector
                    b += learning_rate
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
        if np.dot(W ,vector) + b < 0:
            classified.append(0)
        else:
            classified.append(1)
    return classified



