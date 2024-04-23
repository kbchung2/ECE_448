'''
This is the module you'll submit to the autograder.

There are several function definitions, here, that raise RuntimeErrors.  You should replace
each "raise RuntimeError" line with a line that performs the function specified in the
function's docstring.
'''
import numpy as np

epsilon = 1e-3

def compute_transition(model):
    '''
    Parameters:
    model - the MDP model returned by load_MDP()

    Output:
    P - An M x N x 4 x M x N numpy array. P[r, c, a, r', c'] is the probability that the agent will move from cell (r, c) to (r', c') if it takes action a, where a is 0 (left), 1 (up), 2 (right), or 3 (down).
    '''
    M = model.M
    N = model.N
    
    
    P = np.zeros(( M,N, 4, M, N))
    
    
    for r in range(M): # through rows
        for c in range(N): # through columns
            
            if model.TS[r,c]:
                P[r,c,:,:,:] = 0
                continue
            
            # Trying to move left
            destination_row = r
            destination_column = c-1
            # Only for left
            
            up_row = r - 1
            down_row = r + 1
            right_column  = c+ 1
            left_column = c - 1

            intended_probability = model.D[r,c,0]
            counterclockwise_probability = model.D[r,c,1] # For moving left, this is down
            clockwise_probability = model.D[r,c,2] # For moving left, this is up
            
            # We move left as intended.
            if destination_column >= 0: # If the left cell is in bounds
                if model.W[r,c - 1]: # If there is a wall at (r, c-1)
                    P[r,c,0,r,c] += intended_probability # Add probability of intended direction to same place
                else:
                    P[r,c,0, destination_row, destination_column] += intended_probability # Add probability of intended direction to (r, c-1)
            else:
                P[r,c,0,r,c] += intended_probability
                
            # We accidentally move down
            if down_row < M: # If the down cell is in bounds
                if model.W[down_row, c]:
                    P[r,c,0,r,c] += counterclockwise_probability # add counterclockwise probability
                else:
                    P[r,c,0,down_row,c] += counterclockwise_probability
            else:
                P[r,c,0, r, c] += counterclockwise_probability
                
            # We accidentally move up
            if up_row >= 0 : # If up cell is in bounds
                if model.W[up_row, c]: # IF there is a wall at up cell
                    P[r,c,0,r,c] += clockwise_probability
                else:
                    P[r,c,0,up_row,c] += clockwise_probability
            else:
                P[r,c,0,r,c] += clockwise_probability
                
              
            # trying to move up
            destination_row = r - 1
            destination_column = c
            
            # We move up as intended
            if destination_row >= 0: # If in bounds
                if model.W[destination_row,destination_column]:
                    P[r,c,1,r,c] += intended_probability
                else:
                    P[r,c,1,destination_row, destination_column] += intended_probability
            else:
                P[r,c,1, r,c ] += intended_probability
            # We accidentally move left
            if left_column >= 0:
                if model.W[r,left_column]:
                    P[r,c,1,r,c] += counterclockwise_probability
                else:
                    P[r,c,1,r,left_column] += counterclockwise_probability
            else:
                P[r,c,1,r,c] += counterclockwise_probability
            
            # We accidentally move right
            if right_column < N:
                if model.W[r,right_column]:
                    P[r,c,1,r,c] += clockwise_probability
                else:
                    P[r,c,1,r,right_column] += clockwise_probability
            else:
                P[r,c,1,r,c] += clockwise_probability
            
            # trying to move right
            destination_row = r
            destination_column = c + 1
            # We move right as intended
            if destination_column < N:
                if model.W[destination_row,destination_column]:
                    P[r,c,2,r,c] += intended_probability
                else:
                    P[r,c,2,destination_row,destination_column] += intended_probability
            else:
                P[r,c,2,r,c] += intended_probability
            
            # We accidentally move up
            if up_row >= 0:
                if model.W[up_row,c]:
                    P[r,c,2,r,c] += counterclockwise_probability
                else:
                    P[r,c,2,up_row,c] += counterclockwise_probability
            else:
                P[r,c,2,r,c] += counterclockwise_probability
            # We accidentally move down
            if down_row < M:
                if model.W[down_row,c]:
                    P[r,c,2,r, c] += clockwise_probability
                else:
                    P[r,c,2,down_row,c] += clockwise_probability
            else:
                P[r,c,2,r,c] += clockwise_probability
            # trying to move down
            destination_row = r + 1
            destination_column = c
            # We move down as intended
            if destination_row < M:
                if model.W[destination_row,destination_column]:
                    P[r,c,3,r,c] += intended_probability
                else:
                    P[r,c,3,destination_row, destination_column] += intended_probability
            else:
                P[r,c,3,r,c] += intended_probability
            # We accidentally move right
            if right_column < N:
                if model.W[r, right_column]:
                    P[r,c,3,r,c] += counterclockwise_probability
                else:
                    P[r,c,3,r,right_column] += counterclockwise_probability
            else:
                P[r,c,3,r,c] += counterclockwise_probability
            
            # We accidentally move left
            if left_column >= 0:
                if model.W[r,left_column]:
                    P[r,c,3,r,c] += clockwise_probability
                else:
                    P[r,c,3,r,left_column] += clockwise_probability
            else:
                P[r,c,3,r,c] += clockwise_probability
                    
                
    return P
def compute_utility(model, U_current, P):
    '''
    Parameters:
    model - The MDP model returned by load_MDP()
    U_current - The current utility function, which is an M x N array
    P - The precomputed transition matrix returned by compute_transition()

    Output:
    U_next - The updated utility function, which is an M x N array
    '''
    M = model.M
    N = model.N
    
    reward = model.R
    U_next = np.zeros((M,N))
    for r in range(M):
        for c in range(N):
            PmultUtilList = np.array([ np.sum(P[r,c,state] * U_current)  for state in range(4)])
            U_next[r,c] = reward[r,c] + model.gamma * np.max(PmultUtilList)
    return U_next
    # raise RuntimeError("You need to write this part!")

def value_iterate(model):
    '''
    Parameters:
    model - The MDP model returned by load_MDP()

    Output:
    U - The utility function, which is an M x N array
    '''
    # raise RuntimeError("You need to write this part!")
    P = compute_transition(model)
    iterations = 100
    M = model.M
    N = model.N
    U_current = np.zeros((M,N))
    for iter in range(iterations):
        U_new = compute_utility(model,U_current,P)
        if False in (np.abs(U_new - U_current )< epsilon):
            U_current = U_new
        else:
            break
    return U_current
    

def policy_evaluation(model):
    '''
    Parameters:
    model - The MDP model returned by load_MDP();
    
    Output:
    U - The converged utility function, which is an M x N array
    '''
    raise RuntimeError("You need to write this part!")
