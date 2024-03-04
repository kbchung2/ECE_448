# submitted.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Kelvin Ma (kelvinm2@illinois.edu) on 01/24/2021

"""
This is the main entry point for MP5. You should only modify code
within this file -- the unrevised staff files will be used for all other
files and classes when code is run, so be careful to not modify anything else.
"""
# submitted should return the path.
# The path should be a list of tuples in the form (row, col) that correspond
# to the positions of the path taken by your search algorithm.
# maze is a Maze object based on the maze from the file specified by input filename
# searchMethod is the search method specified by --method flag (bfs,dfs,astar,astar_multi)

import queue
# import numpy as np
def bfs(maze):
    """
    Runs BFS for part 1 of the assignment.

    @param maze: The maze to execute the search on.

    @return path: a list of tuples containing the coordinates of each state in the computed path
    """
    
    #TODO: Implement bfs function
    queue_ = queue.Queue()
    set_ = set()
    start_cell = maze.start
    queue_.put(start_cell)
    current_cell = maze.start
    end = maze.waypoints[0]
    path = []
    backtrace = {} # Would this work? 
    while ( queue_._qsize() != 0 ) : 
        current_cell = queue_.get()
        
        if ( (current_cell[0] == end[0] and current_cell[1] == end[1]) ):
            # Do the backtracing    
            while (not (current_cell[0] == start_cell[0] and current_cell[1] == start_cell[1])):
                path.insert(0,current_cell)
                current_cell = backtrace[current_cell]
            path.insert(0,start_cell)
            return path
          
        valid_neighbors = maze.neighbors_all(current_cell[0],current_cell[1])
        set_.add(current_cell)
        # print("Current Cell: " ,current_cell, "Current Cell's valid neighbors: ", valid_neighbors, "Set: ", set_)
        for idx, neighbor in enumerate(valid_neighbors):
            if neighbor not in set_:    
                backtrace[neighbor] = current_cell
                queue_.put(neighbor)
                set_.add(neighbor)
            
    
    

    return []

def astar_single(maze):
    """
    Runs A star for part 2 of the assignment.

    @param maze: The maze to execute the search on.

    @return path: a list of tuples containing the coordinates of each state in the computed path
    """
    #TODO: Implement astar_single
    path = []
    backtrace = {}
    
    start_cell = maze.start
    end = maze.waypoints[0]
    start_row = start_cell[0]
    start_col = start_cell[1]
    open = queue.PriorityQueue() # Format of this queue is f, row, column
    closed = set() # Format for closed is row, column
    open.put( (0, start_cell[0],start_cell[1] ) ) # f, row, column
    # fgh_mapping = {} # (r,c ) -> [f,g,h]
    
    
    
    fgh_mapping = [  [ [float("inf"),float("inf"),0] for c in range(maze.size.x) ] for r in range(maze.size.y) ] # fgh_mapping[r,c][i] gives you f, g, h at r,c depending on i
    fgh_mapping[start_row][start_col][0] = 0
    fgh_mapping[start_row][start_col][1] = 0
    fgh_mapping[start_row][start_col][2] = 0

    # fgh_mapping[(start_row,start_col)] = [np.inf,np.inf,0]
    
    while open._qsize() > 0:
        q = open.get()
        cur_row = q[1]
        cur_col = q[2]
        closed.add( (q[1],q[2])   )
        valid_neighbors = maze.neighbors_all(q[1],q[2])
        for neighbor in valid_neighbors:
            if neighbor not in closed:
                if neighbor[0] == end[0] and neighbor[1] == end[1]:
                    backtrace[neighbor] = (q[1],q[2]) # Now perform backtracing
                    current_cell = neighbor
                    while not (current_cell[0] == start_cell[0] and current_cell[1] == start_cell[1]):
                        path.insert(0, current_cell)
                        current_cell = backtrace[current_cell]
                    path.insert(0, (start_cell[0],start_cell[1]))
                    return path
                # g_hat = fgh_mapping[(cur_row,cur_col)][1] + 1
                g_hat = fgh_mapping[cur_row][cur_col][1] + 1
                h_hat = max(abs(end[0] - neighbor[0] ) , abs(end[1] - neighbor[1]) )
                f_hat = g_hat + h_hat
                if f_hat <  fgh_mapping[neighbor[0]][neighbor[1]][0]:
                    open.put( (f_hat, neighbor[0], neighbor[1])  )
                    # fgh_mapping[(neighbor[0],neighbor[1])][0] = f_hat
                    # fgh_mapping[(neighbor[0],neighbor[1])][1] = g_hat
                    # fgh_mapping[(neighbor[0],neighbor[1])][2] = h_hat
                    fgh_mapping[neighbor[0]][neighbor[1]][0] = f_hat
                    fgh_mapping[neighbor[0]][neighbor[1]][1] = g_hat
                    fgh_mapping[neighbor[0]][neighbor[1]][2] = h_hat
                    backtrace[neighbor] = (q[1],q[2])
        
    
    return []

# This function is for Extra Credits, please begin this part after finishing previous two functions
def astar_multiple(maze):
    """
    Runs A star for part 3 of the assignment in the case where there are
    multiple objectives.

    @param maze: The maze to execute the search on.

    @return path: a list of tuples containing the coordinates of each state in the computed path
    """
    
    

    return []
