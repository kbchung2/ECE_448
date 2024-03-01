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
