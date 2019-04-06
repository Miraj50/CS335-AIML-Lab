import util
from sudoku import SudokuSearchProblem
from maps import MapSearchProblem

################ Node structure to use for the search algorithm ################
class Node:
    def __init__(self, state, action, path_cost, parent_node, depth):
        self.state = state
        self.action = action
        self.path_cost = path_cost
        self.parent_node = parent_node
        self.depth = depth

########################## DFS for Sudoku ########################
## Choose some node to expand from the frontier with Stack like implementation
def sudokuDepthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.
    Return the final values dictionary, i.e. the values dictionary which is the goal state  
    """

    def convertStateToHash(values):
        """ 
        values as a dictionary is not hashable and hence cannot be used directly in the explored/visited set.
        This function changes values dict into a unique hashable string which can be used in the explored set.
        You may or may not use this
        """
        l = list(sorted(values.items()))
        modl = [a+b for (a, b) in l]
        return ''.join(modl)

    ## YOUR CODE HERE
    lh = set([])
    root = problem.getStartState()
    s = util.Stack()
    s.push(root)
    while True:
        state = s.pop()
        lh.add(convertStateToHash(state))
        if problem.isGoalState(state):
            break
        children = problem.getSuccessors(state)
        for child in children:
            if child not in lh:
                s.push(child[0])
    return state
    # util.raiseNotDefined()

######################## A-Star and DFS for Map Problem ########################
## Choose some node to expand from the frontier with priority_queue like implementation

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def heuristic(state, problem):
    # It would take a while for Flat Earther's to get accustomed to this paradigm
    # but hang in there.

    """
        Takes the state and the problem as input and returns the heuristic for the state
        Returns a real number(Float)
    """
    return util.points2distance(((problem.G.node[state]["x"],0,0),(problem.G.node[state]["y"],0,0)), ((problem.G.node[problem.end_node]["x"],0,0),(problem.G.node[problem.end_node]["y"],0,0)))
    # util.raiseNotDefined()

def AStar_search(problem, heuristic=nullHeuristic):

    """
        Search the node that has the lowest combined cost and heuristic first.
        Return the route as a list of nodes(Int) iterated through starting from the first to the final.
    """
    q = util.PriorityQueue()
    explored = set([])
    depth = 0
    cost = 0
    root = problem.getStartState()
    node = Node(root, cost, 0, None, 0) # state, action, path_cost, parent_node, depth
    q.push(node, heuristic(root, problem))
    while True:
        nd = q.pop()
        cost = nd.action + nd.path_cost
        if nd.state in explored:
            continue
        explored.add(nd.state)
        if problem.isGoalState(nd.state):
            goal = nd
            break
        depth += 1
        children = problem.getSuccessors(nd.state)
        for child in children:
            if child[0] not in explored:
                q.push(Node(child[0], cost, child[2], nd, depth), cost+child[2]+heuristic(child[0], problem))

    path = [goal.state]
    prev = goal.parent_node
    while True:
        if prev.state == root:
            path.append(root)
            break
        else:
            path.append(prev.state)
        prev = prev.parent_node
    return path[::-1]
    # util.raiseNotDefined()