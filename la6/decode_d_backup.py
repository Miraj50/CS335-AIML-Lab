import sys, random

p = 1
if len(sys.argv) == 4:
	p = float(sys.argv[3])

random.seed(0)

def stochastic(cur_ac, x, y):
	no = statemaze[x-1][y]
	ea = statemaze[x][y+1]
	so = statemaze[x+1][y]
	we = statemaze[x][y-1]
	valid = [i for i,j in enumerate([no, we, ea, so]) if j!=-1]
	n_valdir = len(valid)
	pr_ic = (1-p)*1.0/n_valdir
	pr_c = p+pr_ic
	r = random.random()
	if r<=pr_c:
		return cur_ac
	else:
		return random.choice([i for i in valid if i!=cur_ac])

with open(sys.argv[1], "r") as f:
	maze = [map(int, line.strip().split()) for line in f]
y,x = len(maze), len(maze[0])
statemaze = [[-1 for j in range(x)] for i in range(y)]
numState = 0
endStates = []
for i in range(y):
	for j in range(x):
		if maze[i][j]==0:
			statemaze[i][j] = numState
			numState+=1
		elif maze[i][j] == 2: # Start state
			p_x,p_y = i,j
			statemaze[i][j] = numState
			startState = numState
			numState+=1
		elif maze[i][j] == 3: # End state
			statemaze[i][j] = numState
			endStates.append(numState)
			numState+=1

with open(sys.argv[2], "r") as f:
	policy = [int(line.strip().split()[1]) for line in f if line.strip().split()[0] != "iterations"]

s = startState
while s not in endStates:
	if p == 1:
		ac = policy[s]
	else:
		ac = stochastic(policy[s], p_x, p_y)
	if ac == 0:
		p_x = p_x-1
		print "N",
	elif ac == 2:
		p_y = p_y+1
		print "E",
	elif ac == 3:
		p_x = p_x+1
		print "S",
	elif ac == 1:
		p_y = p_y-1
		print "W",
	s = statemaze[p_x][p_y]