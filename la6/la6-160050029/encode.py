import sys

p = 1
if len(sys.argv) == 3:
	p = float(sys.argv[2])

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
			statemaze[i][j] = numState
			startState = numState
			numState+=1
		elif maze[i][j] == 3: # End state
			statemaze[i][j] = numState
			endStates.append(numState)
			numState+=1

print "numStates", numState
print "numActions", 4
print "start", startState
print "end", " ".join(map(str, endStates))

if p==1:
	for i in range(1,y-1):
		for j in range(1,x-1):
			state = statemaze[i][j]
			if state != -1 and state not in endStates:
				no = statemaze[i-1][j]
				ea = statemaze[i][j+1]
				so = statemaze[i+1][j]
				we = statemaze[i][j-1]
				if no == -1: #North
					print "transition", state, 0, state, 0, 1
				elif no in endStates:
					print "transition", state, 0, no, 1, 1
				else:
					print "transition", state, 0, no, 0, 1
				if ea == -1: #East
					print "transition", state, 2, state, 0, 1
				elif ea in endStates:
					print "transition", state, 2, ea, 1, 1
				else:
					print "transition", state, 2, ea, 0, 1
				if so == -1: #South
					print "transition", state, 3, state, 0, 1
				elif so in endStates:
					print "transition", state, 3, so, 1, 1
				else:
					print "transition", state, 3, so, 0, 1
				if we == -1: #West
					print "transition", state, 1, state, 0, 1
				elif we in endStates:
					print "transition", state, 1, we, 1, 1
				else:
					print "transition", state, 1, we, 0, 1
else:
	for i in range(1,y-1):
		for j in range(1,x-1):
			state = statemaze[i][j]
			if state != -1 and state not in endStates:
				no = statemaze[i-1][j]
				ea = statemaze[i][j+1]
				so = statemaze[i+1][j]
				we = statemaze[i][j-1]
				t = [no, we, ea, so]
				n_valdir = 4-t.count(-1)
				pr_ic = (1-p)*1.0/n_valdir
				pr_c = p+pr_ic
				for ac,st in enumerate(t):
					if st == -1:
						print "transition", state, ac, state, 0, 1
					elif st in endStates:
						for k in t:
							if k==st:
								print "transition", state, ac, st, 1, pr_c
							elif k!=-1:
								print "transition", state, ac, k, 0, pr_ic
					else:
						for k in t:
							if k==st:
								print "transition", state, ac, st, 0, pr_c
							elif k!=-1:
								print "transition", state, ac, k, 0, pr_ic

print "discount", 0.9