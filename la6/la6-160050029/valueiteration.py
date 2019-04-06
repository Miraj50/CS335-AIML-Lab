import sys

with open(sys.argv[1], "r") as f:
	nS = int(f.readline().strip().split()[1])
	nA = int(f.readline().strip().split()[1])
	S = int(f.readline().strip().split()[1])
	E = set(map(int, f.readline().strip().split()[1:]))

	md = [[[] for _ in range(nA)] for _ in range(nS)]
	for line in f:
		t = line.strip().split()
		if t[0] == "discount":
			G = float(t[1])
		else:
			p = float(t[5])
			if p != 0:
				md[int(t[1])][int(t[2])].append((int(t[3]), float(t[4]), p))

V = [0.0 for _ in range(nS)]
PI = [-1 for _ in range(nS)]
t = 0 # Iterations Count
while True:
	t = t+1
	VO = V[:]
	for i in range(nS): # For each State

		# m = max()	

		m = float("-inf")
		for j in range(nA): # For each Action
			c = 0
			for k in md[i][j]: # For each transition for an action
				c = c+k[2]*(k[1] + G*V[k[0]])
			if c>m:
				m=c
				pi = j
		if i not in E:
			PI[i] = pi
			V[i] = m

	l1_n = max(map(lambda x,y:abs(x-y), VO, V))
	if l1_n <= 10**-16:
		break

for i in range(nS):
	print V[i], PI[i]
print "iterations ", t