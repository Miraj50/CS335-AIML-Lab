from __future__ import division
import sys, matplotlib.pyplot as plt

plt.plot([i/10 for i in range(0,11)], map(int, sys.argv[1:]))
plt.xlabel('Probability -->')
plt.ylabel('Number of Actions -->')
plt.savefig('plot.png')
plt.show()
