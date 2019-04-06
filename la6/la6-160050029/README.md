I have made a final script named **run.sh**, which runs each of the encoder, decoder and the valueiteration scripts one by one to produce the final answer.

The various ways in which it can be run is :

$ ./run.sh gridfile
==> This is equivalent to deterministic maze.

$ ./run.sh gridfile 0.2
==> This is equivalent to stochastic maze (0.2 is the probability, so it can range from 0 to 1).

$ ./run.sh gridfile 0 p
==> This plots the graph between number of actions and probabilities. Note that 0 is just there as a placeholder (can be anything, not useful) and the alphabet p here means to plot, not the decimal probability p. This command uses the plot.py file to plot the graph.