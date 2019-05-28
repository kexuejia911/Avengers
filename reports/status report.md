##Project summary: I believe you can do this.

Approach: Deep Q-learning by using keras to build a three layer neural net with 25 input as features which is our states and 25 is the actual map size. We have two hidden layer since An introduction to computing with neural nets by Lippmann point out that 2 hidden layer are enough for any model. Output layer has output size 4 since now we just have 4 actions. Using ‘mse’ as loss function optimizer and ‘adam’ since it’s most popular and fits most cases. We are doing experience replay to get better result. A 2000 spot memory to memory past result. After each iteration, we random reply 35 experience to retrain the neural network. Rewards : -1 for each step, -100 for fall in lava, +100 for get the door.

Evaluation: see the proposal

Remaining goals and challenges: involving items, more complex map, agent takes too long to learn.
Used: https://machinelearningmastery.com/how-to-define-your-machine-learning-problem/,
https://datascience.stackexchange.com/questions/20535/what-is-experience-replay-and-what-are-its-benefits,
http://web.stanford.edu/class/psych209/Readings/MnihEtAlHassibis15NatureControlDeepRL.pdf,https://keras.io/models/sequential/#sequential-model-methods

Video Link:https://youtu.be/z6rx2-dZaIE
