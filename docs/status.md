---
layout: default
title: Status
---
## Project Summary

For our project, we create some simple maze and navigate an agent through it. Rewards will be implemented throughout the maze to produce more complex problems and solutions. So we plan to involve a smart item collection AI. For the running process, no user input is needed for the agent. The agent should find a way to the destination automatically. During the process, the agent should also collect some items smartly on the path to ensure it can arrive the destination with the balance time, items collected.Right now our agent is able to solve a single maze with lava. We build a maze randomly with a start point and an end point, and the agent is able to learn and find out the best way from the start point to the end point. I believe we will finish the rest functions in the final report. 

## Approach

Deep Q-learning by using keras to build a three layer neural net with 25 input as features which is our states and 25 is the actual map size. We have two hidden layers since An introduction to computing with neural nets by Lippmann points out that 2 hidden layers are enough for any model. Output layer has output size 4 since now we just have 4 actions. Using ‘mse’ as loss function optimizer and ‘adam’ since it’s most popular and fits most cases. We are doing experience replay to get better result. A 2000 spot memory to memory past result. After each iteration, we randomly reply 35 experience to retrain the neural network. Rewards : -1 for each step, -100 for fall in lava, +100 for get the door.

## Evaluation

Quantitative Evaluation:
Numerical Metrics: Time to process, number of items collected, number of items used, and the total score. The time to train the AI may also be analyzed.
Baselines: The agent should find the destination and generate the shortest path correctly and quickly without involving item collection and selection AI.

Qualitative Evaluation:
Simple Example: In a maze that does not contain any item, the agent should simply use the shortest path to reach the destination.
Super-Impressive Example: In a complex maze that involves traps and items, the agent should reach the highest score. In other words, it smartly gives up some item to save time, uses some item to remove some traps to save time, and bypass some traps to save items to get the perfect result.

## Remaining goals and challenges

In the rest two weeks, we are going to involve items since we can only solve a single maze now. We still need to add some item collection functions and train our model. Secondly, we need to build a more complex map, because we can only build a small map with size of 25. Last, our agent takes too long to learn, we need to simplify our code and decrease the complexity of our functions.

## Resources Used
- https://machinelearningmastery.com/how-to-define-your-machine-learning-problem/
- https://datascience.stackexchange.com/questions/20535/what-is-experience-replay-and-what-are-its-benefits
- https://web.stanford.edu/class/psych209/Readings/MnihEtAlHassibis15NatureControlDeepRL.pdf
- https://keras.io/models/sequential/#sequential-model-methods

<iframe width="560" height="315" src="https://www.youtube.com/embed/z6rx2-dZaIE" frameborder="0" allow="accelerometer; autoplay; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
