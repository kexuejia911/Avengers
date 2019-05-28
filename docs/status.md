---
layout: default
title: Status
---
<iframe width="560" height="315" src="https://www.youtube.com/embed/z6rx2-dZaIE" frameborder="0" allow="accelerometer; autoplay; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>\

## Project Summary

After the discussion with our mentor, we decide to simplify our project to make the AI more testable by having a shorter train time. Our main goal is still creating some simple maze and navigate an agent through it and extra rewards will still be implemented throughout the maze to produce more complex problems and solutions. However, instead of involving both item collection and selection AI, we decide to remove the item selection part to simplify the problem. During the process, the agent will be allowed to get out by just having the key without selecting and using it. Traps are removed, other items are now for higher rewards only. After a long time training, the AI agent should be able to know how to get out the maze, and smartly pick up some reward items for any randomly generated map with the same property.

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
