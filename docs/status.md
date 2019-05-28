---
layout: default
title: Status
---
<iframe width="560" height="315" src="https://www.youtube.com/embed/z6rx2-dZaIE" frameborder="0" allow="accelerometer; autoplay; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>\

## Project Summary

After the discussion with our mentor, we decide to simplify our project to make the AI more testable by having a shorter train time. Our main goal is still creating some simple maze and navigate an agent through it and extra rewards will still be implemented throughout the maze to produce more complex problems and solutions. However, instead of involving both item collection and selection AI, we decide to remove the item selection part to simplify the problem. During the process, the agent will be allowed to get out by just having the key without selecting and using it. Traps are removed, other items are now for higher rewards only. After a long time training, the AI agent should be able to know how to get out the maze, and smartly pick up some reward items for any randomly generated map with the same property.

## Approach

For the status report, we are only making minimum AI. For the minimum AI, we have a single pre-generated 5x5 map (figure 1) with some lava but no items. The agent does not need to pick the key to open the door. The AI agent should just find the shortest path to the door. Rewards map: -1 for each step, -100 for fall in lava, +100 for getting the door. 
![alt text](https://github.com/kexuejia911/avengers/blob/master/docs/figure 1.png "figure1")

We use Deep Q-learning algorithm to train the agent and build a three-layer neural net though Keras library. (figure 2) The input layer accepts 25 input as features which is our states. The states are just a list of converted map information. For example, for the block 2-3 on the map, the state for it will be (grand, 0). The first element indicates what type the block is, and the second element is a boolean value that indicates that if the agent is standing on that block. So it makes sense we have 25 elements because the map has 25 blocks. We have two hidden layers since in the article An introduction to computing with neural nets, Lippmann points out that 2 hidden layers are enough for any model. The output layer has 4 outputs and each indicates the expected Q value of that action. The agent will choose the action has the highest Q value with 85% probability and 15% probability to choose a random action. We use ‘mse’ as our loss function and ‘adam’ as our optimizer because it is the most popular one and it fits most cases. We are also doing experience replay to get a better result. We set the memory size 2000 and minimum batch size 35 by our experiment. By the above approach, our agent is allowed to learn the shortest path to the door after 4000 episodes. 

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
