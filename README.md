# README

## Introduction

Genetic algorithms are optimization techniques inspired by the principles of natural selection and evolution. They work by evolving a population of potential solutions through processes like selection, crossover, and mutation to find increasingly better solutions to a given problem.

The OneMax problem is a simple benchmark problem in genetic algorithms where the goal is to maximize the number of ones in a binary string. It serves as a straightforward test case for evaluating the performance and behavior of genetic algorithms in a controlled environment.

Reading the first pages of `article.pdf` will allow you to understand very well this two concepts, in this article we have done important analysis on genetic algorithms. 

## Requirements

- `python version`: 3.10.0
- `deap version`: 1.4.1
- `numpy`: 1.26.4

## Files

### `oneMaxSteadyState.py`

Allows for various experiments where you can choose to study either the effects of mutation, the impact of population size, the influence of crossover, or selection mechanisms on the evolution of solutions. The main function main() initializes a configuration and an experiment object, then calls the method corresponding to the desired experiment, with mutation currently activated and other types of experiments commented out.

### `OneMaxEstimateDistrib.py`

Implements an experiment on a distribution estimation algorithm, where different configurations are tested by varying the number of "k-best" individuals (2, 4, 8, 10, and 14) that are selected as parents in each generation.

### `OneMaxCompactAlgo.py`

Presents an experiment on the compact genetic algorithm (cGA) where two different configurations of the learning rate α are tested: one with α = 1 over 7000 generations, and another with α = 2 over 3500 generations to compare their impact on convergence.

### `OneMaxAptativeRoulette.py`

Presents an experiment on the adaptive roulette wheel that compares the effectiveness of different mutation operators (1-flip, 3-flip, 5-flip, bit-flip) in a genetic algorithm, where the selection probability of each operator is dynamically adjusted according to their performance. Specifically, two configurations are tested and visualized: one that compares the usage rates of different operators over generations, and another that compares performance between an adaptive roulette and a fixed roulette using only bit-flip. Here, you can also experiment with the mask and the leading ones problem (see the main where you need to follow the comments to launch these experiments).

### `OneMaxUcb.py`

Presents an experiment on the UCB (Upper Confidence Bound) algorithm applied to the adaptive selection of mutation operators in a genetic algorithm, with several tests: a first comparison between different mutation operators (bit-flip, 1-flip, 3-flips, 5-flips) where we observe their usage rate, then a test with a useless operator (identity) to validate the effectiveness of the method, and finally tests on variants of the OneMax problem (with mask) and on the LeadingOnes problem to evaluate the robustness of the approach (always follow the comments in the main to launch these experiments).

### `island` Folder

Contains the implementation of the island model. Simply run the `main.py` file to launch the experiment.

## possible improvement ?
Code refacto.
