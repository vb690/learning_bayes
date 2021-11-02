# Thompson Sampling for Adaptive AB Testing

# Motivation
This repository is my attempt to use PyMC3 for implementing Thompson Sampling and testing its efficiency through simulation.

# Features

* Jupyter notebook for comparing adaptive resources allocation through thompson sampling to conventional 50-50 split in a 2 arms AB test scenario.

# Data
The data are sympulated by drawing at random from 2 binomial distributions with fixed parameter `n=2000` and `p` changing over time.

<p align="center">
  <img width="350" height="350" src="https://github.com/vb690/learning_bayes/blob/main/examples/thompson_sampling/results/1.png">
<p align="center">
  
# Results
  
## Estimated Ratio
  
Thompson Sampling             |  50-50 Split
:-------------------------:|:-------------------------:
![](https://github.com/vb690/learning_bayes/blob/main/examples/thompson_sampling/results/2.png)  |  ![](https://github.com/vb690/learning_bayes/blob/main/examples/thompson_sampling/results/4.png)


## Allocation and Conversions
  
Thompson Sampling             |  50-50 Split
:-------------------------:|:-------------------------:
![](https://github.com/vb690/learning_bayes/blob/main/examples/thompson_sampling/results/3.png)  |  ![](https://github.com/vb690/learning_bayes/blob/main/examples/thompson_sampling/results/5.png)
  
## Perfromance
<p align="center">
  <img width="500" height="400" src="https://github.com/vb690/learning_bayes/blob/main/examples/thompson_sampling/results/6.png">
<p align="center">
  


