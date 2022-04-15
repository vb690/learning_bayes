# Estimating 2020 COVID-19 Deaths Excess <br/> Using Simple Exponential Smoothing

<p align="center">
  <img width="600" height="300" src="https://github.com/vb690/learning_bayes/blob/main/examples/covid_exponential_smoothing/results/moving_average.png">
<p align="center">

## Motivation
The best way for having a rough idea of the impact of COVID-19 pandemic is to look at the excess of deaths that it caused. This is a non-trivial task as it requires to estimate a counterfactual: how many deaths would we observe if the COVID-19 pandemic never occoured (see [this publication](https://projecteuclid.org/journals/annals-of-applied-statistics/volume-9/issue-1/Inferring-causal-impact-using-Bayesian-structural-time-series-models/10.1214/14-AOAS788.pdf) for a more sophisticated approach to the problem).  
  
Here we take a simplicisitc approach to the problem: we fit a time series model (i.e. [simple exponential smoothing](https://otexts.com/fpp3/ses.html)) to yearly deaths data in UK from 1990 to 2019. We then perfom prediction from the year 2020 and compute the differential with the observed data for the same year.  
  
Given the over-simplicistic assumptions of this approach, we develop our model within a bayeasian framework. The hope is that through uncertainity quantification we are able to absorb some of the effect caused by a sub-optimal modelling approach.

## Features
The nootebook implements simple exponential smoothing modelling yearly UK deaths as a weighted average of the past five years. After performing prediction for the year 2020, it then computes and visualize the distribution of differences between the predicted and the observed number of deaths.

## Data
  
The data comes from the [UK Office for National Statistics](https://www.ons.gov.uk/aboutus/transparencyandgovernance/freedomofinformationfoi/deathsintheukfrom1990to2020).

## Results
    
<p align="center">
  <img width="900" height="300" src="https://github.com/vb690/learning_bayes/blob/main/examples/covid_exponential_smoothing/results/exp_smooth.png">
<p align="center">
    


