# Python implementation for "*Nearly Minimax Optimal Regret for Multinomial Logistic Bandit"*

The Python implementation for the experiments described in the paper *"Nearly Minimax Optimal Regret for Multinomial Logistic Bandit."*

## Files

We first introduce the necessary files.

* **Uniform Rewards**: Implementations for the uniform reward setting.
  * `main_uniform.py` is the code for executing the simulation environments under uniform rewards.
  * `models_uniform.py` contains algorithms for uniform rewards.

* **Nonuniform Rewards**: Implementations for the non-uniform reward setting.
  * `main_non_uniform.py` is the code for running the simulation experiments under non-uniform rewards.
  * `models_non_uniform.py` contains algorithms for non-uniform rewards.

## Requirements

We and list the required dependencies.

* matplotlib==3.5.2
* numpy==1.20.1
* tdqm==0.0.1
* seaborn==0.11.2
* scipy==1.7.3
* PuLP==2.8.0

## Experiments

We provide scripts for running experiments for both uniform and non-uniform rewards cases. Run `main_uniform.py` for the uniform rewards case and `main_non_uniform.py` for non-uniform rewards case.

