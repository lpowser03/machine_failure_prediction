# @Title: Data Visualization for Machine Failure dataset
# @Date: 6/25/2025, last mod: 6/25/2025
# @Author: Logan Powser
# @Abstract: MLFlow experimentation script

from machine_learning import run_experiment

if __name__ == '__main__':

    params = {'random_state': 42,
              'min_samples_split':0.01,
              }
    #3 strings, 'logistic', 'decision_tree', and 'random_forest'

    run_experiment('random_forest', params)