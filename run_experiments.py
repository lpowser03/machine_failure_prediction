# @Title: Run Experiments
# @Date: 6/25/2025, last mod: 6/25/2025
# @Author: Logan Powser
# @Abstract: MLFlow experimentation script

from machine_learning import run_experiment, final_testing

if __name__ == '__main__':

    params = {
            'max_iter':1000
            #'random_state':42,
            #'min_samples_split':0.01,
            }
    #3 model strings: 'logistic', 'decision_tree', and 'random_forest'
    #5 feature set strings: "base", 'operational_modes', 'safety_indicators', 'temperature_interactions', 'electrical_system'
    model, score, test_data = run_experiment('logistic', params)
    metrics, experiment = final_testing(model, test_data[0], test_data[1])

    print(f'Run ID: {experiment.info.run_id}')


    # save model to Docker

    # run model in Docker
    # load model from Docker
    # have model wait for a prediction request through Flask
    # send prediction request to model
    # return prediction result
    # based on integer value of prediction request, return appropriate dataset prediction response
    # seeds to choose from are 42, 0, 1, 123, 456