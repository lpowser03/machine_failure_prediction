# @Title: Data Exploration for Machine Failure dataset
# @Date: 6/18/2025, last mod: 6/19/2025
# @Author: Logan Powser
# @Abstract: Data exploration for Machine Failure dataset from Kaggle

from data_viz import corr_map, plot_distr, scatter
import pandas as pd
import seaborn as sns

sns.set_theme('notebook', 'whitegrid')

def explore(df:pd.DataFrame, name='Machine Failure Dataset'):
    corr_map(df, name)
    plot_distr(df, name)
    scatter(df, 'AQ', 'fail')
    scatter(df, 'USS', 'fail')
    scatter(df, 'VOC', 'fail')
    scatter(df, 'CS', 'USS')
    scatter(df, 'RP', 'tempMode')
    scatter(df, 'IP', 'Temperature')
    #pass

if __name__ == '__main__':
    df = pd.read_csv('data/data.csv')
    # print(df[['footfall','tempMode','AQ']].describe())
    # print(df[['USS','CS','VOC']].describe())
    # print(df[['RP','IP', 'Temperature', 'fail']].describe())
    explore(df)
