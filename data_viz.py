# @Title: Data Visualization for Machine Failure dataset
# @Date: 6/18/2025, last mod: 6/19/2025
# @Author: Logan Powser
# @Abstract: Data visualization functions for Machine Failure dataset from Kaggle

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.core.pylabtools import figsize


def corr_map(df:pd.DataFrame, name='Machine Failure Dataset', save=False):
    plt.figure(figsize=(12,10))
    sns.heatmap(df.corr(), annot=True, cmap='inferno')
    plt.title(f'Correlation Matrix of {name}')
    plt.tight_layout()
    if save: plt.savefig(f'graphs/{name}_corr_map.png')
    plt.show()

def plot_distr(df:pd.DataFrame, name='Machine Failure Dataset', save=False):
    for col in df.columns:
        df[col].plot.hist()
        plt.title(f'Distribution of {col}')
        plt.xlabel(f'{col}')
        plt.tight_layout()
        if save: plt.savefig(f'graphs/{name}_{col}_distr.png')
        plt.show()

def scatter(df: pd.DataFrame, x:str, y:str, save=False) -> None:
  """
  Plotting variables against each other
  Args:
    df (pd.DataFrame): DataFrame containing subject data.
    x (str): Variable to plot on x-axis.
    y (str): Variable to plot on y-axis.
    save (bool, optional): Whether to save the plot. Defaults to False.
  """

  sns.scatterplot(df, x=x, y=y, alpha=0.2)
  plt.title(f'{x} vs. {y}')
  plt.xlabel(f'{x}')
  plt.ylabel(f'{y}')
  plt.tight_layout()
  if save: plt.savefig(f'graphs/{x}_vs_{y}.png')
  plt.show()