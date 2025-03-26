import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def numericalStats(df: pd.DataFrame) -> pd.DataFrame:
    stats = {
        'mean': df.mean(),
        'median': df.median(),
        'min': df.min(),
        'max': df.max(),
        'std': df.std(),
        '5th_percentile': df.quantile(0.05),
        '95th_percentile': df.quantile(0.95),
        'missing_values': df.isnull().sum()
    }
    return pd.DataFrame(stats)

def categoricalStats(df: pd.DataFrame) -> pd.DataFrame:
    stats = {
        'unique_classes': df.nunique(),
        'missing_values': df.isnull().sum(),
        'class_proportions': df.apply(lambda x: x.value_counts(normalize=True).to_dict())
    }
    return pd.DataFrame(stats)


if __name__ == "__main__":
    if (os.path.exists(".\\data\\datasetClean.csv")):
        dataf: pd.DataFrame = pd.read_csv(".\\data\\datasetClean.csv")
    else:
        print("First run loadData.py to load the dataset.")
        exit()

    if (not os.path.exists(".\\results")):
        os.mkdir(".\\results")

    numerical_columns = dataf.select_dtypes(include=[np.number]).columns
    numerical_columns = numerical_columns.drop('Application order')
    numerical_stats_df = numericalStats(dataf[numerical_columns])

    categorical_columns = dataf.select_dtypes(include=['object', 'category']).columns.tolist()
    categorical_columns.append("Application order")
    categorical_stats_df = categoricalStats(dataf[categorical_columns])

    numerical_stats_df.to_csv('.\\results\\numerical_stats.csv')
    categorical_stats_df.to_csv('.\\results\\categorical_stats.csv')