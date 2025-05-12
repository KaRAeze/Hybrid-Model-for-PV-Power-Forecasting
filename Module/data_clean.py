# Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split

# SVM data clean
def data_filter_svr(df, sol_col='Surface_irradiance', pwr_col='Value', n=80):
    print('original shape', df.shape)
    df = df[(df[sol_col] < 1200) & (df[sol_col] >= 0)]
    print('0--1200', df.shape)

    # Since the wind direction data is a multi-digit floating-point number, 
    # if duplicated, it is filled data with missing values and needs to be removed
    df = df.drop_duplicates(subset=['wind_speed']) 
    print('Remove the repetitive wind direction', df.shape)
    indexNames = df[(df[sol_col] < 1) & (df[pwr_col] > 5)].index
    df.drop(indexNames, inplace=True)  
    print('sol0', df.shape)
    indexNames = df[(df[sol_col] > 50) & (df[pwr_col] < 0.3)].index
    df.drop(indexNames, inplace=True)  
    print('pwr0', df.shape)
    dd = df
    plt.scatter(dd[sol_col], dd[pwr_col], s=0.8)
    plt.xlabel('ssr')
    plt.ylabel(pwr_col)
    plt.title('Filtered Data')
    plt.show()

    model_out = SVR()
    dd = dd.dropna()
    X = dd[[pwr_col]].values
    y = dd[sol_col].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
    model_out.fit(X_train, y_train.ravel())
    y_pred = model_out.predict(X)
    dd[sol_col + '_filter'] = y_pred
    df_save = dd
    df_save = dd[np.abs(dd[sol_col] - dd[sol_col + '_filter']) < n].drop(columns=[sol_col + '_filter'])
    print('SVM', df_save.shape)
    return df_save

def plot_data_before_after(df_before, df_after, sol_col='Surface_irradiance', pwr_col='Value'):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # befor cleaning
    axes[0].scatter(df_before[sol_col], df_before[pwr_col], s=0.8)
    axes[0].set_xlabel('ssr')
    axes[0].set_ylabel(pwr_col)
    axes[0].set_title('Before Filtering')
    
    # after cleaning
    axes[1].scatter(df_after[sol_col], df_after[pwr_col], s=0.8)
    axes[1].set_xlabel('ssr')
    axes[1].set_ylabel(pwr_col)
    axes[1].set_title('After Filtering')
    
    plt.show()