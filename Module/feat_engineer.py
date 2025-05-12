import pandas as pd
import numpy as np

### This function is used to convert time strings to a ratio of the total seconds in a day ###
def time_ratio(time_strings):
    df = pd.DataFrame(time_strings, columns=['time'])
    df['time'] = pd.to_datetime(df['time'])

    # Extract hour,minute,second
    df['hours'] = df['time'].dt.hour
    df['minutes'] = df['time'].dt.minute
    df['seconds'] = df['time'].dt.second

    df['total_seconds'] = df['hours'] * 3600 + df['minutes'] * 60 + df['seconds']
    total_seconds_in_day = 24 * 3600
    df['ratio'] = df['total_seconds'] / total_seconds_in_day
    
    print(df[['time', 'ratio']].head(5))
    return df['ratio']

### This function creates a sliding window ###
def sliding_windows(data, dt_str_array, window_size):
    x_ = []
    y_ = []

    for i in range(len(data)-window_size):
        
        time_str1, time_str2 = dt_str_array[i], dt_str_array[i + window_size]
        # Remove the second-to-last column
        label_x = data.iloc[i:(i+window_size), data.columns != data.columns[-2]].values
        # The second to last column is "output"
        label_y = data.iloc[(i+window_size), data.columns == data.columns[-2]].values

        # NaN processing
        if label_y[-1] == 0:
            continue
        x_.append(label_x)
        y_.append(label_y)

    return np.array(x_),np.array(y_)