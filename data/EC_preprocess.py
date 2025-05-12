import pandas as pd
import numpy as np

def ec_pre(ecpath):
    # read csv
    df = pd.read_csv(ecpath)
    print(df)

    # calculate wind speed & wind direction
    df['ws'] = np.sqrt(df['u10']**2 + df['v10']**2)
    df['wd'] = (np.arctan2(df['u10'], df['v10']) * 180 / np.pi + 360) % 360

    df = df.loc[:,['time_index', 'ssr','ws', 'wd']]
    df.rename(columns = {'ssr':'Surface Irradiance'}, inplace = True)
    df.drop_duplicates(inplace = True)
    print(df)
    
    # save
    df.to_csv(ecpath, index=False)