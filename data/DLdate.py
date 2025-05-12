import pandas as pd

def DLdata(path):
    file_path = path  
    df = pd.read_csv(file_path)
    
    if 'Date' in df.columns:
        # time format converted
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values(by='Date')
        # get the unique value for date
        unique_dates = df['Date'].dt.date.unique()
        df1 = pd.DataFrame()
        df2 = pd.DataFrame()
        # iteration
        for i in range(0, len(unique_dates), 10):
            # get data from current 10 days
            current_dates = unique_dates[i:i+10]
            # less than 10 days -> add to df1
            if len(current_dates) < 10:
                df1 = pd.concat([df1, df[df['Date'].dt.date.isin(current_dates)]])
                continue
            
            # get data from current 10 days
            current_data = df[df['Date'].dt.date.isin(current_dates)]
            # get data from first 7 days
            first_seven_days = current_data[current_data['Date'].dt.date.isin(current_dates[:7])]
            # get data from last 3 days
            last_three_days = current_data[current_data['Date'].dt.date.isin(current_dates[7:])]
            # save
            df1 = pd.concat([df1, first_seven_days])
            df2 = pd.concat([df2, last_three_days])
        
        # index reset
        df1 = df1.reset_index(drop=True)
        df2 = df2.reset_index(drop=True)
        # concatenate df1 and df2
        final_df = pd.concat([df1, df2])
        final_df = final_df.reset_index(drop=True)
        # save to csv
        final_df.to_csv(path, index=False)
    else:
        print("Please certificate data has a column named after 'Date'")