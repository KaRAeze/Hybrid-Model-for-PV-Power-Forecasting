import pandas as pd
from EC_preprocess import ec_pre
from DLdate import DLdata
ecpath = 'rad.csv' # data extracted from NWP, according to your station's position
stapath = 'sta.csv' # data measured at station
final_filepath = 'rad_a.csv'
# process for EC data
ec_pre(ecpath)
# 10 days split to 7days + 3days
DLdata(ecpath)
DLdata(stapath)
# data merge
df1 = pd.read_csv(ecpath)
print('EC:',df1.shape)
df2 = pd.read_csv(stapath).iloc[:,2:-1]
print('sta:',df2.shape)
df1['Date'] = pd.to_datetime(df1['Date'])
df1.set_index('Date', inplace=True)
df2['Date'] = pd.to_datetime(df2['Date']) + pd.Timedelta(hours=4)
df2.set_index('Date', inplace=True)
merged_df = pd.merge(df1, df2, left_index=True, right_index=True, how='outer')
merged_df.dropna(inplace=True)
merged_df.rename(columns={'ssr_x':'ssr',
                          'ws_x':'ws',
                          'wd_x':'wd',
                          'ssr_y':'ssr(sta)',
                          'ws_y':'ws(sta)',
                          'wd_y':'wd(sta)'},inplace=True)
merged_df.to_csv(final_filepath)
DLdata(final_filepath)
print('merged:',merged_df.shape)