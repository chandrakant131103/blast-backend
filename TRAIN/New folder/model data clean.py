import pandas as pd
import numpy as np

df=pd.read_csv(".csv")
df.isna() # shows where NaNs are
print(df.dropna())  # drops rows with any NaN
df.dropna().to_csv("mine_data_no_nan.csv", index=False)