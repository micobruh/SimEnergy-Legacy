import pandas as pd
from .config import appliances_lst, directory


# Read the default cleaned dataset for the barplot
def get_data(device_lst = appliances_lst, directory = directory):
    df = pd.read_pickle(directory + "mains.pkl")
    for device in device_lst:
        file_name = directory + f"{device}_pred.pkl"
        df1 = pd.read_pickle(file_name)
        df = pd.merge(df, df1, on = ['unix'])
    df[df < 0] = 0
    return df
