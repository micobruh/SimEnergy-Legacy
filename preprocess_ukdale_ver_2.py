import pandas as pd
from tqdm import tqdm
import numpy as np

# c is generally smaller than a & b, with b being the largest
# The smallest value of b & c is larger than 0, while minimum a can be 0
# Column meaning: Active power, reactive power voltage

# 1) UNIX timestamp
# 2) Active power (watts)
# 3) Apparent power (volt-amperes)
# 4) Mains RMS voltage

# house 1:
# Column b is strictly larger than a
# Column b is not always larger than c
# Column a is not always larger than c
# a is between 0 and 8070
# b is between 1.6 and 11570
# c appears to be voltage because the values are around 200 to 270

# house 2:
# Column b is strictly larger than a
# Column b is not always larger than c
# Column a is not always larger than c
# a is between 0 and 6700
# b is between 5 and 6750
# c appears to be voltage because the values are around 210 to 250

# house 5:
# Column b is strictly larger than a & c
# Column a is not always larger than c
# a is between 195 and 8450
# b is between 250 and 8560
# c appears to be voltage because the values are around 200 to 260

def read_dataframe(house_nr, channel_type, channel_nr = "", appliance = "appliance"):
    """
    Parameters
    ----------
    source: Either "redd" or "ukdale" should be chosen
    house_nr: House number of the file
    channel_type: Either "labels" or "channel" should be chosen
    channel_nr: Channel number
    appliance: Name of the appliance

    Returns
    ----------
    A dataframe read from a .dat data file
    """    
    if channel_type == "labels":
        file_name = f"ukdale/house_{house_nr}/labels.dat"
        df = pd.read_table(file_name, sep = "\s+", names = ['unix', 'appliance'])
    elif channel_type == "channel":
        file_name = f"ukdale/house_{house_nr}/channel_{channel_nr}.dat"
        df = pd.read_table(file_name, sep = "\s+", names = ['unix', appliance]) 
    return df

def determine_appl_lst(house_nr):
    """
    Parameters
    ----------
    house_nr: House number of the file

    Returns
    ----------
    A list containing all appliance name with its channel number
    """    
    df_labels = read_dataframe(house_nr, "labels")
    return [df_labels.appliance.tolist()[i] + "_" + str(df_labels.unix.tolist()[i]) for i in range(df_labels.shape[0])][1: ]

def preprocess_house(house_nr, channel_nr, appl_lst, df_mains):
    """
    Parameters
    ----------
    house_nr: House number of the file
    channel_nr: Channel number
    appl_lst: List with names of appliance + channel number (Computed in function "determine_unix_boundary")

    Returns
    ----------
    Downsampled dataset with power values per 6 seconds
    """

    # Get the appliance name created in appl_lst
    appliance = appl_lst[channel_nr - 1]

    df_appliance = read_dataframe(house_nr, "channel", channel_nr, appliance)
    print("Df appliance exists")

    df_appliance['unix'] = round(df_appliance['unix'])
    df_mains["unix"] = round(df_mains["unix"])    

    # Get the range of the unix time
    minimum_unix = df_appliance.unix.min()
    maximum_unix = df_appliance.unix.max()  

    # Extract power record per 10 mins instead of per 1 min to match with collection frequency of appliance power value
    downsampled_unix_lst = [i for i in range(int(minimum_unix), int(maximum_unix) + 1, 10)]

    df_mains = (
        df_mains
        # Only use data within the unix range and in the downsampled_unix_lst
        .query("unix == @downsampled_unix_lst")
        # Remove all missing values of unix
        .dropna(subset = ["unix"])
        .reset_index(drop = True)
        )
    print("Df mains filtered")

    df_appliance = (
        df_appliance
        # Only use data within the unix range and in the downsampled_unix_lst
        .query("unix == @downsampled_unix_lst")        
        # Find unix time difference between every data record
        .assign(unix_diff = df_appliance['unix'].diff(periods = -1).abs())        
    )
    print("Df appliance filtered")

    # Create a full list of unix time values
    df = pd.DataFrame({"unix": downsampled_unix_lst})
    print("Df made")

    # NA will be filled in if the unix time record does not exist before
    df = pd.merge(df_appliance, df, how = "right", on = "unix")
    print("Merge 1 done")
    df = pd.merge(df_mains, df, how = "right", on = "unix")
    print("Merge 2 done")

    # Fill in the missing unix time difference with previous known value
    df["unix_diff"] = df["unix_diff"].fillna(method = 'pad')
    print("1st fill na done")
    df_corrected = (
        pd.concat([
            # Fill the missing power value with previous known value if the unix time difference <= 3 mins
            df.query("unix_diff <= 180").fillna(method = 'ffill'), 
            # Fill the missing power value with 0 if the unix time difference > 3 mins
            df.query("unix_diff > 180").fillna(0)
            ])
        .sort_values(by = ["unix"])
        # Fill in next known value in case there are unknown power values of appliance at the very beginning
        .fillna(method = 'bfill')
        .astype({"unix": "int64"})
        .reset_index(drop = True)
        )
    print("2nd fill na done")

    return df_corrected[["unix", "mains", appliance]]

house_nr = 3
appl_lst = determine_appl_lst(house_nr)
df_mains = read_dataframe(house_nr, "channel", 1, appliance = "mains")
print("Success")
for channel_nr in tqdm(range(1, len(appl_lst) + 1)):
    df_appliance_processed = preprocess_house(house_nr, channel_nr, appl_lst, df_mains)
    df_appliance_processed.to_pickle(f"ukdale_new/house_{house_nr}/{appl_lst[channel_nr - 1]}.pkl")
    print("Done")

house_nr = 4
appl_lst = determine_appl_lst(house_nr)
df_mains = read_dataframe(house_nr, "channel", 1, appliance = "mains")
print("Success")
for channel_nr in tqdm(range(1, len(appl_lst) + 1)):
    df_appliance_processed = preprocess_house(house_nr, channel_nr, appl_lst, df_mains)
    df_appliance_processed.to_pickle(f"ukdale_new/house_{house_nr}/{appl_lst[channel_nr - 1]}.pkl")
    print("Done")
