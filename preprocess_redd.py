import pandas as pd
from tqdm import tqdm

def read_dataframe(source, house_nr, channel_type, channel_nr = "", appliance = "appliance"):
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
        file_name = f"{source}/house_{house_nr}/labels.dat"
    elif channel_type == "channel":
        file_name = f"{source}/house_{house_nr}/channel_{channel_nr}.dat"
    return pd.read_table(file_name, sep = "\s+", names = ['unix', appliance])    

def determine_unix_boundary(source, house_nr):
    """
    Parameters
    ----------
    source: Either "redd" or "ukdale" should be chosen
    house_nr: House number of the file

    Returns
    ----------
    A tuple containing the minimum and maximum unix values where all appliances have power values
    A list containing all appliance name with its channel number
    """
    df_channel_name = read_dataframe(source, house_nr, "labels")

    # Initialize minimum unix and maximum unix
    minimum_unix = 0
    maximum_unix = 0

    for channel_nr in tqdm(range(1, df_channel_name.shape[0] + 1)):
        df_small = read_dataframe(source, house_nr, "channel", channel_nr)
        if channel_nr == 1:
            # Update minimum unix and maximum unix with the first channel's unix values
            minimum_unix = df_small["unix"].min()
            maximum_unix = df_small["unix"].max()
        else:
            # Update minimum unix if the new one is larger
            local_min = df_small["unix"].min()
            if local_min > minimum_unix:
                minimum_unix = local_min
            # Update maximum unix if the new one is smaller
            local_max = df_small["unix"].max()
            if local_max < maximum_unix:
                maximum_unix = local_max
    
    unix_range = [minimum_unix, maximum_unix]

    # Create a list that contains appliance and its corresponding channel
    appl_lst = [df_channel_name.appliance.tolist()[i] + 
                "_" + str(df_channel_name.unix.tolist()[i]) for i in range(df_channel_name.shape[0])]

    return unix_range, appl_lst

def preprocess_house(source, house_nr, channel_nr, unix_range, appl_lst):
    """
    Parameters
    ----------
    source: Either "redd" or "ukdale" should be chosen
    house_nr: House number of the file
    channel_nr: Channel number
    unix_range: The lower and upper limit of unix time
    appl_lst: List with names of appliance + channel number (Computed in function "determine_unix_boundary")

    Returns
    ----------
    Downsampled dataset with power values per 6 seconds
    """

    # Make sure the unix_range list is sorted
    unix_range = sorted(unix_range)

    # Get the appliance name created in appl_lst
    appliance = appl_lst[channel_nr - 1]

    # Get the range of the unix time
    minimum_unix = unix_range[0]
    maximum_unix = unix_range[1]

    df_og = read_dataframe(source, house_nr, "channel", channel_nr, appliance)

    df_og = (
        df_og
        # Only use data within the unix range
        .query("unix >= @minimum_unix & unix <= @maximum_unix")
        # Remove all missing values of unix
        .dropna(subset = ["unix"])
        .reset_index(drop = True)
        # Find unix time difference between every data record
        .assign(unix_diff = df_og['unix'].diff(periods = -1).abs())
        )

    # Create a full list of unix time values
    unix_lst = [i for i in range(int(df_og["unix"].min()), int(df_og["unix"].max()) + 1)]
    df = pd.DataFrame({"unix": unix_lst})

    # NA will be filled in if the unix time record does not exist before
    df = pd.merge(df_og, df, how = "right", on = "unix")

    # Fill in the missing unix time difference with previous known value
    df["unix_diff"] = df["unix_diff"].fillna(method = 'pad')
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
        )

    # Extract power record per 6 mins instead of per 1 min to match with collection frequency of appliance power value
    downsampled_unix_lst = [i for i in range(int(df_og["unix"].min()), int(df_og["unix"].max()) + 1, 6)]
    return df_corrected[["unix", appliance]].query("unix == @downsampled_unix_lst").reset_index(drop = True)

def preprocess_mains(house_nr, unix_range):
    """
    Parameters
    ----------
    house_nr: House number of the file (Should only be 1, 2 or 5 from ukdale)
    unix_range: The lower and upper limit of unix time

    Returns
    ----------
    Downsampled mains dataset with power values per 6 seconds
    """

    # Make sure the unix_range list is sorted
    unix_range = sorted(unix_range)

    # Get the range of the unix time
    minimum_unix = unix_range[0]
    maximum_unix = unix_range[1]

    # Read mains.dat to a dataframe
    df_og = pd.read_table(f"ukdale/house_{house_nr}/mains.dat", sep = "\s+", names = ['unix', 'mains', 'a', 'b'])

    df_og = (
        df_og
        # Only use data within the unix range
        .query("unix >= @minimum_unix & unix <= @maximum_unix")
        # Remove all missing values of unix
        .dropna(subset = ["unix"])
        .reset_index(drop = True)
        # Find unix time difference between every data record
        .assign(unix_diff = df_og['unix'].diff(periods = -1).abs())
        )

    # Create a full list of unix time values
    unix_lst = [i for i in range(int(df_og["unix"].min()), int(df_og["unix"].max()) + 1)]
    df = pd.DataFrame({"unix": unix_lst})

    # NA will be filled in if the unix time record does not exist before
    df = pd.merge(df_og, df, how = "right", on = "unix")

    # Fill in the missing unix time difference with previous known value
    df["unix_diff"] = df["unix_diff"].fillna(method = 'pad')
    df_corrected = (
        pd.concat([
            # Fill the missing power value with previous known value if the unix time difference <= 3 mins
            df.query("unix_diff <= 180").fillna(method = 'ffill'), 
            # Fill the missing power value with 0 if the unix time difference > 3 mins
            df.query("unix_diff > 180").fillna(0)
            ])
        .sort_values(by = ["unix"])
        # Fill in next known value in case there are unknown power values of mains at the very beginning
        .fillna(method = 'bfill')
        )

    # Extract power record per 6 mins instead of per 1 min to match with collection frequency of mains power value
    downsampled_unix_lst = [i for i in range(int(df_og["unix"].min()), int(df_og["unix"].max()) + 1, 6)]
    df_corrected = df_corrected.astype({"unix": "int64"})
    return df_corrected[["unix", "mains"]].query("unix == @downsampled_unix_lst").reset_index(drop = True)

def preprocess_dataset(source):
    """
    Parameters
    ----------
    source: Either "redd" or "ukdale" should be chosen

    Returns
    ----------
    Output pickle files of the dataset folder
    """
    # Determine the number of houses based on the dataset chosen
    if source == "redd":
        nr_of_houses = 6
    elif source == "ukdale":
        nr_of_houses = 5
    
    # Recursive usage of the above functions for each channel at each house
    # And then output the results to a pickle file in a folder
    for house_nr in tqdm(range(1, nr_of_houses + 1)):
        unix_range, appl_lst = determine_unix_boundary(source, house_nr)
        for channel_nr in tqdm(range(1, len(appl_lst) + 1)):
            df_appliance_processed = preprocess_house(source, house_nr, channel_nr, unix_range, appl_lst)
            if source == "redd":
                df_appliance_processed.to_pickle(f"redd_corrected/house_{house_nr}/{appl_lst[channel_nr - 1]}.pkl")
            elif source == "ukdale":
                df_appliance_processed.to_pickle(f"ukdale_corrected/house_{house_nr}/{appl_lst[channel_nr - 1]}.pkl")
        
        # Houses 1, 2, 5 from ukdale have a separate mains.dat file
        # So the function "preprocess_mains" will be used in such scenario
        if source == "ukdale" and house_nr in [1, 2, 5]:
            df_mains_processed = preprocess_mains(house_nr, unix_range)
            df_mains_processed.to_pickle(f"ukdale_corrected/house_{house_nr}/mains.pkl")

house_nr = 5
unix_range, appl_lst = determine_unix_boundary("ukdale", house_nr)
for channel_nr in tqdm(range(1, len(appl_lst) + 1)):
    df_appliance_processed = preprocess_house("ukdale", house_nr, channel_nr, unix_range, appl_lst)
    df_appliance_processed.to_pickle(f"ukdale_corrected/house_{house_nr}/{appl_lst[channel_nr - 1]}.pkl")

# Houses 1, 2, 5 from ukdale have a separate mains.dat file
# So the function "preprocess_mains" will be used in such scenario
df_mains_processed = preprocess_mains(house_nr, unix_range)
df_mains_processed.to_pickle(f"ukdale_corrected/house_{house_nr}/mains.pkl")

unix_range, appl_lst = determine_unix_boundary("ukdale", 1)
for channel_nr in tqdm(range(1, len(appl_lst) + 1)):
    df_appliance_processed = preprocess_house("ukdale", 1, channel_nr, unix_range, appl_lst)
    df_appliance_processed.to_pickle(f"ukdale_corrected/house_1/{appl_lst[channel_nr - 1]}.pkl")

# Houses 1, 2, 5 from ukdale have a separate mains.dat file
# So the function "preprocess_mains" will be used in such scenario
df_mains_processed = preprocess_mains(1, unix_range)
df_mains_processed.to_pickle(f"ukdale_corrected/house_1/mains.pkl")
