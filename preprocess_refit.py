import pandas as pd
from tqdm import tqdm
import numpy as np

column_name_list = [
['timestamp', 'unix', 'mains',
 'fridge_1', 'chest_freezer_2', 'upright_freezer_3',
 'tumble_dryer_4', 'washing_machine_5', 'dishwasher_6',
 'computer_site_7', 'television_site_8', 'electric_heater_9', 'issues'],
['timestamp', 'unix', 'mains',
 'fridge_freezer_1', 'washing_machine_2', 'dishwasher_3',
 'television_4', 'microwave_5', 'toaster_6',
 'hifi_7', 'kettle_8', 'oven_extractor_fan_9', 'issues'],
['timestamp', 'unix', 'mains',
 'toaster_1', 'fridge_freezer_2', 'freezer_3', 
 'tumble_dryer_4', 'dishwasher_5', 'washing_machine_6', 
 'television_7', 'microwave_8', 'kettle_9', 'issues'],
['timestamp', 'unix', 'mains',
 'fridge_1', 'freezer_2', 'fridge_freezer_3',
 'washing_machine_4', 'washing_machine_5', 'computer_site_6',
 'television_site_7', 'microwave_8', 'kettle_9', 'issues'],
['timestamp', 'unix', 'mains',
 'fridge_freezer_1', 'tumble_dryer_2', 'washing_machine_3',
 'dishwasher_4', 'computer_site_5', 'television_site_6',
 'combination_microwave_7', 'kettle_8', 'toaster_9', 'issues'],
['timestamp', 'unix', 'mains',
 'freezer_1', 'washing_machine_2', 'dishwasher_3',
 'mjy_computer_4', 'television_site_5', 'microwave_6',
 'kettle_7', 'toaster_8', 'pgm_computer_9', 'issues'],
['timestamp', 'unix', 'mains',
 'fridge_1', 'freezer_2', 'freezer_3',
 'tumble_dryer_4', 'washing_machine_5', 'dishwasher_6',
 'television_site_7', 'toaster_8', 'kettle_9', 'issues'],
['timestamp', 'unix', 'mains',
 'fridge_1', 'freezer_2', 'dryer_3',
 'washing_machine_4', 'toaster_5', 'computer_6',
 'television_site_7', 'microwave_8', 'kettle_9', 'issues'],
['timestamp', 'unix', 'mains',
 'fridge_freezer_1', 'washer_dryer_2', 'washing_machine_3',
 'dishwasher_4', 'television_site_5', 'microwave_6',
 'kettle_7', 'hifi_8', 'electric_heater_9', 'issues'],
['timestamp', 'unix', 'mains',
 'magimix_1', 'freezer_2', 'chest_freezer_3',
 'fridge_freezer_4', 'washing_machine_5', 'dishwasher_6',
 'television_site_7', 'microwave_8', 'kenwood_kmix_9', 'issues'],
['timestamp', 'unix', 'mains',
 'fridge_1', 'fridge_freezer_2', 'washing_machine_3',
 'dishwasher_4', 'computer_site_5', 'microwave_6', 
 'kettle_7', 'router_8', 'hifi_9', 'issues'],
['timestamp', 'unix', 'mains',
 'fridge_freezer_1', 'television_site_2', 'microwave_3',
 'kettle_4', 'toaster_5', 'television_site_6',
 'not_used_7', 'not_used_8', 'not_used_9', 'issues'],
['timestamp', 'unix', 'mains',
 'television_site_1', 'unknown_2', 'washing_machine_3',
 'dishwasher_4', 'tumble_dryer_5', 'television_site_6',
 'computer_site_7', 'microwave_8', 'kettle_9', 'issues'],
['timestamp', 'unix', 'mains',
 'fridge_freezer_1', 'tumble_dryer_2', 'washing_machine_3',
 'dishwasher_4', 'computer_site_5', 'television_site_6',
 'microwave_7', 'kettle_8', 'toaster_9', 'issues'],
['timestamp', 'unix', 'mains',
 'fridge_freezer_1', 'fridge_freezer_2', 'electric_heater_3',
 'electric_heater_4', 'washing_machine_5', 'dishwasher_6',
 'computer_site_7', 'television_site_8', 'dehumidifier_heater_9', 'issues'],
['timestamp', 'unix', 'mains',
 'freezer_1', 'fridge_freezer_2', 'tumble_dryer_3',
 'washing_machine_4', 'computer_site_5', 'television_site_6',
 'microwave_7', 'kettle_8', 'plug_site_9', 'issues'],
['timestamp', 'unix', 'mains',
 'fridge_1', 'freezer_2', 'fridge_freezer_3',
 'washer_dryer_4', 'washing_machine_5', 'dishwasher_6',
 'desktop_computer_7', 'television_site_8', 'microwave_9', 'issues'],
['timestamp', 'unix', 'mains',
 'fridge_freezer_1', 'washing_machine_2', 'television_site_3',
 'microwave_4', 'kettle_5', 'toaster_6',
 'bread-maker_7', 'lamp_8', 'hifi_9', 'issues'],
['timestamp', 'unix', 'mains',
 'fridge_1', 'freezer_2', 'tumble_dryer_3',
 'washing_machine_4', 'dishwasher_5', 'computer_site_6',
 'television_site_7', 'microwave_8', 'kettle_9', 'issues'],
['timestamp', 'unix', 'mains',
 'fridge_freezer_1', 'tumble_dryer_2', 'washing_machine_3',
 'dishwasher_4', 'food_mixer_5', 'television_6',
 'kettle_toaster_7', 'vivarium_8', 'pond_pump_9', 'issues']
]

def preprocess_house(house_nr):
    """
    Parameters
    ----------
    house_nr: House number of the file

    Returns
    ----------
    Downsampled dataset with power values per 10 seconds
    """
    file_name = f"refit/CLEAN_House{house_nr}.csv"
    df_old = pd.read_csv(file_name)
    df_old.columns = column_name_list[house_nr - 1]

    df_old['unix'] = round(df_old['unix'])  

    # Get the range of the unix time
    minimum_unix = df_old.unix.min()
    maximum_unix = df_old.unix.max()  

    # Extract power record per 10 secs instead of per 1 sec to match with collection frequency of appliance power value
    downsampled_unix_lst = [i for i in range(int(minimum_unix), int(maximum_unix) + 1, 10)]

    df_old = (
        df_old
        # Only use data within the unix range and in the downsampled_unix_lst
        .query("unix == @downsampled_unix_lst")
        # Find unix time difference between every data record
        .assign(unix_diff = df_old['unix'].diff(periods = -1).abs())
        )
    print("Df filtered")

    # Create a full list of unix time values
    df = pd.DataFrame({"unix": downsampled_unix_lst})
    print("Df made")

    # NA will be filled in if the unix time record does not exist before
    df = pd.merge(df_old, df, how = "right", on = "unix")
    print("Merge done")

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

    all_appliances = column_name_list[house_nr - 1][3: -1]
    for i in all_appliances:
        df_corrected[["unix", "mains", i]].to_pickle(f"refit_new\\house_{house_nr}\\{i}.pkl")

for i in tqdm(range(1, 21)):
    preprocess_house(i)
