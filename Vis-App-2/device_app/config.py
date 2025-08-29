import pandas as pd

# Change this directory when running on your computer!
directory = "C:\\Users\\20201242\\Documents\\Honors\\SimEnergy\\Vis-App-2\\device_app\\datasets\\"

# Don't change anything below!
past_lst = ["Past Hour", "Past Day", "Past Week", "Past Month", "Past Year"]
appliances_lst = ["dishwaser", "refrigerator", 
                  "washer_dryer", "oven", 
                  "kettle", "microwave", 
                  "stove", "lighting", 
                  "laptop", "tv"]
images_lst = ["../assets/dishwasher.png", "../assets/refrigerator.png", 
              "../assets/washer_dryer.png", "../assets/oven.png", 
              "../assets/kettle.png", "../assets/microwave.png", 
              "../assets/stove.png", "../assets/lighting.png", 
              "../assets/laptop.png", "../assets/tv.png"]
appliances_power = {
    "dishwaser": {"max_power_watt": 1800, "on_power_threshold_watt": 1200, "min_on_duration_sec": 1800, "min_off_duration_sec": 300},
    "refrigerator": {"max_power_watt": 150, "on_power_threshold_watt": 100, "min_on_duration_sec": 900, "min_off_duration_sec": 1800},
    "washer_dryer": {"max_power_watt": 2500, "on_power_threshold_watt": 1200, "min_on_duration_sec": 1800, "min_off_duration_sec": 600},
    "oven": {"max_power_watt": 5000, "on_power_threshold_watt": 3000, "min_on_duration_sec": 1800, "min_off_duration_sec": 1800},
    "kettle": {"max_power_watt": 1500, "on_power_threshold_watt": 1500, "min_on_duration_sec": 30, "min_off_duration_sec": 300},
    "microwave": {"max_power_watt": 1200, "on_power_threshold_watt": 800, "min_on_duration_sec": 30, "min_off_duration_sec": 300},
    "stove": {"max_power_watt": 8000, "on_power_threshold_watt": 4000, "min_on_duration_sec": 1800, "min_off_duration_sec": 1800},
    "lighting": {"max_power_watt": 100, "on_power_threshold_watt": 50, "min_on_duration_sec": 30, "min_off_duration_sec": 60},
    "laptop": {"max_power_watt": 100, "on_power_threshold_watt": 60, "min_on_duration_sec": 30, "min_off_duration_sec": 300},
    "tv": {"max_power_watt": 400, "on_power_threshold_watt": 100, "min_on_duration_sec": 1800, "min_off_duration_sec": 1800}
}

df_appliances_power = pd.DataFrame.from_dict(appliances_power, orient='index')
max_power_lst = df_appliances_power['max_power_watt'].to_numpy()