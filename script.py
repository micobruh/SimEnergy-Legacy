# -*- coding: utf-8 -*-

# Commented out IPython magic to ensure Python compatibility.
import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view
import tensorflow_addons as tfa
from keras.models import Sequential, load_model, Model
from keras.layers import Input, Dense, LSTM, Flatten, Conv1D, Bidirectional, RNN, Dropout, BatchNormalization
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from keras.optimizers import Adam
import os
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

import tensorflow as tf
device_name = tf.test.gpu_device_name()
if device_name != '/device:GPU:0':
    raise SystemError('GPU device not found')
print('Found GPU at: {}'.format(device_name))

####################################################################################################

class Dataset(object):
    def __init__(self, appliance_name):
        self.appliance_name = appliance_name

        self.appliance_train = ''
        self.mains_train = ''
        self.appliance_test = ''
        self.mains_test = ''
        self.model_name = ''
        self.window_size = 0
        self.batch_size_nr = 0
        self.data_source = ''
        self.house = 0

    def appliance_involved(self):
        """
        Parameters
        ----------
        appliance_name: Self, Name of the appliance

        Returns
        ----------
        Appliance column names used in training and testing datasets: Self
        Model name (LSTM1/LSTM2/Autoencoder/Rectangles): Self
        Window size: Self
        Batch size: Self
        Data source: Self, redd/ukdale
        House: Self
        """
        if self.appliance_name == 'refrigerator':
            self.appliance_train = 'refrigerator_5'
            self.mains_train = 'mains_1'
            self.appliance_test = 'refrigerator_7'
            self.mains_test = 'mains_1'
            self.model_name = 'Autoencoder'
            self.window_size = 512
            self.batch_size_nr = 64
            self.data_source = 'redd'
            self.house = 1

        elif self.appliance_name == 'washer_dryer':
            self.appliance_train = 'washer_dryer_10'
            self.mains_train = 'mains_1'
            self.appliance_test = 'washer_dryer_13'
            self.mains_test = 'mains_1'
            self.model_name = 'Autoencoder'
            self.window_size = 128
            self.batch_size_nr = 64
            self.data_source = 'redd'
            self.house = 1

        elif self.appliance_name == 'dishwaser':
            self.appliance_train = 'dishwaser_6'
            self.mains_train = 'mains_1'
            self.appliance_test = 'dishwaser_9'
            self.mains_test = 'mains_1'
            self.model_name = 'Autoencoder'
            self.window_size = 1536
            self.batch_size_nr = 64
            self.data_source = 'redd'
            self.house = 1
        
        elif self.appliance_name == 'oven':
            self.appliance_train = 'oven_4'
            self.mains_train = 'mains_1'
            self.appliance_test = ''
            self.mains_test = ''
            self.model_name = 'Autoencoder'
            self.window_size = 128
            self.batch_size_nr = 64
            self.data_source = 'redd'
            self.house = 1

        elif self.appliance_name == 'kettle':
            self.appliance_train = 'kettle_8'
            self.mains_train = 'kettle_8'
            self.appliance_test = ''
            self.mains_test = ''
            self.model_name = 'Autoencoder'
            self.window_size = 128
            self.batch_size_nr = 64
            self.data_source = 'ukdale'
            self.house = 2

        elif self.appliance_name == 'microwave':
            self.appliance_train = 'microwave_15'
            self.mains_train = 'microwave_15'
            self.appliance_test = ''
            self.mains_test = ''
            self.model_name = 'Autoencoder'
            self.window_size = 256
            self.batch_size_nr = 64
            self.data_source = 'ukdale'
            self.house = 2

        elif self.appliance_name == 'stove':
            self.appliance_train = 'stove_14'
            self.mains_train = 'mains_1'
            self.appliance_test = ''
            self.mains_test = ''
            self.model_name = 'Autoencoder'
            self.window_size = 128
            self.batch_size_nr = 64
            self.data_source = 'redd'
            self.house = 1

        elif self.appliance_name == 'lighting':
            self.appliance_train = 'lighting_9'
            self.mains_train = 'mains_1'
            self.appliance_test = ''
            self.mains_test = ''
            self.model_name = 'Autoencoder'
            self.window_size = 32
            self.batch_size_nr = 64
            self.data_source = 'redd'
            self.house = 1
        
        elif self.appliance_name == 'laptop':
            self.appliance_train = 'laptop_2'
            self.mains_train = 'laptop_2'
            self.appliance_test = ''
            self.mains_test = ''
            self.model_name = 'Autoencoder'
            self.window_size = 128
            self.batch_size_nr = 64
            self.data_source = 'ukdale'
            self.house = 2
        
        elif self.appliance_name == 'tv':
            self.appliance_train = 'primary_tv_5'
            self.mains_train = 'primary_tv_5'
            self.appliance_test = ''
            self.mains_test = ''
            self.model_name = 'Autoencoder'
            self.window_size = 128
            self.batch_size_nr = 64
            self.data_source = 'ukdale'
            self.house = 5
    
    def generate_file_name(self, col):
        """
        Parameters
        ----------
        house: Self, House number
        data_source: Self, "redd" or "ukdale"
        col: Column name of the appliance in the file

        Returns
        ----------
        Self, A string with the proper file link
        """
        return f"data/{self.data_source}_corrected/house_{self.house}/{col}.pkl"

    def train_val_test_split(self, arr, split_ratio = 0.7):
        """
        Parameters
        ----------
        arr: The original training data
        split_ratio: Ratio between nr of data in training and validation data

        Returns
        ----------
        Numpy arrays with training and validation datasets split
        """
        train = arr[: int(arr.shape[0] * split_ratio)]
        val = arr[int(arr.shape[0] * split_ratio): ]
        return train, val

    def min_max_scaling(self, arr, min_pow, max_pow, train):
        """
        Parameters
        ----------
        arr: Numpy array
        min_pow: Minimum value (From training dataset)
        max_pow: Maximum value (From training dataset)
        train: Boolean, if true, the input is for training

        Returns
        ----------
        Numpy array with all data in min max scaling
        The minimum and maximum values of the column in the training dataset
        """
        if train:
            max_pow = arr.max()
        return arr / max_pow, max_pow

    def sliding_window(self, x_array, window_size, mid_pt = True, min_pow = 0, max_pow = 0, train = True):
        """
        Parameters
        ----------
        x_array: Numpy array of mains
        window_size: Window size of the sliding window
        min_pow: Minimum value (From training dataset)
        max_pow: Maximum value (From training dataset)
        train: Boolean, if true, the input is for training    

        Returns
        ----------
        Reshaped datasets to fulfill the LSTM requirement
        New x array has shape (nr_of_elements, window_size)
        """
        scale_x_array, min_pow, max_pow = self.min_max_scaling(x_array, min_pow, max_pow, train)
        if mid_pt:
            # Add 0 to the x array before AND after sliding window
            pad_x_array = np.pad(scale_x_array, (window_size // 2, window_size // 2 - 1), 'constant', constant_values = (0, 0))
        else:
            # Add 0 to the x array before sliding window
            pad_x_array = np.pad(scale_x_array, (window_size - 1, 0), 'constant', constant_values = (0, 0))            
        reshaped_x_array = sliding_window_view(pad_x_array, window_size)
        return reshaped_x_array, min_pow, max_pow

    def reshape_y(self, y_array, min_pow = 0, max_pow = 0, train = True):
        """
        Parameters
        ----------
        y_array: Numpy array of appliance
        min_pow: Minimum value (From training dataset)
        max_pow: Maximum value (From training dataset)
        train: Boolean, if true, the input is for training  

        Returns
        ----------
        Reshaped datasets to fulfill the LSTM requirement
        New y dataset has shape (nr_of_elements, 1)
        """
        scale_y_array, min_pow, max_pow = self.min_max_scaling(y_array, min_pow, max_pow, train)    
        reshaped_y_array = scale_y_array.reshape(scale_y_array.shape[0], 1)
        return reshaped_y_array, min_pow, max_pow

####################################################################################################

def model_structure(model_type, time_step):
    """
    Parameters
    ----------
    model_type: Type of the model
    time_step: Window size of the sliding window

    Returns
    ----------
    A neural network model structure
    """    
    # create and fit the neural network
    model = Sequential()

    # build the model: 
    if model_type == 'LSTM1':
        main_input = Input(shape = (time_step, 1), name = 'main_input')
        l1 = LSTM(64, return_sequences = True, dropout = 0.3)(main_input)
        l2 = LSTM(128, return_sequences = True, dropout = 0.3)(l1)
        l3 = LSTM(256, return_sequences = False, dropout = 0.3)(l2)
        time_output = Dense(1, name = 'time_output')(l3)
    
    elif model_type == 'LSTM2':
        main_input = Input(shape = (time_step, 1), name = 'main_input')
        c1 = Conv1D(16, 4, strides = 1, activation = 'linear')(main_input)
        # l1 = Bidirectional(RNN(tfa.rnn.PeepholeLSTMCell(128), return_sequences = True))(c1)
        # l2 = Bidirectional(RNN(tfa.rnn.PeepholeLSTMCell(256), return_sequences = True))(l1)
        l1 = Bidirectional(LSTM(128, return_sequences = True, dropout = 0.3))(c1)
        l2 = Bidirectional(LSTM(256, return_sequences = True, dropout = 0.3))(l1)
        d1 = Dense(128, activation = 'tanh')(l2)
        f1 = Flatten()(d1)
        time_output = Dense(1, name = 'time_output', activation = 'linear')(f1)
    
    elif model_type == 'Autoencoder':
        main_input = Input(shape = (time_step, 1), name = 'main_input')
        c1 = Conv1D(16, 4, strides = 1, activation = 'linear')(main_input)
        d1 = Dense((time_step - 3) * 8, activation = 'relu')(c1)
        d2 = Dense(128, activation = 'relu')(d1)
        d3 = Dense((time_step - 3) * 8, activation = 'relu')(d2)
        c2 = Conv1D(1, 4, strides = 1, activation = 'linear')(d3)
        f1 = Flatten()(c2)
        time_output = Dense(1, name = 'time_output')(f1)
    
    model = Model(inputs=[main_input], outputs=[time_output])
    opt = Adam(learning_rate = 0.001, epsilon = 1e-08, decay = 0.004, clipvalue = 3)

    # The loss used in model training is mean_squared_error because it is time prediction
    # The optimizer is Adam
    model.compile(loss = 'mean_squared_error', optimizer = opt)
    
    return model

####################################################################################################

def reverse_min_max_scaling(arr, min_pow, max_pow):
    """
    Parameters
    ----------
    arr: Numpy array of the raw prediction
    min_pow: Minimum value of the appliance (From training dataset)
    max_pow: Maximum value of the appliance (From training dataset)

    Returns
    ----------
    A numpy array with the proper prediction values
    """
    return arr * (max_pow - min_pow) + min_pow

####################################################################################################

def training(appliance_name, data_source, house_train, epochs_nr = 20):
    """
    Parameters
    ----------
    appliance_name: Name of the appliance
    data_source: "redd" or "ukdale"
    house_train: House nr of the training data
    epoch_nr: Nr of epochs when training the model

    Returns
    ----------
    A dataframe containing minimum and maximum mains and appliance power
    """    
    mains_train, _, appliance_train, _, model_name, window_size, batch_size_nr = appliance_involved(appliance_name)

    df_train_x = pd.read_pickle(generate_file_name(house_train, mains_train, data_source))
    df_train_y = pd.read_pickle(generate_file_name(house_train, appliance_train, data_source))

    if data_source == "ukdale":
        mains_train = "mains"

    train_x, val_x = train_val_test_split(df_train_x[mains_train].to_numpy())
    train_y, val_y = train_val_test_split(df_train_y[appliance_train].to_numpy())

    train_x, min_mains, max_mains = sliding_window(train_x, window_size)
    train_y, min_appliance, max_appliance = reshape_y(train_y)
    val_x, _, _ = sliding_window(val_x, window_size, min_mains, max_mains, False)
    val_y, _, _ = reshape_y(val_y, min_appliance, max_appliance, False)

    print("Preprocessing Done!")

    with tf.device(device_name):
        model = model_structure(model_name, window_size)
        # Save the best model
        early_stopping = EarlyStopping(monitor = 'val_loss', patience = 42)
        checkpoint_filepath = f"data/{data_source}_corrected/models/model_{appliance_name}/{model_name}/" + "weights.{epoch:02d}.h5"
        model_checkpoint_callback = ModelCheckpoint(filepath = checkpoint_filepath, monitor = 'val_loss', mode = 'min', save_best_only = True)
        lr_reducer = ReduceLROnPlateau(monitor = 'val_loss', factor = 0.5, patience = 10, verbose = 0, mode = 'auto', min_delta = 0.0001, cooldown = 0, min_lr = 0)

        # Fit the model
        # Validation data is used here for evaluation during the training process
        model.fit(train_x, train_y, validation_data = (val_x, val_y), epochs = epochs_nr, 
                batch_size = batch_size_nr, callbacks = [early_stopping, model_checkpoint_callback, lr_reducer])    

    print("Training Done!")

    return pd.DataFrame({"mains": [min_mains, max_mains], appliance_name: [min_appliance, max_appliance]})

####################################################################################################

def prediction(x, unix, appliance_name, window_size, df_min_max, best_model):
    """
    Parameters
    ----------
    x: Numpy array of the new data
    unix: Numpy array of unix time
    appliance_name: Name of the appliance
    window_size: Window size of the sliding window
    df_min_max: Dataframe with minimum power and maximum power of the appliance
    best_model: A string with a model file link

    Returns
    ----------
    A numpy array with the proper prediction values
    """  
    min_mains = df_min_max.loc[0, "mains"]
    max_mains = df_min_max.loc[1, "mains"]
    min_appliance = df_min_max.loc[0, appliance_name]
    max_appliance = df_min_max.loc[1, appliance_name]

    test_x, _, _ = sliding_window(x, window_size, min_mains, max_mains, False)

    # Load the best model trained
    model = load_model(best_model)

    with tf.device(device_name):
        test_predict = model.predict(test_x)
    
    df_test_y = pd.DataFrame({
        "unix": unix, 
        "pred_" + appliance_name: reverse_min_max_scaling(test_predict.flatten(), min_appliance, max_appliance)
        })

    print("Prediction done!")

    return df_test_y

def generate_dataset(house, col, data_source, split, mains = False):
    """
    Parameters
    ----------
    house: Numpy array of the new data
    col: Column name of the appliance in the file
    data_source: "redd" or "ukdale"
    split: Boolean, if true, train val split will be performed on the dataset

    Returns
    ----------
    Numpy array(s), each contains the appliance power and unix time
    """
    df = pd.read_pickle(generate_file_name(house, col, data_source))
    if data_source == "ukdale" and mains == True:
        col = "mains"
    if split:
        train, val = train_val_test_split(df[col].to_numpy())
        train_unix, val_unix = train_val_test_split(df["unix"].to_numpy())
        return train, val, train_unix, val_unix
    else:
        return df[col].to_numpy(), df["unix"].to_numpy()

def predict_by_new_model(data, unix, appliance_name, df_min_max, model_name, window_size, df_org, data_source):
    """
    Parameters
    ----------
    data: Numpy array of the new data
    unix: Numpy array of unix time
    appliance_name: Name of the appliance
    df_min_max: Dataframe with minimum power and maximum power of the appliance
    model_name: Name of the model
    window_size: Window size of the sliding window
    df_org: Numpy array with the real appliance power

    Returns
    ----------
    A dataframe containing unix time, predicted appliance power and real appliance power
    """    
    directory = f"data/{data_source}_corrected/models/model_{appliance_name}/{model_name}/"
    all_file_lst = [os.path.join(root, file) for root, _, files in os.walk(directory) for file in files]
    best_model = all_file_lst[-1]
    df_test_y = prediction(data, unix, appliance_name, window_size, df_min_max, best_model)
    df_test_y["real_" + appliance_name] = df_org
    return df_test_y

def evaluation(df, appliance_name):
    """
    Parameters
    ----------
    df: Dataframe with predicted power and actual power of the appliance
    appliance_name: Name of the appliance

    Returns
    ----------
    R2
    RMSE
    MAE
    """  
    return r2_score(df["real_" + appliance_name], df["pred_" + appliance_name]), mean_squared_error(df["real_" + appliance_name], df["pred_" + appliance_name]) ** 0.5, mean_absolute_error(df["real_" + appliance_name], df["pred_" + appliance_name])

####################################################################################################

def just_train(appliance_name, house_train, data_source):
    """
    Parameters
    ----------
    appliance_name: Name of the appliance
    house_train: House number used for training
    house_test: House number used for testing
    data_source: "redd" or "ukdale"

    Returns
    ----------
    Create model files and min max power files
    """      
    mains_train, _, appliance_train, _, model_name, window_size, batch_size_nr = appliance_involved(appliance_name)
    df_min_max = training(appliance_name, data_source, house_train)
    df_min_max.to_pickle(f"data/{data_source}_corrected/output/min_max_power_{appliance_train}.pkl")

    print("Done!")

def evaluation_run(appliance_name, house_train, house_test, data_source_train, data_source_test):
    """
    Parameters
    ----------
    appliance_name: Name of the appliance
    house_train: House number used for training
    house_test: House number used for testing
    data_source_train: "redd" or "ukdale"
    data_source_test: "redd" or "ukdale"

    Returns
    ----------
    Print R2, RMSE, MAE of the datasets
    Output pickle files with the predictions
    """      
    mains_train, mains_test, appliance_train, appliance_test, model_name, window_size, batch_size_nr = appliance_involved(appliance_name)
    df_min_max = training(appliance_name, data_source_train, house_train)
    df_train, df_val, unix_train, unix_val = generate_dataset(house_train, mains_train, True, True)
    df_train_appliance, df_val_appliance, _, _ = generate_dataset(house_train, appliance_train, True)
    df_test, unix_test = generate_dataset(house_test, mains_test, data_source_test, False, True)
    df_test_appliance, _ = generate_dataset(house_test, appliance_test, data_source_test, False)
    df_train_pred = predict_by_new_model(df_train, unix_train, appliance_name, df_min_max, model_name, window_size, df_train_appliance, data_source_train)
    df_val_pred = predict_by_new_model(df_val, unix_val, appliance_name, df_min_max, model_name, window_size, df_val_appliance, data_source_train)
    df_test_pred = predict_by_new_model(df_test, unix_test, appliance_name, df_min_max, model_name, window_size, df_test_appliance, data_source_test)

    print("R2, RMSE, MAE: ")
    print(f"Training dataset: {evaluation(df_train_pred, appliance_name)}")
    print(f"Validation dataset: {evaluation(df_val_pred, appliance_name)}")
    print(f"Testing dataset: {evaluation(df_test_pred, appliance_name)}")

    df_train_pred.to_pickle(f"data/{data_source_train}/output/house_{house_train}_{appliance_train}_{model_name}.pkl")
    df_val_pred.to_pickle(f"data/{data_source_train}/output/house_{house_train}_{appliance_train}_{model_name}.pkl")
    df_test_pred.to_pickle(f"data/{data_source_test}_corrected/output/house_{house_test}_{appliance_test}_{model_name}.pkl")
    df_min_max.to_pickle(f"data/{data_source_train}_corrected/output/min_max_power_{appliance_train}.pkl")

    print("Done!")

####################################################################################################

# just_train("oven", 1, "redd")
# print("Major Achievement!!!")

# just_train("kettle", 2, "ukdale")
# print("Major Achievement!!!")

just_train("microwave", 2, "ukdale")
print("Major Achievement!!!")

just_train("stove", 1, "redd")
print("Major Achievement!!!")

just_train("lighting", 1, "redd")
print("Major Achievement!!!")

just_train("laptop", 2, "ukdale")
print("Major Achievement!!!")

just_train("tv", 5, "ukdale")
print("Major Achievement!!!")

# evaluation_run("refrigerator", 1, 3)