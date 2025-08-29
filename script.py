# -*- coding: utf-8 -*-

# Commented out IPython magic to ensure Python compatibility.
import numpy as np
import pandas as pd
from keras.models import Sequential, load_model, Model
from keras.layers import Input, Dense, LSTM, Flatten, Conv1D, Bidirectional
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from keras.optimizers import Adam
import os
from tqdm import tqdm
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

import tensorflow as tf
device_name = tf.test.gpu_device_name()
if device_name != '/device:GPU:0':
    raise SystemError('GPU device not found')
print('Found GPU at: {}'.format(device_name))

from google.colab import drive
drive.mount('/content/drive')

####################################################################################################

class Dataset(object):
    def __init__(self, appliance_name, model_link = '', file_front_link = '', link_nr = 0, start_nr = 0):
        self.appliance_name = appliance_name
        self.train_link = ''
        self.test_link = ''
        self.window_size = 0
        self.batch_size = 0
        self.train_mains_window_lst = ''
        self.train_responding_appliance_power = ''
        self.val_mains_window_lst = ''
        self.val_responding_appliance_power = ''
        self.model_name = ''
        self.model_link = model_link
        self.model = ''
        self.file_front_link = file_front_link
        self.link_nr = link_nr
        self.start_nr = start_nr

    def appliance_involved(self):
        """
        Parameters
        ----------
        appliance_name: Self, Name of the appliance

        Returns
        ----------
        Links to training and testing datasets: Self
        Window size: Self
        Batch size: Self
        Train Std: Self
        Model name (LSTM1/LSTM2/Autoencoder/Rectangles): Self
        """
        if self.appliance_name == 'fridge':
            # "ukdale_new/house_1/fridge_12.pkl",
            self.train_link = ["ukdale_new/house_5/fridge_freezer_19.pkl",
            "refit_new/house_1/fridge_1.pkl", "refit_new/house_2/fridge_freezer_1.pkl",
            "refit_new/house_3/fridge_freezer_2.pkl", "refit_new/house_4/fridge_1.pkl"]
            self.test_link = ["ukdale_new/house_2/fridge_14.pkl", "refit_new/house_5/fridge_freezer_1.pkl"]
            self.window_size = 512
            self.batch_size = 64
            self.train_std = 250
            self.max_power = 300
            self.model_name = 'Seq2Pt'

        elif self.appliance_name == 'washer_dryer':
            self.train_link = ["ukdale_new/house_5/fridge_freezer_19.pkl",
            "refit_new/house_1/fridge_1.pkl", "refit_new/house_2/fridge_freezer_1.pkl",
            "refit_new/house_3/fridge_freezer_2.pkl", "refit_new/house_4/fridge_1.pkl"]
            self.test_link = ["ukdale_new/house_2/fridge_14.pkl", "refit_new/house_5/fridge_freezer_1.pkl"]
            self.window_size = 128
            self.batch_size = 64
            self.train_std = 250
            self.max_power = 300
            self.model_name = 'LSTM'

        elif self.appliance_name == 'dishwasher':
            self.train_link = ["ukdale_new/house_5/fridge_freezer_19.pkl",
            "refit_new/house_1/fridge_1.pkl", "refit_new/house_2/fridge_freezer_1.pkl",
            "refit_new/house_3/fridge_freezer_2.pkl", "refit_new/house_4/fridge_1.pkl"]
            self.test_link = ["ukdale_new/house_2/fridge_14.pkl", "refit_new/house_5/fridge_freezer_1.pkl"]
            self.window_size = 1536
            self.batch_size = 64
            self.train_std = 250
            self.max_power = 300
            self.model_name = 'LSTM'

        elif self.appliance_name == 'oven':
            self.train_link = ["ukdale_new/house_5/fridge_freezer_19.pkl",
            "refit_new/house_1/fridge_1.pkl", "refit_new/house_2/fridge_freezer_1.pkl",
            "refit_new/house_3/fridge_freezer_2.pkl", "refit_new/house_4/fridge_1.pkl"]
            self.test_link = ["ukdale_new/house_2/fridge_14.pkl", "refit_new/house_5/fridge_freezer_1.pkl"]
            self.window_size = 128
            self.batch_size = 64
            self.train_std = 250
            self.max_power = 300
            self.model_name = 'LSTM'

        elif self.appliance_name == 'kettle':
            self.train_link = ["ukdale_new/house_5/fridge_freezer_19.pkl",
            "refit_new/house_1/fridge_1.pkl", "refit_new/house_2/fridge_freezer_1.pkl",
            "refit_new/house_3/fridge_freezer_2.pkl", "refit_new/house_4/fridge_1.pkl"]
            self.test_link = ["ukdale_new/house_2/fridge_14.pkl", "refit_new/house_5/fridge_freezer_1.pkl"]
            self.window_size = 128
            self.batch_size = 64
            self.train_std = 250
            self.max_power = 300
            self.model_name = 'LSTM'

        elif self.appliance_name == 'microwave':
            self.train_link = ["ukdale_new/house_5/fridge_freezer_19.pkl",
            "refit_new/house_1/fridge_1.pkl", "refit_new/house_2/fridge_freezer_1.pkl",
            "refit_new/house_3/fridge_freezer_2.pkl", "refit_new/house_4/fridge_1.pkl"]
            self.test_link = ["ukdale_new/house_2/fridge_14.pkl", "refit_new/house_5/fridge_freezer_1.pkl"]
            self.window_size = 256
            self.batch_size = 64
            self.train_std = 250
            self.max_power = 300
            self.model_name = 'LSTM'

    def sliding_window_standardization(self, org_mains, start_idx, end_idx):
        mains_window_lst = []
        for i in tqdm(range(start_idx, end_idx)):
            small_window = np.array(org_mains[i: i + self.window_size])
            standardized_window = (small_window - np.mean(small_window)) / self.train_std
            mains_window_lst.append(standardized_window)
        return mains_window_lst

    def form_small_train_data(self, df):
        start_idx = self.start_nr * 10000
        if start_idx > round(df.shape[0] * 0.99):
            self.link_nr += 1
            self.start_idx = 0
            start_idx = 0
            end_idx = 10000
        else:
            df = df[: round(df.shape[0] * 0.99)]
            org_mains = np.array([0.0] * (self.window_size // 2) + df.mains.tolist() + [0.0] * (self.window_size // 2 + 1))
            train_responding_appliance_power = df[df.columns[2]].to_numpy() / self.max_power
            if start_idx + 10000 > df.shape[0] or df.shape[0] - (start_idx + 10000) <= self.window_size:
                end_idx = df.shape[0]
            else:
                end_idx = start_idx + 10000
            train_mains_window_lst = self.sliding_window_standardization(org_mains, start_idx, end_idx)
            self.train_mains_window_lst = np.array(train_mains_window_lst)
            train_responding_appliance_power = train_responding_appliance_power[start_idx: end_idx]
            self.train_responding_appliance_power = train_responding_appliance_power.reshape(train_responding_appliance_power.shape[0], 1)

    def form_val_data(self):
        val_mains_window_lst = []
        val_responding_appliance_power = []
        for link_idx, link in tqdm(enumerate(self.train_link)):
            df = pd.read_pickle(self.file_front_link + link)
            # Create train data
            if link_idx == self.link_nr:
                self.form_small_train_data(df)
            # Create val data
            df = df[round(df.shape[0] * 0.99): ]
            org_mains = np.array([0.0] * (self.window_size // 2) + df.mains.tolist() + [0.0] * (self.window_size // 2 + 1))
            sub_responding_appliance_power = list(df[df.columns[2]].to_numpy() / self.max_power)
            sub_mains_window_lst = self.sliding_window_standardization(org_mains, 0, df.shape[0])
            val_mains_window_lst += sub_mains_window_lst
            val_responding_appliance_power += sub_responding_appliance_power
        self.val_mains_window_lst = np.array(val_mains_window_lst)
        val_responding_appliance_power = np.array(val_responding_appliance_power)
        self.val_responding_appliance_power = val_responding_appliance_power.reshape(val_responding_appliance_power.shape[0], 1)

    def model_structure(self):
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
        if self.model_name == 'Seq2Pt':
            main_input = Input(shape = (self.window_size, 1), name = 'main_input')
            c1 = Conv1D(filters = 30, kernel_size = 10, strides = 1, activation = 'relu')(main_input)
            c2 = Conv1D(filters = 30, kernel_size = 8, strides = 1, activation = 'relu')(c1)
            c3 = Conv1D(filters = 40, kernel_size = 6, strides = 1, activation = 'relu')(c2)
            c4 = Conv1D(filters = 50, kernel_size = 5, strides = 1, activation = 'relu')(c3)
            c5 = Conv1D(filters = 50, kernel_size = 5, strides = 1, activation = 'relu')(c4)
            f1 = Flatten()(c5)
            d1 = Dense(1024, activation = 'linear')(f1)
            time_output = Dense(1, name = 'time_output')(d1)

        elif self.model_name == 'LSTM':
            main_input = Input(shape = (self.window_size, 1), name = 'main_input')
            c1 = Conv1D(16, 4, strides = 1, activation = 'linear')(main_input)
            l1 = Bidirectional(LSTM(128, return_sequences = True, dropout = 0.3))(c1)
            l2 = Bidirectional(LSTM(256, return_sequences = True, dropout = 0.3))(l1)
            d1 = Dense(128, activation = 'tanh')(l2)
            f1 = Flatten()(d1)
            time_output = Dense(1, name = 'time_output', activation = 'linear')(f1)

        elif self.model_name == 'Autoencoder':
            main_input = Input(shape = (self.window_size, 1), name = 'main_input')
            c1 = Conv1D(16, 4, strides = 1, activation = 'linear')(main_input)
            d1 = Dense((self.window_size - 3) * 8, activation = 'relu')(c1)
            d2 = Dense(128, activation = 'relu')(d1)
            d3 = Dense((self.window_size - 3) * 8, activation = 'relu')(d2)
            c2 = Conv1D(1, 4, strides = 1, activation = 'linear')(d3)
            f1 = Flatten()(c2)
            time_output = Dense(1, name = 'time_output')(f1)

        model = Model(inputs = [main_input], outputs = [time_output])
        opt = Adam(learning_rate = 0.01, epsilon = 1e-08, clipvalue = 3)

        # The loss used in model training is mean_squared_error because it is time prediction
        # The optimizer is Adam
        model.compile(loss = 'mean_squared_error', optimizer = opt)
        self.model = model

    def training_initial_setting(self):
        self.appliance_involved()
        self.form_val_data()
        if self.model_link == '':
            self.model_structure()
        else:
            self.model = load_model(self.model_link)
            print("Continual Learning")

    def training(self, epochs_nr):
        # Save the best model
        early_stopping = EarlyStopping(monitor = 'val_loss', patience = 50)
        checkpoint_filepath = self.file_front_link + f"{self.appliance_name}_{self.model_name}_{self.link_nr}_{self.start_nr}_"+ "epoch_{epoch:02d}.h5"
        model_checkpoint_callback = ModelCheckpoint(filepath = checkpoint_filepath, monitor = 'val_loss', mode = 'min', save_best_only = True)
        lr_reducer = ReduceLROnPlateau(monitor = 'val_loss', factor = 0.5, patience = 30, verbose = 0, mode = 'auto', min_delta = 0.0001, cooldown = 0, min_lr = 0)

        # Fit the model
        # Validation data is used here for evaluation during the training process
        self.model.fit(self.train_mains_window_lst, self.train_responding_appliance_power,
                validation_data = (self.val_mains_window_lst, self.val_responding_appliance_power),
                epochs = epochs_nr, batch_size = self.batch_size, callbacks = [early_stopping, model_checkpoint_callback, lr_reducer])

        print("Training Done!")

    def prediction(self, link, eval = False):
        self.appliance_involved()
        df = pd.read_pickle(link)
        org_mains = np.array([0.0] * (self.window_size // 2) + df.mains.tolist() + [0.0] * (self.window_size // 2 + 1))
        test_x = self.sliding_window_standardization(org_mains, 0, df.shape[0])
        self.model = load_model(self.model_link)
        test_predict = self.model.predict(test_x)
        test_predict *= self.max_power
        if not eval:
            return test_predict
        else:
            return mean_absolute_error(df[df.columns[2]].to_numpy(), test_predict), mean_squared_error(df[df.columns[2]].to_numpy(), test_predict) ** 0.5

####################################################################################################

dataset = Dataset(appliance_name = "fridge", model_link = "/content/drive/MyDrive/Appliance Models/Software/Models/fridge_Seq2Pt_0_5_epoch_19.h5", file_front_link = "/content/drive/MyDrive/Appliance Models/Software/Models/", link_nr = 0, start_nr = 6)
dataset.training_initial_setting()
with tf.device(device_name):
    dataset.training(epochs_nr = 100)
