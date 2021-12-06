#!/usr/bin/env python3
# -*- coding: utf-8 -*-
##
# @file     scalable_indoor_localization.py
# @author   Kyeong Soo (Joseph) Kim <kyeongsoo.kim@gmail.com>
# @date     2017-11-15
#           2020-12-20 updated for TensorFlow ver. 2.x
#           2021-12-01 use EarlyStopping
#
# @brief    Build and evaluate a scalable indoor localization system
#           based on Wi-Fi fingerprinting using a neural-network-based
#           multi-label classifier.
#
# @remarks  This work is based on the <a href="https://keras.io/">Keras</a>-based
#           implementation of the system described in "<a
#           href="https://arxiv.org/abs/1611.02049v2">Low-effort place
#           recognition with WiFi fingerprints using deep learning</a>".
#
#           The results are published in the following paper:
#           Kyeong Soo Kim, Sanghyuk Lee, and Kaizhu Huang "A scalable deep
#           neural network architecture for multi-building and multi-floor
#           indoor localization based on Wi-Fi fingerprinting," Big Data
#           Analytics, vol. 3, no. 4, pp. 1-17, Apr. 19, 2018. Available online:
#           https://doi.org/10.1186/s41044-018-0031-2
#

# import modules
import math
import numpy as np
import pandas as pd
import random
import streamlit as st
from sklearn.preprocessing import scale
from tensorflow.keras.models import load_model

# app title
st.title('Wi-Fi Indoor Localization')

# define global constants
path_train = '../data/UJIIndoorLoc/trainingData2.csv'           # '-110' for the lack of AP.
path_validation = '../data/UJIIndoorLoc/validationData2.csv'    # ditto
path_model = 'my_model'
N = 8
scaling = 0.2
training_ratio = 0.9

# read both train and test dataframes for consistent label formation through one-hot encoding
train_df = pd.read_csv(path_train, header=0)  # pass header=0 to be able to replace existing names
test_df = pd.read_csv(path_validation, header=0)

train_AP_features = scale(np.asarray(train_df.iloc[:,0:520]).astype(float), axis=1) # convert integer to float and scale jointly (axis=1)
train_df['REFPOINT'] = train_df.apply(lambda row: str(int(row['SPACEID'])) + str(int(row['RELATIVEPOSITION'])), axis=1) # add a new column

blds = np.unique(train_df[['BUILDINGID']])
flrs = np.unique(train_df[['FLOOR']])
x_avg = {}
y_avg = {}
for bld in blds:
    for flr in flrs:
        # map reference points to sequential IDs per building-floor before building labels
        cond = (train_df['BUILDINGID']==bld) & (train_df['FLOOR']==flr)
        _, idx = np.unique(train_df.loc[cond, 'REFPOINT'], return_inverse=True) # refer to numpy.unique manual
        train_df.loc[cond, 'REFPOINT'] = idx
        
        # calculate the average coordinates of each building/floor
        x_avg[str(bld) + '-' + str(flr)] = np.mean(train_df.loc[cond, 'LONGITUDE'])
        y_avg[str(bld) + '-' + str(flr)] = np.mean(train_df.loc[cond, 'LATITUDE'])

# build labels for multi-label classification
len_train = len(train_df)
blds_all = np.asarray(pd.get_dummies(pd.concat([train_df['BUILDINGID'], test_df['BUILDINGID']]))) # for consistency in one-hot encoding for both dataframes
flrs_all = np.asarray(pd.get_dummies(pd.concat([train_df['FLOOR'], test_df['FLOOR']]))) # ditto
blds = blds_all[:len_train]
flrs = flrs_all[:len_train]
rfps = np.asarray(pd.get_dummies(train_df['REFPOINT']))
train_labels = np.concatenate((blds, flrs, rfps), axis=1)

# split the training set into training and validation sets; we will use the
# validation set at a testing set.
train_val_split = np.random.rand(len(train_AP_features)) < training_ratio # mask index array
x_train = train_AP_features[train_val_split]
y_train = train_labels[train_val_split]
x_val = train_AP_features[~train_val_split]
y_val = train_labels[~train_val_split]

### build and train a complete model with the trained SAE encoder and a new classifier
# st.write("\nLoading the saved model ...\n")
model = load_model(path_model)

# turn the given validation set into a testing set
test_AP_features = scale(np.asarray(test_df.iloc[:,0:520]).astype(float), axis=1)  # convert integer to float and scale jointly (axis=1)
x_test_utm = np.asarray(test_df['LONGITUDE'])
y_test_utm = np.asarray(test_df['LATITUDE'])
blds = blds_all[len_train:]
flrs = flrs_all[len_train:]

# row_number = int(input('Insert a number（the row number of test dataset that you choose）'))
n_rows = test_df.shape[0]
row_number = st.slider('Input a row number', min_value=0, max_value=n_rows-1)
# row_number = st.number_input('Input a row number', min_value=0, max_value=n_rows-1)
test_row = test_df.iloc[[row_number]]
test_rss = test_AP_features[[row_number]]

# calculate the accuracy of building and floor estimation
preds = model(test_rss, training=False)[0].numpy()

# calculate positioning error when building and floor are correctly estimated
x = 0.0
x_weighted = 0.0
y = 0.0
y_weighted = 0.0
pos_err = -1.0 # initial value as an indicator of no processing
pos_err_weighted = -1.0 # ditto
if test_row['BUILDINGID'].values[0] == np.argmax(preds[:3]) and test_row['FLOOR'].values[0] == np.argmax(preds[3:8]):
    x_test_utm = x_test_utm[row_number]
    y_test_utm = y_test_utm[row_number]
    blds = blds[row_number]
    flrs = flrs[row_number]
    rfps = preds[8:118]
    idxs = np.argpartition(rfps, -N)[-N:]  # (unsorted) indexes of up to N nearest neighbors
    threshold = scaling*np.amax(rfps)
    xs = []
    ys = []
    ws = []
    for i in idxs:
        rfp = np.zeros(110)
        rfp[i] = 1
        rows = np.where((train_labels == np.concatenate((blds, flrs, rfp))).all(axis=1))[0]
        if rows.size > 0:
            if rfps[i] >= threshold:
                xs.append(train_df.loc[train_df.index[rows[0]], 'LONGITUDE'])
                ys.append(train_df.loc[train_df.index[rows[0]], 'LATITUDE'])
                ws.append(rfps[i])
    if len(xs) > 0:
        x = np.mean(xs)
        y = np.mean(ys)
        pos_err = math.sqrt((x-x_test_utm)**2 + (y-y_test_utm)**2)
        x_weighted = np.average(xs, weights=ws)
        y_weighted = np.average(ys, weights=ws)
        pos_err_weighted = math.sqrt((x_weighted-x_test_utm)**2 + (y_weighted-y_test_utm)**2)
    else:
        key = str(np.argmax(blds)) + '-' + str(np.argmax(flrs))
        x = x_weighted = x_avg[key]
        y = y_weighted = y_avg[key]
        pos_err = pos_err_weighted = math.sqrt((x-x_test_utm)**2 + (y-y_test_utm)**2)

### display input and output
col1, col2 = st.columns(2)

with col1:
    st.write('\nTrue location')
    st.write('- Row number:\t', row_number)
    st.write('- Building ID:\t', test_row['BUILDINGID'].values[0])
    st.write('- Floor ID:\t', test_row['FLOOR'].values[0])
    st.write('- Coordinates')
    st.write('  + X:\t', test_row['LONGITUDE'].values[0])
    st.write('  + Y:\t', test_row['LATITUDE'].values[0])

with col2:
    if pos_err >= 0:
        st.write('\nEsimated location')
        st.write('- Building ID:\t', np.argmax(preds[:3]))
        st.write('- Floor ID:\t', np.argmax(preds[3:8]))
        st.write('- Coordinates')
        st.write('  + X:\t', x)
        st.write('  + Y:\t', y)
        st.write('  + Positioning error [m]:\t', pos_err)
        st.write('- Coordinates (weighted)')
        st.write('  + X:\t', x)
        st.write('  + Y:\t', y)
        st.write('  + Positioning error [m]:\t', pos_err)
    else:
        st.write('  + Building/Floor estimation failed!')