# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import logging

logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)

raw_data_file = r'..\madridDataBase\weather_madrid_LEMD_1997_2015.csv'
labelName = 'CET'
toPredict = 'Mean TemperatureC'


def remove_low_data_features(df, threshold=2 / 3):
    """remove every feature that has at list threshold the data missing"""

    num_of_days = df.shape[0]
    df.dropna(thresh=(num_of_days * threshold), axis=1, inplace=True)


def _sample_dist(sample1, sample2):
    """return the distance between to samples"""
    return (sample1.subtract(sample2).abs()).sum()


def closest_fit(df, trainDf, numOfSamples=100, useLabel=False):
    """fill the nan in data set using closest fit"""
    logging.info("***start closest_fit")

    for idx, sample in df.iterrows():
        if sample.isnull().any():
            null_columns = df.columns[sample.isnull() == True].tolist()
            if not useLabel:
                null_columns.append(labelName)

            samples = trainDf.sample(numOfSamples, axis=0).dropna(axis=0, how='any')
            while samples.empty:
                samples = trainDf.sample(numOfSamples, axis=0).dropna(axis=0, how='any')

            min_dist = np.inf
            min_index = -1

            for idx2, sample2 in samples.iterrows():
                dist = _sample_dist(sample.drop(null_columns), sample2.drop(null_columns))
                if pd.notnull(dist) and dist < min_dist:
                    min_dist = dist
                    min_index = idx2
            if not useLabel:
                null_columns.remove(labelName)
            df.loc[idx, null_columns] = trainDf.loc[min_index, null_columns][0]


def _derive_nth_day_feature(df, feature, N):
    """derive feature for the N day"""

    rows = df.shape[0]
    nth_prior_measurements = [None] * N + [df[feature][i - N] for i in range(N, rows)]
    col_name = "{}_{}".format(feature, N)
    df[col_name] = nth_prior_measurements


def sign_sqrt(x):
    return np.sign(x)*np.sqrt(np.abs(x))


def sign_square(x):
    return np.sign(x)*np.square(x)


def features_trend(df, features, x):
    for feature in features:
        df[feature+"trend_in_last" + str(x) +"_days"] = df[feature].diff()\
                .apply(sign_sqrt).rolling(window=x).mean().apply(sign_square)

def features_deriving(df, features, N):
    """derive all features in features list for all the previus N days"""
    logging.info("***start features_deriving")

    for feature in features:
        if feature != 'date':
            for n in range(1, N + 1):
                _derive_nth_day_feature(df, feature, n)


def prepare_data(output_file_version=''):
    logging.info("START TIME")

    """Load the weather history"""
    df = pd.read_csv(raw_data_file, sep=',', header=0)

    remove_low_data_features(df)
    features = df.columns.tolist()

    # Imputation - fill missing values
    closest_fit(df, df, 10)

    # create feature for each date based on N previous days
    features_to_derive = ['Mean TemperatureC', 'MeanDew PointC', ' Mean Humidity', ' Min Humidity',
                          ' Mean Sea Level PressurehPa', ' Mean VisibilityKm', ' Mean Wind SpeedKm/h',
                          'Precipitationmm', ' CloudCover', 'WindDirDegrees']

    features_deriving(df, features_to_derive, 5)

    # TODO: most of the work is here

    # create feature for each date based on 3 and 6 previous days
    # TODO: add features_to_calculate
    features_to_calculate_trend = ['Mean TemperatureC', 'MeanDew PointC']
    features_trend(df, features_to_calculate_trend, 3)
    features_trend(df, features_to_calculate_trend, 6)

    # drop dates with missing data
    df.dropna(how='any', inplace=True)

    # drop columns not in use
    features.remove(toPredict)
    df.drop(features, axis=1, inplace=True)
    
    # TODO: remove redundant feature using pearson correlation and mutual information

    df.to_csv('..\\madridDataBase\\prepared_data_' + output_file_version + '.csv')


if __name__ == "__main__":
    prepare_data()

