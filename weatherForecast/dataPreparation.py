# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import logging
import datetime

logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)

raw_data_file = r'..\madridDataBase\weather_madrid_LEMD_1997_2015.csv'
labelName = 'CET'
toPredict = 'Mean TemperatureC'


def remove_low_data_features(df, threshold=2 / 3):
    """remove every feature that has at list threshold the data missing"""
    logging.info("remove_low_data_features")

    num_of_days = df.shape[0]
    df.dropna(thresh=(num_of_days * threshold), axis=1, inplace=True)


def _sample_dist(sample1, sample2):
    """return the distance between to samples"""

    return (sample1.subtract(sample2).abs()).sum()


def fill_from_previous_day(df):
    """fill the nan in data set using closest fit"""
    logging.info("fill_from_previous_day")

    df.fillna(method='ffill', inplace=True)


def closest_fit(df, trainDf, numOfSamples=100, useLabel=False):
    """fill the nan in data set using closest fit"""
    logging.info("closest_fit")

    for idx, sample in df.iterrows():
        if sample.isnull().any():
            null_columns = df.columns[sample.isnull()].tolist()
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
    col_name = "{}_{}_days_before".format(feature, N)
    df[col_name] = nth_prior_measurements


def _sign_sqrt(x):
    """calculate sign sqrt"""

    return np.sign(x)*np.sqrt(np.abs(x))


def _sign_square(x):
    """calculate sign sqrt"""

    return np.sign(x)*np.square(x)


def features_last_x_days_average(df, features, x):
    """calculate the average of last x days for feature in features"""
    logging.info("features_last_" +str(x) + "_days_average")

    for feature in features:
        df[feature+"average_of_last_" + str(x) +"_days"] = df[feature].rolling(window=x).mean()


def features_trend_type1(df, features, x):
    """calculate the trend of last x days for feature in features"""
    logging.info("features_trend_type1 of last " + str(x) + " days")

    for feature in features:
        df[feature+"trend_in_last_" + str(x) +"_days"] = df[feature].diff()\
                .apply(_sign_sqrt).rolling(window=x).mean().apply(_sign_square)


def features_trend_type2(df, features, x):
    """calculate the trend of last x days for feature in features"""
    logging.info("features_trend_type2 of last " + str(x) + " days")

    for feature in features:
        df[feature + "trend_in_last_" + str(x) + "_days"] = df[feature].diff() \
            .apply(_sign_square).rolling(window=x).mean().apply(_sign_sqrt)


def features_deriving(df, features, x):
    """derive all features in features list for all the previus N days"""
    logging.info("start features_deriving")

    for feature in features:
        if feature != 'date':
            for n in range(1, x + 1):
                _derive_nth_day_feature(df, feature, n)


def prepare_data(days_before=3, trending_before=7, average_before=7,
                 fill_missing_from_previous_day=True):

    logging.info("START TIME")

    """Load the weather history"""
    df = pd.read_csv(raw_data_file, sep=',', header=0)

    # remove features that dose not have enough days with data on them
    remove_low_data_features(df)
    features = df.columns.tolist()

    # Imputation - fill missing values
    if fill_missing_from_previous_day is True:
        fill_from_previous_day(df)
    else:
        closest_fit(df, df, 10)

    # create feature for each date based on N previous days
    features_to_derive = list(features)
    features_to_derive.remove('CET')
    features_deriving(df, features_to_derive, days_before)

    # create feature of the average for each date based on x previous days
    features_to_calculate_average = features_to_derive
    if average_before > 2:
        for i in range(2, average_before):
            features_last_x_days_average(df, features_to_calculate_average, i)

    # create feature of the trend for each date based on x previous days
    features_to_calculate_trend = features_to_derive

    if trending_before >= 2:
        for i in range(2, trending_before + 1):
            features_trend_type1(df, features_to_calculate_trend, i)
            features_trend_type2(df, features_to_calculate_trend, i)

    # drop dates with missing data
    df.dropna(how='any', inplace=True)

    # drop columns not in use
    features.remove(toPredict)
    df.drop(features, axis=1, inplace=True)

    version = "{}__days_before={}_trending_before={}_average_before={}_fill_missing_from_previous_day={}"\
        .format(datetime.datetime.now().strftime("%Y_%m_%d__%H-%M"), days_before, trending_before, average_before,
                fill_missing_from_previous_day)
    df.to_csv('..\\madridDataBase\\prepared_data_' + version + '.csv')
    return version


if __name__ == "__main__":
    prepare_data()

