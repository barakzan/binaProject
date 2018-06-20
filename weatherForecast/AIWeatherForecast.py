# -*- coding: utf-8 -*-
import pandas as pd
import dataPreparation
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
import logging

logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)
make_new_database = False
use_forest_instead_of_tree = True


def calculate_precision(df, x, feature):
    """calculate the precision of the algorithm up to x degrees"""
    logging.info("calculate_precision")

    for i in range(x):
        df["precision_up_to_" + str(i) + "_degrees"] = np.where(df[feature] <= i, 1, 0)


def forecast():
    if make_new_database:
        version = dataPreparation.prepare_data(fill_missing_from_previous_day=False,
                                               days_before=7, trending_before=14, average_before=14)
        file = open("last_version.txt", 'w')
        file.write(version)
    else:
        file = open("last_version.txt", 'r')
        version = file.read()
    df = pd.read_csv('..\\madridDataBase\\prepared_data_' + version + '.csv', sep=',', header=0)

    # split to train, validation and test sets in chronological order 20 20 60
    train_and_validation, test = train_test_split(df, test_size=0.2, shuffle=False)
    train, validation = train_test_split(train_and_validation, test_size=0.2, shuffle=False)

    if use_forest_instead_of_tree:
        classifier = RandomForestClassifier(n_estimators=22, bootstrap=False)
    else:
        # train the decision tree classifier
        classifier = DecisionTreeClassifier()

    classifier.fit(train.drop(['Mean TemperatureC'], axis=1), train['Mean TemperatureC'])
    test_predictions = classifier.predict(test.drop(['Mean TemperatureC'], axis=1))

    results_df = pd.DataFrame({'real_value': test['Mean TemperatureC'], 'test_predictions': test_predictions})
    results_df['abs_diff'] = abs(results_df['real_value'] - results_df['test_predictions'])
    calculate_precision(results_df, 4, 'abs_diff')
    total = results_df["precision_up_to_1_degrees"].count()
    accurate = results_df["precision_up_to_0_degrees"].sum()
    one_degree = results_df["precision_up_to_1_degrees"].sum()
    two_degrees = results_df["precision_up_to_2_degrees"].sum()
    three_degrees = results_df["precision_up_to_3_degrees"].sum()
    accurate = (accurate/total)*100
    one_degree = (one_degree/total)*100
    two_degrees = (two_degrees/total)*100
    three_degrees = (three_degrees/total)*100
    parameter_to_improve = (3*accurate + 2*one_degree + two_degrees) / 6

    print('******************')
    print('******************')

    print("zero degree accuracy = ", round(accurate, 2))
    print("one degrees accuracy = ", round(one_degree, 2))
    print("two degrees accuracy = ", round(two_degrees, 2))
    print("three degrees accuracy = ", round(three_degrees, 2))
    print("- - - - - - - - - - - - - - - - - - - ")
    print("parameter_to_improve = ", round(parameter_to_improve, 3))

    # sprint(results_df['abs_diff'].value_counts())
    print('******************')
    print('******************')

    logging.info("END TIME")
    # print("features_to_drop:\n", features_to_drop)
    

if __name__ == '__main__':
    forecast()
