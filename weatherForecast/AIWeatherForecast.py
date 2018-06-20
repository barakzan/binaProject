# -*- coding: utf-8 -*-
import pandas as pd
import dataPreparation
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
import logging

logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)
make_new_database = True


def calculate_precision(df, x, feature):
    """calculate the precision of the algorithm up to x degrees"""
    logging.info("calculate_precision")

    for i in range(x):
        df["precision_up_to_" + str(i) + "_degrees"] = np.where(df[feature] <= i, 1, 0)


def calculate_results(results_df, print_results=False):
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

    if print_results:
        print('******************')
        print("zero degree accuracy = ", round(accurate, 2))
        print("one degrees accuracy = ", round(one_degree, 2))
        print("two degrees accuracy = ", round(two_degrees, 2))
        print("three degrees accuracy = ", round(three_degrees, 2))
        print("- - - - - - - - - - - - - - - - - - - ")
        print("parameter_to_improve = ", round(parameter_to_improve, 3))
        print('******************')

    return [accurate, one_degree, two_degrees, three_degrees]


def forecast(days_before=7, use_forest_instead_of_tree=True, fill_missing_from_previous_day=True):
    trending_before = days_before
    average_before = days_before

    madrid_file_name = dataPreparation.get_file_name('madrid', days_before, fill_missing_from_previous_day)
    madrid_file = Path(madrid_file_name)
    if not madrid_file.is_file():
        dataPreparation.prepare_data('madrid', days_before, trending_before, average_before,
                                     fill_missing_from_previous_day)

    austin_file_name = dataPreparation.get_file_name('austin', days_before, fill_missing_from_previous_day)
    austin_file = Path(austin_file_name)
    if not austin_file.is_file():
        dataPreparation.prepare_data('austin', days_before, trending_before, average_before,
                                     fill_missing_from_previous_day)

    madrid_df = pd.read_csv(madrid_file_name, sep=',', header=0)
    austin_df = pd.read_csv(austin_file_name, sep=',', header=0)

    # split to train, validation and test sets in chronological order 20 20 60
    train_and_validation, test = train_test_split(madrid_df, test_size=0.2, shuffle=False)
    train, validation = train_test_split(train_and_validation, test_size=0.2, shuffle=False)

    if use_forest_instead_of_tree:
        classifier = RandomForestClassifier(n_estimators=22, bootstrap=False)
    else:
        # train the decision tree classifier
        classifier = DecisionTreeClassifier()

    classifier.fit(train.drop(['Mean TemperatureC'], axis=1), train['Mean TemperatureC'])
    madrid_test_predictions = classifier.predict(test.drop(['Mean TemperatureC'], axis=1))
    austin_test_predictions = classifier.predict(austin_df.drop(['Mean TemperatureC'], axis=1))

    madrid_results_df = pd.DataFrame({'real_value': test['Mean TemperatureC'],
                                      'test_predictions': madrid_test_predictions})
    austin_results_df = pd.DataFrame({'real_value': austin_df['Mean TemperatureC'],
                                      'test_predictions': austin_test_predictions})
    madrid_position = calculate_results(madrid_results_df, print_results=False)
    austin_position = calculate_results(austin_results_df, print_results=False)

    logging.info("END TIME")

    return [madrid_position, austin_position]


if __name__ == '__main__':
    forecast()
