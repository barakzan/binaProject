# -*- coding: utf-8 -*-
import pandas as pd
import dataPreparation
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
from sklearn import tree
from sklearn.datasets import load_iris
from sklearn.externals.six import StringIO
import pydot
import logging

logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)
make_new_database = True


def calculate_precision(df, x, feature):
    """calculate the precision of the algorithm up to x degrees"""
    #logging.info("calculate_precision")

    for i in range(x):
        df["precision_up_to_" + str(i) + "_degrees"] = np.where(df[feature] <= i, 1, 0)


def calculate_results(results_df, print_results=False):
    results_df['abs_diff'] = abs(round(results_df['real_value'] + 0.5) - results_df['validation_predictions'])
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

    if print_results:
        print("\tzero degree accuracy = ", round(accurate, 2))
        print("\tone degrees accuracy = ", round(one_degree, 2))
        print("\ttwo degrees accuracy = ", round(two_degrees, 2))
        print("\tthree degrees accuracy = ", round(three_degrees, 2))

    return [accurate, one_degree, two_degrees, three_degrees]


def forecast(days_before=7, classifier_type='forest', fill_missing_from_previous_day=True, max_depth=10):
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
    train, validation = train_test_split(train_and_validation, test_size=0.25, shuffle=False)

    # makes a new joined data frame
    joined_df = pd.concat([austin_df, train], ignore_index=True)

    # choose tree, forest or linear classifier
    if classifier_type == 'forest':
        classifier = RandomForestClassifier(n_estimators=22, bootstrap=False, max_depth=max_depth)
        joined_classifier = RandomForestClassifier(n_estimators=22, bootstrap=False, max_depth=max_depth)
    elif classifier_type == 'tree':
        # train the decision tree classifier
        classifier = DecisionTreeClassifier()
        joined_classifier = DecisionTreeClassifier()
    else:
        classifier = LinearRegression()
        joined_classifier = LinearRegression()

    # train madrid classifier
    classifier.fit(train.drop(['Mean TemperatureC'], axis=1), train['Mean TemperatureC'])

    # predict
    madrid_validation_predictions = classifier.predict(validation.drop(['Mean TemperatureC'], axis=1))
    austin_validation_predictions = classifier.predict(austin_df.drop(['Mean TemperatureC'], axis=1))

    # train joined classifier
    joined_classifier.fit(joined_df.drop(['Mean TemperatureC'], axis=1), joined_df['Mean TemperatureC'])

    # predict
    madrid_joined_validation_predictions = joined_classifier.predict(validation.drop(['Mean TemperatureC'], axis=1))

    # get results
    madrid_results_df = pd.DataFrame({'real_value': validation['Mean TemperatureC'],
                                      'validation_predictions': madrid_validation_predictions})
    austin_results_df = pd.DataFrame({'real_value': austin_df['Mean TemperatureC'],
                                      'validation_predictions': austin_validation_predictions})
    madrid_joined_results_df = pd.DataFrame({'real_value': validation['Mean TemperatureC'],
                                             'validation_predictions': madrid_joined_validation_predictions})
    # calculate precision
    madrid_precision = calculate_results(madrid_results_df, print_results=False)
    austin_precision = calculate_results(austin_results_df, print_results=False)
    madrid_joined_precision = calculate_results(madrid_joined_results_df, print_results=False)

    folds = 10
    kf = KFold(n_splits=folds)
    fold_precision = [0, 0, 0, 0]
    for train_index, validation_index in kf.split(train):
        if classifier_type == 'linear':
            break
        classifier.fit(train.iloc[train_index].drop(['Mean TemperatureC'], axis=1), train.iloc[train_index]['Mean TemperatureC'])
        madrid_validation_predictions = classifier.predict(train.iloc[validation_index].drop(['Mean TemperatureC'], axis=1))
        kfold_results_df = pd.DataFrame({'real_value': train.iloc[validation_index]['Mean TemperatureC'],
                                          'validation_predictions': madrid_validation_predictions})
        madrid_precision = calculate_results(kfold_results_df, print_results=False)
        fold_precision = [x + y for x, y in zip(fold_precision, madrid_precision)]

    fold_precision = [fold_precision[0] / folds, fold_precision[1] / folds, fold_precision[2] / folds, fold_precision[3] / folds]

    logging.info("END forecast: " + str(days_before) + " days before, classifier: " + str(classifier_type))
    return [madrid_precision, austin_precision, madrid_joined_precision, fold_precision]


def final_project_results(days_before=7, classifier_type='tree', fill_missing_from_previous_day=True, max_depth=10):
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

    train, test = train_test_split(madrid_df, test_size=0.2, shuffle=False)

    joined_df = pd.concat([austin_df, train], ignore_index=True)

    # choose tree, forest or linear classifier
    if classifier_type == 'forest':
        classifier = RandomForestClassifier(n_estimators=22, bootstrap=False, max_depth=max_depth)
        joined_classifier = RandomForestClassifier(n_estimators=22, bootstrap=False, max_depth=max_depth)
    elif classifier_type == 'tree':
        # train the decision tree classifier
        classifier = DecisionTreeClassifier()
        joined_classifier = DecisionTreeClassifier()
    else:
        classifier = LinearRegression()
        joined_classifier = LinearRegression()

    # train madrid classifier
    classifier.fit(train.drop(['Mean TemperatureC'], axis=1), train['Mean TemperatureC'])

    # train joined classifier
    joined_classifier.fit(joined_df.drop(['Mean TemperatureC'], axis=1), joined_df['Mean TemperatureC'])

    # predict
    madrid_test_predictions = classifier.predict(test.drop(['Mean TemperatureC'], axis=1))
    austin_test_predictions = classifier.predict(austin_df.drop(['Mean TemperatureC'], axis=1))
    madrid_joined_test_predictions = joined_classifier.predict(test.drop(['Mean TemperatureC'], axis=1))

    # get results
    madrid_results_df = pd.DataFrame({'real_value': test['Mean TemperatureC'],
                                      'validation_predictions': madrid_test_predictions})
    austin_results_df = pd.DataFrame({'real_value': austin_df['Mean TemperatureC'],
                                      'validation_predictions': austin_test_predictions})
    madrid_joined_results_df = pd.DataFrame({'real_value': test['Mean TemperatureC'],
                                             'validation_predictions': madrid_joined_test_predictions})
    # calculate precision
    print("\nmadrid precision:")
    calculate_results(madrid_results_df, print_results=True)
    print("\naustin precision:")
    calculate_results(austin_results_df, print_results=True)
    print("\nmadrid joined precision:")
    calculate_results(madrid_joined_results_df, print_results=True)
    print("\nyesterday precision:")
    predict_yesterday(print_results=True)

    # print classifier tree
    iris = load_iris()
    clf = classifier.fit(iris.data, iris.target)
    tree.export_graphviz(clf, out_file='tree.dot')

    dot_data = StringIO()
    tree.export_graphviz(clf, out_file=dot_data)
    graph = pydot.graph_from_dot_data(dot_data.getvalue())
    graph[0].write_pdf("iris.pdf")


def predict_yesterday(print_results):
    madrid_file_name = dataPreparation.get_file_name('madrid', 1, fill_missing_from_previous_day=True)
    madrid_file = Path(madrid_file_name)
    if not madrid_file.is_file():
        dataPreparation.prepare_data('madrid', 1, 1, 1, fill_missing_from_previous_day=True)

    madrid_df = pd.read_csv(madrid_file_name, sep=',', header=0)

    madrid_yesterday_results_df = pd.DataFrame({'real_value': madrid_df['Mean TemperatureC'],
                                                'validation_predictions': madrid_df['Mean TemperatureC'].shift(1)})
    return calculate_results(madrid_yesterday_results_df, print_results)


if __name__ == '__main__':
    forecast()
