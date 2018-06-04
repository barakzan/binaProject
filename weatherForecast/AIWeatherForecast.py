# -*- coding: utf-8 -*-
import pandas as pd
import datetime
import dataPreparation
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
import logging

logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)


def main():
    # TODO: take last ver by default
    version = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M")
    dataPreparation.prepare_data(version)
    df = pd.read_csv('..\\madridDataBase\\prepared_data_' + version + '.csv', sep=',', header=0)

    # split to train, validation and test sets
    # TODO: change test size to be a parameter
    train_and_validation, test = train_test_split(df, test_size=0.2, shuffle=False)
    train, validation = train_test_split(train_and_validation, test_size=0.2, shuffle=False)
    
    # train knn algorithm
    # TODO: make K a parameter
    knn_classifier = KNeighborsClassifier(n_neighbors=3)
    knn_classifier.fit(train.drop(['Mean TemperatureC'], axis=1), train['Mean TemperatureC'])
    test_predictions = knn_classifier.predict(test.drop(['Mean TemperatureC'], axis=1))

    print('accuracy:', metrics.accuracy_score(test['Mean TemperatureC'], test_predictions) * 100, '%')
    # print('precision:', metrics.precision_score(test['Mean TemperatureC'], test_predictions))
    # print('recall:', metrics.recall_score(test['Mean TemperatureC'], test_predictions))
    # print('f1 score:', metrics.f1_score(test['Mean TemperatureC'], test_predictions))
    
    logging.info("END TIME")
    # print("features_to_drop:\n", features_to_drop)
    

if __name__ == '__main__':
    main()
