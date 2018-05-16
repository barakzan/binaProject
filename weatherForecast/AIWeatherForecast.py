# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
#imports
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold
from sklearn.feature_selection import mutual_info_regression, mutual_info_classif, RFECV
from sklearn import metrics
import math
import logging

logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)

fileName = r'C:\Users\Ofir\Desktop\Technion\ProjectB\binaProject\madrid DataBase\weather_madrid_LEMD_1997_2015.csv'
labelName = 'CET'
toPredict = 'Mean TemperatureC'


def removeLowDataFeatures(df):
    """remove every feature that has at list half the data missing"""
    
    numOfDates = df.shape[0]
    df.dropna(thresh=(numOfDates - numOfDates/3), axis=1, inplace=True)


def _samp_dist(sample1, sample2):
    """return the distance between to samples"""
    return (sample1.subtract(sample2).abs()).sum()
   
    
def closest_fit(df, trainDf, useLabel=True, numOfSamples=100):
    """fill the nan in data set using closest fit"""
    logging.info("***start closest_fit")
    
    for idx, sample in df.iterrows():
        if(sample.isnull().any()):
            nacols = df.columns[sample.isnull() == True].tolist()
            if(not useLabel):
                nacols.append(labelName)
            
            samples = trainDf.sample(numOfSamples, axis=0).dropna(axis=0, how='any')
            while(samples.empty):
                samples = trainDf.sample(numOfSamples, axis=0).dropna(axis=0, how='any')
                
            minDist = np.inf
            minIdx = -1
            dist = np.inf
            for idx2, sample2 in samples.iterrows():
                dist = _samp_dist(sample.drop(nacols), sample2.drop(nacols))
                if (pd.notnull(dist) and dist < minDist):
                    minDist = dist
                    minIdx = idx2
            if(not useLabel):
                nacols.remove(labelName)
            df.loc[idx, nacols] = trainDf.loc[minIdx, nacols][0]


def _derive_nth_day_feature(df, feature, N):
    """derive feature for the N day"""
    
    rows = df.shape[0]
    nth_prior_measurements = [None]*N + [df[feature][i-N] for i in range(N, rows)]
    col_name = "{}_{}".format(feature, N)
    df[col_name] = nth_prior_measurements
    

def featuresDeriving(df, features, N):
    """derive all features in features list for all the previus N days"""
    logging.info("***start featuresDeriving")
    
    for feature in features:  
        if feature != 'date':
            for n in range(1, N+1):
                _derive_nth_day_feature(df, feature, n)
        

def main():
    logging.info("START TIME")
    
    """Load the weather history"""
    df = pd.read_csv(fileName, sep=',', header=0)
    rawData = df.copy()
    
    removeLowDataFeatures(df)
    features = df.columns.tolist()
    
    #Imputation - fill missing values
    closest_fit(df, df, False, 10)
    
    #create feature for each date based on N previous days
    featuresToDerive = ['Mean TemperatureC', 'MeanDew PointC', ' Mean Humidity', ' Min Humidity', ' Mean Sea Level PressurehPa',
                        ' Mean VisibilityKm', ' Mean Wind SpeedKm/h', 'Precipitationmm', ' CloudCover', 'WindDirDegrees']
    featuresDeriving(df, featuresToDerive, 5)
    
    #drop dates with missing data
    df.dropna(how='any', inplace=True)
    
    #drop columns not in use
    features.remove(toPredict)
    df.drop(features, axis=1, inplace=True)
    
    #remove redandunt feature using pearson corralation and mutual information
    
    #split to train, validation and test sets
    train, test = train_test_split(df, test_size=0.2, shuffle=False)
    train, validation = train_test_split(train, test_size=0.2, shuffle=False)
    
    #train knn algorithem
    neigh = KNeighborsClassifier(n_neighbors=3)
    neigh.fit(train.drop(['Mean TemperatureC'], axis=1), train['Mean TemperatureC'])
    test_pred = neigh.predict(test.drop(['Mean TemperatureC'], axis=1))

    print('accuracy:', metrics.accuracy_score(test['Mean TemperatureC'], test_pred))
    #print('precision:', metrics.precision_score(test['Mean TemperatureC'], test_pred))
    #print('recall:', metrics.recall_score(test['Mean TemperatureC'], test_pred))
    #print('f1 score:', metrics.f1_score(test['Mean TemperatureC'], test_pred))
    
    logging.info("END TIME")
    #print("features_to_drop:\n", features_to_drop)
    

if __name__ == '__main__':
    main()
