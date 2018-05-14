# -*- coding: utf-8 -*-
#imports
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import mutual_info_regression, mutual_info_classif, RFE
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsClassifier
import math
import logging

logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)

LabelName = 'VoteInt'
dictFeatures = {'Occupation_Satisfaction' : 'Uniform',
       'Avg_monthly_expense_when_under_age_21' : 'None' ,
        'AVG_lottary_expanses' : 'None',
       'Avg_Satisfaction_with_previous_vote' : 'None',
       'Garden_sqr_meter_per_person_in_residancy_area' : 'Normal',
       'Financial_balance_score_(0-1)' : 'Uniform',
        '%Of_Household_Income' : 'Uniform',
       'Avg_government_satisfaction' : 'Uniform',
        'Avg_education_importance' : 'Uniform',
       'Avg_environmental_importance' : 'Uniform',
        'Avg_Residancy_Altitude' : 'Uniform',
       'Yearly_ExpensesK' : 'Uniform', 
        '%Time_invested_in_work' : 'Uniform',
        'Yearly_IncomeK' : 'Normal',
       'Avg_monthly_expense_on_pets_or_plants' : 'Normal',
        'Avg_monthly_household_cost' : 'Normal',
       'Phone_minutes_10_years' : 'None', 
        'Avg_size_per_room' : 'Normal',
       'Weighted_education_rank' : 'Normal', 
        '%_satisfaction_financial_policy' : 'Uniform',
       'Avg_monthly_income_all_years' : 'None', 
        'Last_school_grades' : 'None',
       'Number_of_differnt_parties_voted_for' : 'Normal',
       'Political_interest_Total_Score' : 'Normal', 
        'Number_of_valued_Kneset_members' : 'Uniform',
       'Overall_happiness_score' : 'Normal', 
        'Num_of_kids_born_last_10_years' : 'None',
       'Age_groupInt' : 'Uniform', 
        'Voting_TimeInt' : 'Uniform'}


def identifyFeatures(ds):
    """Identify and set the correct type of each attribute"""
    logging.info("***start identifyFeatures")
    
    ObjFeat=ds.keys()[ds.dtypes.map(lambda x: x=='object')]
    ObjFeatOrdinal = ['Voting_Time', 'Age_group', 'Vote']
    ObjFeatNominal = ObjFeat.drop(ObjFeatOrdinal)
    
    # Transform the original features to categorical
    # Creat new 'int' features, resp.
    f = 'Age_group'
    ds[f] = ds[f].astype("category", ordered=True, categories=["Below_30", "30-45", "45_and_up"])
    ds[f+"Int"] = ds[f].cat.rename_categories(range(ds[f].nunique())).astype(int)
    ds.loc[ds[f].isnull(), f+"Int"] = np.nan #fix NaN conversion
    
    f = 'Voting_Time'
    ds[f] = ds[f].astype("category", ordered=True, categories=["By_16:00", "After_16:00"])
    ds[f+"Int"] = ds[f].cat.rename_categories(range(ds[f].nunique())).astype(int)
    ds.loc[ds[f].isnull(), f+"Int"] = np.nan #fix NaN conversion
        
    f = 'Vote'
    ds[f] = ds[f].astype("category")
    ds[f+"Int"] = ds[f].cat.rename_categories(range(ds[f].nunique())).astype(int)
    ds.loc[ds[f].isnull(), f+"Int"] = np.nan #fix NaN conversion

    # Create one hot features
    for f in ObjFeatNominal:
        f_dummies = pd.get_dummies(ds[f], prefix=f)
        ds = pd.concat([ds, f_dummies], axis=1)
        
    #Cleaning - drop the newly created columns
    ds.drop(ObjFeat, axis=1, inplace=True)
    return ds


def samp_dist(sample1, sample2):
    """return the distance between to samples"""
    return (sample1.subtract(sample2).abs()).sum()
   
    
def closest_fit(ds, trainDs, useLabel=True, numOfSamples=1000):
    """fill the nan in data set using closest fit"""
    logging.info("***start closest_fit")
    
    for idx, sample in ds.iterrows():
        if(sample.isnull().any()):
            nacols = ds.columns[sample.isnull() == True].tolist()
            if(not useLabel):
                nacols.append(LabelName)
            
            samples = trainDs.sample(numOfSamples, axis=0).dropna(axis=0, how='any')
            while(samples.empty):
                samples = trainDs.sample(numOfSamples, axis=0).dropna(axis=0, how='any')
                
            #print(samples.shape)
            minDist = np.inf
            minIdx = -1
            dist = np.inf
            for idx2, sample2 in samples.iterrows():
                dist = samp_dist(sample.drop(nacols), sample2.drop(nacols))
                #print(dist)
                if (pd.notnull(dist) and dist < minDist):
                    minDist = dist
                    minIdx = idx2
            if(not useLabel):
                nacols.remove(LabelName)
            ds.loc[idx, nacols] = trainDs.loc[minIdx, nacols][0]


def outlierDetection(ds):
    """remove every sample that is more than 4 std from mean value for every feature"""
    logging.info("***start outlierDetection")
    
    for feature, distribution in dictFeatures.items():
        if (distribution == 'Normal'):
            ds = ds[np.abs(ds[feature] - ds[feature].mean()) <= (4*ds[feature].std())]


def normalization(ds, trainDs):
    """we use min max scaling for fetures with Uniform distribution
    and z score scaling for Normal and other distribution"""
    logging.info("***start normalization")
    
    for feature, distribution in dictFeatures.items():
        if (distribution == 'Uniform'):
            ds[feature] -= trainDs[feature].min()
            ds[feature] /= 0.5*((trainDs[feature].max() - trainDs[feature].min()))
            ds[feature] += -1
        else:
            std = trainDs[feature].std()
            mean = trainDs[feature].mean()
            ds[feature] -= mean
            ds[feature] /= std
            
            #apply decimal scaling
            maxN = trainDs[feature].max()
            k = math.log10(maxN)
            ds[feature] /= pow(10,k)
        

def pearsonCorrelation(ds, threshold=0.85):
    """calc corr matrix for all features and return list of features with linear dependancies, drop features from ds"""
    logging.info("***start pearsonCorrelation")
    
    trainCorrMat = ds.corr(method='pearson').abs()
    upper = trainCorrMat.where(np.triu(np.ones(trainCorrMat.shape), k=1).astype(np.bool))
    features_to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
    ds.drop(features_to_drop, axis=1, inplace=True)
    print("pearsonCorrelation:\n", features_to_drop) 
    return features_to_drop        


def mutualInformation(ds, threshold=1):
    """calc mutual information for all features and return list of features with dependancies, drop features from ds"""
    logging.info("***start mutualInformation")
    
    features_to_drop = []
    for column in ds.columns.drop(LabelName):
        
        if(column in dictFeatures.keys()):
            miMat = mutual_info_regression(ds.drop([LabelName, column], axis=1), ds[column], discrete_features='auto', n_neighbors=3, copy=True, random_state=None)
        else:
            miMat = mutual_info_classif(ds.drop([LabelName, column], axis=1), ds[column], discrete_features='auto', n_neighbors=3, copy=True, random_state=None)
        #print(column, miMat)
        if(any(mi > threshold for mi in miMat)):
            ds.drop(column, axis=1, inplace=True)
            features_to_drop.append(column)
            
    print("mutualInformation:\n", features_to_drop)        
    return features_to_drop
    

def SBS(ds):
    """Feature ranking with recursive feature elimination and cross-validated selection of the best number of features.
    drop features from ds"""
    logging.info("***start SBS")
    
    estimator = SVR(kernel="linear")
    selector = RFE(estimator, 20, step=1)
    selector = selector.fit(ds.drop(['VoteInt'], axis=1), ds['VoteInt'])
    features_to_drop = np.setdiff1d(ds.columns, ds.columns[selector.get_support(indices=True)]).tolist()
    ds.drop(features_to_drop, axis=1, inplace=True)
    print("SBS:\n", features_to_drop)
    return features_to_drop


def utilityScoreWrapper(x, y, clf):
    return clf.score(x, y)


def sfs(x, y, k, clf, score):
    """
    :param x: feature set to be trained using clf. list of lists.
    :param y: labels corresponding to x. list.
    :param k: number of features to select. int
    :param clf: classifier to be trained on the feature subset.
    :param score: utility function for the algorithm, that receives clf, feature subset and labeles, returns a score. 
    :return: list of chosen feature indexes
    """
    logging.info("***start sfs")
    
    X = np.array(x, dtype=np.int)
    Y = np.array(y, dtype=np.int)
    selected_features = np.array([], dtype=np.int)
    indexes = np.array(range(np.size(X, axis=1)), dtype=np.int)
    
    for _ in range(k):
        max_score = 0
        max_feature = 0
        for i in (index for index in indexes if index not in selected_features):
            tmp_features = selected_features
            tmp_features = np.append(tmp_features, i)
            tmp_x = np.array(X[:,tmp_features], dtype=np.int)
            clf.fit(tmp_x, Y)
            tmp_score = score(tmp_x, Y, clf)
            if tmp_score > max_score:
                max_score = tmp_score
                max_feature = i
        selected_features = np.append(selected_features, max_feature)
    
    features_to_drop = np.setdiff1d(x.columns, selected_features).tolist()
    x.drop(features_to_drop, axis=1, inplace=True)
    print("SFS:\n", features_to_drop)
    return features_to_drop


def main():
    logging.info("START TIME")
    """Load the Election Challenge data from the ElectionsData.csv file
    and generate a CSV file with selected features"""
    
    rawData = pd.read_csv('ElectionsData.csv', sep=',', header=0)
    rawData_original = rawData.copy()
    rawData = identifyFeatures(rawData)
    
    #split to train, validation and test
    train_original, test_original = train_test_split(rawData_original, test_size=0.2) 
    train_original, validation_original = train_test_split(train_original, test_size=0.25)
    train = rawData.loc[train_original.index]
    validation = rawData.loc[validation_original.index]
    test = rawData.loc[test_original.index]
    
    #Imputation
    #convert negetive values to NaN
    train[train < 0] = np.nan
    validation[validation < 0] = np.nan
    test[test < 0] = np.nan
    
    closest_fit(train, train)
    closest_fit(validation, train)
    closest_fit(test, train)
    
    #Data Cleansing
    outlierDetection(train)
    #Normalization (scaling)
    normalization(train, train)
    normalization(validation, train)
    normalization(test, train)
    
    #Feature selection
    #Pearson Correlation
    features_to_drop = pearsonCorrelation(train)
    #Mutual Information
    features_to_drop = features_to_drop + mutualInformation(train)
    #SBS
    features_to_drop = features_to_drop + SBS(train)
    #sfs
    #clf = KNeighborsClassifier(n_neighbors=5)
    #features_to_drop = features_to_drop + sfs(train.drop(['VoteInt'], axis=1), train['VoteInt'], 10, clf, utilityScoreWrapper)
    
    #Apply for validation and test sets
    validation.drop(features_to_drop, axis = 1, inplace = True)
    test.drop(features_to_drop, axis = 1, inplace = True)
    print("features_in_train:\n", train.columns)
    
    #Save datasets
    train_original.to_csv('dataSets/train_original.csv')
    test_original.to_csv('dataSets/test_original.csv')
    validation_original.to_csv('dataSets/validation_original.csv')
    train.to_csv('dataSets/train.csv')
    validation.to_csv('dataSets/validation.csv')
    test.to_csv('dataSets/test.csv')
    
    logging.info("END TIME")
    print("features_to_drop:\n", features_to_drop)
    

if __name__ == '__main__':
    main()