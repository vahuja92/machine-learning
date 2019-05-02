
# Import Statements
from __future__ import division

import pandas as pd
import numpy as np
import os
import matplotlib
import matplotlib.pyplot as plt
import pylab as pl
from datetime import timedelta
from datetime import datetime
import random
from scipy import optimize
import time
import seaborn as sns
import csv

from ml_helperfunctions import *
from read_and_clean_helperfunctions import *

# def main():

    # print 'Number of arguments:', len(sys.argv), 'arguments.'
    # print 'Argument List:', str(sys.argv)
    #
    # # parse input parameters
    #
    # # csv data file to be used as input
    # infile = sys.argv[1]
    #
    # # the filename we want to write results to
    # outfile = sys.argv[2]
    #
    # # which model(s) to run
    # model = sys.argv[3]
    #
    # # which parameter grid do we want to use (test, small, large)
    # grid_size = sys.argv[4]

'''
now do this with the data!
'''

def magicloop(model, pre_df, outfile):

    '''
    For now, the magicloop function takes..

    Inputs:
        - pre_df: the preprocessed pandas dataframe (this is the preprocessed data. For now, inputting
                this as a dataframe for ease of playing around with it. Rayid Inputs
                a csv file instead)
        - outfile: the file to write the output of the models
    '''
    #read the csv data - this might have to call the preprocessing function
    # data = pd.read_csv(infile)

    # which variable to use for prediction_time - NOT SURE
    # prediction_time = 'dis_date'

    # outcome variables we want to loop over
    outcomes = ['less_60']

    # validation dates we want to loop over - NOT SURE
    # validation_dates = ['2012-04-01', '2012-10-01', '2013-04-01']

    # define feature groups
    #how do you deal with time? use month or year posted as a predictor? Seems reasonable
    #also location??
    school_predictors = ['school_magnet',
                        'eligible_double_your_impact_match',
                        'total_price_including_optional_support_cat',
                        'students_reached_cat']
    primary_subject_predictors = []
    secondary_subject_predictors = []
    poverty_level_predictors = []
    grade_level_predictors = []
    teacher_gender_predictors = []
    resource_type = []
    for col in pre_df.columns:
        if ("primary_focus_area_" in col) or ("primary_focus_subject_" in col):
            primary_subject_area_predictors.append(col)
        elif ("secondary_focus_area_" in col) or ("secondary_focus_subject_" in col):
            secondary_subject_area_predictors.append(col)
        elif "poverty_level_" in col:
            poverty_level_predictors.append(col)
        elif "grade_level_Grades" in col:
            grade_level_predictors.append(col)
        elif "teacher_prefix_" in col:
            teacher_gender_predictors.append(col)
        elif "resource_type_" in col:
            resource_type.append(col)

    all_predictors = school_predictors + primary_subject_predictors + \
                    secondary_subject_predictors + \
                    poverty_level_predictors + \
                    grade_level_predictors + \
                    teacher_gender_predictors + \
                    resource_type

    # models_to_run=['RF','AB', 'LR', 'SVM', 'GB', 'DT', 'KNN'
    if (model == 'all'):
        models_to_run= ['RF','AB', 'LR', 'SVM', 'GB', 'DT', 'KNN']
    else:
        models_to_run = []
        models_to_run.append(model)

    clfs, grid = define_clfs_params(grid_size)

    #FOR NOW, USE ALL THE PREDICTORS
    # which feature/predictor sets do we want to use in our analysis
    # predictor_sets = [primary_subject_predictors,
    #                 secondary_subject_predictors,
    #                 poverty_level_predictors,
    #                 grade_level_predictors,
    #                 teacher_gender_predictors,
    #                 resource_type]
    #
    # #FIGURE OUT WHAT THIS DOES
    # # generate all possible subsets of the feature/predictor groups
    # predictor_subsets = get_subsets(predictor_sets)
    #
    # all_predictors=[]
    # for p in predictor_subsets:
    #     merged = list(itertools.chain.from_iterable(p))
    #     all_predictors.append(merged)

    # write header for the csv
    with open(outfile, "w") as myfile:
        myfile.write("model_type ,clf, parameters, outcome, validation_date, group,train_set_size, validation_set_size,predictors,baseline,precision_at_5,precision_at_10,precision_at_20,precision_at_30,precision_at_40,precision_at_50,recall_at_5,recall_at_10,recall_at_20,recall_at_30,recall_at_40, ecall_at_50,auc-roc")

    # define dataframe to write results to
    results_df =  pd.DataFrame(columns=('model_type','clf', 'parameters', 'outcome', 'validation_date', 'group',
                                        'train_set_size', 'validation_set_size','predictors',
                                        'baseline','precision_at_5','precision_at_10','precision_at_20','precision_at_30','precision_at_40',
                                        'precision_at_50','recall_at_5','recall_at_10','recall_at_20','recall_at_30','recall_at_40',
                                        'recall_at_50','auc-roc'))

    # Logistic Regression, K-Nearest Neighbor, Decision Trees, SVM, Random Forests, Boosting, and Bagging.


    # the magic loop starts here
    # we will loop over models, parameters, outcomes, validation_Dates
    # and store several evaluation metrics
    #COME BACK HERE, follow this... see if you can just create your own dictionaries here for parameters
        # the magic loop starts here
    # we will loop over models, parameters, outcomes, validation_Dates
    # and store several evaluation metrics


def clf_loop(models_to_run, clfs, grid, X, y):
    """Runs the loop using models_to_run, clfs, gridm and the data
    """
    results_df =  pd.DataFrame(columns=('model_type','clf', 'parameters', 'auc-roc','p_at_5', 'p_at_10', 'p_at_20'))
    for n in range(1, 2):
        # create training and valdation sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)
        for index,clf in enumerate([clfs[x] for x in models_to_run]):
            print(models_to_run[index])
            parameter_values = grid[models_to_run[index]]
            for p in ParameterGrid(parameter_values):
                try:
                    clf.set_params(**p)
                    y_pred_probs = clf.fit(X_train, y_train).predict_proba(X_test)[:,1]
                    # you can also store the model, feature importances, and prediction scores
                    # we're only storing the metrics for now
                    y_pred_probs_sorted, y_test_sorted = zip(*sorted(zip(y_pred_probs, y_test), reverse=True))
                    results_df.loc[len(results_df)] = [models_to_run[index],clf, p,
                                                       roc_auc_score(y_test, y_pred_probs),
                                                       precision_at_k(y_test_sorted,y_pred_probs_sorted,5.0),
                                                       precision_at_k(y_test_sorted,y_pred_probs_sorted,10.0),
                                                       precision_at_k(y_test_sorted,y_pred_probs_sorted,20.0)]
                    if NOTEBOOK == 1:
                        plot_precision_recall_n(y_test,y_pred_probs,clf)
                except IndexError as e:
                    print('Error:',e)
                    continue
    return results_df



def main():

    # define grid to use: test, small, large
    grid_size = 'test'
    clfs, grid = define_clfs_params(grid_size)

    # define models to run
    models_to_run=['RF','DT','KNN', 'ET', 'AB', 'GB', 'LR', 'NB']

    # load data from csv
    df = pd.read_csv("/Users/rayid/Projects/uchicago/Teaching/MLPP-2017/Homeworks/Assignment 2/credit-data.csv")

    # select features to use
    features  =  ['RevolvingUtilizationOfUnsecuredLines', 'DebtRatio', 'age', 'NumberOfTimes90DaysLate']
    X = df[features]

    # define label
    y = df.SeriousDlqin2yrs

    # call clf_loop and store results in results_df
    results_df = clf_loop(models_to_run, clfs,grid, X,y)
    if NOTEBOOK == 1:
        results_df

    # save to csv
    results_df.to_csv('results.csv', index=False)


if __name__ == '__main__':
    main()
