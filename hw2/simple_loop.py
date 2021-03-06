from __future__ import division
import pandas as pd
import numpy as np
from sklearn import preprocessing, svm, metrics, tree, decomposition, svm
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression, Perceptron, SGDClassifier, OrthogonalMatchingPursuit, RandomizedLogisticRegression
from sklearn.neighbors.nearest_centroid import NearestCentroid
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from  sklearn.model_selection import train_test_split
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import *
from sklearn.preprocessing import StandardScaler
import random
import matplotlib.pyplot as plt
from scipy import optimize
import time
import seaborn as sns
import preprocess_helper as rc
import preprocess as pre
from datetime import timedelta
from datetime import datetime


# for jupyter notebooks
#%matplotlib inline

# if you're running this in a jupyter notebook, print out the graphs
NOTEBOOK = 0

def define_clfs_params(grid_size):
    """
    Define defaults for different classifiers.
    Define three types of grids:
    Test: for testing your code
    Small: small grid
    Large: Larger grid that has a lot more parameter sweeps
    """

    clfs = {'RF': RandomForestClassifier(n_estimators=50, n_jobs=-1),
        'ET': ExtraTreesClassifier(n_estimators=10, n_jobs=-1, criterion='entropy'),
        'AB': AdaBoostClassifier(DecisionTreeClassifier(max_depth=1), algorithm="SAMME", n_estimators=200),
        'LR': LogisticRegression(penalty='l1', C=1e5),
        'SVM': svm.SVC(kernel='linear', probability=True, random_state=0),
        'GB': GradientBoostingClassifier(learning_rate=0.05, subsample=0.5, max_depth=6, n_estimators=10),
        'NB': GaussianNB(),
        'DT': DecisionTreeClassifier(),
        'SGD': SGDClassifier(loss="hinge", penalty="l2"),
        'KNN': KNeighborsClassifier(n_neighbors=3)
            }

    large_grid = {
    'RF':{'n_estimators': [1,10,100,1000,10000], 'max_depth': [1,5,10,20,50,100], 'max_features': ['sqrt','log2'],'min_samples_split': [2,5,10], 'n_jobs': [-1]},
    'LR': { 'penalty': ['l1','l2'], 'C': [0.00001,0.0001,0.001,0.01,0.1,1,10]},
    'SGD': { 'loss': ['hinge','log','perceptron'], 'penalty': ['l2','l1','elasticnet']},
    'ET': { 'n_estimators': [1,10,100,1000,10000], 'criterion' : ['gini', 'entropy'] ,'max_depth': [1,5,10,20,50,100], 'max_features': ['sqrt','log2'],'min_samples_split': [2,5,10], 'n_jobs': [-1]},
    'AB': { 'algorithm': ['SAMME', 'SAMME.R'], 'n_estimators': [1,10,100,1000,10000]},
    'GB': {'n_estimators': [1,10,100,1000,10000], 'learning_rate' : [0.001,0.01,0.05,0.1,0.5],'subsample' : [0.1,0.5,1.0], 'max_depth': [1,3,5,10,20,50,100]},
    'NB' : {},
    'DT': {'criterion': ['gini', 'entropy'], 'max_depth': [1,5,10,20,50,100],'min_samples_split': [2,5,10]},
    'SVM' :{'C' :[0.00001,0.0001,0.001,0.01,0.1,1,10],'kernel':['linear']},
    'KNN' :{'n_neighbors': [1,5,10,25,50,100],'weights': ['uniform','distance'],'algorithm': ['auto','ball_tree','kd_tree']}
           }

    small_grid = {
    'RF':{'n_estimators': [10,100], 'max_depth': [5,50], 'max_features': ['sqrt','log2'],'min_samples_split': [2,10], 'n_jobs': [-1]},
    'LR': { 'penalty': ['l1','l2'], 'C': [0.00001,0.001,0.1,1,10]},
    'SGD': { 'loss': ['hinge','log','perceptron'], 'penalty': ['l2','l1','elasticnet']},
    'ET': { 'n_estimators': [10,100], 'criterion' : ['gini', 'entropy'] ,'max_depth': [5,50], 'max_features': ['sqrt','log2'],'min_samples_split': [2,10], 'n_jobs': [-1]},
    'AB': { 'algorithm': ['SAMME', 'SAMME.R'], 'n_estimators': [1,10,100,1000,10000]},
    'GB': {'n_estimators': [10,100], 'learning_rate' : [0.001,0.1,0.5],'subsample' : [0.1,0.5,1.0], 'max_depth': [5,50]},
    'NB' : {},
    'DT': {'criterion': ['gini', 'entropy'], 'max_depth': [1,5,10,20,50,100],'min_samples_split': [2,5,10]},
    'SVM' :{'C' :[0.00001,0.0001,0.001,0.01,0.1,1,10],'kernel':['linear']},
    'KNN' :{'n_neighbors': [1,5,10,25,50,100],'weights': ['uniform','distance'],'algorithm': ['auto','ball_tree','kd_tree']}
           }

    test_grid = {
    'RF':{'n_estimators': [1], 'max_depth': [1], 'max_features': ['sqrt'],'min_samples_split': [10]},
    'LR': { 'penalty': ['l1'], 'C': [0.01]},
    'SGD': { 'loss': ['perceptron'], 'penalty': ['l2']},
    'ET': { 'n_estimators': [1], 'criterion' : ['gini'] ,'max_depth': [1], 'max_features': ['sqrt'],'min_samples_split': [10]},
    'AB': { 'algorithm': ['SAMME'], 'n_estimators': [1]},
    'GB': {'n_estimators': [1], 'learning_rate' : [0.1],'subsample' : [0.5], 'max_depth': [1]},
    'NB' : {},
    'DT': {'criterion': ['gini'], 'max_depth': [1],'min_samples_split': [10]},
    'SVM' :{'C' :[0.01],'kernel':['linear']},
    'KNN' :{'n_neighbors': [5],'weights': ['uniform'],'algorithm': ['auto']}
           }

    if (grid_size == 'large'):
        return clfs, large_grid
    elif (grid_size == 'small'):
        return clfs, small_grid
    elif (grid_size == 'test'):
        return clfs, test_grid
    else:
        return 0, 0

# a set of helper function to do machine learning evalaution

def joint_sort_descending(l1, l2):
    # l1 and l2 have to be numpy arrays
    idx = np.argsort(l1)[::-1]
    return l1[idx], l2[idx]

def generate_binary_at_k(y_scores, k):
    '''
    Classifies the prediction as 1 or 0 given the k threshold.
    The threshold is actually a percentile (?)
    '''
    cutoff_index = int(len(y_scores) * (k / 100.0))
    test_predictions_binary = [1 if x < cutoff_index else 0 for x in range(len(y_scores))]
    return test_predictions_binary

def precision_at_k(y_true, y_scores, k):
    y_scores, y_true = joint_sort_descending(np.array(y_scores), np.array(y_true))
    preds_at_k = generate_binary_at_k(y_scores, k)
    precision = precision_score(y_true, preds_at_k)

    return precision


def recall_at_k(y_true, y_scores, k):
    y_scores_sorted, y_true_sorted = joint_sort_descending(np.array(y_scores), np.array(y_true))
    preds_at_k = generate_binary_at_k(y_scores_sorted, k)
    recall = recall_score(y_true_sorted, preds_at_k)

    return recall

def plot_precision_recall_n(y_true, y_prob, model_name):
    from sklearn.metrics import precision_recall_curve
    y_score = y_prob
    precision_curve, recall_curve, pr_thresholds = precision_recall_curve(y_true, y_score)
    precision_curve = precision_curve[:-1]
    recall_curve = recall_curve[:-1]
    pct_above_per_thresh = []
    number_scored = len(y_score)
    for value in pr_thresholds:
        num_above_thresh = len(y_score[y_score>=value])
        pct_above_thresh = num_above_thresh / float(number_scored)
        pct_above_per_thresh.append(pct_above_thresh)
    pct_above_per_thresh = np.array(pct_above_per_thresh)

    plt.clf()
    fig, ax1 = plt.subplots()
    ax1.plot(pct_above_per_thresh, precision_curve, 'b')
    ax1.set_xlabel('percent of population')
    ax1.set_ylabel('precision', color='b')
    ax2 = ax1.twinx()
    ax2.plot(pct_above_per_thresh, recall_curve, 'r')
    ax2.set_ylabel('recall', color='r')
    ax1.set_ylim([0,1])
    ax1.set_ylim([0,1])
    ax2.set_xlim([0,1])

    name = model_name
    plt.title(name)
    #come back to this
    # plt.savefig(name)
    # plt.show()



def clf_loop(models_to_run, clfs, grid, data, features, outfile):
    """
    Runs the loop using models_to_run, clfs, gridm and the data
    """
    results_df =  pd.DataFrame(columns=('model_type', 'validation_date', 'clf', 'parameters', 'auc-roc', \
                                        'p_at_1', 'p_at_2', 'p_at_5', 'p_at_10', 'p_at_20', 'p_at_30', 'p_at_50', \
                                        'r_at_1', 'r_at_2', "r_at_5", "r_at_10", 'r_at_20', 'r_at_30', 'r_at_50', \
                                         'baseline', 'len_x_train'))
    validation_dates = ['2012-07-01', '2013-01-01', '2013-07-01']

    for index,clf in enumerate([clfs[x] for x in models_to_run]):
        for validation_date in validation_dates:
            print(validation_date)
            # create training and valdation sets
            train_set = data.loc[data['date_posted'] <= datetime.strptime(validation_date, '%Y-%m-%d') - timedelta(days=60)]
            rc.fill_missing_w_median(train_set)
            X_train = train_set[features]
            y_train = train_set['less_60']
            # fill in missing values for train set using just the train set
            # we'll do it a very naive way here but you should think more carefully about this first
            train_set.dropna(axis=1, how='any', inplace=True)

            validation_set = data.loc[data['date_posted'] > datetime.strptime(validation_date, '%Y-%m-%d') - timedelta(days=0)]
            # fill in missing values for validation set using all the data
            # we'll do it a very naive way here but you should think more carefully about this first
            rc.fill_missing_w_median(validation_set)
            validation_set.dropna(axis=1, how='any', inplace=True)
            X_test = validation_set[features]
            y_test = validation_set['less_60']
            print(models_to_run[index])
            parameter_values = grid[models_to_run[index]]
            for p in ParameterGrid(parameter_values):
                print(p)
                try:
                    clf.set_params(**p)
                    y_pred_probs = clf.fit(X_train, y_train).predict_proba(X_test)[:,1]
                    # you can also store the model, feature importances, and prediction scores
                    # we're only storing the metrics for now
                    y_pred_probs_sorted, y_test_sorted = zip(*sorted(zip(y_pred_probs, y_test), reverse=True))
                    baseline = y_pred_probs.sum()/len(y_pred_probs)
                    results_df.loc[len(results_df)] = [models_to_run[index],validation_date, clf, p,
                                                       roc_auc_score(y_test, y_pred_probs),
                                                       precision_at_k(y_test_sorted,y_pred_probs_sorted,1.0),
                                                       precision_at_k(y_test_sorted,y_pred_probs_sorted,2.0),
                                                       precision_at_k(y_test_sorted,y_pred_probs_sorted,5.0),
                                                       precision_at_k(y_test_sorted,y_pred_probs_sorted,10.0),
                                                       precision_at_k(y_test_sorted,y_pred_probs_sorted,20.0),
                                                       precision_at_k(y_test_sorted,y_pred_probs_sorted,30.0),
                                                       precision_at_k(y_test_sorted,y_pred_probs_sorted,50.0),
                                                       recall_at_k(y_test_sorted,y_pred_probs_sorted,1.0),
                                                       recall_at_k(y_test_sorted,y_pred_probs_sorted,2.0),
                                                       recall_at_k(y_test_sorted,y_pred_probs_sorted,5.0),
                                                       recall_at_k(y_test_sorted,y_pred_probs_sorted,10.0),
                                                       recall_at_k(y_test_sorted,y_pred_probs_sorted,20.0),
                                                       recall_at_k(y_test_sorted,y_pred_probs_sorted,30.0),
                                                       recall_at_k(y_test_sorted,y_pred_probs_sorted,50.0),
                                                       baseline, len(X_train)]

                    # plot_precision_recall_n(y_test,y_pred_probs,clf)
                    print("got here 1")

                except IndexError as e:
                    print('Error:',e)
                    continue
        print("Reading to file")
        csv_to_output = outfile + models_to_run[index] + ".csv"
        results_df.to_csv(csv_to_output, index=False)

    return results_df



# def main(infile, outfile):
def main(infile, outfile, run_on_sample, grid_size):

    df = pre.pre_process(infile)
    # define grid to use: test, small, large
    clfs, grid = define_clfs_params(grid_size)
    df_sub = df.sample(frac=.25)

    # define models to run
    # models_to_run=['RF','DT','KNN', 'ET', 'AB', 'GB', 'LR', 'NB']
    # models_to_run=['RF','DT','KNN', 'ET', 'AB', 'GB', 'LR', 'NB']
 # Logistic Regression, K-Nearest Neighbor, Decision Trees, SVM, Random Forests, Boosting, and Bagging.
    # models_to_run=['KNN', 'RF', 'LR', 'DT', 'AB', 'SVM']
    models_to_run=['DT', 'RF', 'AB', 'KNN', 'SVM']
    # models_to_run=['RF']
    # models_to_run=['LR']

    # load data from csv
    # df = pd.read_csv("/Users/rayid/Projects/uchicago/Teaching/MLPP-2017/Homeworks/Assignment 2/credit-data.csv")

    # COME BACK HERE - select features to use
    features  = [col for col in df if col not in ["projectid",
                        "projectid", "teacher_acctid", "schoolid", "school_ncesid",
                        "school_latitude", "school_longitude", "school_city",
                        "school_state", "school_metro", "school_district", "school_county",
                        "teacher_prefix", "primary_focus_subject", "primary_focus_area"
                        "secondary_focus_subject", "secondary_focus_area", "resource_type",
                        "poverty_level", "grade_level", "projectid", "teacher_acctid", "schoolid"
                        "school_ncesid", "school_latitude", "school_longitude",
                        "school_city", "school_state", "school_metro", "school_district",
                        "school_county", "teacher_prefix", "primary_focus_subject",
                        "primary_focus_area", "secondary_focus_subject", "secondary_focus_area",
                        "resource_type", "poverty_level", "grade_level",
                        "total_price_including_optional_support", "students_reached",
                        "date_posted", "datefullyfunded", "dif", "less_60"]]


    if run_on_sample == 1:
        results_df = clf_loop(models_to_run, clfs, grid, df_sub, features, "output/sample_mod_v2")
    else:
        results_df = clf_loop(models_to_run, clfs, grid, df, features, "output/sample_mod_v2_")
    # save to csv
    results_df.to_csv(outfile, index=False)


if __name__ == '__main__':
    main()
