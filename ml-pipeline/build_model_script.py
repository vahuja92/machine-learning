import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
import read_and_clean as rc


DATAFILE = "data/credit-data.csv"
#adjust this to be more modular
#create a few different libraries - one for preprocessing, one for them models

def main():
    '''
    Read, explore, pre-process, and build model.

    Outputs the accuracy score of the model used on the training set.
    '''
    #read dataset
    raw_data = rc.read_dataset(DATAFILE)
    pre_df = raw_data.copy(deep=True)

    #explore dataset
    rc.describe_data(pre_df)
    rc.distributions(pre_df, 'data-exploration/histograms_v2.png')
    rc.correlations(pre_df, 'SeriousDlqin2yrs',
                         'data-exploration/correlation_heat_map_v2.png')


    #Pre-process the data
    rc.fill_missing_w_median(pre_df)

    #convert columns to the correct type
    data_types = {'SeriousDlqin2yrs' : 'int',
                'RevolvingUtilizationOfUnsecuredLines' : 'float',
                'age' : 'int',
                'NumberOfTime30-59DaysPastDueNotWorse' : 'int',
                'zipcode' : 'int',
                'DebtRatio' : 'float',
                'MonthlyIncome' : 'float',
                'NumberOfOpenCreditLinesAndLoans' : 'int',
                'NumberOfTimes90DaysLate' : 'int',
                'NumberRealEstateLoansOrLines' : 'int',
                'NumberOfTime60-89DaysPastDueNotWorse' : 'int',
                'NumberOfDependents' : 'int'}

    pre_df = pre_df.astype(data_types)
    # explore data


    #check to see if I even need to return anything for these functions to change
    #the dataframe
    pre_df = rc.discretize_cont_var(pre_df, "MonthlyIncome", q_num_quantiles = 4)

    #make this modular - create checks for the types of data that can be turned into dummies
    pre_df = rc.create_dummies(pre_df, 'NumberOfDependents', 'int')
    return pre_df
    #Build Model
    predictor_vars = ['RevolvingUtilizationOfUnsecuredLines',
                      'age',
                      'zipcode',
                      'NumberOfTime30-59DaysPastDueNotWorse',
                      'DebtRatio',
                      'NumberOfOpenCreditLinesAndLoans',
                      'NumberOfTimes90DaysLate',
                      'NumberRealEstateLoansOrLines',
                      'NumberOfTime60-89DaysPastDueNotWorse',
                      'MonthlyIncome_cat',
                      'NumberOfDependents_0', 'NumberOfDependents_1',
                      'NumberOfDependents_2', 'NumberOfDependents_3', 'NumberOfDependents_4',
                      'NumberOfDependents_5', 'NumberOfDependents_6', 'NumberOfDependents_7',
                      'NumberOfDependents_8', 'NumberOfDependents_9',
                      'NumberOfDependents_13']

    target_var = "SeriousDlqin2yrs"

    logistic_regr = rc.train_logistic_model(pre_df, predictor_vars, target_var)

    score = rc.evaluate_logistic_model(logistic_regr, pre_df, predictor_vars, target_var)

    print("accuracy score = ", score)

if __name__ == '__main__':
    main()
