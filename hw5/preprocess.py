'''
Preprocess the project projects_2012_2013.csv file,
using the helper functions from preprocess_helper.py library
'''
import preprocess_helper as rc
import pandas as pd
import numpy as np
from datetime import timedelta
from datetime import datetime

 #You discarded many important features in line 259-172, simple_loop.py, such as teacher_prefix.
 #fix the readme, make sure this runs from the command line.
def pre_process(raw_data, dummy_vars, boolean_vars, vars_not_to_include, columns_to_datetime):
    '''
    Pre-process a read dataframe

    pre_df: (dataframe) the raw data to process
    dummy_vars: (list) a list of variables to create dummies of
    '''
    pre_df = raw_data.copy(deep=True)
    #fill missing values w/ the median of the column
    rc.fill_missing_w_median(pre_df)

    #convert date columns to datetime and add a variable to the dataset with the
    # month of the date, to be used as a feature in the dataset
    for col in columns_to_datetime:
        print(col)
        pre_df[col] = pd.to_datetime(pre_df[col])
        col_month = col + '_month_year'
        pre_df[col_month] = pre_df[col].dt.to_period('M')
        pre_df[col_month].head()
    #create the output variable  - 1 if NOT funded within 60 days, 0 if funded within 60 days
    pre_df = create_predictor_variable(pre_df)

    # #discritize necessary variables
    # discretize_vars = ['total_price_including_optional_support',
    #                    'students_reached']
    # for var in discretize_vars:
    #     pre_df[var] = rc.discretize_cont_var(pre_df, var, q_num_quantiles = 4)
            # does this work, or do i have to convert again to different type?

    #make this modular - create checks for the types of data that can be turned into dummies

    for var in dummy_vars:
        pre_df = rc.create_dummies(pre_df, var, 'int')
    #drop the original categorical variables, because I've made them into dummies
    pre_df = pre_df.drop(dummy_vars, axis=1)

    #make t/f variables dummies
    for var in boolean_vars:
        pre_df[var] = np.where(pre_df[var]=='t', 1, 0)

    #drop the date variables given in the pre_df. They will be used as col_month
    #instead, as defined in line 34 and 35 above.
    pre_df = pre_df.drop(columns_to_datetime, axis=1)

    #drop the variables not to include (id variables)
    pre_df = pre_df.drop(vars_not_to_include, axis=1)

    return pre_df

def create_predictor_variable(pre_df):
    '''
    This function is specific to the fundraising dataset.
    In order to create the binary prediction variable,
    (1 if not funded within 60 days, 0 for funded within 60 days), this function
    uses the 'datefullyfunded' and 'date_posted' variables.
    '''
    pre_df['dif'] = pre_df['datefullyfunded'] - pre_df['date_posted']
    pre_df['dif'] = pre_df['dif'].astype('timedelta64[D]')
    pre_df['not_funded'] = np.where(pre_df['dif'] > 60, 1, 0)      # create output variable
    pre_df.drop(['dif'], axis=1, inplace=True)

    return pre_df
