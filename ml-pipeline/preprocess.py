'''
Preprocess the project projects_2012_2013.csv file,
using the helper functions from preprocess_helper.py library
'''
import preprocess_helper as rc
import pandas as pd
import numpy as np

def pre_process(infile):
    '''
    Read, explore, pre-process, and build model.

    Outputs the accuracy score of the model used on the training set.
    '''
    #read dataset
    raw_data = rc.read_dataset(infile)
    pre_df = raw_data.copy(deep=True)

    #explore read_dataset
    rc.describe_data(pre_df)

    #fill missing values w/ the median of the column - GO BACK TO This
    #This shouldn't be here, but I'm doing it to make the rest of the code run right now.
    rc.fill_missing_w_median(pre_df)

    #convert date columns to datetime
    columns_to_date_time = ['datefullyfunded', 'date_posted']
    for cols in columns_to_date_time:
        pre_df[cols] = pd.to_datetime(pre_df[cols])

    #create the output variable (whether the project was funded within 60 days)
    pre_df['dif'] = pre_df['datefullyfunded'] - pre_df['date_posted']
    pre_df['dif'] = pre_df['dif'].astype('timedelta64[D]')
    pre_df['less_60'] = np.where(pre_df['dif'] < 60, 1, 0)      # create output variable

    #discritize necessary variables
    discretize_vars = ['total_price_including_optional_support',
                       'students_reached']
    for var in discretize_vars:
        pre_df[var] = rc.discretize_cont_var(pre_df, var, q_num_quantiles = 4)
            # does this work, or do i have to convert again to different type?

    #make this modular - create checks for the types of data that can be turned into dummies
    dummy_vars = ['teacher_prefix',
                  'primary_focus_subject',
                  'primary_focus_area', 'secondary_focus_subject',
                  'secondary_focus_area', 'resource_type',
                  'poverty_level', 'grade_level']

    for var in dummy_vars:
        pre_df = rc.create_dummies(pre_df, var, 'int')

    #make t/f variables dummies
    bool_vars = ['eligible_double_your_impact_match', 'school_charter', 'school_magnet']
    for var in bool_vars:
        pre_df[var] = np.where(pre_df[var]=='t', 1, 0)

    return pre_df
