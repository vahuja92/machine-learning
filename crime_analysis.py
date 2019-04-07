import sys
import pandas as pd
import numpy as np
import datetime
from sodapy import Socrata

'''
Vedika Ahuja
crime_analysis.py

Read and Analyze Reported Crime data in Chicago for 2017 and 2018
'''

def query_crime_data():
    '''
    Queries 2017 and 2018 crime data from the Chicago data portal. 

    Come back to this to allow user to choose years? 
    '''
    client = Socrata("data.cityofchicago.org", None)

    # results = client.get("6zsd-86xi", where="year=2017 OR year=2018", limit=np.inf)
    results = client.get("6zsd-86xi", where="year=2017 OR year=2018", limit=1000000) #how to I not include a limit?

    results_df = pd.DataFrame.from_records(results)

    return results_df

def clean_dataset(crime_df):
    '''
    Convert the date column into a week and month column,
    reorder columns, specifiy datatypes for each column.
    '''
    #clean dates
    crime_df['date_formatted'] = pd.to_datetime(crime_df['date'])
    crime_df['month'] = crime_df['date_formatted'].dt.month
    crime_df['week'] = crime_df['date_formatted'].dt.week

    #specify types and order for each column
    clean = crime_df[['id', 
                      'case_number',
                      'date_formatted',
                      'month',
                      'week',
                      'year'
                      'block', 
                      'iucr',
                      'primary_type',
                      'description',
                      'location_description',
                      'arrest',
                      'domestic',
                      'beat',
                      'district',
                      'ward',
                      'community_area',
                      'fbi_code',
                      'x_coordinate',
                      'y_coordinate',
                      'latitude',
                      'longitude',
                      'location_city',
                      'location_address',
                      'location_zip',
                      'location_state']]
    
    convert_dict = {'id' : 'int',
                  'case_number': 'str',
                  'block': 'str', 
                  'iucr' : 'str',
                  'primary_type' : 'str',
                  'description' : 'str',
                  'location_description' :'str',
                  'arrest' : 'bool',
                  'domestic' : 'bool',
                  'beat' : 'str' ,
                  'district' : 'str',
                  'ward' : 'int',
                  'community_area' : 'str',
                  'fbi_code' : 'str',
                  'x_coordinate' : 'float',
                  'y_coordinate' : 'float',
                  'latitude' : 'float',
                  'longitude' : 'float' ,
                  'location_city' : 'str',
                  'location_address' : 'str',
                  'location_zip' : 'str',
                  'location_state' : 'str'}

    clean = clean.astype(convert_dict)

    return clean

def descriptive_stat(crime_dataframe):
    '''
    Generate summary statistics for the crime reports data including but not 
    limited to number of crimes of each type, 

    
    how they change over time, and 
    how they are different by neighborhood. Please use a combination of tables 
    and graphs to present these summary stats.
    '''
    pass


def export_dfs(output_file, list_dataframes):
    '''
    Takes a list of dataframes and exports them as separate sheets in an excel
    workbook

    see - http://pandas-docs.github.io/pandas-docs-travis/reference/api/pandas.ExcelWriter.html
    '''
    pass


def go():
    crimes = query_crime_data()


    return crimes