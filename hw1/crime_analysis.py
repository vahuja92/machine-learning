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
                      'year',
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
                      'longitude']]
    
    convert_dict = {
                    'id' : 'int',
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
                  'ward' : 'str',
                  'community_area' : 'str',
                  'fbi_code' : 'str',
                  'x_coordinate' : 'float',
                  'y_coordinate' : 'float',
                  'latitude' : 'float',
                  'longitude' : 'float'}

    clean = clean.astype(convert_dict)

    return clean

def crime_type(crime_df):
    '''
    Generate summary statistics for the crime reports data including but not 
    limited to number of crimes of each type, 

    
    how they change over time, and 
    how they are different by neighborhood. 

    Please use a combination of tables and graphs to present these summary stats.
    '''
    #types of crimes
    year_groups = crime_df.groupby(['year', 'primary_type'])
    value_counts = year_groups.agg({'primary_type' : 'count'})
    by_type = value_counts.unstack(level=0)
    by_type = by_type['primary_type'].reset_index()
    by_type["percent_change"] = (by_type['2018'] - by_type['2017'])/by_type['2017']
    
    # year_groups = crime_df.groupby(['year', 'primary_type']).size()
    return by_type

def crime_over_time(crime_df):
    '''
    Plot month-year on x axis and total crimes on y-axis for the whole dataset.


    '''

    # year_groups = crime_df.groupby(['year', 'primary_type'])
    # year_groups = crime_df.groupby(['year', 'primary_type'])
    crime_df["month-year"] = crime_df['date_formatted'].dt.to_period("M")
    group = crime_df.groupby("month-year").size()
    #plot this series now.

    return group

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