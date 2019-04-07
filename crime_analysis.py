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

def add_date_columns(crime_df):
    '''
    Convert the date column into a week and month column
    Come back to this - nice to have
    '''
    crime_df['date_formatted'] = pd.to_datetime(crime_df['date'])
    

    
def descriptive_stat(crime_dataframe):
    '''
    Generate summary statistics for the crime reports data including but not 
    limited to number of crimes of each type, 

    
    how they change over time, and 
    how they are different by neighborhood. Please use a combination of tables 
    and graphs to present these summary stats.
    '''
    crimes_by_type = 

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