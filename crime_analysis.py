import sys
import pandas as pd
import numpy as np

'''
Vedika Ahuja
crime_analysis.py

Read and Analyze Reported Crime data in Chicago for 2017 and 2018
'''

CRIME_2017 = "data/reported_crimes_2017.csv"
CRIME_2018 = "data/reported_crimes_2018.csv"

def read_crime_csv(file):
    '''
    Takes the filepath and name of a dataset of reported crimes downloaded from 
    the Chicago data protal and outputs a pandas dataframe.

    Input: (str) crime data csv filepath/filename
    Output: (pandas dataframe)
    '''

    cleaned = pd.read_csv(file)

    return cleaned

def concat_years(list_files):
    '''
    Joins different years of data into one dataset.

    Input: 
        - list_files: (list) of pandas dataframes
    Output:
        combined: pandas dataframe
    '''
    concat = pd.concat(list_files)

    return concat

def go():
    crime_2017 = read_crime_csv(CRIME_2017)
    crime_2018 = read_crime_csv(CRIME_2018)
    crimes = concat_year([crime_2017, crime_2018])


    return crimes