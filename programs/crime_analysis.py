import sys
import pandas as pd
import numpy as np

'''
Vedika Ahuja
crime_analysis.py

Read and Analyze Reported Crime data in Chicago for 2017 and 2018
'''

def read_crime_csv(file):
	'''
	Takes the filepath and name of a dataset of reported crimes downloaded from 
	the Chicago data protal and outputs a pandas dataframe.

	Input: (str) crime data csv filepath/filename
	Output: (pandas dataframe)
	'''
	