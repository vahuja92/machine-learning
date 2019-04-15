import pandas as pd



def read_dataset(filename):
	'''
	Takes a csv filepath/filename and saves as a pandas dataframe, with the
	first row used as the header.
	'''

	dataframe =pd.read_csv(filename)

	return dataframe

'''
def explore_data(df):
