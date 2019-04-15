import pandas as pd



def read_dataset(filename):
	'''
	Takes a csv filepath/filename and saves as a pandas dataframe, with the
	first row used as the header.
	'''

	dataframe = read.csv(filename)

	return dataframe