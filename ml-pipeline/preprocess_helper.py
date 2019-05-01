import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
# from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score as accuracy
import graphviz # If you don't have this, install via pip/conda

'''
Set Pyplot rcParams
'''
SMALL_SIZE = 6
MEDIUM_SIZE = 8
BIGGER_SIZE = 12

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=MEDIUM_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=SMALL_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

'''
other useful functions for preprocessing:

num_unique() (doesn't work for strings)
value_counts()
'''

def read_dataset(filename, data_types=None):
	'''
	Takes a csv filepath/filename and saves as a pandas dataframe, with the
	first row used as the header.

	data_taypes: (dict)
		The user can also specify the variable names and data types for each var
		in a dictionary
	'''
	dataframe = pd.read_csv(filename, dtype=data_types)

	return dataframe

'''
Explore the dataframe
'''

def describe_data(df):
	'''
	Print descriptive statistics of the dataset
	'''
	print("Columns in the data")
	print(df.columns)
	print("-------------------------------------")
	print("Summary of data")
	print("-------------------------------------")
	print(df.describe())
	print("-------------------------------------")
	print ('The data has {} rows and {} columns'.format(df.shape[0],df.shape[1]))
	print("--------------------------------------")
	print("The data has the following missing values for each variable")
	print("--------------------------------------")
	print(df.isnull().sum().sort_values(ascending=False))

def distributions(df, output_file):
	'''
	Print and export graphs and tables about the distributions of variables
	in the df.

	Specify the target variable to find target specific correlations
	'''
	#clean this up at some point
	fig = df.hist()
	plt.savefig(output_file, bbox_inches='tight')
	plt.close()

def correlations(df, target_var, output_file):
	'''
	Print and export correlations between the variables.

	Specify the target variable to find target specific correlations
	'''
	# seperate out the Categorical and Numerical features
	numerical_feature = df.dtypes[(df.dtypes == 'float') | (df.dtypes == 'int')].index
	categorical_feature=df.dtypes[df.dtypes== 'object'].index

	corr= df[numerical_feature].corr()
	sns.heatmap(corr, cmap="YlGnBu")
	plt.savefig(output_file, bbox_inches='tight')
	plt.close()

	print("5 most postivively correlated variables with target variable")
	print("")
	print (corr[target_var].sort_values(ascending=False)[:5], '\n') #top 5 values
	print("-------------------------------------------")
	print("5 most negatively correlated variables with target variable")
	print("")
	print (corr[target_var].sort_values(ascending=True)[:5], '\n') #top 5 values

'''
Pre-Process Data:
'''

def fill_missing_w_median(df):
	'''
	Fill all missing values with the median of the variable.

	Input:
		- df: pandas dataframe

	Side Effect: modifies the dataframe
	'''
	col = list(df)
	for c in col:
		if df[c].isnull().any() & (df[c].dtype == 'float' or df[c].dtype == 'int'):
			median = df[c].median()
			df[c] = df[c].fillna(median)


def discretize_cont_var(df, varname, q_num_quantiles = 4):
	'''
	Discretizes a the variable specified (varname) in the df.
	This function automatically discretizes the variable into quartiles.
	The user can change this by specifying the number of quantiles.

	Side Effect:
		- Changes input df by adding discretized variable

	 Output:
	 	- df (pandas dataframe)
	'''
	new_var = varname + "_cat"
	df[new_var] = pd.qcut(df[varname],
	 					  q_num_quantiles,
						  labels=list(range(1,(q_num_quantiles+1))),
						  duplicates='drop')
	df[new_var] = df[new_var].astype('int')

	return df


def create_dummies(df, varname, data_type):
	'''
	Create dummy variables from the categorical variable specified as varname.

	Inputs:
		- df: pandas dataframe
		- varname: str varname
	'''
	#go back to make checks on the type of data that can be made into dummies.
	#make this more functional - create checks on this
	add_col = pd.get_dummies(df[varname], prefix=varname, dtype = data_type)
	df = df.join(add_col)

	return df
