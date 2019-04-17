import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

'''
4/18
Tomorrow:
	2. Do the explore data section
	3. Do write-up
'''

def read_dataset(filename, data_types=None, parameters=None):
	'''
	Takes a csv filepath/filename and saves as a pandas dataframe, with the
	first row used as the header.

	A JSON file with the following format:
		Dataset Name: 'str'
		with names of the variables and the datatypes can be provided
	as well.

	(Go back and do this.)
	'''

	dataframe = pd.read_csv(filename, dtype=data_types)

	return dataframe

'''
Explore the dataframe
Should run these functions separately in Jupyter notebooks
Qs - how do I output the describe() table using a function?
df.describe()

ExploreData:You can use the code you wrote for assignment 1 heretogenerate
distributions of variables, correlations between them, find outliers,
and data summaries.
'''

def var_distributions(df):
	print(df.describe())

def explore(df):
	df.describe()

def cr_box_plot(df):
	df.boxplot()
	plt.show()

	df.plot()
	plt.show()

'''
Pre-ProcessData:Forthisassignment,youcanlimitthistofilling
in missing values for the variables that have missing values.
You can use any simple method to do it (use mean or median to
fill in missing values).
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
		if df[c].isnull().any():
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


#################################################################
 				# BUILD MODEL
#################################################################

def train_logistic_model(training_df, predictor_vars, target_var):
	'''
	Trains a logistic regression model using the dataframe provided,
	a list of the predictor_vars in the dataframe, and the target variable.

	Inputs:
		- training_df: pandas dataframe
		- predictor_vars: list of variables in training_df to use as predictor
			variables in the model
		- target_var: (str) the dependent variable in the model.
	'''
	pred_data = training_df[predictor_vars]
	dep_data = training_df[[target_var]].to_numpy().ravel()
	logistic_regr = LogisticRegression(solver='liblinear', penalty='l1')
	logistic_regr.fit(pred_data, dep_data.ravel())

	return logistic_regr

def evaluate_logistic_model(model, test_df, predictor_vars, target_var):
	'''
	Takes a trained model and evaluates the accuracy of the model on the given
	testing_df.

	Inputs:
		- model: a scikit learn model class
		- testing_df: pandas dataframe (the testing dataframe to test predicted
					  vs. actual accuracy)

	Outputs:
		- score: (float) The percent of classifiers the model predicted
				correctly.
	'''
	pred_data = test_df[predictor_vars]
	dep_data = test_df[[target_var]].to_numpy().ravel()
	score = model.score(pred_data, dep_data)

	return score
# # Replace using median
# median = df['NUM_BEDROOMS'].median()
# df['NUM_BEDROOMS'].fillna(median, inplace=True)
