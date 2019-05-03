import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import train_test_split
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
def define_model(model, parameters, n_cores=None):
	'''
	Code derived (with small modifications from) Rayid Ghani:
	https://github.com/dssg/police-eis/blob/master/eis/models.py
	'''
	if model == "RandomForest":
        return ensemble.RandomForestClassifier(
            n_estimators=parameters['n_estimators'],
            max_features=parameters['max_features'],
            criterion=parameters['criterion'],
            max_depth=parameters['max_depth'],
            min_samples_split=parameters['min_samples_split'],
            random_state=parameters['random_state'],
            n_jobs=n_cores)

    elif model == "RandomForestBagging":
        #TODO Make Model Bagging #COME BACK HERE - not done.
        return ensemble.BaggingClassifier(
                    ensemble.RandomForestClassifier(
                        n_estimators=parameters['n_estimators'],
                        max_features=parameters['max_features'],
                        criterion=parameters['criterion'],
                        max_depth=parameters['max_depth'],
                        min_samples_split=parameters['min_samples_split'],
                        random_state=parameters['random_state'],
                        n_jobs=n_cores),
                    #Bagging parameters
                    n_estimators=parameters['n_estimators_bag'],
                    max_samples=parameters['max_samples'],
                    max_features=parameters['max_features_bag'],
                    bootstrap=parameters['bootstrap'],
                    bootstrap_features=parameters['bootstrap_features'],
                    n_jobs=n_cores
                    )

    elif model == "RandomForestBoosting":
        #TODO Make Model Boosting
        return ensemble.AdaBoostClassifier(
            ensemble.RandomForestClassifier(
                n_estimators=parameters['n_estimators'],
                max_features=parameters['max_features'],
                criterion=parameters['criterion'],
                max_depth=parameters['max_depth'],
                min_samples_split=parameters['min_samples_split'],
                random_state=parameters['random_state'],
                n_jobs=n_cores),
            #Boosting parameters
            learning_rate=parameters['learning_rate'],
            algorithm=parameters['algorithm'],
            n_estimators=parameters['n_estimators_boost']
			)

	elif model == 'SVM':
        return svm.SVC(C=parameters['C_reg'],
                       kernel=parameters['kernel'],
                       probability=True,
                       random_state=parameters['random_state'])

    elif model == 'LogisticRegression':
        return linear_model.LogisticRegression(**parameters)

	elif model == 'DecisionTreeClassifier':
        return tree.DecisionTreeClassifier(
            max_features=parameters['max_features'],
            criterion=parameters['criterion'],
            max_depth=parameters['max_depth'],
            min_samples_split=parameters['min_samples_split'],
			random_state=parameters['random_state'])

	elif model == 'KNeighborsClassifier':
        return neighbors.KNeighborsClassifier(**parameters)


def build_model(modelobj, X_train, y_train):
	'''
	Takes a model class and fits the model on the training data given.
	'''
	modelobj = model.fit(X_train, y_train)


def train_logistic_model(training_df, predictor_vars, target_var, model_type):
	'''
	Trains a logistic regression model using the dataframe provided,
	a list of the predictor_vars in the dataframe, and the target variable.

	Inputs:
		- training_df: pandas dataframe
		- predictor_vars: list of variables in training_df to use as predictor
			variables in the model
		- target_var: (str) the dependent variable in the model.
		- model_type:
			- This is the scikit learn model class the user wants to run:
				options:
				-LogisticRegression
				-KNeighborsClassifier
	'''
	pred_data = training_df[predictor_vars]
	dep_data = training_df[[target_var]].to_numpy().ravel()
	logistic_regr = LogisticRegression()
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
	# class_probability = model.predict_proba(pred_data)
	# predicted_scores = np.where(class_probability >= classifier_threshold, 1, 0)
	score = model.score(pred_data, dep_data)

	return score

# def create_confusion_matrix():
