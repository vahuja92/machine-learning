import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
import read_clean as rc

raw_data = read_dataset(filename, parameters)
# explore data
pre_df = raw_data.copy(deep=True)

#Pre-process the data
fill_missing_w_median(pre_df)
#check to see if I even need to return anything for these functions to change
#the dataframe
pre_df = discretize_cont_var(pre_df, discretized_var, q_num_quantiles = 4)
pre_df = create_dummies(pre_df, dummy_var)

#Build Model
predictor_vars = None
target_var = None
#fill the predicor and target_var in (or make a Dataset class? )
logistic_regr = train_logistic_model(training_df, predictor_vars, target_var)
score = evaluate_logistic_model(logistic_regr, x_test, y_test)
