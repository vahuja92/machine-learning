To create the outputs, run simple_loop.py from the command line, with the following arguments:

infile outfile run_on_sample models_to_run grid size

For example, you could run: 

python simple_loop.py "data/projects_2012_2013.csv" "output/test.csv" ["DT"] 1 "test"

Description of the arguments:
 
  Infile: a csv file with the data
  Outfile: a csv file to write the results of running the models
  run_on_sample: (int) 1 to run on a randomly selected quarter of the sample
                  0 to run on the full sample
  models_to_run: (list) list of models to run (choices are: RF, ET, AB, LR, SVM, GB, NB, DT, SGD, KNN)    
  grid_size: (str) "test", "small", "large"
                  - corresponds to a dictionary with a given amount of parameters for each model.
                  
  
