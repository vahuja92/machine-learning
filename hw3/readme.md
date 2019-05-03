To create the outputs, import simple_loop.py into Ipython.

From Ipython, run following function:

main(infile, outfile, run_on_sample, grid_size)
 
 Description of function - 
 '''
  Infile: a csv file with the data
  Outfile: a csv file to write the results of running the models
  run_on_sample: 1 to run on a randomly selected quarter of the sample
                  0 to run on the full sample
  grid_size: (str) "test", "small", "large"
                  - corresponds to a dictionary with a given amount of parameters for each model.
                  
  '''
