 README

To run the model, type python3 script.py from the command line. The final output is in the output folder, named 'final_output_hw5.csv'.

All the globals used to in the functions are specified in the script.py program. One could change the infile, outfile and other preprocessing and cleaning parameters in the file and run from the command line if desired.

script.py imports functions from three python programs:
  - preprocess_helper: Helper functions used to clean the dataset. Imported and used by the preprocess.py script
  - preprocess: The script to clean the datasets (used on the cleaning and test sets separately)
  - ml_pipeline.py: Functions used to build the and evaluate the machine learning models.
