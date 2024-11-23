# CSE6242 - Data & Visual Analytics - Fall 2024
## Contributors

Carlos Moncada Gonzalez,  
Christian Park,  
Kenny Phan,  
Julian Johnson,  
Parker Brotman

**CSE 6242 course project on predicts fantasy football scores with an accuracy that rivals major sporting platforms**

## Desription:
This repository contains the necessary files needed to run the interactive nfl fantasy player prediction tool. Below will include the following relevant files:
- build_nfl_model_prod.py
    - Module that is used to perform all the modeling from start to finish.
    - Includes feature engineering, model training, and generating predictions
- requirements.txt 
    - Text file that lists out the required packages needed to use the tool properly
- plotly-dash-viz.py 
    - Module to run for final visualization tool
- generate_nfl_prediction_data.py
    - Python script used to aggregate the csv used for the backend of the interactive tool.
- generate_nfl_prediction_plots.py
    - Python script used to generate fantasy model interactive visualization for season 2024.

The steps that are outlined below are imperative to using the tool properly.


## Installation: 
### Prerequisites
 
	- Python 3.8+
	- Ensure you have pip or conda installed.
 
Clone the Repository

git clone https://github.com/yourusername/Team-134-CSE-6242-Project.git
cd Team-134-CSE-6242-Project




### Execution:

#### *1. Setup:*

- Install Dependencies
 
	1.	Install the required Python packages: `pip install -r requirements.txt`
        - Alternatively, you can recreate the environment using conda:
            - conda env create -f environment.yml
            - conda activate team-134-project
	2.	Verify that all required packages are installed using the utility script:

This will ensure that you have the necessary packages needed to run the code using package versions this project was developed in.

#### *2. Preparing the data for the backend:*

This step requires running the file `generate_backend_data.py` in the terminal. Run the following command in the terminal:
```
python .\generate_nfl_prediction_data.py
```

The output of running this command should provide you a final csv called *"fantasy_prediction_data.csv"* that will be saved to your working directory. This csv is what will be used 
for the backend of the interactive visualization. NOTE: This will take time due to generating features and training multiple models.
If you are less concerned about the accuracy of the model, go into the "generate_backend_data.py" file and change the start date to a more recent
date ie. (2015). This should reduce run time but decrease the amount of data allowed for the model training. 

#### *3. Visualization Re-Produce Process*

##### Visualization (Final):
Make sure the data file created by *generate_nfl_prediction_data.py* is located at `fantasy_prediction_data.csv`.

 Run the following command in the terminal:
```
python .\plotly-dash-viz.py
```

The terminal output should include a line similar to the following:
```
Dash is running on http://127.0.0.1:8050/
```

Navigate to this URL in your browser.


##### Visualization (Deployed):

**Note, this is a prototype interactive viz tool on season 2024 for cross-checking with the final interactive tool. Please refer to and run the final version for all seasons.**

A step-by-step process to reproduce the plots deployed at:
  - https://cse-6242-t134.github.io


Make sure the data file created by *generate_nfl_prediction_data.py* is located at `fantasy_prediction_data.csv`.

 Run the following command in the terminal:
```
python .\generate_nfl_prediction_plots.py
```

The output of running this command should provide you an `index.html` file that will be saved to your working directory. This `html` is the homepage for interactive viz and which will be used 
for the deployment to github pages.
