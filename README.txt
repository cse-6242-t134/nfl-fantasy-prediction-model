README.txt - a concise, short README.txt file, corresponding to the "user guide". This file should contain:

    DESCRIPTION - Describe the package in a few paragraphs
    INSTALLATION - How to install and setup your code
    EXECUTION - How to run a demo on your code
    [Optional, but recommended] DEMO VIDEO - Include the URL of a 1-minute *unlisted* YouTube video in this txt file. The video would show how to install and execute your system/tool/approach (e.g, from typing the first command to compile, to system launching, and running some examples). Feel free to speed up the video if needed (e.g., remove less relevant video segments). This video is optional (i.e., submitting a video does not increase scores; not submitting one does not decrease scores). However, we recommend teams to try and create such a video, because making the video helps teams better think through what they may want to write in the README.txt, and generally how they want to "sell" their work.


Desription:
 This module is necessary for generating the model object that is used in the backend for the interactive visualization tool that tracks 
 NFL Players fantasy performance. The methods present in this module are listed below with a brief outline of what each of them accopmlishes:

 load_data(): This method is used to load in the necessary data used to generate features for each player. This data spans from 1999 to 2024 and is uploaded weekly
 as games are played.

 generate_features(): This method is used to generate features for a given position group. This takes in the outputs from the load_data() method and aggregates features. 
 The feature level data is returned from this method.

 train_model(): This method takes in the feature level data and provides the necessary preprocessing steps required for this data before undergoing model fitting.
 These steps include the handling of categorical variables,isolating the variables of interest (features that will be used for the model), and splitting the data into the necessary test/train splits.
 After these preprocessing steps after completed, the model will be fit. The model object and preprocessed data will be returned.



Installation: 


Execution:

To execute the code successfully and return the model object, all the user must do is run the following method.

- run_entire_model_process()

The only parameter necessary for this method is the position group of interest. For ease of use purposes, the default value is set 
to 'rb_wr'. This method will return the same outputs as the train_model() parameter but will also run all the prior methods before it. 
This method is just a shortcut to returning the model object without having the run all the required steps beforehand as this method 
will handle it all for the user.