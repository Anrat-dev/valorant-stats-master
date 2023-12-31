# Valorant Folder
There are two folders in the Valorant folder, one for running with Sklearn and one for running with Pyspark

## Files

* ```run.py``` - Primary file which runs the classifier for cores in range (1, 8). 

* ```run_agents.py``` - File that uses Pyspark to create a Random Forest classifier for agent data. It classifies the Win rate into 3 categories [low(<30%), average(30%-70%), high(>70%)]. This python script creates a spark session and then creates a Random Forest using RandomForestClassifier from pyspark.ml.classification 

* ```run_weapons.py``` - File that uses Pyspark to create a Random Forest classifier for weapon data. It classifies the Damage done per round into 3 categories [low(<100), average(100-120) and high(>120)]. This python script creates a spark session and then creates a Random Forest using RandomForestClassifier from pyspark.ml.classification. To execute the Pyspark Classifier execute the run.py file. To pick which dataset to run the classifier on, comment/uncomment the function call on lines 13 and 14

* ```AgentsModel.py``` - File that uses Scikit-learn library to utilize its random forest classifier for agent data. This classifies the Win rate into 3 categories [low(<30%), average(30%-70%), high(>70%)]. The whole file can be opened as a .pynb file (Jupyter Notebook) and the code cells can be run sequentially.

* ```weaponsModel.py``` - File that uses Scikit-learn library to utilize its random forest classifier for agent data. This classifies the Damage done per round into 3 categories [low(<100), average(100-120) and high(>120)]. The whole file can be opened as a .pynb file (Jupyter Notebook) and the code cells can be run sequentially.


## Folders (Data)

* ```complete_file_agents.csv``` - agent abilities usage. Only includes data from the aggregate of all maps.
* ```complete_file_weapons.csv``` - essential weapon data, such as headshot percentage and average damage per round. Includes indvidual map data too.

# CSGO Folder
There are two folders in the CSGO folder, one for running with Sklearn and one for running with Pyspark.

## Files

* ```Random_fr.ipynb``` - Primary file which runs the classifier for cores in range (1, 8) for the Sklearn Model. The file can be run sequentially by running individual code blocks
* ```CS_GO_532_project.py``` -  Primary file to run the classifier for the pyspark implementation. Running the script will run the entire classifier right away


## Folders (Data)

* ```mm_master_demos.csv``` - Include all the relevant rows for the dataset. The preprocessing for this takes place in the Python notebook and python script respectively

