# Valorant Folder
There are two folders in the Valorant folder, one for running with Sklearn and one for running with Pyspark

## Files

* ```run.py``` - Primary file which runs the classifier for cores in range (1, 8). 

* ```run_agents.py``` - File that uses Pyspark to create a Random Forest classifier for agent data. It classifies the Win rate into 3 categories [low(<30%), average(30%-70%), high(>70%)]. This python script creates a spark session and then creates a Random Forest using RandomForestClassifier from pyspark.ml.classification 

* ```run_weapons.py``` - File that uses Pyspark to create a Random Forest classifier for weapon data. It classifies the Damage done per round into 3 categories [low(<100), average(100-120) and high(>120)]. This python script creates a spark session and then creates a Random Forest using RandomForestClassifier from pyspark.ml.classification 

## Folders (Data)

* ```complete_file_agents.csv``` - agent abilities usage. Only includes data from the aggregate of all maps.
* ```complete_file_weapons.csv``` - essential weapon data, such as headshot percentage and average damage per round. Includes indvidual map data too.

# CSGO Folder
There are two folders in the Valorant folder, one for running with Sklearn and one for running with Pyspark

# Files

* ```run.py``` - Primary file which runs the classifier for cores in range (1, 8). 


# Folders (Data)

* ```complete_file_agents.csv``` - agent abilities usage. Only includes data from the aggregate of all maps.

