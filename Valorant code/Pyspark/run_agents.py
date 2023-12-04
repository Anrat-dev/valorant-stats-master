import pandas as pd
import os
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.sql import SparkSession
import pyspark.sql.functions as F
import time
import psutil
import matplotlib.pyplot as plt

# creates a random forest using pyspark.ml
def random_forest(func_num_cores):
    num_cores = func_num_cores
    # create a spark session with 'i' number of cores for execution
    spark = SparkSession.builder.config("spark.executor.cores", str(num_cores)).getOrCreate()

    # path to the file being classified
    path = '../Data/complete_file_agents.csv'
    df = spark.read.csv(path, inferSchema=True, header=True)
    # df.show()

    # this converts the columns into a single column called features
    # this is done because the PySpark classifier requires the format label, features -> where features is an array of all the associated values and label is a column that is being predicted
    assembler =  VectorAssembler(inputCols=['Agent Name', 'KD', 'Kills', 'Deaths', 'Assists', 'Pick Rate', 'ACS', 'First Blood', 'Num Matches'], outputCol='features')
    output = assembler.transform(df)
    # output.show()
    # output.select(['features', 'Label']).show(truncate=False)


    model_df = output.select(['features', 'Label'])

    exec_times = []
    cpu_usage = []
    mem_usage  = []
    accuracy =[]
    # iterating the random forest 10 times just to make sure we get a good average value for every core number execution
    # sometimes the values produced can be extreme values and not mean values therefore to get an accurate analysis we take the average of ten runs
    for i in range(10):
        start_time = time.time()
        training_df, test_df = model_df.randomSplit([0.7,0.3])
        # print(model_df.count())
        # print(training_df.count())
        # print(test_df.count())

        # PySpark classifier that create a Random forest with 50 decision trees. 
        # For the valorant data set we have not added a max_depth as it is a smaller dataset and runs only to a max depth of 15
        rf_classifier = RandomForestClassifier(labelCol="Label", numTrees=50).fit(training_df)
        rf_predictions = rf_classifier.transform(test_df)
        # rf_predictions.show()

        # this is a PySpark function that can evaluate a given PySpark classifier
        rf_auc = MulticlassClassificationEvaluator(labelCol='Label').evaluate(rf_predictions)
        
        # here we append the required to data to measure system metric as well as the accuracy
        exec_times.append(time.time() - start_time)
        cpu_usage.append(psutil.cpu_percent())
        mem_usage.append(psutil.virtual_memory()[2])
        accuracy.append(rf_auc)

    return (sum(accuracy)/len(accuracy), sum(exec_times)/len(exec_times), sum(cpu_usage)/len(cpu_usage), sum(mem_usage)/len(mem_usage))

