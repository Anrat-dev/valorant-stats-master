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


def random_forest(func_num_cores):
    num_cores = func_num_cores
    spark = SparkSession.builder.config("spark.executor.cores", str(num_cores)).getOrCreate()


    path = '../Data/complete_file_weapons.csv'
    df = spark.read.csv(path, inferSchema=True, header=True)


    assembler =  VectorAssembler(inputCols=['Weapon Name', 'Kills Per Match', 'Headshot', 'Bodyshot', 'Legshot'], outputCol='features')
    output = assembler.transform(df)
    # output.show()
    # output.select(['features', 'Label']).show(truncate=False)


    model_df = output.select(['features', 'Label'])

    exec_times = []
    cpu_usage = []
    mem_usage  = []
    accuracy =[]
    for i in range(10):
        start_time = time.time()
        training_df, test_df = model_df.randomSplit([0.7,0.3])
        # print(model_df.count())
        # print(training_df.count())
        # print(test_df.count())

        rf_classifier = RandomForestClassifier(labelCol="Label", numTrees=50).fit(training_df)
        rf_predictions = rf_classifier.transform(test_df)
        # rf_predictions.show()

        rf_auc = MulticlassClassificationEvaluator(labelCol='Label').evaluate(rf_predictions)
        exec_times.append(time.time() - start_time)
        cpu_usage.append(psutil.cpu_percent())
        mem_usage.append(psutil.virtual_memory()[2])
        accuracy.append(rf_auc)
    
    return (sum(accuracy)/len(accuracy), sum(exec_times)/len(exec_times), sum(cpu_usage)/len(cpu_usage), sum(mem_usage)/len(mem_usage))


    # print("accuracy : ", sum(accuracy)/len(accuracy))
    # print(f"Execution time: {sum(exec_times)/len(exec_times)}")
    # print('The CPU usage is: ', sum(cpu_usage)/len(cpu_usage))
    # print('RAM memory % used:', sum(mem_usage)/len(mem_usage))

    # plt.figure(figsize=(12, 8))


    # plt.subplot(3, 1, 1)
    # plt.plot(range(num_cores), cpu_usage, marker='o')
    # plt.title('CPU Usage per core')
    # plt.xlabel('Core')
    # plt.ylabel('CPU Usage (%)')


# plt.subplot(3, 1, 2)
# plt.plot(range(num_cores), mem_usage, marker='o', color='orange')
# plt.title('Memory Usage per core')
# plt.xlabel('Core')
# plt.ylabel('Memory Usage (%)')


# plt.subplot(3, 1, 3)
# plt.plot(range(num_cores), exec_times, marker='o', color='green')
# plt.title('Time Taken to execute the Random forest per core')
# plt.xlabel('Core')
# plt.ylabel('Time Taken (seconds)')

# plt.tight_layout()
# plt.show()