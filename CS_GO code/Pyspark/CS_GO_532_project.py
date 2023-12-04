#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().system('pip install pyspark')


# In[ ]:


from pyspark.ml import Pipeline
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.context import SparkContext
from pyspark.sql.session import SparkSession
import pandas as pd
import matplotlib.pyplot as plt
from pyspark.ml.feature import StringIndexer
from pyspark.sql.functions import col
import numpy as np


# In[ ]:


from pyspark.ml import Pipeline
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
import psutil
import time
from threading import Thread
from pyspark.ml.feature import StringIndexer
from pyspark.ml.feature import OneHotEncoder, VectorAssembler
from pyspark.sql.functions import when


# In[ ]:


def get_system_stats():
    cpu_percent = psutil.cpu_percent(interval=1)
    memory_info = psutil.virtual_memory()

    return cpu_percent, memory_info.percent

num_cores = 8

spark = SparkSession.builder \
    .appName("CS_GO") \
    .config("spark.executor.cores", str(num_cores)) \
    .getOrCreate()


# In[ ]:


master_demo_data = spark.read.csv("mm_master_demos.csv", inferSchema=True, header=True)
master_demo_data_drop = ['_c0','index','file', 'date', 'att_team', 'att_side', 'vic_team', 'att_id', 'vic_id', 'winner_team',
                         'att_pos_x', 'is_bomb_planted','att_pos_y', 'vic_pos_x', 'vic_pos_y', 'avg_match_rank', 'att_rank', 'vic_rank','tick','hitbox','award','wp']
master_df = master_demo_data.drop(*master_demo_data_drop)

feature_columns = ["map", "round", "seconds", "hp_dmg", "arm_dmg", "is_bomb_planted", "bomb_site",
                    "wp_type", "winner_side", "round_type", "ct_eq_val", "t_eq_val"]


# In[9]:


cat =  ['map', 'bomb_site', 'wp_type', 'winner_side', 'round_type']
num = ['round', 'seconds', 'hp_dmg', 'arm_dmg', 'ct_eq_val', 't_eq_val']

indexers = [StringIndexer(inputCol=col, outputCol=f"{col}_index", handleInvalid="skip") for col in cat]

encoders = [OneHotEncoder(inputCol=f"{col}_index", outputCol=f"{col}_encoded") for col in cat]

assembler = VectorAssembler(inputCols=[f"{col}_encoded" for col in cat] + num, outputCol="features")

pipeline = Pipeline(stages=indexers + encoders + [assembler])

pipeline_model = pipeline.fit(master_df)
result_df = pipeline_model.transform(master_df)

result_df = result_df.withColumn('vic_side_int', when(result_df.vic_side=="Terrorist", 1).otherwise(0))


result_df = result_df.select("features","vic_side_int")

time_taken = []
cpu_usage = []
memory_usage = []
accuracy_core = []
for i in range(num_cores):
    start_time = time.time()
    training_df, validation_df = result_df.randomSplit([0.7,0.3])
    rf = RandomForestClassifier(labelCol="vic_side_int", featuresCol="features", numTrees=10, maxDepth=30)
    rf_model = rf.fit(training_df)

    predictions = rf_model.transform(validation_df)


    evaluator = MulticlassClassificationEvaluator(labelCol="vic_side_int", predictionCol="prediction", metricName="accuracy")
    accuracy_core.append(evaluator.evaluate(predictions)*100)
    cpu_percent, memory_percent = get_system_stats()
    time_taken.append(time.time() - start_time)
    cpu_usage.append(cpu_percent)
    memory_usage.append(memory_percent)



plt.figure(figsize=(12, 8))


plt.subplot(4, 1, 1)
plt.plot(range(num_cores), cpu_usage, marker='o')
plt.title('CPU Usage per core')
plt.xlabel('Core')
plt.ylabel('CPU Usage (%)')


plt.subplot(4, 1, 2)
plt.plot(range(num_cores), memory_usage, marker='o', color='orange')
plt.title('Memory Usage per core')
plt.xlabel('Core')
plt.ylabel('Memory Usage (%)')


plt.subplot(4, 1, 3)
plt.plot(range(num_cores), time_taken, marker='o', color='green')
plt.title('Time Taken to execute the Random forest per core')
plt.xlabel('Core')
plt.ylabel('Time Taken (seconds)')

plt.subplot(4, 1, 4)
plt.plot(range(num_cores), accuracy_core, marker='o')
plt.title('Accuracy per core')
plt.xlabel('Core')
plt.ylabel('Accuracy (%)')


plt.tight_layout()
plt.show()

