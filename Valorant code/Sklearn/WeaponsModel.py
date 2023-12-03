# %% [markdown]
# ### RANDOM FOREST CLASSIFICATION USING SCIKIT-LEARN ON WEAPONS DATASET 

# %% [markdown]
# Importing Libraries

# %%
import pandas as pd
from memory_profiler import profile
import time
import psutil
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

%reload_ext memory_profiler

# %% [markdown]
# Method for getting system statistics to capture utilization of system resources during training

# %%
def get_system_stats():
    cpu_percent = psutil.cpu_percent(interval=1)
    memory_info = psutil.virtual_memory()[2]

    return cpu_percent, memory_info

# %% [markdown]
# Code cell for the training process including data pre-processing

# %%
path = '../complete_file_weapons.csv'
df = pd.read_csv(path)
feature_columns = [col for col in df.columns if col != 'Label']
X = df[feature_columns]
y = df['Label']


def train_evaluate_model(X_train, y_train, X_test, y_test, cores, random_state, estimators):

    cpu_percent_list = []
    memory_percent_list = []
    time_taken_list = []
    accuracy_list = []

    for i in range(10):

        start_time = time.time()

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
        
        rf_classifier = RandomForestClassifier(n_estimators=estimators, n_jobs=cores, random_state=random_state, max_depth=10)
        rf_classifier.fit(X_train, y_train)

        end_time = time.time()
        training_time = (end_time - start_time) * 1000  # Convert to milliseconds

        predictions = rf_classifier.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)

        cpu_percent, memory_percent = get_system_stats()
        accuracy_list.append(accuracy)
        time_taken_list.append(time.time() - start_time)
        cpu_percent_list.append(cpu_percent)
        memory_percent_list.append(memory_percent)

    
    finalAcc = sum(accuracy_list)/10

    finalCPU = sum(cpu_percent_list)/10
    cpu_usage.append(finalCPU)
    finalmem = sum(memory_percent_list)/10
    memory_usage.append(finalmem)
    finaltimetaken = sum(time_taken_list)/10
    time_taken.append(finaltimetaken)
    
    
    print(f"Accuracy with {cores} cores: {finalAcc}")
    print(f"Training time with {cores} cores: {finaltimetaken:.2f} ms")
    print(f"CPU Usage with {cores} cores: {finalCPU}")
    print(f"Memory Usage with {cores} cores: {finalmem}")

# %% [markdown]
# Calling the classifier function and running it by incrementing the number of cores used

# %%
time_taken = []
cpu_usage = []
memory_usage = [] 

for cores in range(1,9):

    random_state = [0, 42, 34, 120, 100, 89, 24, 60, 123]
    trees = 50
    maxDepth = 10
    train_evaluate_model(X_train, y_train, X_test, y_test, cores, 42, trees)

# %% [markdown]
# Plotting the results

# %%
plt.figure(figsize=(12, 8))


plt.subplot(3, 1, 1)
plt.plot(range(1,9), cpu_usage, marker='o')
plt.title('CPU Usage per core')
plt.xlabel('Core')
plt.ylabel('CPU Usage (%)')


plt.subplot(3, 1, 2)
plt.plot(range(1,9), memory_usage, marker='o', color='orange')
plt.title('Memory Usage per core')
plt.xlabel('Core')
plt.ylabel('Memory Usage (%)')

plt.subplot(3, 1, 3)
plt.plot(range(1,9), time_taken, marker='o', color='green')
plt.title('Time Taken to execute the Random forest per core')
plt.xlabel('Core')
plt.ylabel('Time Taken (seconds)')

plt.tight_layout()
plt.show()


