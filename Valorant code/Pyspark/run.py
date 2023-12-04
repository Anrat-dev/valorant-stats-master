from run_agents import random_forest as random_forest_agent
from run_weapons import random_forest as random_forest_weapons

import matplotlib.pyplot as plt

exec_times = []
cpu_usage = []
mem_usage  = []
accuracy =[]

# executes the classifier for increasin gnumber of cores, starting from a single core to 8 cores
for i in range(1, 9):
    # acc, exec, cpu, mem = random_forest_agent(i)
    acc, exec, cpu, mem = random_forest_weapons(i)
    exec_times.append(exec)
    cpu_usage.append(cpu)
    mem_usage.append(mem)
    accuracy.append(acc)

plt.figure(figsize=(12, 8))


plt.subplot(3, 1, 1)
plt.plot(range(1, 9), cpu_usage, marker='o')
plt.title('CPU Usage per core')
plt.xlabel('Core')
plt.ylabel('CPU Usage (%)')


plt.subplot(3, 1, 2)
plt.plot(range(1, 9), mem_usage, marker='o', color='orange')
plt.title('Memory Usage per core')
plt.xlabel('Core')
plt.ylabel('Memory Usage (%)')


plt.subplot(3, 1, 3)
plt.plot(range(1, 9), exec_times, marker='o', color='green')
plt.title('Time Taken to execute the Random forest per core')
plt.xlabel('Core')
plt.ylabel('Time Taken (seconds)')

plt.tight_layout()
plt.show()


print("accuracy : ", sum(accuracy)/len(accuracy))
print(f"Execution time: {sum(exec_times)/len(exec_times)}")
print('The CPU usage is: ', sum(cpu_usage)/len(cpu_usage))
print('RAM memory % used:', sum(mem_usage)/len(mem_usage))

# # agent data
# accuracy :  0.8494471800353159
# Execution time: 0.5032490730285644
# The CPU usage is:  20.045
# RAM memory % used: 74.30250000000001

# # weapons data 
# accuracy :  0.8500716120731413
# Execution time: 0.5453400433063507
# The CPU usage is:  20.696250000000003
# RAM memory % used: 74.22749999999999