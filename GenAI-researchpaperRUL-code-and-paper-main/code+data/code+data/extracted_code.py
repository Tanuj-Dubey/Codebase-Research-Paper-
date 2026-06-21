import datetime
import pandas as pd
from scipy.io import loadmat
from pandas import DataFrame
import keras

print("Input which battery you need to extract data from. Choose from the following")
print("Battery Number: B0005,B0006,B0007,B0018")
B = input()

def to_df(mat_db):
    """Returns one pd.DataFrame per cycle type"""

    # Features common for every cycle
    cycles_cols = ['type', 'ambient_temperature', 'time']

    # Features monitored during the cycle
    features_cols = {
        'charge': ['Voltage_measured', 'Current_measured', 'Temperature_measured',
                'Current_charge', 'Voltage_charge', 'Time'],
        'discharge': ['Voltage_measured', 'Current_measured', 'Temperature_measured',
                    'Current_charge', 'Voltage_charge', 'Time', 'Capacity'],
        'impedance': ['Sense_current', 'Battery_current', 'Current_ratio',
                    'Battery_impedance', 'Rectified_impedance', 'Re', 'Rct']
    }

    # Define one pd.DataFrame per cycle type
    df = {key: pd.DataFrame() for key in features_cols.keys()}

    # Get every cycle
    print(f'Number of cycles: {mat_db[0][0][0].shape[1]}')
    cycles = [[row.flat[0] for row in line] for line in mat_db[0][0][0][0]]

    # Get measures for every cycle
    for cycle_id, cycle_data in enumerate(cycles):
        tmp = pd.DataFrame()

        # Data series for every cycle
        features_x_cycle = cycle_data[-1]

        # Get features for the specific cycle type
        features = features_cols[cycle_data[0]]

        for feature, data in zip(features, features_x_cycle):
            if len(data[0]) > 1:
                # Correct number of records
                tmp[feature] = data[0]
            else:
                # Single value, so assign it to all rows
                tmp[feature] = data[0][0]

        # Add columns common to the cycle measurements
        tmp['id_cycle'] = cycle_id
        for k, col in enumerate(cycles_cols):
            tmp[col] = cycle_data[k]

        # Append cycle data to the right pd.DataFrame
        cycle_type = cycle_data[0]
        df[cycle_type] = df[cycle_type].append(tmp, ignore_index=True)

    return df

mat_db = loadmat(B+'.mat') [B]
dfs = to_df(mat_db)

dfs["charge"].to_csv("charge_"+B+".csv")
dfs["discharge"].to_csv("discharge_"+B+".csv")
chdf=dfs["charge"]
disdf=dfs["discharge"]

import numpy as np
import matplotlib.pyplot as plt
# Plotting the 2D graph
plt.figure(figsize=(8, 6))
plt.scatter(chdf['Time'], chdf['Current_charge'],label='Current', c=chdf['id_cycle'], cmap='jet')  # Color-coded scatter plot
plt.scatter(chdf['Time'], chdf['Voltage_charge'],label='Voltage', c=chdf['id_cycle'], cmap='jet')
plt.xlabel('Time(secs)')
plt.ylabel('Current(amp) and Volatge(v)')
plt.title('Change of charge current and voltage with respect to time and charge cycles')
current_label = 'Current'
voltage_label = 'Voltage'
plt.text(0.5, 0.55, current_label, transform=plt.gca().transAxes, ha='center')
plt.text(0.5, 0.94, voltage_label, transform=plt.gca().transAxes, ha='center')
plt.colorbar()  # Add a colorbar to indicate the third attribute
plt.grid(True)
plt.show()

import numpy as np
import matplotlib.pyplot as plt

# Plotting the 2D graph
plt.figure(figsize=(8, 6))
plt.scatter(disdf['Time'], disdf['Voltage_charge'], c=disdf['id_cycle'], cmap='viridis')  # Color-coded scatter plot
plt.xlabel('Time(secs)')
plt.ylabel('Volatge(v)')
plt.title('Change of voltage with respect to time and discharge cycles')
plt.colorbar()  # Add a colorbar to indicate the third attribute
plt.grid(True)
plt.show()


# load .mat data
import scipy.io
def loadMat(matfile):
    data = scipy.io.loadmat(matfile)
    filename = matfile.split("/")[-1].split(".")[0]
    col = data[filename]
    col = col[0][0][0][0]
    size = col.shape[0]

    data = []
    for i in range(size):
        k = list(col[i][3][0].dtype.fields.keys())
        d1, d2 = {}, {}
        if str(col[i][0][0]) != 'impedance':
            for j in range(len(k)):
                t = col[i][3][0][0][j][0];
                l = [t[m] for m in range(len(t))]
                d2[k[j]] = l
        d1['type'], d1['temp'],  d1['data'] = str(col[i][0][0]), int(col[i][1][0]),  d2
        data.append(d1)

    return data

def getBatteryCapacity(Battery):
  cycle, capacity = [], []
  current, voltage=[],[]
  temperature=[]
  i = 1
  for Bat in Battery:
    if Bat['type'] == 'discharge':
      capacity.append(Bat['data']['Capacity'][0])
      current.append(Bat['data']['Current_load'][1])
      voltage.append(Bat['data']['Voltage_load'][1])
      temperature.append(Bat['data']['Temperature_measured'][1])
      cycle.append(i)
      i += 1
  return [cycle, capacity,current,voltage,temperature]

fig, ax = plt.subplots(1, figsize=(6, 4))
data = loadMat(B + '.mat')
result= getBatteryCapacity(data)
ax.plot(disdf['id_cycle'],disdf['Capacity'], 'r-', label=B)
ax.set(xlabel='Discharge cycles', ylabel='Capacity (Ah)', title='Capacity degradation at ambient temperature of 24°C')
plt.legend()

# Store capacity and dischage cycles as HI's also store 2nd value of current_load,voltage_load and Temperature_measured during discharge cycles
# 2nd value because first value often contains noise
HI5 = result[0]  #discharge cycles
HI6=result[1]    #capacity
current = result[2]
voltage=result[3]
temperature=result[4]

# Extraction of constant current charge cycle data
chcurrent = []
chvoltage=[]
chtime=[]
chbat_cycle=[]
for temp in range(len(chdf['id_cycle'])):
  if(4.19<=chdf['Voltage_measured'][temp]<=4.22):
    chbat_cycle.append(chdf['id_cycle'][temp])
    chcurrent.append(chdf['Current_measured'][temp])
    chtime.append(chdf['Time'][temp])
    chvoltage.append(chdf['Voltage_measured'][temp])

IC_V=[]
IC_HI=[]
HI1=[]
HI2=[]
IC_HIV={}
ch_cycle=[]
for j in range(len(chcurrent)-1):
  if(chbat_cycle[j]==chbat_cycle[j+1]):
    IC = (chcurrent[j]*(chtime[j+1]-chtime[j]))/(chvoltage[j+1]-chvoltage[j])
    IC_V.append([IC,chvoltage[j],chbat_cycle[j]])
    IC_HI.append(IC)
    IC_HIV[IC]=chvoltage[j]
  else:
    try:
      a=max(IC_HI)
      HI1.append(a)
      HI2.append(IC_HIV[a])
      IC_HI=[]
      IC_HIV={}
      ch_cycle.append(chbat_cycle[j])
    except:
      pass
try:
  a=max(IC_HI)
  HI1.append(a)
  HI2.append(IC_HIV[a])
  ch_cycle.append(chbat_cycle[-1])
  IC_HI=[]
  IC_HIV={}
except:
  pass
print(len(HI1))
print(len(HI2))

IC_V=[]
IC_HI=[]
HI6=[]
HI7=[]
IC_HIV={}
ch_cycle=[]
for j in range(len(chvoltage)-1):
  if(chbat_cycle[j]==chbat_cycle[j+1]):
    IC = (chcurrent[j+1]*chtime[j+1]-chcurrent[j]*chtime[j])/(chvoltage[j])
    IC_V.append([IC,chvoltage[j],chbat_cycle[j]])
    IC_HI.append(IC)
    IC_HIV[IC]=chvoltage[j]
  else:
    try:
      a=max(IC_HI)
      HI6.append(a)
      HI7.append(IC_HIV[a])
      IC_HI=[]
      IC_HIV={}
      ch_cycle.append(chbat_cycle[j])
    except:
      pass
try:
  a=max(IC_HI)
  HI6.append(a)
  HI7.append(IC_HIV[a])
  ch_cycle.append(chbat_cycle[-1])
  IC_HI=[]
  IC_HIV={}
except:
  pass

print(len(HI6))
print(len(HI7))

import csv
df=pd.read_csv(f"/content/{B}_health_index_updated.csv")
a=len(df)

rows = zip(df['cycle'], df['capacity'],df['IC_C_H'],df['IC_C_P'], df['IC_D_H'],df['IC_D_P'],HI6[:a],HI7[:a])
csv_file = f'{B}_health_index_updated_v1.csv'
# Write the rows to the CSV file
with open(csv_file, 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['cycle', 'capacity', 'IC_C_H', 'IC_C_P','IC_D_H','IC_D_P',"IC_CV_H","IC_CV_D"])
    # writer.writerow(['cycle', 'capacity', 'current', 'IC_D_H', 'voltage'])  # Write header
    writer.writerows(rows)

print(f"CSV file '{csv_file}' created successfully.")


chcycle=[]
chIC=[]
chVol=[]
for i,j,k in IC_V:
  chcycle.append(k)
  chIC.append(i)
  chVol.append(j)

# # uncomment to draw the point scatter IC curve for different Discharge cycles
# import numpy as np
# import matplotlib.pyplot as plt
# plt.figure(figsize=(8, 6))
# plt.scatter(chVol,chIC,c=chcycle, cmap='jet')  # Color-coded scatter plot
# plt.ylabel('dQ/dV')
# plt.xlabel('Volatge(v)')
# plt.title('IC curve for different charge cycles')
# plt.colorbar()  # Add a colorbar to indicate the third attribute
# plt.grid(True)
# plt.show()

import matplotlib.pyplot as plt
fig, ax = plt.subplots()
for cycle in set(chcycle):
    # Filter data for the current cycle
    cycle_IC = [ic for ic, c in zip(chIC, chcycle) if c == cycle]
    cycle_voltage = [v for v, c in zip(chVol, chcycle) if c == cycle]
    # each cycle has diff colour print the label for first,middle,and last cycle only
    if cycle == 1 or cycle == len(set(chcycle))/2 or cycle == len(set(chcycle)):
        ax.plot(cycle_voltage, cycle_IC, 'bo')
    else:
        ax.plot(cycle_voltage, cycle_IC,'bo')
ax.set_xlabel('Voltage')
ax.set_ylabel('IC(dQ/dV)')
ax.set_title('IC vs Voltage for Different Charge Cycles')
ax.legend()
plt.show()


print((cmap))

import matplotlib.pyplot as plt
import matplotlib.cm as cm

fig, ax = plt.subplots()
cmap = cm.get_cmap('tab20')  # Choose a colormap with many colors

# Generate a list of colors for each cycle
colors = [cmap(i) for i in range(len(set(chcycle)))]

for cycle, color in zip(set(chcycle), colors):
    # Filter data for the current cycle
    cycle_IC = [ic for ic, c in zip(chIC, chcycle) if c == cycle]
    cycle_voltage = [v for v, c in zip(chVol, chcycle) if c == cycle]
    # Plot each cycle with its assigned color as a scatter plot
    ax.scatter(cycle_voltage, cycle_IC, color=color)

ax.set_xlabel('Voltage')
ax.set_ylabel('IC(dQ/dV)')
ax.set_title('IC vs Voltage for Different Charge Cycles')
ax.legend()
plt.show()


discurrent = []
disvoltage=[]
distime=[]
disbat_cycle=[]
for temp in range(len(disdf['id_cycle'])):
  disbat_cycle.append(disdf['id_cycle'][temp])
  discurrent.append(disdf['Current_measured'][temp])
  distime.append(disdf['Time'][temp])
  disvoltage.append(disdf['Voltage_measured'][temp])

disIC_V=[]
disIC_HI=[]
disIC_HIV={}
HI3=[]
HI4=[]
dis_cycle=[]
for j in range(len(disdf['id_cycle'])-1):
  if(disbat_cycle[j]==disbat_cycle[j+1]):
    disIC = (discurrent[j]*(distime[j+1]-distime[j]))/(disvoltage[j+1]-disvoltage[j])
    disIC_V.append([disIC,disvoltage[j],disbat_cycle[j]])
    disIC_HI.append(disIC)
    disIC_HIV[disIC]=disvoltage[j]

  else:
    a=max(disIC_HI)
    HI3.append(a)
    HI4.append(disIC_HIV[a])
    dis_cycle.append(disbat_cycle[j])
    disIC_HI=[]
    disIC_HIV={}
try:
  a=max(disIC_HI)
  HI3.append(a)
  HI4.append(disIC_HIV[a])
  dis_cycle.append(disbat_cycle[-1])
  IC_HI=[]
  IC_HIV={}
except:
  pass
print(len(HI3))
print(len(HI4))

discycle=[]
disIC=[]
disVol=[]
for i,j,k in disIC_V:
  discycle.append(k)
  disIC.append(i)
  disVol.append(j)

# uncomment to draw the point scatter IC curve for different Discharge cycles
# import numpy as np
# import matplotlib.pyplot as plt
# plt.figure(figsize=(8, 6))
# plt.scatter(disVol,disIC,c=discycle, cmap='jet')  # Color-coded scatter plot
# plt.ylabel('IC(dQ/dV)')
# plt.xlabel('Volatge(v)')
# plt.title('IC curve for different Discharge cycles')
# plt.colorbar()  # Add a colorbar to indicate the third attribute
# plt.grid(True)
# plt.show()

import matplotlib.pyplot as plt
fig, ax = plt.subplots()
for cycle in set(discycle):
    # Filter data for the current cycle
    cycle_IC = [ic for ic, c in zip(disIC, discycle) if c == cycle]
    cycle_voltage = [v for v, c in zip(disVol, discycle) if c == cycle]
    # each cycle has diff colour print the label for first,middle,and last cycle only
    if cycle == 1 or cycle == len(set(discycle))/2 or cycle == len(set(discycle)):
        ax.plot(cycle_voltage, cycle_IC, label=f'Cycle {cycle}')
    else:
        ax.plot(cycle_voltage, cycle_IC)

ax.set_xlabel('Voltage')
ax.set_ylabel('IC(dQ/dV)')
ax.set_title('IC vs Voltage for Different Discharge Cycles')
ax.legend()
plt.show()


print(len(HI1))
print(len(HI2))
print(len(HI3))
print(len(HI4))
print(len(HI5))
print(len(HI6))
print(len(current))
print(len(voltage))
print(len(temperature))

a=min(len(HI1),len(HI2),len(HI3),len(HI4),len(HI5),len(HI6))
print(a)

fig, ax = plt.subplots(1, figsize=(6, 4))
ax.plot(HI5,HI1[:a], 'bo')

ax.set(xlabel='HI1', ylabel='HI5')
plt.legend()
fig, ax = plt.subplots(1, figsize=(6, 4))
ax.plot(HI5,HI2[:a], 'bo')

ax.set(xlabel='HI2', ylabel='HI5')
plt.legend()
fig, ax = plt.subplots(1, figsize=(6, 4))
ax.plot(HI5,HI3[:a], 'bo')
ax.set(xlabel='HI3', ylabel='HI5')
plt.legend()
fig, ax = plt.subplots(1, figsize=(6, 4))
ax.plot(HI5,HI4[:a],  'bo')
ax.set(xlabel='HI4', ylabel='HI5')
plt.legend()


fig, ax = plt.subplots(1, figsize=(6, 4))
ax.plot(HI1[:a],HI6, 'bo')

ax.set(xlabel='HI1', ylabel='HI6')
plt.legend()
fig, ax = plt.subplots(1, figsize=(6, 4))
ax.plot(HI2[:a],HI6, 'bo')
plt.xlim(3.8,4.5)
ax.set(xlabel='HI2', ylabel='HI6')
plt.legend()
fig, ax = plt.subplots(1, figsize=(6, 4))
ax.plot(HI3[:a],HI6, 'bo')
plt.xlim(10000, 50000)
ax.set(xlabel='HI3', ylabel='HI6')
plt.legend()
fig, ax = plt.subplots(1, figsize=(6, 4))
ax.plot(HI4[:a],HI6,  'bo')
plt.xlim(3.35,3.6)
ax.set(xlabel='HI4', ylabel='HI6')
plt.legend()
fig, ax = plt.subplots(1, figsize=(6, 4))
ax.plot(HI5[:a],HI6, 'bo')
ax.set(xlabel='HI5', ylabel='HI6')
plt.legend()

import pandas as pd
from sklearn.cluster import KMeans

# Load your original dataset into a DataFrame
original_data = pd.read_csv('/content/B0018_health_index_updated.csv')  # Replace 'your_dataset.csv' with the path to your dataset

# Select the features you want to use for clustering
X = original_data[['cycle', 'capacity', 'IC_C_H', 'IC_C_P', 'IC_D_H', 'IC_D_P']]  # Include all relevant features

# Initialize and fit the K-Means model with k=3
kmeans = KMeans(n_clusters=3, random_state=0)
kmeans.fit(X)

# Add the cluster labels to the original dataset
original_data['cluster'] = kmeans.labels_

# Create new attributes based on cluster assignments
original_data['cluster_mean_IC_C_P'] = original_data.groupby('cluster')['IC_C_P'].transform('mean')
original_data['cluster_mean_IC_D_H'] = original_data.groupby('cluster')['IC_D_H'].transform('mean')
original_data['cluster_mean_IC_D_P'] = original_data.groupby('cluster')['IC_D_P'].transform('mean')
original_data['cluster_mean_capacity'] = original_data.groupby('cluster')['capacity'].transform('mean')
# Save the modified dataset with new attributes
original_data.to_csv('B0018_cluster.csv', index=False)


import csv
B="B0018"
df=pd.read_csv(f"/content/new_{B}_health_index_updated.csv")
df1=pd.read_csv(f"/content/{B}_cluster.csv")
rows = zip(df['cycle'], df['capacity'],df['IC_C_H'],df['IC_C_P'], df['IC_D_H'],df['IC_D_P'],df['check_points'],df1['cluster'])
csv_file = f'{B}_health_index_updated_v2.csv'
# Write the rows to the CSV file
with open(csv_file, 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['cycle', 'capacity', 'IC_C_H', 'IC_C_P','IC_D_H','IC_D_P',"check_points","cluster"])
    # writer.writerow(['cycle', 'capacity', 'current', 'IC_D_H', 'voltage'])  # Write header
    writer.writerows(rows)

print(f"CSV file '{csv_file}' created successfully.")


import pandas as pd
import numpy as np
from scipy.stats import pearsonr
# data = {
#     'IC_C_H': HI1[:a],
#     'IC_C_P': HI2[:a],
#     'IC_D_H': HI3[:a],
#     'IC_D_P': HI4[:a],
#     'Cycles': HI5[:a],
#     'capacity': HI6[:a],
#     'voltage':voltage[:a],
#     'temperature':temperature[:a],
#     'current':current[:a]
# }

# concatenated_df = pd.DataFrame(data)

# correlation_matrix = concatenated_df.corr()
# # Get the correlation coefficients between 'capacity' and other attributes
# capacity_correlation = correlation_matrix['Cycles'].drop('Cycles')

# # Sort the correlation coefficients in descending order
# sorted_correlation = capacity_correlation.abs().sort_values(ascending=False)

# # Print the sorted correlation coefficients
# print(sorted_correlation)


data=pd.read_csv("/content/B0005_health_index_updated_v3.csv")
corr,_ = pearsonr(data["capacity"],data["IC_C_H"])
print(f"IC_C_H : {corr}")
corr,_ = pearsonr(data["capacity"],data["IC_C_P"])
print(f"IC_C_P : {corr}")
corr,_ = pearsonr(data["capacity"],data["IC_D_H"])
print(f"IC_D_H : {corr}")
corr,_ = pearsonr(data["capacity"],data["IC_D_P"])
print(f"IC_D_P : {corr}")
corr,_ = pearsonr(data["capacity"],data["distance"])
print(f"distance : {corr}")
corr,_ = pearsonr(data["capacity"],data["ratio"])
print(f"clusratioter : {corr}")
corr,_ = pearsonr(data["capacity"],data["moving_average"])
print(f"moving_average : {corr}")
corr,_ = pearsonr(data["capacity"],data["exponential_moving_average"])
print(f"exponential_moving_average : {corr}")



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# Create a random correlation matrix
df = pd.read_csv("/content/B0005_health_index_updated_v1.csv")
corr_matrix = df.corr(method='spearman')

# Create a DataFrame from the correlation matrix
df = pd.DataFrame(corr_matrix, columns=['cycle', 'capacity', 'IC_CC_H', 'IC_CC_P','IC_D_H','IC_D_P','IC_CV_H','IC_CV_P'])

# Create the heatmap
sns.heatmap(df, cmap="coolwarm", annot=True)
plt.title("Spearman Correlation Heatmap")
plt.show()



import numpy as np
import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (9,4)
df=pd.read_csv("/content/B0005_health_index_updated.csv")
df1=pd.read_csv("/content/B0006_health_index_updated.csv")
df2=pd.read_csv("/content/B0007_health_index_updated.csv")
df3=pd.read_csv("/content/B0018_health_index_updated.csv")

plt.plot(df["capacity"],color='red',label='B0005')
plt.plot(df1["capacity"],color='blue',label='B0006')
plt.plot(df2["capacity"],color='black',label='B0007')
plt.plot(df3["capacity"],color='green',label='B0018')
plt.title(' capacity degradation')
plt.xlabel('cycle')
# plt.ylim(0,2)
plt.ylabel('capacity')
plt.legend()
plt.show()



import csv
rows = zip(HI5, HI6,HI1,HI2, HI3,HI4)
csv_file = f'{B}_health_index_updated.csv'
# Write the rows to the CSV file
with open(csv_file, 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['cycle', 'capacity', 'IC_C_H', 'IC_C_P','IC_D_H','IC_D_P'])
    # writer.writerow(['cycle', 'capacity', 'current', 'IC_D_H', 'voltage'])  # Write header
    writer.writerows(rows)

print(f"CSV file '{csv_file}' created successfully.")


final_data

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.callbacks import ModelCheckpoint
from keras.models import load_model
from keras.optimizers import SGD

# Load the dataset
dataset = final_data
# Normalize the dataset
scaler = MinMaxScaler(feature_range=(0, 1))
normalized_data = scaler.fit_transform(dataset)

# Split the dataset into input (X) and output (y) variables
X = normalized_data[:, 1:9]  # Assuming health index features are in columns 1 to 6
# X = normalized_data[:, 1:5]
y = normalized_data[:, 0]   # number of cycles is in the first column

# Split the dataset into training and testing sets
df=pd.read_csv(f'{target_battery}_health_index_updated.csv')
normalized_data = scaler.fit_transform(df)
x_target = normalized_data[:, 1:9]  # Assuming health index features are in columns 1 to 6
# X = normalized_data[:, 1:5]
y_target = normalized_data[:, 0]

X_train, X_test = X,x_target
y_train, y_test = y,y_target

# Reshape the input data for LSTM [samples, time steps, features]
X_train = np.reshape(X_train, (X_train.shape[0],1,  X_train.shape[1]))
X_test = np.reshape(X_test, (X_test.shape[0],1, X_test.shape[1]))


## Assuming X_train and X_test are your input data
#X_train = np.reshape(X_train, (X_train.shape[0], 1, 10))
#X_test = np.reshape(X_test, (X_test.shape[0], 1, 10))

# Build the LSTM model
model = Sequential()
model.add(LSTM(units=60, input_shape=(1, 5)))  # Assuming 50 LSTM units
model.add(Dense(units=64,activation='relu'))
model.add(Dense(units=32,activation='relu'))
model.add(Dense(units=1))
sgd_optimizer = SGD(lr=0.1)

model.compile(loss='mean_squared_error', optimizer=sgd_optimizer)
print(model.summary())


checkpoint = ModelCheckpoint("best_model.h5", monitor='val_loss', save_best_only=True, mode='min', verbose=1)

# Train the model with the callback
history = model.fit(X_train, y_train, epochs=200, batch_size=1, validation_data=(X_test, y_test), callbacks=[checkpoint])

# Load the best model
model = load_model("best_model.h5")


# Evaluate the model
train_loss = model.evaluate(X_train, y_train, verbose=0)
print(f'Training loss: {train_loss}')

test_loss = model.evaluate(X_test, y_test, verbose=0)
print(f'Testing loss: {test_loss}')


# Make predictions
train_predictions = model.predict(X_train)
test_predictions = model.predict(X_test)

train_rul_predictions = train_predictions[:, 0]
test_rul_predictions = test_predictions[:, 0]
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
train_mae = mean_absolute_error(y_train, train_rul_predictions)
test_mae = mean_absolute_error(y_test, test_rul_predictions)

# Calculate RMSE
train_rmse = mean_squared_error(y_train, train_rul_predictions, squared=False)
test_rmse = mean_squared_error(y_test, test_rul_predictions, squared=False)

# Calculate R^2 score
train_r2 = r2_score(y_train, train_rul_predictions)
test_r2 = r2_score(y_test, test_rul_predictions)

# Print the error metrics
# print(f'Training re: {train_re}')
# print(f'Testing re: {test_re}')
print(f'Training MAE: {train_mae}')
print(f'Testing MAE: {test_mae}')
print(f'Training RMSE: {train_rmse}')
print(f'Testing RMSE: {test_rmse}')
print(f'Training R^2 score: {train_r2}')
print(f'Testing R^2 score: {test_r2}')

with open(f'LSTM_{target_battery}.txt','w') as f:
  f.write(f'Training loss: {train_loss}\n')
  f.write(f'Testing loss: {test_loss}\n')
  f.write(f'Training MAE: {train_mae}\n')
  f.write(f'Testing MAE: {test_mae}\n')
  f.write(f'Training RMSE: {train_rmse}\n')
  f.write(f'Testing RMSE: {test_rmse}\n')
  f.write(f'Training R^2 score: {train_r2}\n')
  f.write(f'Testing R^2 score: {test_r2}\n')
f.close()

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.callbacks import ModelCheckpoint
from keras.models import load_model
from keras.optimizers import SGD

# Load the dataset
dataset = pd.read_csv('/content/B0007.csv')
target_battery = "B0007"

# Normalize the dataset
scaler = MinMaxScaler(feature_range=(0, 1))
normalized_data = scaler.fit_transform(dataset)

# Split the dataset into input (X) and output (y) variables
X = normalized_data[:, 1:13]  # Assuming health index features are in columns 1 to 12
y = normalized_data[:, 0]   # Number of cycles is in the first column

# Split the dataset into training and testing sets
train_size = int(0.8 * len(dataset))
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Reshape the input data for LSTM [samples, time steps, features]
X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
X_test = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))

# Build the LSTM model
model = Sequential()
model.add(LSTM(units=60, input_shape=(1, 10)))
model.add(Dense(units=64, activation='relu'))
model.add(Dense(units=32, activation='relu'))
model.add(Dense(units=1))
sgd_optimizer = SGD(lr=0.1)

model.compile(loss='mean_squared_error', optimizer=sgd_optimizer)
print(model.summary())

checkpoint = ModelCheckpoint("best_model.h5", monitor='val_loss', save_best_only=True, mode='min', verbose=1)

# Train the model with the callback
history = model.fit(X_train, y_train, epochs=200, batch_size=1, validation_data=(X_test, y_test), callbacks=[checkpoint])

# Load the best model
model = load_model("best_model.h5")

# Evaluate the model
train_loss = model.evaluate(X_train, y_train, verbose=0)
print(f'Training loss: {train_loss}')

test_loss = model.evaluate(X_test, y_test, verbose=0)
print(f'Testing loss: {test_loss}')

# Make predictions
train_predictions = model.predict(X_train)
test_predictions = model.predict(X_test)

train_rul_predictions = train_predictions[:, 0]
test_rul_predictions = test_predictions[:, 0]

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

train_mae = mean_absolute_error(y_train, train_rul_predictions)
test_mae = mean_absolute_error(y_test, test_rul_predictions)

# Calculate RMSE
train_rmse = mean_squared_error(y_train, train_rul_predictions, squared=False)
test_rmse = mean_squared_error(y_test, test_rul_predictions, squared=False)

# Calculate R^2 score
train_r2 = r2_score(y_train, train_rul_predictions)
test_r2 = r2_score(y_test, test_rul_predictions)

# Print the error metrics
# print(f'Training re: {train_re}')
# print(f'Testing re: {test_re}')
print(f'Training MAE: {train_mae}')
print(f'Testing MAE: {test_mae}')
print(f'Training RMSE: {train_rmse}')
print(f'Testing RMSE: {test_rmse}')
print(f'Training R^2 score: {train_r2}')
print(f'Testing R^2 score: {test_r2}')

with open(f'LSTM_{target_battery}.txt','w') as f:
  f.write(f'Training loss: {train_loss}\n')
  f.write(f'Testing loss: {test_loss}\n')
  f.write(f'Training MAE: {train_mae}\n')
  f.write(f'Testing MAE: {test_mae}\n')
  f.write(f'Training RMSE: {train_rmse}\n')
  f.write(f'Testing RMSE: {test_rmse}\n')
  f.write(f'Training R^2 score: {train_r2}\n')
  f.write(f'Testing R^2 score: {test_r2}\n')
f.close()


import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import GRU, Dense
from keras.callbacks import ModelCheckpoint
from keras.models import load_model

# Load the dataset
dataset = final_data
# Normalize the dataset
scaler = MinMaxScaler(feature_range=(0, 1))
normalized_data = scaler.fit_transform(dataset)

# Split the dataset into input (X) and output (y) variables
X = normalized_data[:, 1:9]  # Assuming health index features are in columns 1 to 6
# X = normalized_data[:, 1:5]
y = normalized_data[:, 0]   # number of cycles is in the first column

# Split the dataset into training and testing sets
df=pd.read_csv(f'new_{target_battery}_health_index_updated.csv')
normalized_data = scaler.fit_transform(df)
x_target = normalized_data[:, 1:9]  # Assuming health index features are in columns 1 to 6
# X = normalized_data[:, 1:5]
y_target = normalized_data[:, 0]

X_train, X_test = X,x_target
y_train, y_test = y,y_target

# Reshape the input data for LSTM [samples, time steps, features]
X_train = np.reshape(X_train, (X_train.shape[0],1,  X_train.shape[1]))
X_test = np.reshape(X_test, (X_test.shape[0],1, X_test.shape[1]))


# Build the LSTM model
model = Sequential()
model.add(GRU(units=60, input_shape=(1,6)))  # Assuming 60 GRU units
model.add(Dense(units=64, activation='relu'))
model.add(Dense(units=32, activation='relu'))
model.add(Dense(units=1))
model.compile(loss='mean_squared_error', optimizer='adam')
print(model.summary())

# units = 60,dense_unit=64,dense_unit=32,unit=1-->output layer,epochs=90,batchsize=4
# Train the model
checkpoint = ModelCheckpoint("best_model.h5", monitor='val_loss', save_best_only=True, mode='min', verbose=1)

# Train the model with the callback
history = model.fit(X_train, y_train, epochs=200, batch_size=1, validation_data=(X_test, y_test), callbacks=[checkpoint])

# Load the best model
model = load_model("best_model.h5")
# Evaluate the model
train_loss = model.evaluate(X_train, y_train, verbose=0)
print(f'Training loss: {train_loss}')

test_loss = model.evaluate(X_test, y_test, verbose=0)
print(f'Testing loss: {test_loss}')


# Make predictions
train_predictions = model.predict(X_train)
test_predictions = model.predict(X_test)

train_rul_predictions = train_predictions[:, 0]
test_rul_predictions = test_predictions[:, 0]

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
train_mae = mean_absolute_error(y_train, train_rul_predictions)
test_mae = mean_absolute_error(y_test, test_rul_predictions)

# Calculate RMSE
train_rmse = mean_squared_error(y_train, train_rul_predictions, squared=False)
test_rmse = mean_squared_error(y_test, test_rul_predictions, squared=False)

# Calculate R^2 score
train_r2 = r2_score(y_train, train_rul_predictions)
test_r2 = r2_score(y_test, test_rul_predictions)

# Print the error metrics
# print(f'Training re: {train_re}')
# print(f'Testing re: {test_re}')
print(f'Training MAE: {train_mae}')
print(f'Testing MAE: {test_mae}')
print(f'Training RMSE: {train_rmse}')
print(f'Testing RMSE: {test_rmse}')
print(f'Training R^2 score: {train_r2}')
print(f'Testing R^2 score: {test_r2}')

with open(f'GRU_{target_battery}.txt','w') as f:
  f.write(f'Training loss: {train_loss}\n')
  f.write(f'Testing loss: {test_loss}\n')
  f.write(f'Training MAE: {train_mae}\n')
  f.write(f'Testing MAE: {test_mae}\n')
  f.write(f'Training RMSE: {train_rmse}\n')
  f.write(f'Testing RMSE: {test_rmse}\n')
  f.write(f'Training R^2 score: {train_r2}\n')
  f.write(f'Testing R^2 score: {test_r2}\n')
f.close()

import numpy as np
import pandas as pd
import keras
from keras.models import Sequential
from keras.layers import GRU, Dense, Attention
from keras.layers import Input
from keras.models import Model
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import GRU, Dense,Attention,LSTM
from keras.callbacks import ModelCheckpoint
from keras.models import load_model
from keras.optimizers import SGD

# Load the dataset
dataset = final_data
# Normalize the dataset
scaler = MinMaxScaler(feature_range=(0, 1))
normalized_data = scaler.fit_transform(dataset)

# Split the dataset into input (X) and output (y) variables
X = normalized_data[:, 1:9]  # Assuming health index features are in columns 1 to 6
# X = normalized_data[:, 1:5]
y = normalized_data[:, 0]   # number of cycles is in the first column

# Split the dataset into training and testing sets
df=pd.read_csv(f'new_{target_battery}_health_index_updated.csv')
normalized_data = scaler.fit_transform(df)
x_target = normalized_data[:, 1:9]  # Assuming health index features are in columns 1 to 6
# X = normalized_data[:, 1:5]
y_target = normalized_data[:, 0]

X_train, X_test = X,x_target
y_train, y_test = y,y_target

# Reshape the input data for LSTM [samples, time steps, features]
X_train = np.reshape(X_train, (X_train.shape[0],1,  X_train.shape[1]))
X_test = np.reshape(X_test, (X_test.shape[0],1, X_test.shape[1]))

# Build the LSTM model
attention_input = Input(shape=(1, 6))

# GRU layer
gru_output = LSTM(units=60, return_sequences=True)(attention_input)

# Attention layer
attention = Attention()([gru_output, gru_output])

# Apply attention to GRU output
attended_gru_output = keras.layers.Dot(axes=1)([attention, gru_output])

# Additional layers
dense1 = Dense(units=64, activation='relu')(attended_gru_output)
dense2 = Dense(units=32, activation='relu')(dense1)
output = Dense(units=1)(dense2)

# Create the model
model = Model(inputs=attention_input, outputs=output)

sgd_optimizer = SGD(lr=0.01)

# Compile the model with SGD optimizer
model.compile(loss='mean_squared_error', optimizer=sgd_optimizer)

# Compile the model
print(model.summary())

# units = 60,dense_unit=64,dense_unit=32,unit=1-->output layer,epochs=90,batchsize=4
# Train the model
checkpoint = ModelCheckpoint("best_model.h5", monitor='val_loss', save_best_only=True, mode='min', verbose=1)

# Train the model with the callback
history = model.fit(X_train, y_train, epochs=200, batch_size=1, validation_data=(X_test, y_test), callbacks=[checkpoint])

# Load the best model
model = load_model("best_model.h5")
# Evaluate the model
train_loss = model.evaluate(X_train, y_train, verbose=0)
print(f'Training loss: {train_loss}')

test_loss = model.evaluate(X_test, y_test, verbose=0)
print(f'Testing loss: {test_loss}')


# Make predictions
train_predictions = model.predict(X_train)
test_predictions = model.predict(X_test)

train_rul_predictions = train_predictions[:, 0]
test_rul_predictions = test_predictions[:, 0]

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
train_mae = mean_absolute_error(y_train, train_rul_predictions)
test_mae = mean_absolute_error(y_test, test_rul_predictions)

# Calculate RMSE
train_rmse = mean_squared_error(y_train, train_rul_predictions, squared=False)
test_rmse = mean_squared_error(y_test, test_rul_predictions, squared=False)

# Calculate R^2 score
train_r2 = r2_score(y_train, train_rul_predictions)
test_r2 = r2_score(y_test, test_rul_predictions)

# Print the error metrics
# print(f'Training re: {train_re}')
# print(f'Testing re: {test_re}')
print(f'Training MAE: {train_mae}')
print(f'Testing MAE: {test_mae}')
print(f'Training RMSE: {train_rmse}')
print(f'Testing RMSE: {test_rmse}')
print(f'Training R^2 score: {train_r2}')
print(f'Testing R^2 score: {test_r2}')

with open(f'LSTM_attension_{target_battery}.txt','w') as f:
  f.write(f'Training loss: {train_loss}\n')
  f.write(f'Testing loss: {test_loss}\n')
  f.write(f'Training MAE: {train_mae}\n')
  f.write(f'Testing MAE: {test_mae}\n')
  f.write(f'Training RMSE: {train_rmse}\n')
  f.write(f'Testing RMSE: {test_rmse}\n')
  f.write(f'Training R^2 score: {train_r2}\n')
  f.write(f'Testing R^2 score: {test_r2}\n')
f.close()

import numpy as np
import pandas as pd
import keras
from keras.models import Sequential
from keras.layers import GRU, Dense, Attention
from keras.layers import Input
from keras.models import Model
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import GRU, Dense,Attention,LSTM
from keras.callbacks import ModelCheckpoint
from keras.models import load_model
from keras.optimizers import SGD

# Load the dataset
dataset = final_data
# Normalize the dataset
scaler = MinMaxScaler(feature_range=(0, 1))
normalized_data = scaler.fit_transform(dataset)

# Split the dataset into input (X) and output (y) variables
X = normalized_data[:, 1:9]  # Assuming health index features are in columns 1 to 6
# X = normalized_data[:, 1:5]
y = normalized_data[:, 0]   # number of cycles is in the first column

# Split the dataset into training and testing sets
df=pd.read_csv(f'new_{target_battery}_health_index_updated.csv')
normalized_data = scaler.fit_transform(df)
x_target = normalized_data[:, 1:9]  # Assuming health index features are in columns 1 to 6
# X = normalized_data[:, 1:5]
y_target = normalized_data[:, 0]

X_train, X_test = X,x_target
y_train, y_test = y,y_target

# Reshape the input data for LSTM [samples, time steps, features]
X_train = np.reshape(X_train, (X_train.shape[0],1,  X_train.shape[1]))
X_test = np.reshape(X_test, (X_test.shape[0],1, X_test.shape[1]))

# Build the LSTM model
attention_input = Input(shape=(1, 6))

# GRU layer
gru_output = GRU(units=60, return_sequences=True)(attention_input)

# Attention layer
attention = Attention()([gru_output, gru_output])

# Apply attention to GRU output
attended_gru_output = keras.layers.Dot(axes=1)([attention, gru_output])

# Additional layers
dense1 = Dense(units=64, activation='relu')(attended_gru_output)
dense2 = Dense(units=32, activation='relu')(dense1)
output = Dense(units=1)(dense2)

# Create the model
model = Model(inputs=attention_input, outputs=output)

sgd_optimizer = SGD(lr=0.01)

# Compile the model with SGD optimizer
model.compile(loss='mean_squared_error', optimizer=sgd_optimizer)

# Compile the model
print(model.summary())

# units = 60,dense_unit=64,dense_unit=32,unit=1-->output layer,epochs=90,batchsize=4
# Train the model
checkpoint = ModelCheckpoint("best_model.h5", monitor='val_loss', save_best_only=True, mode='min', verbose=1)

# Train the model with the callback
history = model.fit(X_train, y_train, epochs=200, batch_size=1, validation_data=(X_test, y_test), callbacks=[checkpoint])

# Load the best model
model = load_model("best_model.h5")
# Evaluate the model
train_loss = model.evaluate(X_train, y_train, verbose=0)
print(f'Training loss: {train_loss}')

test_loss = model.evaluate(X_test, y_test, verbose=0)
print(f'Testing loss: {test_loss}')


# Make predictions
train_predictions = model.predict(X_train)
test_predictions = model.predict(X_test)

train_rul_predictions = train_predictions[:, 0]
test_rul_predictions = test_predictions[:, 0]

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
train_mae = mean_absolute_error(y_train, train_rul_predictions)
test_mae = mean_absolute_error(y_test, test_rul_predictions)

# Calculate RMSE
train_rmse = mean_squared_error(y_train, train_rul_predictions, squared=False)
test_rmse = mean_squared_error(y_test, test_rul_predictions, squared=False)

# Calculate R^2 score
train_r2 = r2_score(y_train, train_rul_predictions)
test_r2 = r2_score(y_test, test_rul_predictions)

# Print the error metrics
# print(f'Training re: {train_re}')
# print(f'Testing re: {test_re}')
print(f'Training MAE: {train_mae}')
print(f'Testing MAE: {test_mae}')
print(f'Training RMSE: {train_rmse}')
print(f'Testing RMSE: {test_rmse}')
print(f'Training R^2 score: {train_r2}')
print(f'Testing R^2 score: {test_r2}')

with open(f'GRU_attension_{target_battery}.txt','w') as f:
  f.write(f'Training loss: {train_loss}\n')
  f.write(f'Testing loss: {test_loss}\n')
  f.write(f'Training MAE: {train_mae}\n')
  f.write(f'Testing MAE: {test_mae}\n')
  f.write(f'Training RMSE: {train_rmse}\n')
  f.write(f'Testing RMSE: {test_rmse}\n')
  f.write(f'Training R^2 score: {train_r2}\n')
  f.write(f'Testing R^2 score: {test_r2}\n')
f.close()

import csv
B="B0018"
import pandas as pd
df=pd.read_csv(f"/content/{B}_health_index_updated_d1.csv")
df1=pd.read_csv(f"/content/{B}_health_index_updated_v2.csv")
rows = zip(df['cycle'], df['capacity'],df['IC_CV_H'],df['IC_C_P'], df['IC_D_H'],df['IC_D_P'],df1['cluster'],df1['check_points'])
csv_file = f'{B}_health_index_updated_d2.csv'
# Write the rows to the CSV file
with open(csv_file, 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['cycle', 'capacity', 'IC_CV_H', 'IC_CV_D','IC_D_H','IC_D_P','cluster','check_points'])
    # writer.writerow(['cycle', 'capacity', 'current', 'IC_D_H', 'voltage'])  # Write header
    writer.writerows(rows)

print(f"CSV file '{csv_file}' created successfully.")

target_battery="B0018"
model="d2"

import pandas as pd
final_data=pd.DataFrame()
three_battries=input()
print("target battery")
target_battery=input()
for i in three_battries.split(","):
  final_data = pd.concat([final_data, pd.read_csv(f'{i}_health_index_updated_d1.csv')], ignore_index=True)

final_data.to_csv("final.csv")

# !pip install autokeras
import autokeras as ak
import numpy as np
import pandas as pd
import keras
from keras.models import Sequential
from keras.layers import GRU, Dense, Attention
from keras.layers import Input
from keras.models import Model
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import GRU, Dense,Attention,LSTM
from keras.callbacks import ModelCheckpoint
from keras.models import load_model

# Load the training and validation datasets
# Load the dataset
dataset = final_data
# # Normalize the dataset
scaler = MinMaxScaler(feature_range=(0, 1))
normalized_data = scaler.fit_transform(dataset)

# Split the dataset into input (X) and output (y) variables
X = normalized_data[:, 1:9]  # Assuming health index features are in columns 1 to 6
# X = normalized_data[:, 1:5]
y = normalized_data[:, 0]   # number of cycles is in the first column

# # Split the dataset into training and testing sets
df=pd.read_csv(f'{target_battery}_health_index_updated_d2.csv')
normalized_data = scaler.fit_transform(df)
x_target = normalized_data[:, 1:9]  # Assuming health index features are in columns 1 to 6
# X = normalized_data[:, 1:5]
y_target = normalized_data[:, 0]

X_train, X_test = X,x_target
y_train, y_test = y,y_target


# Reshape the input data for LSTM [samples, time steps, features]
# X_train = np.reshape(X_train, (X_train.shape[0],1,  X_train.shape[1]))
# X_test = np.reshape(X_test, (X_test.shape[0],1, X_test.shape[1]))


# Create an AutoKeras regression model

clf=ak.StructuredDataRegressor(

    loss="mean_squared_error",
    metrics=None,
    project_name="structured_data_regressor",
    max_trials=10,
    objective="val_loss"
)


# Train the model on the training dataset
clf.fit( x=X_train, y=y_train, epochs=200, callbacks=None, validation_split=0.1, validation_data=(X_test, y_test),batch_size=1)




# Make predictions on the validation dataset
train_loss = clf.evaluate(X_train, y_train, verbose=0)
print(f'Training loss: {train_loss}')

test_loss = clf.evaluate(X_test, y_test, verbose=0)
print(f'Testing loss: {test_loss}')


train_predictions = clf.predict(X_train)
test_predictions =clf.predict(X_test)
train_rul_predictions = train_predictions[:, 0]
test_rul_predictions = test_predictions[:, 0]

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
train_mae = mean_absolute_error(y_train, train_rul_predictions)
test_mae = mean_absolute_error(y_test, test_rul_predictions)

# Calculate RMSE
train_rmse = mean_squared_error(y_train, train_rul_predictions, squared=False)
test_rmse = mean_squared_error(y_test, test_rul_predictions, squared=False)

# Calculate R^2 score
train_r2 = r2_score(y_train, train_rul_predictions)
test_r2 = r2_score(y_test, test_rul_predictions)

# Print the error metrics
# print(f'Training re: {train_re}')
# print(f'Testing re: {test_re}')
print(f'Training MAE: {train_mae}')
print(f'Testing MAE: {test_mae}')
print(f'Training RMSE: {train_rmse}')
print(f'Testing RMSE: {test_rmse}')
print(f'Training R^2 score: {train_r2}')
print(f'Testing R^2 score: {test_r2}')

with open(f'autokeras_{target_battery}.txt','w') as f:
  f.write(f'Training loss: {train_loss}\n')
  f.write(f'Testing loss: {test_loss}\n')
  f.write(f'Training MAE: {train_mae}\n')
  f.write(f'Testing MAE: {test_mae}\n')
  f.write(f'Training RMSE: {train_rmse}\n')
  f.write(f'Testing RMSE: {test_rmse}\n')
  f.write(f'Training R^2 score: {train_r2}\n')
  f.write(f'Testing R^2 score: {test_r2}\n')
f.close()


import matplotlib.pyplot as plt

# Assuming 'df' contains your original dataset and 'y_pred' contains the predicted values

# Extract the 'trade_date' column as the time axis

# Create a new figure
plt.figure(figsize=(12, 6))

  # Set the x-axis limits from 1 to 5
  # Set the y-axis limits from 10 to 30

# Plot the actual "close" values
plt.plot( y_test, label='Actual', color='blue', linewidth=2)

# print(time_axis)
# print(y_pred)
# Plot the predicted "close" values
plt.plot(test_rul_predictions, label='Predicted', color='red', linestyle='--',linewidth=2)

# Set axis labels and a title
plt.xlabel('cycle')
plt.ylabel('Capacity')
plt.title('Actual vs. Predicted Capacity Over Time')

# Add a legend
plt.legend()

# Rotate x-axis labels for better readability (optional)
plt.xticks(rotation=45)

# Display the plot
plt.grid()
plt.tight_layout()
plt.show()

# !pip install tpot

import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Replace 'your_dataset.csv' with the actual file name
data = final_data
# # Normalize the dataset
scaler = MinMaxScaler(feature_range=(0, 1))
normalized_data = scaler.fit_transform(data)

# Split the dataset into input (X) and output (y) variables
X = normalized_data[:, 1:9]  # Assuming health index features are in columns 1 to 6
# X = normalized_data[:, 1:5]
y = normalized_data[:, 0]   # number of cycles is in the first column

df=pd.read_csv(f'{target_battery}_health_index_updated_d1.csv')
normalized_data = scaler.fit_transform(df)
x_target = normalized_data[:, 1:9]  # Assuming health index features are in columns 1 to 6
# X = normalized_data[:, 1:5]
y_target = normalized_data[:, 0]

X_train, X_test = X,x_target
y_train, y_test = y,y_target

# Assuming 'capacity' is your target variable, and the rest are features
target = data['capacity']
features = data.drop(columns=['capacity'])
from tpot import TPOTRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.metrics import r2_score
import numpy as np


# Create and configure the TPOTRegressor
tpot = TPOTRegressor(generations=3, population_size=20, random_state=42, verbosity=3)

# Fit TPOT to the training data
tpot.fit(X_train, y_train)

# Make predictions on the test data
y_pred = tpot.predict(X_test)

# Calculate Mean Absolute Error (MAE)
mae = mean_absolute_error(y_test, y_pred)

# Calculate Mean Squared Error (MSE) and then RMSE
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

# Evaluate the best model on the test data
r2 = tpot.score(X_test, y_test)

r2_scr = r2_score(y_test, y_pred)

print("R-squared score:", r2)
print("Mean Absolute Error (MAE):", mae)
print("Root Mean Squared Error (RMSE):", rmse)
print("R2 score:",r2_scr)


with open(f'tpot_{target_battery}.txt','w') as f:
  f.write(f'Testing MAE: {mae}\n')
  f.write(f'Testing RMSE: {rmse}\n')
  f.write(f'Testing R^2 score: {r2}\n')
f.close()



import matplotlib.pyplot as plt

# Assuming 'df' contains your original dataset and 'y_pred' contains the predicted values

# Extract the 'trade_date' column as the time axis

# Create a new figure
plt.figure(figsize=(12, 6))

  # Set the x-axis limits from 1 to 5
  # Set the y-axis limits from 10 to 30

# Plot the actual "close" values
plt.plot( y_test, label='Actual', color='blue', linewidth=2)

# print(time_axis)
# print(y_pred)
# Plot the predicted "close" values
plt.plot(y_pred, label='Predicted', color='red', linestyle='--',linewidth=2)

# Set axis labels and a title
plt.xlabel('cycle')
plt.ylabel('Capacity')
plt.title('Actual vs. Predicted Capacity Over Time')

# Add a legend
plt.legend()

# Rotate x-axis labels for better readability (optional)
plt.xticks(rotation=45)

# Display the plot
plt.grid()
plt.tight_layout()
plt.show()

# !pip install h2o

import h2o
from h2o.automl import H2OAutoML
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.metrics import r2_score

# Initialize H2O cluster

h2o.init()
# Load your dataset
data =h2o.import_file(f"final.csv")
data1 = h2o.import_file(f"/content/{target_battery}_health_index_updated_d2.csv")
target = "capacity"
target1 = "capacity"


# Split the data into training and testing sets
train, test = data.split_frame(ratios=[0.9])

# Define the predictor columns (features)
predictors = [col for col in data.columns if col != target]

predictors1 = [col for col in data1.columns if col != target1]


a = len(data)
X_train, X_test=predictors,predictors1
y_train, y_test=target,target1

# Train the AutoML model
aml = H2OAutoML(max_runtime_secs=60)  # You can adjust the maximum runtime
aml.train(x=X_train, y=y_train, training_frame=train)

# Get the leader model
leader = aml.leader

# Make predictions on the test data
y_predictions = leader.predict(test)

# Evaluate model performance
performance = leader.model_performance(test)

# Calculate the requested metrics
rmse = performance.rmse()
mae = performance.mae()
mse = performance.mse()
r2 = performance.r2()

# Print the performance metrics
print(f"Root Mean Squared Error (RMSE): {rmse}")
print(f"Mean Absolute Error (MAE): {mae}")
print(f"Mean Squared Error (MSE): {mse}")
print(f"R-squared (R^2): {r2}")

with open(f'H20_{target_battery}.txt','w') as f:
  f.write(f'Testing mse: {mse}\n')
  f.write(f'Testing MAE: {mae}\n')
  f.write(f'Testing RMSE: {rmse}\n')
  f.write(f'Testing R^2 score: {r2}\n')
f.close()


# Shutdown H2O cluster
h2o.cluster().shutdown()

import matplotlib.pyplot as plt


# Create a new figure
plt.figure(figsize=(12, 6))
# Plot the actual "close" values
plt.plot( y_test, label='Actual', color='blue', linewidth=2)

# print(time_axis)
# print(y_pred)
# Plot the predicted "close" values
plt.plot(y_predictions, label='Predicted', color='red', linestyle='--',linewidth=2)

# Set axis labels and a title
plt.xlabel('cycle')
plt.ylabel('Capacity')
plt.title('Actual vs. Predicted Capacity Over Time')

# Add a legend
plt.legend()

# Rotate x-axis labels for better readability (optional)
plt.xticks(rotation=45)

# Display the plot
plt.grid()
plt.tight_layout()
plt.show()

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
train_mae = mean_absolute_error(y_train, train_rul_predictions)
test_mae = mean_absolute_error(y_test, test_rul_predictions)

# Calculate RMSE
train_rmse = mean_squared_error(y_train, train_rul_predictions, squared=False)
test_rmse = mean_squared_error(y_test, test_rul_predictions, squared=False)

# Calculate R^2 score
train_r2 = r2_score(y_train, train_rul_predictions)
test_r2 = r2_score(y_test, test_rul_predictions)

# Print the error metrics
# print(f'Training re: {train_re}')
# print(f'Testing re: {test_re}')
print(f'Training MAE: {train_mae}')
print(f'Testing MAE: {test_mae}')
print(f'Training RMSE: {train_rmse}')
print(f'Testing RMSE: {test_rmse}')
print(f'Training R^2 score: {train_r2}')
print(f'Testing R^2 score: {test_r2}')

with open(f'{target_battery}.txt','w') as f:
  f.write(f'Training loss: {train_loss}\n')
  f.write(f'Testing loss: {test_loss}\n')
  f.write(f'Training MAE: {train_mae}\n')
  f.write(f'Testing MAE: {test_mae}\n')
  f.write(f'Training RMSE: {train_rmse}\n')
  f.write(f'Testing RMSE: {test_rmse}\n')
  f.write(f'Training R^2 score: {train_r2}\n')
  f.write(f'Testing R^2 score: {test_r2}\n')
f.close()


import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (8,4)
plt.figure()
plt.ylabel('loss'); plt.xlabel('epoch')
plt.semilogy(history.history['loss'])

# Denormalizing number of cycle leads to inconsistency, so plot on this value
#Visualization
test_capacity=list(dataset['capacity'])
plt.rcParams["figure.figsize"] = (9,4)
plt.plot(test_rul_predictions,color='red',label='Predicted RUL')
plt.plot(y_test,color='blue',label='Real RUL')
plt.plot(y_test-test_rul_predictions,color='green',label='Prediction error RUL')
plt.title(' RUL Prediction')
# plt.xlabel('capacity')
# plt.ylim(0,2)
plt.ylabel('RUL')
plt.legend()
plt.show()



# !pip install ruptures

import ruptures as rpt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def detect_change_points(time_series1, time_series2, model="rbf"):

    # Convert the input data to NumPy arrays
    time_series1 = np.array(time_series1)
    time_series2 = np.array(time_series2)
    combined_series = np.concatenate((time_series1, time_series2))

    # Create a change point detection model with subsegmentation
    algo = rpt.Pelt(model=model, min_size=5).fit(combined_series)

    # Detect change points
    change_points = algo.predict(pen=2)  # You can adjust the 'pen' parameter for precision

    # Separate the change points based on the lengths of the input time series
    change_points = [cp for cp in change_points if cp <= len(time_series1)]

    return change_points

# Example usage:
df= pd.read_csv("/content/B0005_health_index_updated.csv")
time_series1 = df["cycle"]
time_series2 = df["capacity"]

change_points = detect_change_points(time_series1, time_series2)

# Create a plot to visualize the change points
plt.figure(figsize=(10, 4))
plt.plot(time_series1, time_series2, marker='o')

# Highlight the regions with change points
for cp in change_points:
    plt.axvline(cp, color='red', linestyle='--', linewidth=1)

plt.legend()
plt.title("Time Series with Change Point Detection")
plt.xlabel("Time")
plt.ylabel("Value")
plt.show()


import pandas as pd

def detect_abrupt_changes(signal, percent_change, max_time_interval):
    """
    Detect abrupt changes in a time series signal based on the specified parameters.

    Parameters:
        signal (pandas.Series): One-dimensional time series signal.
        percent_change (float): The minimum percentage change in the signal range that is considered an abrupt change.
        max_time_interval (int): The maximum time interval in samples between two data points to be considered part of the same change.

    Returns:
        pandas.Series: A boolean mask indicating the positions of the abrupt changes in the input signal.
    """
    # Calculate the absolute change in the signal.
    abs_change = abs(signal.diff())

    # Calculate the threshold for the minimum change required to be considered an abrupt change.
    change_threshold = (signal.max() - signal.min()) * percent_change

    # Initialize a mask of False values to indicate no abrupt changes have been detected yet.
    abrupt_changes = pd.Series(False, index=signal.index)

    # Loop over each data point in the signal.
    for i in range(1, len(signal)):
        # If the absolute change is greater than the threshold, mark this data point as the start of an abrupt change.
        if abs_change[i] >= change_threshold:
            abrupt_changes[i] = True

            # Keep track of the end of the current change.
            end_of_change = i

            # Continue checking subsequent data points to see if they are still part of the same change.
            for j in range(i+1, min(i+max_time_interval, len(signal))):
                if abs_change[j] >= change_threshold:
                    # If the change is still above the threshold, mark this data point as part of the same change.
                    abrupt_changes[j] = True

                    # Update the end of the current change.
                    end_of_change = j
                else:
                    # If the change has fallen below the threshold, stop checking subsequent data points.
                    break

            # Skip checking data points that are already part of the current change.
            i = end_of_change

    return abrupt_changes



import matplotlib.pyplot as plt
df= pd.read_csv("/content/B0018_health_index_updated.csv")
signal = df["capacity"]

# Plot the time series
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(signal, label='signal')

# Plot the abrupt changes
abrupt_changes = detect_abrupt_changes(signal, 0.04, 10)
abrupt_changes = abrupt_changes.astype(int)
ax.plot(signal[abrupt_changes==1], 'ro', label='abrupt changes')

# Add labels, grid, and legend
ax.set_xlabel('Time')
ax.set_ylabel('Signal')
ax.set_title('Time Series with Abrupt Changes')
ax.legend()
ax.grid(True)

# Show the plot
plt.show()

change_points=[]
for i in range(len(abrupt_changes)):
  if abrupt_changes[i]==1:
    change_points.append(i)

average=[]
distance=[]
for i in range(len(signal)):
  nearest=0
  k=63456
  for j in change_points:
    if k>=abs(j-i):
      k=abs(j-i)
      nearest=j
  if k==0:
    average.append(0)
    distance.append(0)
  else:
    distan=abs(signal[i]-signal[j])
    ratio=abs(signal[i]-signal[j])/k
    average.append(ratio)
    distance.append(distan)

print(average)


ma=[]
for i in range(len(signal)-2):
  a=(signal[i]+signal[i+1]+signal[i+2])/3
  ma.append(a)

a=(signal[len(signal)-2]+signal[len(signal)-1])/3
ma.append(a)
a=signal[len(signal)-1]/3
ma.append(a)
print(ma)


def calculate_ema(data, n):
    ema = [data[0]]  # Initial EMA is the first data point
    smoothing_factor = 2 / (n + 1)

    for i in range(1, len(data)):
        ema_today = (data[i] * smoothing_factor) + (ema[i - 1] * (1 - smoothing_factor))
        ema.append(ema_today)

    return ema
ema=calculate_ema(signal,10)
print(ema)

import pandas as pd

def detect_abrupt_changes(signal, percent_change,max_num):

    # Calculate the threshold for the minimum change required to be considered an abrupt change.
    change_threshold = (signal.max() - signal.min()) * percent_change

    # Initialize a mask of False values to indicate no abrupt changes have been detected yet.
    abrupt_changes = pd.Series(False, index=signal.index)

    # Loop over each data point in the signal.
    for i in range(1, len(signal)):
      # if abs(signal[i]-signal[i-1])>=change_threshold:
      #   abrupt_changes[i]=True
      if i > max_num and (i+max_num)<(len(signal)-1):
        max_id=0
        min_id=685465
        for j in range(i-max_num,i+max_num):
          max_id = max(max_id,signal[j])
          min_id = min(min_id,signal[j])
        if signal[i]>=max_id:
          abrupt_changes[i] = True
        if signal[i]<=min_id:
          abrupt_changes[i] = True

    return abrupt_changes



import matplotlib.pyplot as plt
df= pd.read_csv("/content/B0018_health_index_updated.csv")
signal = df["capacity"]

# Plot the time series
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(signal, label='signal')

# Plot the abrupt changes
abrupt_changes = detect_abrupt_changes(signal, 0.035,3)
abrupt_changes = abrupt_changes.astype(int)
ax.plot(signal[abrupt_changes==1], 'ro', label='abrupt changes')

# Add labels, grid, and legend
ax.set_xlabel('Time')
ax.set_ylabel('Signal')
ax.set_title('Time Series with Abrupt Changes')
ax.legend()
ax.grid(True)

# Show the plot
plt.show()

import csv
rows = zip(df["cycle"], df["capacity"],df["IC_C_H"],df["IC_C_P"], df["IC_D_H"],df["IC_D_P"],distance,average,ma,ema)
csv_file = f'B0018_health_index_updated_v3.csv'
# Write the rows to the CSV file
with open(csv_file, 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['cycle', 'capacity', 'IC_C_H', 'IC_C_P','IC_D_H','IC_D_P','distance','ratio','moving_average','exponential_moving_average'])
    # writer.writerow(['cycle', 'capacity', 'current', 'IC_D_H', 'voltage'])  # Write header
    writer.writerows(rows)

print(f"CSV file '{csv_file}' created successfully.")


import numpy as np
import pandas as pd

def z_score(data):
    mean = data.mean()
    std = data.std()
    return (data - mean) / std


if __name__ == "__main__":
    # Load the data using Pandas
    df = pd.read_csv("/content/B0005_health_index_updated.csv")
    data = df["capacity"]
    z_scores = z_score(data)
    print(z_scores)


import numpy as np

df = pd.read_csv("/content/B0006_health_index_updated_d1.csv")
data = df["capacity"]


fig, ax = plt.subplots(1, figsize=(6, 4))
ax.plot(df['cycle'],df['capacity'], 'r-')
ax.set(xlabel='cycles', ylabel='capacity ')
plt.legend()
fig, ax = plt.subplots(1, figsize=(6, 4))
ax.plot(df['cycle'],df['IC_CV_H'], 'r-')
ax.set(xlabel='cycles', ylabel='IC_CV_H ')
plt.legend()
fig, ax = plt.subplots(1, figsize=(6, 4))
ax.plot(df['cycle'],df['IC_C_P'], 'r-')
ax.set(xlabel='cycles', ylabel='IC_C_P ')
plt.legend()
fig, ax = plt.subplots(1, figsize=(6, 4))
ax.plot(df['cycle'],df['IC_D_H'], 'r-')
ax.set(xlabel='cycles', ylabel='IC_D_H ')
plt.legend()
fig, ax = plt.subplots(1, figsize=(6, 4))
ax.plot(df['cycle'],df['IC_D_P'], 'r-')
ax.set(xlabel='cycles', ylabel='IC_D_P ')
plt.legend()


import matplotlib.pyplot as plt
import numpy as np
from scipy.cluster.vq import kmeans2
import pandas as pd

def kmeans_clustering(data, k):
  """Performs K-means clustering on the given data.

  Args:
    data: A numpy array of data with dimensions [n_samples, n_features].
    k: The number of clusters.

  Returns:
    A list of labels, where each label is the cluster index of the corresponding data point.
  """

  # Reshape the data to have two dimensions.

  # Convert the data to type float.
  data = data.astype(np.float32)

  # Perform K-means clustering.
  centroids, labels = kmeans2(data, k)

  return centroids,labels

def plot_clusters(data, labels, centroids):
  """Plots the data points and cluster centroids."""

  # Create a scatter plot of the data points.
  plt.scatter(data[:, 0], data[:, 1])

  # Plot the cluster centroids.
  for centroid in centroids:
    plt.plot(centroid[0], centroid[1], 'o', markersize=10, markerfacecolor='red')

  # Show the plot.
  plt.show()

if __name__ == '__main__':
  # Read the data from the CSV file.
  data = pd.read_csv("/content/B0005_health_index_updated.csv")
  data = data.to_numpy()

  # Perform K-means clustering with k=2.
  centroids,labels = kmeans_clustering(data, k=3)

  # Plot the clusters.
  plot_clusters(data.reshape(-1,2), labels, centroids)


# !pip install autots

import pandas as pd
final_data=pd.DataFrame()
three_battries=input()
print("target battery")
target_battery=input()
for i in three_battries.split(","):
  final_data = pd.concat([final_data, pd.read_csv(f'{i}_health_index_updated_d2.csv')], ignore_index=True)

from autots import AutoTS, load_daily
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# sample datasets can be used in either of the long or wide import shapes
data = final_data

a=len(data)
train_data=data[:int(0.9*a)]
validation_data=data[int(0.9*a):]
b=len(validation_data)
model = AutoTS(
    forecast_length=b,
    frequency='infer',
    prediction_interval=0.9,
    ensemble='auto',
    model_list="default",
    transformer_list="default",
    drop_most_recent=1,
    max_generations=3,
    num_validations=2,
    validation_method="backwards"
)
model = model.fit(
    train_data,
    date_col='cycle' ,
    value_col='capacity',
)

prediction = model.predict()


y_true = data['capacity'].values[-len(prediction.forecast):]  # True values
y_pred = prediction.forecast.values  # Predicted values
rmse = np.sqrt(mean_squared_error(y_true, y_pred))
print(f"RMSE: {rmse}")

r2 = r2_score(y_true, y_pred)

print(f"r2_score: {r2}")

# Print the details of the best model
print(model)

# point forecasts dataframe
forecasts_df = prediction.forecast
# upper and lower forecasts
forecasts_up, forecasts_low = prediction.upper_forecast, prediction.lower_forecast

# accuracy of all tried model results
model_results = model.results()
# and aggregated from cross validation
validation_results = model.results("validation")

import matplotlib.pyplot as plt

# Assuming 'df' contains your original dataset and 'y_pred' contains the predicted values

# Extract the 'trade_date' column as the time axis
time_axis = validation_data['cycle']
pred=prediction.forecast
# Create a new figure
plt.figure(figsize=(12, 6))

  # Set the x-axis limits from 1 to 5
  # Set the y-axis limits from 10 to 30

# Plot the actual "close" values
plt.plot(time_axis, validation_data['capacity'], label='Actual', color='blue', linewidth=2)

# print(time_axis)
# print(y_pred)
# Plot the predicted "close" values
plt.plot(time_axis, pred, label='Predicted', color='red', linestyle='--',linewidth=2)

# Set axis labels and a title
plt.xlabel('cycle')
plt.ylabel('capacity')
plt.title('Actual vs. Predicted  capacity Over Time')

# Add a legend
plt.legend()

# Rotate x-axis labels for better readability (optional)
plt.xticks(rotation=45)

# Display the plot
plt.grid()
plt.tight_layout()
plt.show()

# !pip list


import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Load your dataset into a DataFrame
data = pd.read_csv('/content/B0018_health_index_updated.csv')  # Replace 'your_dataset.csv' with the path to your dataset

# Select the features you want to use for clustering
X = data[['cycle', 'capacity', 'IC_C_H', 'IC_C_P', 'IC_D_H', 'IC_D_P']]

# Initialize and fit the K-Means model with k=3
kmeans = KMeans(n_clusters=3, random_state=0)
kmeans.fit(X)

# Add the cluster labels to your dataset
data['cluster'] = kmeans.labels_

# Create a scatter plot of the data points, color-coded by cluster
plt.scatter(X['cycle'], X['capacity'], c=data['cluster'], cmap='viridis')
plt.xlabel('Cycle')
plt.ylabel('Capacity')
plt.title('K-Means Clustering (k=3)')

# Show the plot
plt.show()

import numpy as np

def find_change_points(data, threshold):
    """
    Detect change points in a list using Cumulative Sum (CUSUM) method.

    Args:
    data (list): Input data as a list.
    threshold (float): Threshold for change point detection.

    Returns:
    change_points (list): List of detected change points.
    """
    data = np.array(data)
    mean = np.mean(data)
    cumulative_sum = np.cumsum(data - mean)
    print(cumulative_sum)
    change_points = []
    current_cp = None
    for i, c in enumerate(cumulative_sum):
        if c > threshold:
            if current_cp is None:
                print(i)
                current_cp = i
        elif current_cp is not None:
            change_points.append((current_cp, i - 1))
            current_cp = None

    if current_cp is not None:
        change_points.append((current_cp, len(data) - 1))

    return change_points

# Example usage:

df= pd.read_csv("/content/B0005_health_index_updated.csv")
input_data = df["capacity"]
threshold_value = 3  # Adjust this threshold as needed.

change_points = find_change_points(input_data, threshold_value)

for start, end in change_points:
    print(f"Change detected from index {start} to {end}")

# You can customize the threshold_value and the input_data list as needed.


import numpy as np
from scipy.cluster.vq import kmeans
import pandas as pd
from sklearn.preprocessing import MinMaxScaler



def kmeans_clustering(data, k):

  # Perform K-means clustering.
  centroids, labels = kmeans(data, k)

  return labels

if __name__ == '__main__':
  # Get the input data.

  dataset = pd.read_csv("/content/B0005_health_index_updated.csv")
  # Normalize the dataset
  scaler = MinMaxScaler(feature_range=(0, 1))
  normalized_data = scaler.fit_transform(dataset)


  data = normalized_data

  # Perform K-means clustering with k=2.
  labels = kmeans_clustering(data, k=3)

  # Print the cluster labels.
  print(labels)


from sklearn.preprocessing import MinMaxScaler
dataset = pd.read_csv("/content/B0005_health_index_updated.csv")
scaler = MinMaxScaler(feature_range=(0, 1))
normalized_data = scaler.fit_transform(dataset)
print(normalized_data.shape)

import pandas as pd

# Sample DataFrame (replace this with your actual DataFrame)
data = {
    'ts_code': [601988, 601988, 601988, 601988, 601988, 601988, 601988, 601988],
    'trade_date': [20070104, 20070105, 20070108, 20070109, 20070110, 20070111, 20070112, 20070115],
    'open': [5.69, 5.3, 4.87, 5.06, 5.25, 5.07, 4.88, 4.71],
    'high': [5.97, 5.34, 5.14, 5.19, 5.29, 5.07, 4.97, 5.0],
    'low': [5.37, 5.07, 4.83, 4.95, 5.05, 4.9, 4.7, 4.65],
    'close': [5.63, 5.07, 5.08, 5.18, 5.1, 4.93, 4.73, 4.99]
}

df = pd.DataFrame(data)

# Convert trade_date to datetime format
df['trade_date'] = pd.to_datetime(df['trade_date'], format='%Y%m%d')

# Print the DataFrame with the updated trade_date format
print(df)


import os
os.mkdir("./result")
os.mkdir("./model")


import zipfile
import os

def add_folder_to_zip(zip_file, folder_path):
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            file_path = os.path.join(root, file)
            zip_file.write(file_path, os.path.relpath(file_path, folder_path))

parent_folder = "/content/result"  # Path to the parent folder

with zipfile.ZipFile("result.zip", "w") as zip_file:
    add_folder_to_zip(zip_file, parent_folder)


