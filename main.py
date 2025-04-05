import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import h5py
import random
from sklearn.model_selection import train_test_split

# This file creates the HDF5 file required for the assignment. It also prints the graphs with the moving average

# Load all the CSV's
# Jumping data
matt_jump_BP = pd.read_csv("Matt/Jumping_Back_Pocket.csv")
matt_jump_FP = pd.read_csv("Matt/Jumping_Front_Pocket.csv")
matt_jump_H = pd.read_csv("Matt/Jumping_In_Hand.csv")
matt_jump_JP = pd.read_csv("Matt/Jumping_Jacket_Pocket.csv")

ben_jump_H = pd.read_csv("Ben/Jumping Hand.csv")
ben_jump_JP = pd.read_csv("Ben/Jumping Jacket.csv")
ben_jump_FP = pd.read_csv("Ben/Jumping Pocket.csv")

guntas_jump_H = pd.read_csv("Guntas/jumpingphoneinhand.csv")
guntas_jump_FP = pd.read_csv("Guntas/jumpingphoneleftpocket.csv")
guntas_jump_BP = pd.read_csv("Guntas/jumpingphonebackpocket.csv")

# Walking data
matt_walk_H = pd.read_csv("Matt/Walking_In_Hand.csv")
matt_walk_FP = pd.read_csv("Matt/Walking_Front_Pocket.csv")
matt_walk_BP = pd.read_csv("Matt/Walking_Back_Pocket.csv")

ben_walk_H = pd.read_csv("Ben/Walking Hand.csv")
ben_walk_JP = pd.read_csv("Ben/Walking Jacket.csv")
ben_walk_FP = pd.read_csv("Ben/Walking Pocket.csv")

guntas_walk_H = pd.read_csv("Guntas/Walking.csv")



#Plots for visualization

#Plotting Matt's jumping data

#Extracting data columns for each data column
time = matt_jump_FP.iloc[:, 0]
matt_jump_FP_x_acc = matt_jump_FP.iloc[:, 1]
matt_jump_FP_y_acc = matt_jump_FP.iloc[:, 2]
matt_jump_FP_z_acc = matt_jump_FP.iloc[:, 3]
matt_jump_FP_total_acc = matt_jump_FP.iloc[:, 4]

#Creating subplot
matt_jump_FP_fig, matt_jump_FP_axes = plt.subplots(ncols=1, nrows=4, figsize=(10, 10))

#Plot X acceleration
matt_jump_FP_axes[0].plot(time, matt_jump_FP_x_acc, label='X Acceleration', color='r')
matt_jump_FP_axes[0].set_title("Matt Jumping FP - Acceleration X Component", fontsize=20)
matt_jump_FP_axes[0].set_ylabel("Acceleration [m/s^2]", fontsize=15)
matt_jump_FP_axes[0].legend()

#Plot Y acceleration
matt_jump_FP_axes[1].plot(time, matt_jump_FP_y_acc, label='Y Acceleration', color='g')
matt_jump_FP_axes[1].set_title("Matt Jumping FP - Acceleration Y Component", fontsize=20)
matt_jump_FP_axes[1].set_ylabel("Acceleration [m/s^2]", fontsize=15)
matt_jump_FP_axes[1].legend()

#Plot Z acceleration
matt_jump_FP_axes[2].plot(time, matt_jump_FP_z_acc, label='Z Acceleration', color='b')
matt_jump_FP_axes[2].set_title("Matt Jumping FP - Acceleration Z Component", fontsize=20)
matt_jump_FP_axes[2].set_ylabel("Acceleration [m/s^2]", fontsize=15)
matt_jump_FP_axes[2].legend()

#Plot Total acceleration
matt_jump_FP_axes[3].plot(time, matt_jump_FP_total_acc, label='Total Acceleration', color='k')
matt_jump_FP_axes[3].set_title("Matt Jumping FP - Total Acceleration", fontsize=20)
matt_jump_FP_axes[3].set_xlabel("Time [s]", fontsize=15)
matt_jump_FP_axes[3].set_ylabel("Acceleration [m/s^2]", fontsize=15)
matt_jump_FP_axes[3].legend()

plt.tight_layout()

#Plotting Ben's jumping data

#Extracting data columns for each data column
time = ben_jump_H.iloc[:, 0]
ben_jump_H_x_acc = ben_jump_H.iloc[:, 1]
ben_jump_H_y_acc = ben_jump_H.iloc[:, 2]
ben_jump_H_z_acc = ben_jump_H.iloc[:, 3]
ben_jump_H_total_acc = ben_jump_H.iloc[:, 4]

#Creating subplot
ben_jump_H_fig, ben_jump_H_axes = plt.subplots(ncols=1, nrows=4, figsize=(10, 10))

#Plot X acceleration
ben_jump_H_axes[0].plot(time, ben_jump_H_x_acc, label='X Acceleration', color='r')
ben_jump_H_axes[0].set_title("Ben Jumping H - Acceleration X Component", fontsize=20)
ben_jump_H_axes[0].set_ylabel("Acceleration [m/s^2]", fontsize=15)
ben_jump_H_axes[0].legend()

#Plot Y acceleration
ben_jump_H_axes[1].plot(time, ben_jump_H_y_acc, label='Y Acceleration', color='g')
ben_jump_H_axes[1].set_title("Ben Jumping H - Acceleration Y Component", fontsize=20)
ben_jump_H_axes[1].set_ylabel("Acceleration [m/s^2]", fontsize=15)
ben_jump_H_axes[1].legend()

#Plot Z acceleration
ben_jump_H_axes[2].plot(time, ben_jump_H_z_acc, label='Z Acceleration', color='b')
ben_jump_H_axes[2].set_title("Ben Jumping H - Acceleration Z Component", fontsize=20)
ben_jump_H_axes[2].set_ylabel("Acceleration [m/s^2]", fontsize=15)
ben_jump_H_axes[2].legend()

#Plot Total acceleration
ben_jump_H_axes[3].plot(time, ben_jump_H_total_acc, label='Total Acceleration', color='k')
ben_jump_H_axes[3].set_title("Ben Jumping H - Total Acceleration", fontsize=20)
ben_jump_H_axes[3].set_xlabel("Time [s]", fontsize=15)
ben_jump_H_axes[3].set_ylabel("Acceleration [m/s^2]", fontsize=15)
ben_jump_H_axes[3].legend()

plt.tight_layout()

#Plotting Guntas' jumping data

#Extracting data columns for each data column
time = guntas_jump_BP.iloc[:, 0]
guntas_jump_BP_x_acc = guntas_jump_BP.iloc[:, 1]
guntas_jump_BP_y_acc = guntas_jump_BP.iloc[:, 2]
guntas_jump_BP_z_acc = guntas_jump_BP.iloc[:, 3]
guntas_jump_BP_total_acc = guntas_jump_BP.iloc[:, 4]

#Creating subplot
fig3, axes3 = plt.subplots(ncols=1, nrows=4, figsize=(10,10))

#Plot X acceleration
axes3[0].plot(time, guntas_jump_BP_x_acc, label='X Acceleration', color='r')
axes3[0].set_title("Guntas Jumping BP - Acceleration X Component", fontsize=20)
axes3[0].set_ylabel("Acceleration [m/s^2]", fontsize=15)
axes3[0].legend()

#Plot Y acceleration
axes3[1].plot(time, guntas_jump_BP_y_acc, label='Y Acceleration', color='g')
axes3[1].set_title("Guntas Jumping BP - Acceleration Y Component", fontsize=20)
axes3[1].set_ylabel("Acceleration [m/s^2]", fontsize=15)
axes3[1].legend()

#Plot Z acceleration
axes3[2].plot(time, guntas_jump_BP_z_acc, label='Z Acceleration', color='b')
axes3[2].set_title("Guntas Jumping BP - Acceleration Z Component", fontsize=20)
axes3[2].set_ylabel("Acceleration [m/s^2]", fontsize=15)
axes3[2].legend()

#Plot Total acceleration
axes3[3].plot(time, guntas_jump_BP_total_acc, label='Total Acceleration', color='k')
axes3[3].set_title("Guntas Jumping BP - Total Acceleration", fontsize=20)
axes3[3].set_xlabel("Time [s]", fontsize=15)
axes3[3].set_ylabel("Acceleration [m/s^2]", fontsize=15)
axes3[3].legend()

plt.tight_layout()

#Plotting Matt's walking data

#Extracting data columns for each data column
time = matt_walk_BP.iloc[:, 0]
matt_walk_BP_x_acc = matt_walk_BP.iloc[:, 1]
matt_walk_BP_y_acc = matt_walk_BP.iloc[:, 2]
matt_walk_BP_z_acc = matt_walk_BP.iloc[:, 3]
matt_walk_BP_total_acc = matt_walk_BP.iloc[:, 4]

#Creating subplot
matt_walk_BP_fig, matt_walk_BP_axes = plt.subplots(ncols=1, nrows=4, figsize=(10, 10))

#Plot X acceleration
matt_walk_BP_axes[0].plot(time, matt_walk_BP_x_acc, label='X Acceleration', color='r')
matt_walk_BP_axes[0].set_title("Matt Walking BP - Acceleration X Component", fontsize=20)
matt_walk_BP_axes[0].set_ylabel("Acceleration [m/s^2]", fontsize=15)
matt_walk_BP_axes[0].legend()

#Plot Y acceleration
matt_walk_BP_axes[1].plot(time, matt_walk_BP_y_acc, label='Y Acceleration', color='g')
matt_walk_BP_axes[1].set_title("Matt Walking BP - Acceleration Y Component", fontsize=20)
matt_walk_BP_axes[1].set_ylabel("Acceleration [m/s^2]", fontsize=15)
matt_walk_BP_axes[1].legend()

#Plot Z acceleration
matt_walk_BP_axes[2].plot(time, matt_walk_BP_z_acc, label='Z Acceleration', color='b')
matt_walk_BP_axes[2].set_title("Matt Walking BP - Acceleration Z Component", fontsize=20)
matt_walk_BP_axes[2].set_ylabel("Acceleration [m/s^2]", fontsize=15)
matt_walk_BP_axes[2].legend()

#Plot Total acceleration
matt_walk_BP_axes[3].plot(time, matt_walk_BP_total_acc, label='Total Acceleration', color='k')
matt_walk_BP_axes[3].set_title("Matt Walking BP - Total Acceleration", fontsize=20)
matt_walk_BP_axes[3].set_xlabel("Time [s]", fontsize=15)
matt_walk_BP_axes[3].set_ylabel("Acceleration [m/s^2]", fontsize=15)
matt_walk_BP_axes[3].legend()

plt.tight_layout()

#Plotting Ben's walking data

#Extracting data columns for each data column
time = ben_walk_JP.iloc[:, 0]
ben_walk_JP_x_acc = ben_walk_JP.iloc[:, 1]
ben_walk_JP_y_acc = ben_walk_JP.iloc[:, 2]
ben_walk_JP_z_acc = ben_walk_JP.iloc[:, 3]
ben_walk_JP_total_acc = ben_walk_JP.iloc[:, 4]

#Creating subplot
ben_walk_JP_fig, ben_walk_JP_axes = plt.subplots(ncols=1, nrows=4, figsize=(10, 10))

#Plot X acceleration
ben_walk_JP_axes[0].plot(time, ben_walk_JP_x_acc, label='X Acceleration', color='r')
ben_walk_JP_axes[0].set_title("Ben Walking JP - Acceleration X Component", fontsize=20)
ben_walk_JP_axes[0].set_ylabel("Acceleration [m/s^2]", fontsize=15)
ben_walk_JP_axes[0].legend()

#Plot Y acceleration
ben_walk_JP_axes[1].plot(time, ben_walk_JP_y_acc, label='Y Acceleration', color='g')
ben_walk_JP_axes[1].set_title("Ben Walking JP - Acceleration Y Component", fontsize=20)
ben_walk_JP_axes[1].set_ylabel("Acceleration [m/s^2]", fontsize=15)
ben_walk_JP_axes[1].legend()

#Plot Z acceleration
ben_walk_JP_axes[2].plot(time, ben_walk_JP_z_acc, label='Z Acceleration', color='b')
ben_walk_JP_axes[2].set_title("Ben Walking JP - Acceleration Z Component", fontsize=20)
ben_walk_JP_axes[2].set_ylabel("Acceleration [m/s^2]", fontsize=15)
ben_walk_JP_axes[2].legend()

#Plot Total acceleration
ben_walk_JP_axes[3].plot(time, ben_walk_JP_total_acc, label='Total Acceleration', color='k')
ben_walk_JP_axes[3].set_title("Ben Walking JP - Total Acceleration", fontsize=20)
ben_walk_JP_axes[3].set_xlabel("Time [s]", fontsize=15)
ben_walk_JP_axes[3].set_ylabel("Acceleration [m/s^2]", fontsize=15)
ben_walk_JP_axes[3].legend()

plt.tight_layout()

#Plotting Guntas' walking data

#Extracting data columns for each data column
time = guntas_walk_H.iloc[:, 0]
guntas_walk_H_x_acc = guntas_walk_H.iloc[:, 1]
guntas_walk_H_y_acc = guntas_walk_H.iloc[:, 2]
guntas_walk_H_z_acc = guntas_walk_H.iloc[:, 3]
guntas_walk_H_total_acc = guntas_walk_H.iloc[:, 4]

#Creating subplot
guntas_walk_H_fig, guntas_walk_H_axes = plt.subplots(ncols=1, nrows=4, figsize=(10, 10))

#Plot X acceleration
guntas_walk_H_axes[0].plot(time, guntas_walk_H_x_acc, label='X Acceleration', color='r')
guntas_walk_H_axes[0].set_title("Guntas Walking H - Acceleration X Component", fontsize=20)
guntas_walk_H_axes[0].set_ylabel("Acceleration [m/s^2]", fontsize=15)
guntas_walk_H_axes[0].legend()

#Plot Y acceleration
guntas_walk_H_axes[1].plot(time, guntas_walk_H_y_acc, label='Y Acceleration', color='g')
guntas_walk_H_axes[1].set_title("Guntas Walking H - Acceleration Y Component", fontsize=20)
guntas_walk_H_axes[1].set_ylabel("Acceleration [m/s^2]", fontsize=15)
guntas_walk_H_axes[1].legend()

#Plot Z acceleration
guntas_walk_H_axes[2].plot(time, guntas_walk_H_z_acc, label='Z Acceleration', color='b')
guntas_walk_H_axes[2].set_title("Guntas Walking H - Acceleration Z Component", fontsize=20)
guntas_walk_H_axes[2].set_ylabel("Acceleration [m/s^2]", fontsize=15)
guntas_walk_H_axes[2].legend()

#Plot Total acceleration
guntas_walk_H_axes[3].plot(time, guntas_walk_H_total_acc, label='Total Acceleration', color='k')
guntas_walk_H_axes[3].set_title("Guntas Walking H - Total Acceleration", fontsize=20)
guntas_walk_H_axes[3].set_xlabel("Time [s]", fontsize=15)
guntas_walk_H_axes[3].set_ylabel("Acceleration [m/s^2]", fontsize=15)
guntas_walk_H_axes[3].legend()

plt.tight_layout()

#Create scatter plots for walking acceleration components vs magnitude
fig1, ax1 = plt.subplots(3, 1, figsize=(10, 12))

#Walking dataset acceleration components
ax1[0].scatter(matt_walk_H["Absolute acceleration (m/s^2)"], matt_walk_H["Acceleration x (m/s^2)"], color='red', alpha=0.5, label='X Component')
ax1[1].scatter(matt_walk_H["Absolute acceleration (m/s^2)"], matt_walk_H["Acceleration y (m/s^2)"], color='green', alpha=0.5, label='Y Component')
ax1[2].scatter(matt_walk_H["Absolute acceleration (m/s^2)"], matt_walk_H["Acceleration z (m/s^2)"], color='blue', alpha=0.5, label='Z Component')

#Labels and title
ax1[0].set_ylabel("X Acceleration (m/s²)")
ax1[1].set_ylabel("Y Acceleration (m/s²)")
ax1[2].set_ylabel("Z Acceleration (m/s²)")
ax1[2].set_xlabel("Absolute Acceleration (m/s²)")
ax1[0].set_title("Matt Walking H - Component Acceleration vs. Absolute Acceleration", fontsize=15)

#Legends
for a in ax1:
    a.legend()
    a.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()

# Create scatter plots for jumping acceleration components vs magnitude
fig2, ax2 = plt.subplots(3, 1, figsize=(10, 12))

#Jumping dataset acceleration components
ax2[0].scatter(guntas_jump_H["Absolute acceleration (m/s^2)"], guntas_jump_H["Acceleration x (m/s^2)"], color='red', alpha=0.5, label='X Component')
ax2[1].scatter(guntas_jump_H["Absolute acceleration (m/s^2)"], guntas_jump_H["Acceleration y (m/s^2)"], color='green', alpha=0.5, label='Y Component')
ax2[2].scatter(guntas_jump_H["Absolute acceleration (m/s^2)"], guntas_jump_H["Acceleration z (m/s^2)"], color='blue', alpha=0.5, label='Z Component')

#Labels and title
ax2[0].set_ylabel("X Acceleration (m/s²)")
ax2[1].set_ylabel("Y Acceleration (m/s²)")
ax2[2].set_ylabel("Z Acceleration (m/s²)")
ax2[2].set_xlabel("Absolute Acceleration (m/s²)")
ax2[0].set_title("Matt Walking H - Component Acceleration vs. Absolute Acceleration", fontsize=15)

# Legends
for a in ax2:
    a.legend()
    a.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()

plt.show()


# Map each dataset's name to the dataset
datasets = {
    "matt_jump_BP": matt_jump_BP,
    "matt_jump_FP": matt_jump_FP,
    "matt_jump_H": matt_jump_H,
    "matt_jump_JP": matt_jump_JP,
    "ben_jump_H": ben_jump_H,
    "ben_jump_JP": ben_jump_JP,
    "ben_jump_FP": ben_jump_FP,
    "guntas_jump_H": guntas_jump_H,
    "guntas_jump_FP": guntas_jump_FP,
    "guntas_jump_BP": guntas_jump_BP,
    "matt_walk_H": matt_walk_H,
    "matt_walk_FP": matt_walk_FP,
    "matt_walk_BP": matt_walk_BP,
    "ben_walk_H": ben_walk_H,
    "ben_walk_JP": ben_walk_JP,
    "ben_walk_FP": ben_walk_FP,
    "guntas_walk_H": guntas_walk_H
}


# Open the file and add the raw data to it
with h5py.File('./hdf5_data.h5', 'w') as hdf:
    rawData = hdf.create_group('/Raw Data')
    matt_rawData = hdf.create_group('Raw Data/Matt')
    matt_rawData.create_dataset('Jumping_BP', data=matt_jump_BP)
    matt_rawData.create_dataset('Jumping_FP', data=matt_jump_FP)
    matt_rawData.create_dataset('Jumping_JP', data=matt_jump_JP)
    matt_rawData.create_dataset('Jumping_H', data=matt_jump_H)
    matt_rawData.create_dataset('Walking_H', data=matt_walk_H)
    matt_rawData.create_dataset('Walking_FP', data=matt_walk_FP)
    matt_rawData.create_dataset('Walking_BP', data=matt_walk_BP)

    ben_rawData = hdf.create_group('Raw Data/Ben')
    ben_rawData.create_dataset('Jumping_H', data=ben_jump_H)
    ben_rawData.create_dataset('Jumping_JP', data=ben_jump_JP)
    ben_rawData.create_dataset('Jumping_FP', data=ben_jump_FP)
    ben_rawData.create_dataset('Walking_H', data=ben_walk_H)
    ben_rawData.create_dataset('Walking_JP', data=ben_walk_JP)
    ben_rawData.create_dataset('Walking_FP', data=ben_walk_FP)

    guntas_rawData = hdf.create_group('Raw Data/Guntas')
    guntas_rawData.create_dataset('Jumping_H', data=guntas_jump_H)
    guntas_rawData.create_dataset('Jumping_FP', data=guntas_jump_FP)
    guntas_rawData.create_dataset('Jumping_BP', data=guntas_jump_BP)
    guntas_rawData.create_dataset('Walking_H', data=guntas_walk_H)

    # Create the other groups in HDF file
    preproc_group = hdf.create_group("/Preprocessed Data")
    train_test_group = hdf.create_group("/Train_Test Data")
    train_group = train_test_group.create_group("Train")
    test_group = train_test_group.create_group("Test")

    # Go through each dataset to create the moving averages and store them
    for name, i in datasets.items():
        df = i

        # Clean NA values in the dataset if they exist
        naIndex = np.where(df.isna())
        if len(naIndex[0]) != 0:
            df.interpolate(method="linear", inplace=True)

        # MA length
        window = 49

        # Create Moving average
        xAccMa = df['Acceleration x (m/s^2)'].rolling(window=window).mean()
        yAccMa = df['Acceleration y (m/s^2)'].rolling(window=window).mean()
        zAccMa = df['Acceleration z (m/s^2)'].rolling(window=window).mean()

        # Save the moving averages into preprocessed group
        personName = name.split("_")[0].capitalize()
        if personName in preproc_group:
            grm = preproc_group[personName]
        else:
            grm = preproc_group.create_group(personName)

        # Create a dataset with the corresponding data and MA
        grm.create_dataset(f"xMA {name}", data=xAccMa)
        grm.create_dataset(f"yMA {name}", data=yAccMa)
        grm.create_dataset(f"zMA {name}", data=zAccMa)

        # Create and save segmented data
        # Get the time series values from the time dataframe
        time_Series = df["Time (s)"]

        # Calculate the differences between consecutive time values
        time_differences = time_Series.diff().dropna()  # .diff() gets the difference, .dropna() removes the first NaN

        # Calculate the average time difference between all the segments
        average_delta = time_differences.mean()

        # Calculate number of data entries in each 5-second window
        num_entries = int(5 / average_delta)

        # Calculate total number of 5-second windows
        num_windows = len(df) // num_entries

        # Arrays to store windows for each column in the dataset
        time_windows = []
        x_windows = []
        y_windows = []
        z_windows = []

        # Split the data into 5 second segments by using math and the numbers calculated above
        for j in range(num_windows):
            start = j * num_entries
            end = start + num_entries

            time_windows.append(time_Series.iloc[start:end])
            x_windows.append(xAccMa.iloc[start:end])
            y_windows.append(yAccMa.iloc[start:end])
            z_windows.append(zAccMa.iloc[start:end])

        # Shuffle all the lists, keeping the shuffle consistent among the different axis
        combined = list(zip(time_windows, x_windows, y_windows, z_windows))
        random.shuffle(combined)
        time_windows, x_windows, y_windows, z_windows = zip(*combined)

        # If dataset is jumping, label it 1 and put into hdf5 file, if walking, label it 0
        # Check if jumping or walking
        if name.split("_")[1] == "jump":
            label = 1
        else:
            label = 0

        # Convert lists of Series to NumPy arrays
        x_array = np.array([w.values for w in x_windows])
        y_array = np.array([w.values for w in y_windows])
        z_array = np.array([w.values for w in z_windows])
        labels_array = np.full((len(x_windows),), label)

        # Save to Preprocessed Group
        # Split into train (90%) and test (10%) sets
        x_train, x_test, y_train, y_test, z_train, z_test, labels_train, labels_test = train_test_split(
            x_array, y_array, z_array, labels_array, test_size=0.1, random_state=42
        )

        # Create train and test groups inside Train_Test Data
        train_subgroup = train_group.create_group(name)
        test_subgroup = test_group.create_group(name)

        # Save training data
        train_subgroup.create_dataset("x", data=x_train)
        train_subgroup.create_dataset("y", data=y_train)
        train_subgroup.create_dataset("z", data=z_train)
        train_subgroup.create_dataset("label", data=labels_train)

        # Save testing data
        test_subgroup.create_dataset("x", data=x_test)
        test_subgroup.create_dataset("y", data=y_test)
        test_subgroup.create_dataset("z", data=z_test)
        test_subgroup.create_dataset("label", data=labels_test)

        # Plot MA's (This plots so many graphs that an error occurs due to too many requests from the API
        # That's why the Y and Z are commented out, you can uncomment them to check the data
        # X plot
        plt.figure(figsize=(10, 6))
        plt.plot(df['Time (s)'], df['Acceleration x (m/s^2)'], label='Original X Acceleration', color='blue')
        plt.plot(df['Time (s)'], xAccMa, label=f'Moving Average {window}', color='red', linewidth=2)

        activity = name.split("_")[1].capitalize() + "ing"
        plt.title(f'X Acceleration with Moving Average for {personName} {activity}', fontsize=15)
        plt.xlabel('Time (s)', fontsize=12)
        plt.ylabel('Acceleration (m/s^2)', fontsize=12)
        plt.legend()
        plt.grid(True, alpha=0.7)
        plt.tight_layout()
        plt.show()

        '''
        # Y plot
        plt.plot(df['Time (s)'], df['Acceleration y (m/s^2)'], label='Original Y Acceleration', color='blue')
        plt.plot(df['Time (s)'], yAccMa100, label='Moving Average', color='red', linewidth=2)

        activity = name.split("_")[1].capitalize() + "ing"
        plt.title(f'Y Acceleration with Moving Average for {personName} {activity}', fontsize=15)
        plt.xlabel('Time (s)', fontsize=12)
        plt.ylabel('Acceleration (m/s^2)', fontsize=12)
        plt.legend()
        plt.grid(True, alpha=0.7)
        plt.tight_layout()
        plt.show()
        
        # Z plot
        plt.plot(df['Time (s)'], df['Acceleration z (m/s^2)'], label='Original Z Acceleration', color='blue')
        plt.plot(df['Time (s)'], zAccMa100, label='Moving Average', color='red', linewidth=2)
    
        activity = name.split("_")[1].capitalize() + "ing"
        plt.title(f'Z Acceleration with Moving Average for {personName} {activity}', fontsize=15)
        plt.xlabel('Time (s)', fontsize=12)
        plt.ylabel('Acceleration (m/s^2)', fontsize=12)
        plt.legend()
        plt.grid(True, alpha=0.7)
        plt.tight_layout()
        plt.show() '''


