import pandas
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import h5py
import random
from sklearn.model_selection import train_test_split

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

    # Create groups in HDF file
    preproc_group = hdf.create_group("/Preprocessed Data")
    train_test_group = hdf.create_group("/Train_Test Data")
    train_group = train_test_group.create_group("Train")
    test_group = train_test_group.create_group("Test")

    for name, i in datasets.items():
        df = i

        # Clean NA values
        naIndex = np.where(df.isna())
        if len(naIndex[0]) != 0:
            df.interpolate(method="linear", inplace=True)

        # MA length
        window = 49

        # Moving average
        xAccMa100 = df['Acceleration x (m/s^2)'].rolling(window=window).mean()
        yAccMa100 = df['Acceleration y (m/s^2)'].rolling(window=window).mean()
        zAccMa100 = df['Acceleration z (m/s^2)'].rolling(window=window).mean()

        # Save the moving averages into preprocessed group
        personName = name.split("_")[0].capitalize()
        if personName in preproc_group:
            grm = preproc_group[personName]
        else:
            grm = preproc_group.create_group(personName)

        grm.create_dataset(f"xMA {name}", data=xAccMa100)
        grm.create_dataset(f"yMA {name}", data=yAccMa100)
        grm.create_dataset(f"zMA {name}", data=zAccMa100)

        # Create and save segmented data
        # Split data into 5 second segments
        # Get the time series values from the time dataframe
        time_Series = df["Time (s)"]

        # Calculate the differences between consecutive time values
        time_differences = time_Series.diff().dropna()  # .diff() gets the difference, .dropna() removes the first NaN

        # Calculate the average time difference
        average_delta = time_differences.mean()

        # Calculate number of data entries in each 5-second window
        num_entries = int(5 / average_delta)

        # Total number of 5-second windows
        num_windows = len(df) // num_entries

        # Arrays to store windows for each column in the dataset
        time_windows = []
        x_windows = []
        y_windows = []
        z_windows = []

        # Split the data into 5 second segments
        for j in range(num_windows):
            start = j * num_entries
            end = start + num_entries

            time_windows.append(time_Series.iloc[start:end])
            x_windows.append(xAccMa100.iloc[start:end])
            y_windows.append(yAccMa100.iloc[start:end])
            z_windows.append(zAccMa100.iloc[start:end])

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

        # Plot MA's
        '''
        # X plot
        plt.figure(figsize=(10, 6))
        plt.plot(df['Time (s)'], df['Acceleration x (m/s^2)'], label='Original X Acceleration', color='blue')
        plt.plot(df['Time (s)'], xAccMa100, label='Moving Average', color='red', linewidth=2)

        activity = name.split("_")[1].capitalize() + "ing"
        plt.title(f'X Acceleration with Moving Average for {personName} {activity}', fontsize=15)
        plt.xlabel('Time (s)', fontsize=12)
        plt.ylabel('Acceleration (m/s^2)', fontsize=12)
        plt.legend()
        plt.grid(True, alpha=0.7)
        plt.tight_layout()
        plt.show()
    
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


