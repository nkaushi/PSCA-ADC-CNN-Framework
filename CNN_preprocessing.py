#Preprocess the data for CNN after allignment - Iref is the only freature so it creates number of samples for Iref 
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Load the data
df = pd.read_csv('Filepath/aligned_data.csv')  # Adjust this if necessary

# Standardize 'Iref' (remove 'Time' column as it's not needed)
scaler = StandardScaler()
df[['Iref']] = scaler.fit_transform(df[['Iref']])  # Only standardizing Iref

# Step 2: Extract Iref as feature
X_iref = df[['Iref']].values  # Shape: (n_samples, 1)

# Step 3: Extract the target bits (separate bits columns)
bit_columns = ['bit0', 'bit1', 'bit2', 'bit3', 'bit4', 'bit5', 'bit6', 'bit7']
y_bits = df[bit_columns].values  # Shape: (n_samples, 8)

# Step 4: Create windows of 40 time steps for CNN input
window_size = 40

# Create sequences for the CNN input (sliding window approach)
X = []
Y = []

# Slide over the data to create windows of 10 time steps
for i in range(len(df) - window_size):
    X.append(X_iref[i:i+window_size])  # 10 time steps (each with 1 feature, which is Iref)
    Y.append(y_bits[i+window_size-1])  # Target is the bit values at the last time step of the window

X = np.array(X)  # Shape: (n_samples - window_size, window_size, 1)
Y = np.array(Y)  # Shape: (n_samples - window_size, 8)


# Print the shape of the input (X) and labels (y)
print("Input shape (X):", X.shape)
print("Label shape (y):", Y.shape)

np.savez('Filepath/cnn_data_1_bit_preprocessing.npz', X=X, Y=Y)