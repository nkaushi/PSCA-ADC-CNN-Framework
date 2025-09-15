#Allign binary code with time
import pandas as pd
import numpy as np

# ---------------------------
# Step 1: Read the Data
# ---------------------------
# Skip the second row (index 1) because it is a reset state.
df = pd.read_csv("your/file/path.csv")

print(df.head(10))



# Rename columns based on the new headers // Example headers
df.columns = ['VINP_RAMP (VINDC=0) X',  
              'VINP_RAMP (VINDC=0) Y',  
              '/I55/VREF (VINDC=0) X',  
              '/I55/VREF (VINDC=0) Y',  
              'Binary_code X',  
              'Binary_code LogicBus',
              'Code (VINDC=0) X',
              'Code (VINDC=0) LogicBus'] 


# Extracting the relevant columns for decimal_data and vref_data
decimal_data = df[['Binary_code X' , 'Binary_code LogicBus']]  # Select decimal data columns (binary code-related)
vref_data = df[['/I55/VREF (VINDC=0) X', '/I55/VREF (VINDC=0) Y']]  # Select the VREF columns

# Rename columns for easier access
decimal_data.columns = ['Time', 'Binary_Code']
vref_data.columns = ['Time', 'Iref']

# Check the data types of the 'Time' columns
print("Data type of 'Time' in decimal_data:", decimal_data['Time'].dtype)
print("Data type of 'Time' in vref_data:", vref_data['Time'].dtype)

# Strip spaces from the 'Time' column and handle non-numeric values
decimal_data.loc[:, 'Time'] = decimal_data['Time'].str.strip()  # Remove extra spaces
decimal_data.loc[:, 'Time'] = pd.to_numeric(decimal_data['Time'], errors='coerce')  # Convert to numeric, 'coerce' turns invalid to NaN
decimal_data = decimal_data.dropna(subset=['Time'])  # Drop rows where 'Time' is NaN

# Handle the 'Binary_Code' column (if you have zeros or unwanted values)
decimal_data = decimal_data[decimal_data['Binary_Code'] != '00000000']  # Remove rows with '00000000' binary code


#Scale current if the value drops
vref_data.loc[:, 'Iref'] = vref_data['Iref'].abs() * 1
print("Scaled and absolute VREF data (first 10 rows):")

# Now convert 'Time' to float64 and round to - This is decimation based on clock speed, if the ADC clock is 10ns then round(9) gives 10 samples per clock.
decimal_data.loc[:, 'Time'] = decimal_data['Time'].astype('float64')
decimal_data.loc[:, 'Time'] = decimal_data['Time'].round(8)

# Verify the cleaned and rounded decimal data
print("\nCleaned and Rounded Decimal Data:")
print(decimal_data.head())

# Preserve original VREF data before downsampling
vref_data_before_downsampling = vref_data.copy()

# Round 'Time' to 1 ns (avoid SettingWithCopyWarning) - Round to the same value as time above to not loose information.
vref_data = vref_data.copy()
vref_data.loc[:, 'Time'] = vref_data['Time'].round(8)  

# Downsample: Keep one unique entry per timestamp (e.g., mean value)
vref_data = vref_data.groupby('Time', as_index=False).mean()

# Ensure sorted order
vref_data = vref_data.sort_values('Time')

# Print last 10 rows of processed VREF data
print("Last 10 rows of downsampled & sorted VREF data:")
print(vref_data.tail(10))

# Compare unique time values before and after
print("Unique Time Values in Original VREF Data:", len(vref_data_before_downsampling['Time'].unique()))
print("Unique Time Values in Downsampled VREF Data:", len(vref_data['Time'].unique()))

# Check the data types of columns
print("\nData types of columns in decimal_data:")
print(decimal_data.dtypes)

print("\nData types of columns in vref_data:")
print(vref_data.dtypes)

# Check the first few rows of the data for an overview
print("\nFirst few rows of decimal_data:")
print(decimal_data.head())

print("\nFirst few rows of vref_data:")
print(vref_data.head())

# Directly handle NaN values or any potential issues without the string operations
vref_data['Time'] = pd.to_numeric(vref_data['Time'], errors='coerce')  # Ensure numeric values and handle errors

# Check the data types of the 'Time' columns
print("Data type of 'Time' in decimal_data:", decimal_data['Time'].dtype)
print("Data type of 'Time' in vref_data:", vref_data['Time'].dtype)

# Step 1: Convert 'Time' columns to float64
decimal_data['Time'] = pd.to_numeric(decimal_data['Time'], errors='coerce')  # Convert to float64
vref_data['Time'] = pd.to_numeric(vref_data['Time'], errors='coerce')  # Convert to float64

# Check the data types of the 'Time' columns
print("Data type of 'Time' in decimal_data:", decimal_data['Time'].dtype)
print("Data type of 'Time' in vref_data:", vref_data['Time'].dtype)

# Sort both datasets by 'Time'
decimal_data = decimal_data.sort_values(by='Time').reset_index(drop=True)
vref_data = vref_data.sort_values(by='Time').reset_index(drop=True)

# Check the first few rows of the data for an overview
print("\nFirst few rows of decimal_data:")
print(decimal_data.head())

print("\nFirst few rows of vref_data:")
print(vref_data.head())

# Remove the first row (reset state)
decimal_data = decimal_data.iloc[1:].reset_index(drop=True)
vref_data = vref_data.iloc[1:].reset_index(drop=True)

# Display the updated first few rows
print("Updated decimal_data:\n", decimal_data.head())
print("Updated vref_data:\n", vref_data.head())

# Merge using the closest previous value for 'Binary_Code'
aligned_data = pd.merge_asof(vref_data, decimal_data, on="Time", direction="backward")

# Ensure that the original Iref from vref_data remains unchanged
aligned_data['Iref'] = vref_data['Iref'].values

# Display first few rows to check alignment
print("Aligned Data:")
print(aligned_data.head())


# Function to separate each bit of Binary_Code into individual columns
def separate_bits(binary_code):
    return [int(bit) for bit in binary_code]

# Apply the function to the 'Binary_Code' column to separate bits
bit_columns = pd.DataFrame(aligned_data['Binary_Code'].apply(separate_bits).tolist(), columns=[f'bit{7-i}' for i in range(8)])

# Concatenate the new bit columns with the original aligned_data
aligned_data = pd.concat([aligned_data, bit_columns], axis=1)

# Display the updated data with separate bit columns
print(aligned_data.head())
print(aligned_data.tail())

aligned_data.to_csv('Filepath/alligned_data.csv', index=False)

