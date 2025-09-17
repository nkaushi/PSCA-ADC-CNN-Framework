# PSCA-ADC-CNN-Framework

This framework is based on NVIDIA-SMI 525.147.05   Driver Version: 525.147.05   CUDA Version: 12.2. Install the required libraries for matplotlib and TensorFlow before proceeding with the script.

This is a list of scripts to test power side-channel attack (PSCA) on ADC CDAC current traces via a full-scale ramp. The secure and unsecure ADCs are single-ended 8-bit Flash-SAR ADCs and 8-bit SAR ADC based on a split capacitor scheme. The results are based on transistor-level transient simulation results.
If you are new to this, this framework enables characterization of SAR ADC against Power Side Channel Attacks (PSCA) by utilizing the current trace from the Capacitive DAC (CDAC). The CDAC shows the current related to the bit cycling during the conversion process. The data required is the current signature from the VREF node of the CDAC. The input signal should span a full range with at least 20 LSB per conversion. This gives enough data for each code.

A transient simulation of the ramp should have the following traces.
1. Current from the VREF node
2. ADC digital code output.
3. Input ramp (optional).
   
The following section describes the scripts in order.

1. Data_preprocessing_allignment.py
This script takes the raw data from the simulation/lab_bench measurement. It loads the .csv file, renames the headers, removes NaN values if any in the data, decimates the data, and updates based on clock speed and data requirement.

As the current changes more frequently than the digital code due to a slow ramp, the code is backward filled until there is an update in the ADC code. The ramp used for this characterization is a monotonic ramp from 0 to FS.

The script also segregates the 8-bit code into bitwise columns for saving and further data processing.

It has various print options through the run to verify the data quality.

The file is saved as a .csv for verification. This file is used in the next script. 


2. CNN_preprocessing.py
This will load the processed file and prepare it for CNN processing. It uses Iref as a feature with a window size of 40. This file creates the cnn_data_1_bit_preprocessing.npz file to be used on the next script.

3. CNN_and_Plots.py
This script loads the .npz file from the preprocessing script and trains the CNN model for each bit. It is a simple model used on previous PSCA research (S2ADC Jeong 2020).

The script uses a binary classifier during training and testing. It creates the following plots;

1. Training accuracy per bit
2. Testing accuracy per bit
3. Bitwise RMSE contribution
4. Normalized bitwise RMSE contribution
5. Errors per bit
6. Bit error rate vs N-RMSE
7. Prediction error histogram
8. RMSE comparison (Full code random vs Total RMSE and summer bitwise RMSE)
9. Heatmap per bit

The files are saved in a summaries directory for comparison with other ADC versions. A comparison script can be used to plot the security of different ADC versions. 

The comparison script compares the parameters obtained for different ADCs. It plots the following:

1. Bitwise RMSE contribution
2. Normalize RMSE plots - Based on 8-bit
3. Bitwise Error Rate
4. Bit predictability based on error rate
5. Total RMSE comparison
6. Correlation heat maps

