# PSCA-ADC-CNN-Framework
This is a list of scripts to test power side channel attack(PSCA) on ADC CDAC current trace via a full scale ramp.
If you are new to this, this framework allows characerization of SAR ADC against pweor side channel attacks. The CDAC shows current realted to the bit cycling during conversion process. The data required is current signature from the VREF node of the CDAC. The input signal should span a full range with atleast 20 LSB per conversion. This gives enough data for each code.

A transient simulation of the ramp should have the following traces.
1. Current from VREF node
2. ADC digital code output.
3. Input ramp (optional).
   
The foolowing section describes the scripts in order.

1. Data_preprocessing_allignment.py
This script takes the raw data from from simulation/lab_bench measurement. It loads the .csv file, renames the headers, remove NaN values if any in the data, decimates the data - update based on clock speed and data set requirement.

Current has more points than code, the code is backward filled untill there is an update in the ADC code. The ramp used for this characertization is a monntonic ramp from 0 to FS.

The script also segregates the 8-bit code into bitwise columns for saving and further data processing.

It has various print options through the run to verfity the data quality.

The file is saved as an .csv for verifcation. This file is used in the next script. 


2. CNN_preprocessing.py
This will load the processed file and prepare it for CNN processing. It uses Iref as a feature with a windown size of 40. This file create the cnn_data_1_bit_preprocessing.npz file to be used on the next scirpt.

3. CNN_and_Plots.py
This script loads the .npz file from the preprocessing script and train the CNN model for each bit. It is a simple model used on previosu PSCA research (S2ADC Jeong 2020).

The script uses binary classifier during traning and testing. It creates the follwoing plots;

1. Training accuracy per bit
2. Testing accuracy per bit
3. Bitwise RMSE contribution
4. Normalized bitwise RMSE contribution
5. Errors per bit
6. Bit error rate vs N-RMSE
7. Prediction error histogram
8. RMSE comparison ( FUll code random vs Total RMSE and summer bitwise RMSE)
9. Heatmap per bit

The files are saved in a summaries directory for comparison with other ADC versions. A comparison script can be used to plot security of different ADC versions. 

The comparison script goes through comparison of different ADC performance. It plots the follwing:

1. Bitwise RMSE contribution
2. Normalize RMSE plots - Based on 8-bit
3. Bitwise Error Rate
4. Bit predictibility based on error rate
5. Total RMSE comparison
6. Correlation heat maps


This framwork is based on NVIDIA-SMI 525.147.05   Driver Version: 525.147.05   CUDA Version: 12.2. Install the required libraries for matplot lib, tensorflow before proceeding with the script.

Thank you,
Dr. Kaushik
