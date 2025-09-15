#Time to get results of the processing

import tensorflow as tf

#Train a CNN model for each bit
import numpy as np
import pandas as pd
from tensorflow.keras.layers import Flatten, Dense, Softmax
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import mean_squared_error
import math
from tensorflow.keras.models import load_model
from sklearn.metrics import accuracy_score
import os
import seaborn as sns
from pathlib import Path
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, LSTM, Dense, BatchNormalization, Dropout, LeakyReLU
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Input


# Create 'plots' directory in current directory
plot_dir = os.path.join(os.getcwd(), "plots")
os.makedirs(plot_dir, exist_ok=True)

# Load the data from the .npz file
data = np.load('Filepath/cnn_data_1_bit_preprocessing.npz')

# Extract the arrays
X = data['X']
Y = data['Y']

# Print shapes to verify
print("Input shape (X):", X.shape)
print("Label shape (Y):", Y.shape)


# Step 1: Define the function to create the CNN model
def create_cnn_model():
    model = Sequential()

    model.add(Input(shape=(40, 1)))  # Specify input shape as the first layer

    # Add the first convolutional layer with 'same' padding
    model.add(Conv1D(5, kernel_size=5, activation='relu', padding='same'))
    model.add(BatchNormalization())
    # Add the first max pooling layer
    model.add(MaxPooling1D(pool_size=5))
    
    # Add the second convolutional layer with 'same' padding
    model.add(Conv1D(5, kernel_size=5, activation='relu', padding='same'))
    model.add(BatchNormalization())
    # Add the second max pooling layer
    model.add(MaxPooling1D(pool_size=5))

    # Flatten the output for the fully connected layers
    model.add(Flatten())

    # Add the first fully connected layer
    model.add(Dense(100, activation='relu'))
    model.add(Dropout(0.3))

    # Add the second fully connected layer
    model.add(Dense(100, activation='relu'))
    model.add(Dropout(0.3))

    # Add the third fully connected layer
    model.add(Dense(100, activation='relu'))
    model.add(Dropout(0.3))

    # Add the output layer with sigmoid activation for binary classification (1 bit per model)
    model.add(Dense(1, activation='sigmoid'))

    # Compile the model
    model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])

    return model


X_train, X_temp, Y_train, Y_temp = train_test_split(X, Y, test_size=0.3, random_state=42)
X_val, X_test, Y_val, Y_test = train_test_split(X_temp, Y_temp, test_size=0.5, random_state=42)
# Store threshold for later decoding
bit_thresholds = {}

print("Checking first 5 labels (Y_train):")
for i in range(5):
    print(f"Sample {i} bits: {Y_train[i]}")

# Also check how many 1's each bit has in training set
print("\nBit distribution in Y_train:")
for b in range(Y_train.shape[1]):
    ones = np.sum(Y_train[:, b])
    zeros = Y_train.shape[0] - ones
    print(f"Bit {b}: 1's = {ones}, 0's = {zeros}, ratio = {ones / Y_train.shape[0]:.2f}")


# Flip bits once after splitting
Y_train = Y_train[:, ::-1]
Y_val = Y_val[:, ::-1]
Y_test = Y_test[:, ::-1]



# --- Training function ---
def train_cnn_for_bit(bit_index):

     # Diagnostic for bit balance
    unique, counts = np.unique(Y_train[:, bit_index], return_counts=True)
    print(f"\n[Bit {bit_index}] Label distribution: {dict(zip(unique, counts))}")

    model = create_cnn_model()

    #Training and validation
    Y_train_bit = Y_train[:, bit_index]
    Y_val_bit = Y_val[:, bit_index]
    Y_test_bit = Y_test[:, bit_index]

    early_stopping = EarlyStopping(monitor='val_accuracy', patience=25, restore_best_weights=True)

    reduce_lr = ReduceLROnPlateau(
        monitor='val_accuracy', 
        factor=0.5,         # reduce LR by half
        patience=5,         # wait 5 epochs before reducing
        min_lr=1e-6,        # lower bound on LR
        verbose=1
    )
    
    history = model.fit(
        X_train, Y_train_bit,
        validation_data=(X_val, Y_val_bit),
        epochs=200,
        batch_size=32,
        callbacks=[early_stopping, reduce_lr],
        verbose=1
    )

    # Evaluate the model on the test data
    test_loss, test_acc = model.evaluate(X_test, Y_test_bit)

    print(f"Bit {bit_index} Model Test Accuracy: {test_acc * 100:.2f}%")
    
    # Save model for reuse
    model.save(f'../modelsavepath/bit_model_{bit_index}.keras')
    
    # Return only the test accuracy and the training history
    return history, test_acc


print(f"Y_train shape: {Y_train.shape}, type: {type(Y_train)}")
print(f"Y_train sample row: {Y_train[0]}")
print(f"Number of bits (columns): {Y_train.shape[1]}")

# Step 4: Train the CNN for each bit from 0 to 7 (LSB to MSB)
# Train all bits and store their accuracies & thresholds
bit_accuracies = []
training_accuracies = []
test_accuracies = []

for bit_index in range(Y_train.shape[1]):
    print(f"\nTraining for bit {bit_index}...")
    history, test_acc = train_cnn_for_bit(bit_index)

    bit_accuracies.append(max(history.history['val_accuracy']))
    training_accuracies.append(history.history['accuracy'][-1])
    test_accuracies.append(test_acc)

# Use a single clean color
bar_color = '#4c72b0'

# Step 6: Plot training accuracy as a bar chart
plt.figure(figsize=(10, 6))

# Plot the training accuracy bar chart with customized colors
plt.bar(range(Y_train.shape[1]), test_accuracies, color=bar_color[:len(test_accuracies)], alpha=0.8,edgecolor='black')

# Correct x-axis labeling with desired format ("Bit 0 to 8", "Bit 1 to 7", ..., "Bit 7 to 0")
plt.xticks(range(Y_train.shape[1]), [f'Bit{i}' for i in range(7, -1, -1)], fontsize=11)

# Add the title and axis labels
plt.title("Training Accuracy for Each Bit", fontsize=13)
plt.xlabel("Bit Index (MSB to LSB)", fontsize=12)
plt.ylabel("Training Accuracy", fontsize=12)

# Set y-axis limit to [0, 1] for accuracy range
plt.ylim(0, 1)
plt.grid(axis='y', linestyle='--', alpha=0.6)

#save png
plt.savefig(os.path.join(plot_dir, 'training_accuracy_bar.png'), dpi=300)

# Display the plot
plt.show()

#Plot table for training accuracy
# Step 7: Print training accuracies in a table format
print("\nTraining Accuracy per Bit (bit0 = MSB, bit7 = LSB):")
print("-" * 40)
print(f"{'Bit Index':<10}{'Training Accuracy':>15}")
print("-" * 40)

for i in range(len(training_accuracies)):
    bit_label = f"Bit{7 - i}"  # Reversing index to match MSB to LSB view
    acc = training_accuracies[i]
    print(f"{bit_label:<10}{acc*100:>13.2f}%")

print("-" * 40)


# Step 7: Print test accuracies in a table format
print("\nTest Accuracy per Bit (bit0 = MSB, bit7 = LSB):")
print("-" * 40)
print(f"{'Bit Index':<10}{'Test Accuracy':>15}")
print("-" * 40)

for i in range(len(test_accuracies)):
    bit_label = f"Bit{7 - i}"  # Reversing index to match MSB to LSB view
    acc = test_accuracies[i]
    print(f"{bit_label:<10}{acc*100:>13.2f}%")

print("-" * 40)

#RMSE Calculation
predicted_bits = []
true_bits = []

for bit_index in range(8):
    print(f"Loading trained model for bit {bit_index}...")
    model = load_model(f'../modelsavepath/bit_model_{bit_index}.keras')

    model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])

    Y_test_bit = Y_test[:, bit_index]
    Y_pred_prob = model.predict(X_test)
    Y_pred_bin = (Y_pred_prob > 0.5).astype(np.uint8).flatten()

    predicted_bits.append(Y_pred_bin)
    true_bits.append(Y_test_bit.astype(np.uint8))

# Step: Plot per-bit test accuracy using actual model predictions
bitwise_test_accuracy_from_preds = [
    accuracy_score(true_bits[i], predicted_bits[i]) for i in range(8)
]

# Plotting
bits = [f'Bit{i}' for i in range(7, -1, -1)]  # MSB to LSB

# Use a single clean color
bar_color = '#4c72b0' 

plt.figure(figsize=(10, 6))
plt.bar(range(8), bitwise_test_accuracy_from_preds, color=bar_color, alpha=0.8,edgecolor='black')

# Formatting
plt.xticks(range(8), bits, fontsize=11)
plt.xlabel("Bit Index (MSB to LSB)", fontsize=12)
plt.ylabel("Test Accuracy", fontsize=12)
plt.title("Test Accuracy per Bit (From Model Predictions on Test Set)", fontsize=13)
plt.ylim(0, 1)
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.tight_layout()
#save png
plt.savefig(os.path.join(plot_dir, 'test_accuracy_bar.png'), dpi=300)
plt.show()

# Print table for comparison
print("\nTest Accuracy per Bit (from model predictions):")
print("-" * 40)
print(f"{'Bit Index':<10}{'Accuracy':>15}")
print("-" * 40)

for i in range(8):
    bit_label = f"Bit{7 - i}"  # MSB to LSB
    acc = bitwise_test_accuracy_from_preds[i]
    print(f"{bit_label:<10}{acc*100:>13.2f}%")
print("-" * 40)

# Step 8: Stack bits and convert to integer values
Dout_attacker_bits = np.stack(predicted_bits, axis=1) # bit0 is MSB coming from above code being consistent
Dout_true_bits     = np.stack(true_bits, axis=1)


def bits_to_int(bits_array):
    weights = 2 ** np.arange(bits_array.shape[1]-1, -1, -1)  # [128, ..., 1]
    return np.dot(bits_array, weights)



Dout_attacker = bits_to_int(Dout_attacker_bits)
Dout_true     = bits_to_int(Dout_true_bits)

# Step 9: Compute RMSE and NRMSE ##############################################################

rmse = math.sqrt(mean_squared_error(Dout_true, Dout_attacker))  # <-- FUll code RMSE
print("\n===== Full-Code RMSE Evaluation =====")
print(f"RMSE from predicted vs true 8-bit codes: {rmse:.4f}")

max_adc_code = 2**8
normalized_rmse = rmse / max_adc_code
print(f"Normalized RMSE (for 8-bit ADC): {normalized_rmse:.4f}")

# Step: Bitwise RMSE contributions
bit_weights = 2 ** np.arange(7, -1, -1)  # [128, 64, ..., 1]
bit_errors = Dout_attacker_bits != Dout_true_bits  # Boolean Mask (samples, 8)
bit_error_rates = np.mean(bit_errors, axis=0) #per-bit BER

bitwise_rmse_contrib = np.sqrt(bit_error_rates) * bit_weights
# Calculate normalized bitwise RMSE per bit
normalized_bitwise_rmse = bitwise_rmse_contrib / max_adc_code
total_bitwise_rmse = np.sqrt(np.sum((bit_weights ** 2) * bit_error_rates))

print("\n===== Bitwise RMSE Breakdown (Weighted by Bit Significance) =====")
print(f"{'Bit':<6}{'Bit Error Rate':>16}{'Weight':>10}{'Bitwise RMSE':>18}")
print("-" * 55)
for i in range(8):
    print(f"Bit {7 - i:<2} {bit_error_rates[i]:>16.4f}{bit_weights[i]:>10}{bitwise_rmse_contrib[i]:>18.4f}")
print("-" * 55)
print(f"{'Total Bitwise-weighted RMSE':<30}: {total_bitwise_rmse:.4f}")


# Step 10: Save results
np.savez('../Summary/cnn_predicted_vs_true.npz', Dout_attacker=Dout_attacker, Dout_true=Dout_true)

df_results = pd.DataFrame({
    'Dout_true': Dout_true,
    'Dout_attacker': Dout_attacker
})
df_results.to_csv('../Summary/cnn_results_RMSE_test_set.csv', index=False)

# --- Step 11: Theoretical baselines ---

# Full-code uniform random guessing RMSE (codes 0 to 255)
codes = np.arange(256)
mean_code = np.mean(codes)
rmse_full_code_random = np.sqrt(np.mean((codes - mean_code) ** 2))  # ~73.9 for 0–255
bit_error_rate_random = 0.5
bitwise_rmse_random = np.sqrt(bit_error_rate_random * (bit_weights ** 2))
rmse_bitwise_random = np.sqrt(np.sum(bitwise_rmse_random ** 2))

print("\n===== Theoretical Baselines (for Comparison) =====")
print(f"Full-code random guessing RMSE (0–255 uniform): {rmse_full_code_random:.4f}")
print(f"Bitwise 50% error RMSE (random bits):          {rmse_bitwise_random:.4f}")

print("\n=== Theoretical Random Guessing RMSE ===")
print(f"Full-code uniform random guessing RMSE (0-255): {rmse_full_code_random:.4f}")
print(f"Bitwise random guessing RMSE (50% error per bit): {rmse_bitwise_random:.4f}")

# Linear sum of bitwise RMSE (reference)
rmse_bitwise_linear_sum = np.sum(bitwise_rmse_contrib)
print(f"Linear sum of bitwise RMSE contributions:       {rmse_bitwise_linear_sum:.4f}")


# --- Step 12: Plot bitwise RMSE contributions and total RMSE ---

# Bit labels
bits_labels = [f'Bit {i}' for i in range(7, -1, -1)]
x = np.arange(len(bits_labels))

# === Y-axis: Dynamically adapt upper limit ===
y_max = max(np.max(bitwise_rmse_contrib), total_bitwise_rmse, rmse) * 1.20  # 20% buffer

# Plot
plt.figure(figsize=(10, 6))
plt.bar(x, bitwise_rmse_contrib, color=bar_color, alpha=0.8, edgecolor='black',label='Bitwise RMSE Contribution')

# === Add horizontal lines for reference ===
plt.axhline(y=total_bitwise_rmse, color='orange', linestyle='--', linewidth=2, label=f'Summed Bitwise RMSE = {total_bitwise_rmse:.2f}')
plt.axhline(y=rmse, color='red', linestyle='-.', linewidth=2, label=f'Full-Code RMSE = {rmse:.2f}')

# === Annotate percentage contribution above each bar ===
percent_contrib = 100 * bitwise_rmse_contrib / np.sum(bitwise_rmse_contrib)
for i, pct in enumerate(percent_contrib):
    plt.text(x[i], bitwise_rmse_contrib[i] + y_max * 0.015, f'{pct:.1f}%', ha='center', fontsize=9, color='black')

# Axis config
plt.xticks(x, bits_labels, fontsize=11)
plt.xlabel('Bit Index (MSB to LSB)', fontsize=12)
plt.ylabel('RMSE Contribution', fontsize=12)
plt.title('Per-Bit RMSE Contribution with % and Total RMSE', fontsize=14)
plt.ylim(0, y_max)
plt.grid(axis='y', linestyle='--', alpha=0.6)

# Legend in top-right, always visible
plt.legend(loc='upper right', fontsize=11)

plt.tight_layout()
plt.savefig(os.path.join(plot_dir, 'bitwise_rmse_contrib.png'), dpi=300)
plt.show()

#Plot noramlized bitwise accuracy
bits_labels = [f'Bit {i}' for i in range(7, -1, -1)]
x = np.arange(len(bits_labels))

y_max = max(np.max(normalized_bitwise_rmse), normalized_rmse) * 1.15  # 15% headroom

plt.figure(figsize=(10, 6))

# Bar plot with consistent publication style color and black edges
plt.bar(x, normalized_bitwise_rmse, color=bar_color, alpha=0.85, edgecolor='black', linewidth=0.8)

# Total normalized RMSE line
plt.axhline(normalized_rmse, color='red', linestyle='--', linewidth=1.8,
            label=f'Normalized Total RMSE = {normalized_rmse:.4f}')

# === Annotate percentage contribution above each bar ===
percent_contrib_norm = 100 * normalized_bitwise_rmse / np.sum(normalized_bitwise_rmse)
for i, pct in enumerate(percent_contrib_norm):
    plt.text(x[i], normalized_bitwise_rmse[i] + y_max * 0.01, f'{pct:.1f}%', ha='center', fontsize=9)

# Axis labels & title with consistent font styling
plt.xticks(x, bits_labels, fontsize=11,)
plt.xlabel('Bit Index (MSB to LSB)', fontsize=12,)
plt.ylabel('Normalized Bitwise RMSE Contribution', fontsize=12,)
plt.title('Normalized Bitwise RMSE Contribution per Bit', fontsize=13)

# Set y-limit and grid consistent with publication style
plt.ylim(0, y_max)
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.minorticks_on()
plt.grid(which='minor', axis='y', linestyle=':', alpha=0.25)

# Legend top right consistent style
plt.legend(loc='upper right', fontsize=11, frameon=True, edgecolor='black')

plt.tight_layout()
plt.savefig(os.path.join(plot_dir, 'normalized_bitwise_rmse.png'), dpi=300)
plt.show()


# Step 16: Plot histogram of prediction error
errors = Dout_attacker - Dout_true

plt.figure(figsize=(10, 6))
plt.hist(errors, bins=np.arange(-10, 11, 1), color=bar_color, edgecolor='black', alpha=0.8)

plt.title(r"Histogram of Prediction Errors ($DOUT_{Attacker} - DOUT_{True}$)", fontsize=14)

plt.xlabel("Error (in codes)", fontsize=12)
plt.ylabel("Frequency", fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.axvline(0, color='red', linestyle='--', linewidth=2, label='Zero Error')

plt.legend(loc='upper right', frameon=True, edgecolor='black',fontsize=11)
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, 'prediction_error_histogram.png'), dpi=300)
plt.show()

# Step 17: Plot Bitwise Error Rate and Bitwise RMSE
bits_labels = [f'Bit {i}' for i in range(7, -1, -1)]  # MSB to LSB

fig, ax1 = plt.subplots(figsize=(10, 6))

# --- Left Y-axis: Bit Error Rate ---

bars = ax1.bar(  # <-- capture the bar containers here
    bits_labels, bit_error_rates,
    color=bar_color, alpha=0.8,
    edgecolor='black', linewidth=0.7,
    label='Bit Error Rate'
)

ax1.set_ylabel('Bit Error Rate', fontsize=12, color='black')
ax1.set_ylim(0, 1.05)
ax1.tick_params(axis='y', labelcolor='black')
ax1.grid(axis='y', linestyle='--', alpha=0.6)

# --- Right Y-axis: Per-Bit RMSE ---
ax2 = ax1.twinx()
ax2.plot(
    bits_labels, normalized_bitwise_rmse,
    'o-', color='darkred',
    linewidth=2, markersize=6,
    label='Weighted RMSE'
)
ax2.set_ylabel('Weighted NRMSE (√Error × Weight)', fontsize=12, color='black')
ax2.set_ylim(0, max(normalized_bitwise_rmse) * 1.25)
ax2.tick_params(axis='y', labelcolor='black')

# --- Labels and Title ---
ax1.set_xlabel('Bit Index (MSB to LSB)', fontsize=12)
plt.title('Bitwise Error Rate and Weighted NRMSE per Bit', fontsize=14)

# --- Annotate bars with error %
for i, bar in enumerate(bars):
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width() / 2, height + 0.02,
             f'{bit_error_rates[i]*100:.1f}%', ha='center', va='bottom', fontsize=9)

# --- Combine and Place Legend ---
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
fig.legend(
    lines1 + lines2, labels1 + labels2,
    loc='upper right', bbox_to_anchor=(0.92, 0.92), fontsize=11,
    frameon=True, edgecolor='black'
)

plt.tight_layout()
plt.savefig(os.path.join(plot_dir, 'bit_error_vs_nrmse.png'), dpi=300)
plt.show()

#Step 18: RMSE with theoretical values

categories = [
    'Theoretical RMSE\n(full-code random)',
    'Actual RMSE\n(from predictions)',
    'Bitwise RMSE\n(quadrature sum)'
]

x = np.arange(len(categories))
rmse_values = [rmse_full_code_random, rmse, total_bitwise_rmse]
colors = ['gray', 'blue', 'orange']

fig, ax = plt.subplots(figsize=(10, 6))

# Bar plot for each category
bars = ax.bar(x, rmse_values, color=colors, edgecolor='black')

# Annotate each bar with RMSE value
for i, bar in enumerate(bars):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, height + 2, f"{rmse_values[i]:.2f}",
            ha='center', va='bottom', fontsize=11)

# Add horizontal reference line: Bitwise Random RMSE (50% error)
ax.axhline(rmse_bitwise_random, color='red', linestyle='--', linewidth=1.8,
           label=f'Bitwise Random RMSE (50% error) = {rmse_bitwise_random:.2f}')

# Axis formatting
ax.set_xticks(x)
ax.set_xticklabels(categories, fontsize=11)
ax.set_ylabel('RMSE', fontsize=12)
ax.set_title('RMSE Comparison: Theoretical, Actual, and Bitwise (Quadrature)', fontsize=14)
ax.set_ylim(0, max(rmse_values) * 1.20)
ax.grid(axis='y', linestyle='--', alpha=0.6)

# Save and show
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, 'rmse_comparison_clean.png'), dpi=300)
plt.show()


#Two bar plot
# Use first two categories and corresponding RMSE values
categories_two = categories[:2]
x_two = np.arange(len(categories_two))
values_two = [rmse_full_code_random, rmse]

fig, ax = plt.subplots(figsize=(8, 5))

bars = ax.bar(x_two, values_two, color=bar_color, edgecolor='black', alpha=0.8)

# Annotate bars with values
for i, val in enumerate(values_two):
    ax.text(x_two[i], val + 2, f"{val:.2f}", ha='center', fontsize=11, fontweight='bold')

# Formatting
ax.set_xticks(x_two)
ax.set_xticklabels(categories_two, fontsize=12)
ax.set_ylabel('RMSE', fontsize=12)
ax.set_title('RMSE Comparison: Theoretical vs Actual', fontsize=14)
ax.set_ylim(0, max(values_two) * 1.15)
ax.grid(axis='y', linestyle='--', alpha=0.7)

plt.tight_layout()
plt.savefig(os.path.join(plot_dir, 'rmse_comparison_theoretical_vs_actual.png'), dpi=300)
plt.show()



#additional tests
# Count total bit errors per bit (number of samples with error in each bit)
bit_error_counts = np.sum(bit_errors, axis=0)

bits_labels = [f'Bit {i}' for i in range(7, -1, -1)]
x = np.arange(len(bits_labels))

plt.figure(figsize=(10, 6))
plt.bar(x, bit_error_counts, color=bar_color, alpha=0.85, edgecolor='black')
plt.xticks(x, bits_labels, fontsize=11)
plt.xlabel('Bit Index (MSB to LSB)', fontsize=12)
plt.ylabel('Number of Bit Errors', fontsize=12)
plt.title('Total Bit Error Counts per Bit', fontsize=14)
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, 'total_errors_per_bit.png'), dpi=300)
plt.show()

print("\nBit Error Count Summary (from most to least significant bit):")
print("-" * 45)
print(f"{'Bit':<8}{'Error Count':>15}{'Error Rate (%)':>18}")
print("-" * 45)
for i in range(8):
    bit_index = 7 - i  # MSB to LSB
    count = bit_error_counts[i]
    rate_percent = bit_error_rates[i] * 100
    print(f"Bit {bit_index:<2}{count:>15}{rate_percent:>18.2f}")
print("-" * 45)

bit_errors_int = bit_errors.astype(int)

# Correlation matrix between bits on error occurrence
corr_matrix = np.corrcoef(bit_errors_int, rowvar=False)

plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm',
            xticklabels=bits_labels, yticklabels=bits_labels)
plt.title('Correlation Matrix of Bit Error Occurrence', fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, 'heatmap.png'), dpi=300)
plt.show()

print("\nBit Error Correlation Matrix (rounded to 2 decimals):")
print(pd.DataFrame(np.round(corr_matrix, 2), 
                   columns=[f'Bit {i}' for i in range(7, -1, -1)],
                   index=[f'Bit {i}' for i in range(7, -1, -1)]))


#Save data for comparison

# Ensure bit labels are included
bit_labels = [f'Bit {i}' for i in range(7, -1, -1)]
df_summary = pd.DataFrame(index=bit_labels)

# Core RMSE data
df_summary['Bit_Error_Rate'] = bit_error_rates
df_summary['Bit_Weight'] = bit_weights
df_summary['Bitwise_RMSE_Contribution'] = bitwise_rmse_contrib
df_summary['Normalized_Bitwise_RMSE'] = normalized_bitwise_rmse

# Summary RMSE values
df_summary['Total_RMSE'] = rmse
df_summary['RMSE_From_Bits'] = total_bitwise_rmse
df_summary['NRMSE'] = normalized_rmse

# Accuracy info
df_summary['Train_Accuracy'] = test_accuracies  # (bitwise train accuracy)
df_summary['Test_Accuracy'] = bitwise_test_accuracy_from_preds  # (bitwise test accuracy)

# Bit error stats
df_summary['Bit_Error_Count'] = bit_error_counts
df_summary['Num_Samples'] = len(X_test)


# Save as CSV — distinguish between runs (e.g., secure)
output_dir = '/home/nipun/Desktop/ADC_Security_Char/Data/FD/Unsecure/CNN/Summaries'
os.makedirs(output_dir, exist_ok=True)
df_summary.to_csv(os.path.join(output_dir, 'summary_secure.csv'), index=False)

df_corr_matrix = pd.DataFrame(
    np.round(corr_matrix, 4),
    index=[f'Bit {i}' for i in range(7, -1, -1)],
    columns=[f'Bit {i}' for i in range(7, -1, -1)]
)

df_corr_matrix.to_csv(os.path.join(output_dir, 'bit_error_correlation_secure.csv'))