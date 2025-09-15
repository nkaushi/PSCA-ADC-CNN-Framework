# Comparison of SAR and Jun 30 20 LSB Flash SAR faster conversion

import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from tensorflow.keras.models import load_model

# Load summary files
df_unsecure = pd.read_csv('summary_unsecure.csv')
df_secure = pd.read_csv('summary_secure.csv')

print(df_unsecure.columns)
print(df_secure.columns)

#bits = df_unsecure['Bit']
#x = np.arange(len(bits))
num_bits = len(df_unsecure)  # should be 8 if you have 8 bits
bit_indices = np.arange(num_bits)  # 0, 1, ..., 7
bit_labels = [f'Bit {i}' for i in range(7, -1, -1)]
x = np.arange(len(bit_labels))  # x-axis indices: 0 to 7


# Define directory to save plots
plot_dir = '/home/../Comparison'
os.makedirs(plot_dir, exist_ok=True)  # Create directory if it doesn't exist

bar_width = 0.35
unsecure_color = '#d62728'  # Professional red
secure_color = '#2ca02c'    # Professional green
neutral_color = 'gray'      # For theoretical

plt.figure(figsize=(10, 8))

# --- Plot bars with solid colors (alpha=1) ---
plt.bar(x - bar_width/2, df_unsecure['Bitwise_RMSE_Contribution'], bar_width,
        label='Unsecure ADC', color=unsecure_color, edgecolor='black', alpha=1.0,linewidth=1.0)
plt.bar(x + bar_width/2, df_secure['Bitwise_RMSE_Contribution'], bar_width,
        label='Secure ADC', color=secure_color, edgecolor='black', alpha=1.0,linewidth=1.0)

# --- Axis labels and formatting ---
plt.xticks(x, bit_labels, fontsize=20)
plt.yticks(fontsize=20)
plt.xlabel("Bit Index (MSB to LSB)", fontsize=24)
plt.ylabel("RMSE", fontsize=24)
plt.title("Bitwise RMSE Contribution", fontsize=26, fontweight='bold', pad=10)

plt.legend(loc='upper right', fontsize=16)
plt.grid(axis='y', linestyle='--', alpha=0.6,linewidth=1.0)

plt.tight_layout()

# --- Save in both formats ---
plt.savefig(os.path.join(plot_dir, "bitwise_rmse_contribution.png"), dpi=1200, bbox_inches="tight")  # High-DPI PNG
plt.savefig(os.path.join(plot_dir, "bitwise_rmse_contribution.pdf"), bbox_inches="tight")            # Vector PDF

plt.show()


#Plot 2: Normalize RMSE plots
plt.figure(figsize=(10, 8))
#plt.plot(x, y, marker='o', linewidth=2)
plt.bar(x - bar_width/2, df_unsecure['Normalized_Bitwise_RMSE'], 
        width=bar_width, label='Unsecure ADC', color=unsecure_color, edgecolor='black', alpha=1.0,linewidth=1.0)

plt.bar(x + bar_width/2, df_secure['Normalized_Bitwise_RMSE'], 
        width=bar_width, label='Secure ADC', color=secure_color, edgecolor='black', alpha=1.0,linewidth=1.0)

plt.xticks(x, bit_labels, fontsize=20)
plt.yticks(fontsize=20)
plt.xlabel('Bit Index (MSB to LSB)', fontsize=24)
plt.ylabel('Bitwise NRMSE', fontsize=24)
plt.title('NRMSE Contribution', fontsize=26, fontweight='bold',pad=10)

plt.ylim(0, max(df_unsecure['Normalized_Bitwise_RMSE'].max(), 
                 df_secure['Normalized_Bitwise_RMSE'].max()) * 1.15)

plt.grid(axis='y', linestyle='--', alpha=0.6, linewidth=1.0)
plt.minorticks_on()
plt.grid(which='minor', axis='y', linestyle=':', alpha=0.25)

plt.legend(fontsize=16, loc='upper right', frameon=True, edgecolor='black')

plt.tight_layout()
plt.savefig(os.path.join(plot_dir, 'normalized_bitwise_rmse.png'), dpi=1200, bbox_inches="tight")
plt.savefig(os.path.join(plot_dir, "normalized_bitwise_rmse.pdf"), bbox_inches="tight")            # Vector PDF
plt.show()


# Plot 3: Bitwise Error Rate
plt.figure(figsize=(10, 8))
#plt.plot(x, y, marker='o', linewidth=2)
plt.bar(x - bar_width/2, df_unsecure['Bit_Error_Rate'], bar_width,
        label='Unsecure ADC', color=unsecure_color, edgecolor='black', alpha=1.0,linewidth=1.0)
plt.bar(x + bar_width/2, df_secure['Bit_Error_Rate'], bar_width,
        label='Secure ADC', color=secure_color, edgecolor='black', alpha=1.0,linewidth=1.0)

plt.xticks(x, bit_labels, fontsize=20)
plt.yticks(fontsize=20)
plt.xlabel("Bit Index (MSB to LSB)", fontsize=24)
plt.ylabel("Normalized Bit Error Rate", fontsize=24)
plt.title("Normalized Bit Error Rate", fontsize=26, fontweight='bold',pad=10)
plt.legend(fontsize=16)
plt.grid(axis='y', linestyle='--', alpha=0.6, linewidth=1.0)
plt.tight_layout()

plt.savefig(os.path.join(plot_dir, 'normalized_bit_error_rate_comparison.png'), dpi=1200, bbox_inches="tight")
plt.savefig(os.path.join(plot_dir, "normalized_bit_error_rate_comparison.pdf"), bbox_inches="tight")            # Vector PDF
plt.show()

print("\nBitwise Error Rate Comparison (Secure vs Unsecure):")
print("-" * 60)
print(f"{'Bit':<8}{'Unsecure Rate':>15}{'Secure Rate':>15}{'Difference':>15}")
print("-" * 60)

for i in range(8):
    bit = bit_labels[i]
    unsecure_rate = df_unsecure['Bit_Error_Rate'].iloc[i]
    secure_rate = df_secure['Bit_Error_Rate'].iloc[i]
    diff = unsecure_rate - secure_rate
    print(f"{bit:<8}{unsecure_rate:15.5f}{secure_rate:15.5f}{diff:15.5f}")

print("-" * 60)

# Plot 4: Bit predictibilty

# Bit labels from MSB to LSB
bits_labels = [f'Bit {i}' for i in range(7, -1, -1)]
x = np.arange(len(bits_labels))

# Calculate predictability = 1 - Bit_Error_Rate
predict_unsecure = 1 - df_unsecure['Bit_Error_Rate']
predict_secure = 1 - df_secure['Bit_Error_Rate']

plt.figure(figsize=(10, 8))
#plt.plot(x, y, marker='o', linewidth=2)
plt.plot(x, predict_unsecure, marker='o', linestyle='-', color=unsecure_color, label='Unsecure ADC',linewidth=1.0)
plt.plot(x, predict_secure, marker='s', linestyle='--', color=secure_color, label='Secure ADC',linewidth=1.0)

plt.xticks(x, bits_labels, fontsize=20)
plt.yticks(np.linspace(0, 1, 11), fontsize=20)
plt.xlabel('Bit Index (MSB to LSB)', fontsize=24)
plt.ylabel('Bitwise Predictability', fontsize=24)
plt.title('Bitwise Predictability', fontsize=26, fontweight='bold',pad=10)
plt.legend(fontsize=16)
plt.grid(True, linestyle='--', alpha=0.6,linewidth=1.0)

plt.tight_layout()
plt.savefig(os.path.join(plot_dir, 'bitwise_predictability_comparison.png'), dpi=1200, bbox_inches="tight")
plt.savefig(os.path.join(plot_dir, "bitwise_predictability_comparison.pdf"), bbox_inches="tight") 
plt.show()

print("\nBitwise Predictability Comparison (1 - Bit Error Rate):")
print("-" * 60)
print(f"{'Bit':<8}{'Unsecure Predict':>20}{'Secure Predict':>20}{'Difference':>15}")
print("-" * 60)

for i in range(8):
    bit = bits_labels[i]
    unsecure_predict = predict_unsecure.iloc[i]
    secure_predict = predict_secure.iloc[i]
    diff = secure_predict - unsecure_predict
    print(f"{bit:<8}{unsecure_predict:>20.5f}{secure_predict:>20.5f}{diff:>15.5f}")

print("-" * 60)


# Plot 5: RMSE Comparison

# RMSE values
rmse_values = [
    73.9,  # Theoretical
    df_unsecure['Total_RMSE'].iloc[0],
    df_secure['Total_RMSE'].iloc[0]
]
labels = ['Random Guessing', 'Unsecure ADC', 'Secure ADC']
colors = [neutral_color, unsecure_color, secure_color]

# Plotting
plt.figure(figsize=(10, 8))
#plt.plot(x, y, marker='o', linewidth=2)
bars = plt.bar(labels, rmse_values, color=colors, edgecolor='black', width=0.5, alpha=0.1, linewidth=1.0)

# Annotate each bar with bold values
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2, height + 3, f'{height:.2f}',
             ha='center', va='bottom', fontsize=12, fontweight='bold')

# Formatting
plt.ylabel('Total RMSE', fontsize=24)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.title('Total RMSE Comparison', fontsize=26, fontweight='bold',pad=10)
plt.ylim(0, max(rmse_values) * 1.25)
plt.grid(axis='y', linestyle='--', alpha=0.6, linewidth=1.0)

plt.tight_layout()
plt.savefig(os.path.join(plot_dir, 'total_rmse_comparison.png'), dpi=1200,bbox_inches="tight")
plt.savefig(os.path.join(plot_dir, "total_rmse_comparison.pdf"), bbox_inches="tight") 
plt.show()

#Plot 5: Correlation heat maps
# Paths to your saved correlation matrices (adjust paths as needed)
corr_unsecure_path = '/home/../Summaries/bit_error_correlation_secure.csv'
corr_secure_path = '/home/../Summaries/bit_error_correlation_secure.csv'

# Load correlation matrices as DataFrames
corr_unsecure = pd.read_csv(corr_unsecure_path, index_col=0)
corr_secure = pd.read_csv(corr_secure_path, index_col=0)

bit_labels = [f'Bit {i}' for i in range(7, -1, -1)]

# Ensure columns and indices match expected bit labels (optional sanity check)
corr_unsecure.columns = bit_labels
corr_unsecure.index = bit_labels
corr_secure.columns = bit_labels
corr_secure.index = bit_labels

# Plot side-by-side heatmaps for easy comparison
plt.figure(figsize=(14, 7))
#plt.plot(x, y, marker='o', linewidth=2)
# Overall figure title
plt.suptitle('Bit Error Correlation Comparison', fontsize=26, fontweight='bold')

plt.subplot(1, 2, 1)
sns.set_context("notebook", font_scale=1.2)
sns.heatmap(corr_unsecure, annot=True, fmt=".2f", cmap='coolwarm', vmin=-1, vmax=1,
            cbar=True, square=True)
plt.xticks(fontsize=14, rotation=0)
plt.yticks(fontsize=14, rotation=0)
plt.title('Unsecure ADC', fontsize=20)

plt.subplot(1, 2, 2)
sns.set_context("notebook", font_scale=1.2)
sns.heatmap(corr_secure, annot=True, fmt=".2f", cmap='coolwarm', vmin=-1, vmax=1,
            cbar=True, square=True)
plt.xticks(fontsize=14, rotation=0)
plt.yticks(fontsize=14, rotation=0)
plt.title('Secure ADC', fontsize=20)

plt.tight_layout()
plt.subplots_adjust(top=0.88)
plt.savefig(os.path.join(plot_dir, 'bit_error_correlation_comparison.pdf'), dpi=1200)
plt.show()

# Print them nicely
print("\n Unsecure ADC Bit Error Correlation Matrix:")
print(corr_unsecure.round(2))

print("\n Secure ADC Bit Error Correlation Matrix:")
print(corr_secure.round(2))


