import pandas as pd
import matplotlib.pyplot as plt
import glob
import os

folder_path = 'data/metrics'  # Change this to your folder path
csv_files = glob.glob(os.path.join(folder_path, '*.csv'))

print(f"Found {len(csv_files)} CSV files:")
for f in csv_files:
    print(f"  - {os.path.basename(f)}")

# Read and merge all CSV files
dfs = []
for file in csv_files:
    temp_df = pd.read_csv(file)
    # Optional: add filename as a column to track source
    temp_df['source_file'] = os.path.basename(file)
    dfs.append(temp_df)

# Merge all dataframes
df = pd.concat(dfs, ignore_index=True)

print(f"\nMerged DataFrame shape: {df.shape}")
print(f"Columns: {list(df.columns)}")
print("\nFirst few rows:")
print(df.head())

# Read the CSV data
# df = pd.read_csv('./data/metrics/action_probs_20251027_211354_main.csv')
# df = pd.read_csv('./data/metrics/action_probs_20251027_211354_2025-01-02_2025-10-23.csv')
# df = pd.read_csv('./data/metrics/action_probs_20251027_211354_2024-01-02_2024-12-31.csv')

# Create figure with subplots
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1. Line plot - All probabilities over time
ax1 = axes[0, 0]
ax1.plot(df['step'], df['prob_hold'], label='Hold', linewidth=2, alpha=0.8)
ax1.plot(df['step'], df['prob_buy'], label='Buy', linewidth=2, alpha=0.8)
ax1.plot(df['step'], df['prob_sell'], label='Sell', linewidth=2, alpha=0.8)
ax1.set_xlabel('Step', fontsize=11)
ax1.set_ylabel('Probability', fontsize=11)
ax1.set_title('Probability Distributions Over Time', fontsize=13, fontweight='bold')
ax1.legend(fontsize=10)
ax1.grid(alpha=0.3)

# 2. Stacked area chart
ax2 = axes[0, 1]
ax2.fill_between(df['step'], 0, df['prob_buy'], label='Buy', alpha=0.7)
ax2.fill_between(df['step'], df['prob_buy'], df['prob_buy'] + df['prob_hold'], 
                 label='Hold', alpha=0.7)
ax2.fill_between(df['step'], df['prob_buy'] + df['prob_hold'], 1, 
                 label='Sell', alpha=0.7)
ax2.set_xlabel('Step', fontsize=11)
ax2.set_ylabel('Cumulative Probability', fontsize=11)
ax2.set_title('Stacked Probability Distribution', fontsize=13, fontweight='bold')
ax2.legend(fontsize=10)
ax2.grid(alpha=0.3)

# 3. Box plots - Distribution summary
ax3 = axes[1, 0]
box_data = [df['prob_hold'], df['prob_buy'], df['prob_sell']]
bp = ax3.boxplot(box_data, labels=['Hold', 'Buy', 'Sell'], patch_artist=True)
for patch, color in zip(bp['boxes'], ['#ff9999', '#66b3ff', '#99ff99']):
    patch.set_facecolor(color)
ax3.set_ylabel('Probability', fontsize=11)
ax3.set_title('Probability Distribution Summary', fontsize=13, fontweight='bold')
ax3.grid(alpha=0.3, axis='y')

# 4. Heatmap-style visualization
ax4 = axes[1, 1]
prob_matrix = df[['prob_hold', 'prob_buy', 'prob_sell']].T
im = ax4.imshow(prob_matrix, aspect='auto', cmap='YlOrRd', interpolation='nearest')
ax4.set_yticks([0, 1, 2])
ax4.set_yticklabels(['Hold', 'Buy', 'Sell'])
ax4.set_xlabel('Step', fontsize=11)
ax4.set_title('Probability Heatmap', fontsize=13, fontweight='bold')
plt.colorbar(im, ax=ax4, label='Probability')

plt.tight_layout()
plt.show()

# Print summary statistics
print("\n=== Probability Statistics ===")
print(df[['prob_hold', 'prob_buy', 'prob_sell']].describe())
print("\n=== Dominant Action per Step ===")
print(f"Hold dominates: {(df['prob_hold'] > df[['prob_buy', 'prob_sell']].max(axis=1)).sum()} steps")
print(f"Buy dominates: {(df['prob_buy'] > df[['prob_hold', 'prob_sell']].max(axis=1)).sum()} steps")
print(f"Sell dominates: {(df['prob_sell'] > df[['prob_hold', 'prob_buy']].max(axis=1)).sum()} steps")