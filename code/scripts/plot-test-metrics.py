import pandas as pd
import matplotlib.pyplot as plt

# Read data from CSV
df = pd.read_csv('../metrics/test/test-CustomUNet2-10E.csv')

# Extract model names and metrics
models = df['Model']
test_loss = df['Test Loss']
test_dice = df['Test Dice']
test_iou = df['Test IoU']
test_accuracy = df['Test Accuracy']

# Set the statistics as x-axis masks
statistics = ['Test Loss', 'Test Dice', 'Test IoU', 'Test Accuracy']

# Plot
plt.figure(figsize=(10, 6))

# Plot each model for each statistic with different colors
for model in models:
    plt.plot(statistics, df.loc[df['Model'] == model].iloc[0][1:], marker='o', label=model)

plt.title('Model Evaluation Metrics CustomUNet2')
plt.xlabel('Statistics')
plt.ylabel('Score')
plt.xticks(rotation=45)
plt.legend()
plt.grid(True)
plt.tight_layout()

plt.show()
