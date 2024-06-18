import os

import pandas as pd
import matplotlib.pyplot as plt

def plot_comparison(csv_file_path1, csv_file_path2):
    # Leer los archivos CSV
    df1 = pd.read_csv(csv_file_path1)
    df2 = pd.read_csv(csv_file_path2)
    name1 = os.path.splitext(os.path.basename(csv_file_path1))[0]
    name2 = os.path.splitext(os.path.basename(csv_file_path2))[0]


    # Colores para los modelos
    colors = ['blue', 'orange']

    # Crear una figura y ejes para las gráficas
    fig, axes = plt.subplots(4, 1, figsize=(10, 20))

    # Plot de Train y Validation Loss
    for i, (df, label) in enumerate(zip([df1, df2], [name1, name2])):
        axes[0].plot(df['Epoch'], df['Train Loss'], label=f'Train Loss {label}', color=colors[i])
        axes[0].plot(df['Epoch'], df['Val Loss'], label=f'Validation Loss {label}', linestyle='--', color=colors[i])
    axes[0].set_title('Train and Validation Loss Comparison')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()

    # Plot de Train y Validation Dice Score
    for i, (df, label) in enumerate(zip([df1, df2], [name1, name2])):
        axes[1].plot(df['Epoch'], df['Train Dice'], label=f'Train Dice {label}', color=colors[i])
        axes[1].plot(df['Epoch'], df['Val Dice'], label=f'Validation Dice {label}', linestyle='--', color=colors[i])
    axes[1].set_title('Train and Validation Dice Score Comparison')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Dice Score')
    axes[1].legend()

    # Plot de Train y Validation IoU
    for i, (df, label) in enumerate(zip([df1, df2], [name1, name2])):
        axes[2].plot(df['Epoch'], df['Train IoU'], label=f'Train IoU {label}', color=colors[i])
        axes[2].plot(df['Epoch'], df['Val IoU'], label=f'Validation IoU {label}', linestyle='--', color=colors[i])
    axes[2].set_title('Train and Validation IoU Comparison')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('IoU')
    axes[2].legend()

    # Plot de Train y Validation Accuracy
    for i, (df, label) in enumerate(zip([df1, df2], [name1, name2])):
        axes[3].plot(df['Epoch'], df['Train Accuracy'], label=f'Train Accuracy {label}', color=colors[i])
        axes[3].plot(df['Epoch'], df['Val Accuracy'], label=f'Validation Accuracy {label}', linestyle='--', color=colors[i])
    axes[3].set_title('Train and Validation Accuracy Comparison')
    axes[3].set_xlabel('Epoch')
    axes[3].set_ylabel('Accuracy')
    axes[3].legend()

    # Ajustar el layout y mostrar las gráficas
    plt.tight_layout()
    plt.show()


# Ejemplo de uso
csv_file_path2 = '../metrics/custom/10-epochs/CustomUNet1.csv'
csv_file_path1 = '../metrics/custom/10-epochs/CustomUNet2.csv'
plot_comparison(csv_file_path1, csv_file_path2)
