import pandas as pd
import matplotlib.pyplot as plt

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

def plot_statistics(csv_file_path):
    # Leer el archivo CSV
    df = pd.read_csv(csv_file_path)

    # Crear una figura y ejes para las gráficas
    fig, axes = plt.subplots(4, 1, figsize=(10, 20))

    # Plot de Train y Validation Loss
    axes[0].plot(df['Epoch'], df['Train Loss'], label='Train Loss')
    axes[0].plot(df['Epoch'], df['Val Loss'], label='Validation Loss')
    axes[0].set_title('Train and Validation Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()

    # Plot de Train y Validation Dice Score
    axes[1].plot(df['Epoch'], df['Train Dice'], label='Train Dice')
    axes[1].plot(df['Epoch'], df['Val Dice'], label='Validation Dice')
    axes[1].set_title('Train and Validation Dice Score')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Dice Score')
    axes[1].legend()

    # Plot de Train y Validation IoU
    axes[2].plot(df['Epoch'], df['Train IoU'], label='Train IoU')
    axes[2].plot(df['Epoch'], df['Val IoU'], label='Validation IoU')
    axes[2].set_title('Train and Validation IoU')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('IoU')
    axes[2].legend()

    # Plot de Train y Validation Accuracy
    axes[3].plot(df['Epoch'], df['Train Accuracy'], label='Train Accuracy')
    axes[3].plot(df['Epoch'], df['Val Accuracy'], label='Validation Accuracy')
    axes[3].set_title('Train and Validation Accuracy')
    axes[3].set_xlabel('Epoch')
    axes[3].set_ylabel('Accuracy')
    axes[3].legend()

    # Ajustar el layout y mostrar las gráficas
    plt.tight_layout()
    plt.show()


# Ejemplo de uso
csv_file_path = '../metrics/custom/10-epochs/CustomUNet2.csv'
plot_statistics(csv_file_path)
