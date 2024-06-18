import pandas as pd
import matplotlib.pyplot as plt

def plot_comparison(csv_file_paths, model_names):
    # Colores para los modelos
    colors = ['blue', 'orange', 'green', 'red']

    # Leer los archivos CSV y añadir el nombre del modelo
    dfs = []
    for i, (csv_file_path, model_name) in enumerate(zip(csv_file_paths, model_names)):
        df = pd.read_csv(csv_file_path)
        df['Model'] = model_name
        df['Color'] = colors[i]
        dfs.append(df)

    # Crear una figura y ejes para las gráficas
    fig, axes = plt.subplots(4, 1, figsize=(10, 20))

    # Plot de Train y Validation Loss
    for df in dfs:
        axes[0].plot(df['Epoch'], df['Train Loss'], label=f'Train Loss {df["Model"][0]}', color=df['Color'][0])
        axes[0].plot(df['Epoch'], df['Val Loss'], label=f'Validation Loss {df["Model"][0]}', linestyle='--', color=df['Color'][0])
    axes[0].set_title('Train and Validation Loss Comparison')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()

    # Plot de Train y Validation Dice
    for df in dfs:
        axes[1].plot(df['Epoch'], df['Train Dice'], label=f'Train Dice {df["Model"][0]}', color=df['Color'][0])
        axes[1].plot(df['Epoch'], df['Val Dice'], label=f'Validation Dice {df["Model"][0]}', linestyle='--', color=df['Color'][0])
    axes[1].set_title('Train and Validation Dice Score Comparison')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Dice Score')
    axes[1].legend()

    # Plot de Train y Validation IoU
    for df in dfs:
        axes[2].plot(df['Epoch'], df['Train IoU'], label=f'Train IoU {df["Model"][0]}', color=df['Color'][0])
        axes[2].plot(df['Epoch'], df['Val IoU'], label=f'Validation IoU {df["Model"][0]}', linestyle='--', color=df['Color'][0])
    axes[2].set_title('Train and Validation IoU Comparison')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('IoU')
    axes[2].legend()

    # Plot de Train y Validation Accuracy
    for df in dfs:
        axes[3].plot(df['Epoch'], df['Train Accuracy'], label=f'Train Accuracy {df["Model"][0]}', color=df['Color'][0])
        axes[3].plot(df['Epoch'], df['Val Accuracy'], label=f'Validation Accuracy {df["Model"][0]}', linestyle='--', color=df['Color'][0])
    axes[3].set_title('Train and Validation Accuracy Comparison')
    axes[3].set_xlabel('Epoch')
    axes[3].set_ylabel('Accuracy')
    axes[3].legend()

    # Ajustar el layout y mostrar las gráficas
    plt.tight_layout()
    plt.show()

# Paths to CSV files and their respective model names
csv_file_paths = [
    'metrics/second/fcn_resnet50-SGD-5e.csv',
    'metrics/second/deeplabv3_resnet50-SGD-5e.csv',
    'metrics/first/fcn_resnet50-Adam-5e.csv',
    'metrics/first/deeplabv3_resnet50-Adam-5e.csv'
]

model_names = [
    'fcn_resnet50-5e-SGD',
    'deeplabv3_resnet50-5e-SGD',
    'fcn_resnet50-5e-Adam',
    'deeplabv3_resnet50-5e-Adam'
]

plot_comparison(csv_file_paths, model_names)
