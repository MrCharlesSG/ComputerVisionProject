import pandas as pd
import matplotlib.pyplot as plt


def plot_selected_models():
    # Read data from CSV files
    df_fcn = pd.read_csv('../metrics/test/test-fcn-resnet-Adam-10E-Full.csv')
    df_unet = pd.read_csv('../metrics/test/test-CustomUNet2-10E.csv')
    df_segnet = pd.read_csv('../metrics/test/test-CustomSegNet2.csv')
    # Extract specific models

    selected_model_segnet = df_segnet[df_segnet['Model'] == 'model-6']
    selected_model_fcn = df_fcn[df_fcn['Model'] == 'model-1']
    selected_model_unet = df_unet[df_unet['Model'] == 'model-8']
    selected_model_unet2 = df_unet[df_unet['Model'] == 'model-9']

    # Combine the selected models into one DataFrame
    combined_df = pd.concat([selected_model_segnet, selected_model_fcn, selected_model_unet, selected_model_unet2 ])

    # Extract model names and metrics
    models = combined_df['Model']
    test_loss = combined_df['Test Loss']
    test_dice = combined_df['Test Dice']
    test_iou = combined_df['Test IoU']
    test_accuracy = combined_df['Test Accuracy']

    # Set the statistics as x-axis masks
    statistics = ['Test Loss', 'Test Dice', 'Test IoU', 'Test Accuracy']

    # Plot
    plt.figure(figsize=(10, 6))

    # Plot each model for each statistic with different colors
    for model in models:
        plt.plot(statistics, combined_df.loc[combined_df['Model'] == model].iloc[0][1:], marker='o', label=model)

    plt.title('Selected Model Evaluation Metrics')
    plt.xlabel('Statistics')
    plt.ylabel('Score')
    plt.xticks(rotation=45)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plt.show()


# Call the function to plot the data
plot_selected_models()
