# Define the hyperparameters
import torch


# Load the dataset and create the data loader
train_data_dir = 'datasets/dataset/train'
val_data_dir = 'datasets/dataset/valid'
image_dir = 'images'
mask_dir = 'masks'



batch_size = 16
learning_rate = 0.001
num_epochs = 10
stats_dir = f'metrics/custom/{num_epochs}-epochs'

# Define the loss function and optimizer
criterion = torch.nn.CrossEntropyLoss()


def optimizer(model_params):
    return torch.optim.Adam(model_params, lr=learning_rate)
'''
# Fisrts

batch_size = 16
learning_rate = 0.001
num_epochs = 5
stats_dir = 'metrics/5-epochs'

# Define the loss function and optimizer
criterion = torch.nn.CrossEntropyLoss()


def optimizer(model_params):
    return torch.optim.Adam(model_params, lr=learning_rate)
'''

'''

# Second
batch_size = 32
learning_rate = 0.01
num_epochs = 5
stats_dir = 'metrics/second'

# Define the loss function and optimizer
criterion = torch.nn.CrossEntropyLoss()


def optimizer(model_params):
    return torch.optim.SGD(model_params, lr=learning_rate, momentum=0.9)

'''
'''

# Third
batch_size = 8
learning_rate = 0.001
num_epochs = 5
stats_dir = 'metrics/third'

# Define the loss function and optimizer
criterion = torch.nn.CrossEntropyLoss()


def optimizer(model_params):
    return torch.optim.AdamW(model_params, lr=learning_rate, weight_decay=1e-4)

'''

num_classes = 5