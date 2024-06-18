from torchvision import models

import train_custom_models
import train_torch_models
from custom_unet_models import CustomUNet2
from hyperparams import *
from datasetseg import *

train_dataset = SegmentationDataset(train_data_dir, image_dir, mask_dir)
train_data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

val_dataset = SegmentationDataset(val_data_dir, image_dir, mask_dir)
val_data_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

# Load the models architecture
model = models.segmentation.deeplabv3_resnet50(pretrained=False, num_classes=num_classes)
# model = CustomUNet2(num_classes=num_classes)


# Train the models
train_custom_models.train_model(model, train_data_loader, val_data_loader, criterion, optimizer(model.parameters()), num_epochs,
                   num_classes, stats_dir)

# Save the trained models, if needed
# torch.save(model.state_dict(), 'models/model.pth')
