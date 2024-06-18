from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score
from mapcalc import calculate_map
from torchmetrics.classification import Dice
from torchmetrics.detection import IntersectionOverUnion

from metrics_functions import dice_score, iou, pixel_accuracy
from utils import create_directory, create_csv_file_for_training, write_csv_stats_for_training

import torch
import numpy as np


def train(model, train_loader, optimizer, criterion, device, epoch, num_epochs):
    model.train()
    total_loss = 0
    total_dice = 0
    total_accuracy = 0
    total_iou = 0
    print(f'Start Training epoch {epoch}')
    for i, (images, masks) in enumerate(train_loader):
        images, masks = images.to(device), masks.to(device).long()

        masks_squeezed = masks.squeeze(1)
        optimizer.zero_grad()

        outputs = model(images)['out']
        loss = criterion(outputs, masks_squeezed)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        # Compute Dice Score
        predicted_masks = torch.argmax(outputs, dim=1).float()
        dice = dice_score(masks_squeezed, predicted_masks)
        iou_train = iou(masks_squeezed, predicted_masks)
        accuracy_train = pixel_accuracy(masks_squeezed, predicted_masks)
        total_dice += dice
        total_iou += iou_train
        total_accuracy += accuracy_train
        print(f"Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(train_loader)}], Loss: {total_loss:.4f}"
              f" Dice: {total_dice}, IoU: {total_iou}, Accuracy: {total_accuracy}")

    return total_loss / len(train_loader), total_dice / len(train_loader), total_iou / len(
        train_loader), total_accuracy / len(train_loader)


def evaluate(model, val_loader, criterion, num_classes, device, epoch, num_epochs):
    model.eval()
    total_loss = 0
    total_dice = 0
    total_accuracy = 0
    total_iou = 0

    print(f'Start Evaluation epoch {epoch}')
    with torch.no_grad():
        for i, (images, masks) in enumerate(val_loader):
            images, masks = images.to(device), masks.to(device).long()
            masks_squeezed = masks.squeeze(1)
            outputs = model(images)['out']
            loss = criterion(outputs, masks_squeezed)
            total_loss += loss.item()

            # Compute Dice Score
            predicted_masks = torch.argmax(outputs, dim=1)
            dice = dice_score(masks_squeezed, predicted_masks)
            iou_train = iou(masks_squeezed, predicted_masks)
            accuracy_train = pixel_accuracy(masks_squeezed, predicted_masks)

            total_dice += dice
            total_iou += iou_train
            total_accuracy += accuracy_train
            print(f"Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(val_loader)}], Loss: {total_loss:.4f}"
                  f" Dice: {total_dice}, IoU: {total_iou}, Accuracy: {total_accuracy}")

    return total_loss / len(val_loader), total_dice / len(val_loader), total_iou / len(
        val_loader), total_accuracy / len(val_loader)


def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, num_classes, stats_dir='stats'):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Starting Training")
    model.to(device)
    create_directory(stats_dir)
    csv_file_path = create_csv_file_for_training(stats_dir)
    for epoch in range(num_epochs):
        print(f"Starting Epoch {epoch + 1}, Time: {datetime.now()}")
        train_loss, train_dice, train_iou, train_accuracy = train(model, train_loader, optimizer, criterion, device,
                                                                  epoch, num_epochs)
        val_loss, val_dice, val_iou, val_accuracy = evaluate(model, val_loader, criterion, num_classes, device, epoch,
                                                             num_epochs)

        print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
              f"Train Dice: {train_dice:.4f}, Train IoU: {train_iou:.4f}, Train Accuracy: {train_accuracy:4f} "
              f"Val Dice: {val_dice:.4f}, Val IoU: {val_iou:.4f}, Val Accuracy: {val_accuracy:.4f}")

        write_csv_stats_for_training(csv_file_path, epoch + 1, train_loss, train_dice, train_iou, train_accuracy,
                                     val_loss, val_dice, val_iou, val_accuracy)

        torch.save(model.state_dict(), f'models/final2/model-{epoch}.pth')

    print(datetime.now())
