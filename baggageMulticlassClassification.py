# -*- coding: utf-8 -*-
"""assignmentPart2a.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1z30V7ATJ_tQAPauBGYXoaYObNM--qgB2
"""

from google.colab import files
uploaded = files.upload()

!apt-get install unrar
!unrar x /content/labeled_data.rar /content/extracted_files/

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, models, transforms
import time
import copy
import os
import random
from PIL import Image
import matplotlib.pyplot as plt

# Check for available device (GPU if available)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Directory containing images organized by class (replace with your data path)
data_dir = '/content/extracted_files/labeled_data'

# Improved Data Augmentation
data_transforms = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(45),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.2),
    transforms.RandomAffine(degrees=0, translate=(0.2, 0.2)),
    transforms.RandomGrayscale(p=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Load the entire dataset
full_dataset = datasets.ImageFolder(data_dir, transform=data_transforms)

# Calculate the split (80% training, 20% validation)
train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size

# Split the dataset into train and validation sets
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

# Create DataLoaders for both train and validation sets
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)

# Define the dataloaders dictionary
dataloaders = {
    'train': train_loader,
    'val': val_loader
}

# Store dataset sizes and class names for later use
dataset_sizes = {
    'train': len(train_dataset),
    'val': len(val_dataset)
}
class_names = full_dataset.classes

# Load pre-trained ResNet model and unfreeze layers for fine-tuning
model_ft = models.resnet50(pretrained=True)
for name, param in model_ft.named_parameters():
    if "layer3" in name or "layer4" in name or "fc" in name:  # Unfreeze last two blocks and FC layer
        param.requires_grad = True
    else:
        param.requires_grad = False

# Replace the final fully connected layer with the number of classes
num_ftrs = model_ft.fc.in_features
model_ft.fc = nn.Sequential(
    nn.Dropout(0.5),  # Add dropout to prevent overfitting
    nn.Linear(num_ftrs, len(class_names))
)
model_ft = model_ft.to(device)

# Define the loss function and optimizer with a smaller learning rate
criterion = nn.CrossEntropyLoss()
optimizer_ft = optim.Adam(filter(lambda p: p.requires_grad, model_ft.parameters()), lr=1e-4)
scheduler = lr_scheduler.CosineAnnealingLR(optimizer_ft, T_max=10)

# Function to train the model
def train_model(model, criterion, optimizer, scheduler, num_epochs=25, patience=10):
    best_acc = 0.0
    epochs_no_improve = 0
    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Training mode
            else:
                model.eval()  # Evaluation mode
            running_loss = 0.0
            running_corrects = 0
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            if phase == 'train':
                scheduler.step()
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            print(f'{phase} Loss: {running_loss / dataset_sizes[phase]:.4f} Acc: {epoch_acc:.4f}')
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                epochs_no_improve = 0  # Reset patience
                best_model_wts = copy.deepcopy(model.state_dict())
            else:
                if phase == 'val':
                    epochs_no_improve += 1
        if epochs_no_improve >= patience:
            print(f"Early stopping at epoch {epoch}")
            break
    model.load_state_dict(best_model_wts)
    return model

# Train the model
model_ft = train_model(model_ft, criterion, optimizer_ft, scheduler, num_epochs=50)

# Save the model
torch.save(model_ft.state_dict(), 'baggage_material_classifier.pth')


import matplotlib.pyplot as plt
import random
from PIL import Image
import os

# Randomly select multiple images from the labeled dataset for inference
def get_random_images(data_dir, num_images=10):
    image_paths = []
    true_classes = []

    class_folders = os.listdir(data_dir)

    for _ in range(num_images):
        selected_class = random.choice(class_folders)  # Randomly choose a class folder
        class_path = os.path.join(data_dir, selected_class)

        image_file = random.choice(os.listdir(class_path))  # Randomly choose an image from that folder
        image_path = os.path.join(class_path, image_file)

        image_paths.append(image_path)
        true_classes.append(selected_class)  # Keep track of the true class

    return image_paths, true_classes

# Perform inference on a single image
def predict_image(image_path, model):
    image = Image.open(image_path)
    image_tensor = data_transforms(image).unsqueeze(0)  # Apply transforms and add batch dimension
    image_tensor = image_tensor.to(device)

    with torch.no_grad():
        outputs = model(image_tensor)
        _, preds = torch.max(outputs, 1)

    return class_names[preds.item()], image  # Also return the original image for display

# Evaluate model on multiple images and show the actual image along with predictions
def evaluate_on_random_images(model, num_images=10):
    image_paths, true_classes = get_random_images(data_dir, num_images=num_images)
    correct_predictions = 0

    for i, image_path in enumerate(image_paths):
        predicted_class, image = predict_image(image_path, model)
        true_class = true_classes[i]

        plt.imshow(image)  # Display the image
        plt.axis('off')  # Hide axis for better display

        # Show the predicted and true class labels on the image
        plt.title(f"True: {true_class} | Predicted: {predicted_class}")
        plt.show()

        if predicted_class == true_class:
            correct_predictions += 1

    accuracy = correct_predictions / num_images * 100
    print(f'Accuracy on {num_images} random images: {accuracy:.2f}%')

# Test the model on 10 random images and display them
evaluate_on_random_images(model_ft, num_images=10)

