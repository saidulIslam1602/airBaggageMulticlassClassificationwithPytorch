import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from PIL import Image
import shutil

# Path to labeled subset dataset (organized into class folders)
data_dir = r"C:\Users\simon\Desktop\baggageData\dataset\labeled_data"

# Define transformations for  dataset
data_transforms = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize all images to 224x224
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # ImageNet normalization
])

# Load the labeled dataset using ImageFolder
train_dataset = datasets.ImageFolder(root=data_dir, transform=data_transforms)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Define the number of classes (number of folders in dataset)
num_classes = len(train_dataset.classes)

# Load a pre-trained ResNet50 model
model = models.resnet50(pretrained=True)

# Freeze the pre-trained layers so we only train the final layers
for param in model.parameters():
    param.requires_grad = False

# Replace the final fully connected layer to match your number of classes
model.fc = nn.Linear(model.fc.in_features, num_classes)

# Move the model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.fc.parameters(), lr=0.001)

# Train the model for a few epochs
epochs = 5
for epoch in range(epochs):
    running_loss = 0.0
    model.train()

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # Backward pass and optimize
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f'Epoch [{epoch + 1}/{epochs}], Loss: {running_loss / len(train_loader)}')

print("Training complete.")

# Save the model for future use
torch.save(model.state_dict(), 'trained_resnet50.pth')

# Load and preprocess image for prediction
def predict_image(image_path, model, transform):
    image = Image.open(image_path)
    image = transform(image).unsqueeze(0)  # Add batch dimension

    # Move image to device (GPU/CPU)
    image = image.to(device)

    # Make prediction
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():
        output = model(image)
        _, predicted_class = torch.max(output, 1)

    return predicted_class.item()

# Path to folder with unlabeled images
unlabeled_dir = r"C:\Users\simon\Desktop\baggageData\dataset\data"

# Load the list of image filenames from the unlabeled folder
unlabeled_images = [img for img in os.listdir(unlabeled_dir) if img.endswith(('.jpg', '.png', '.jpeg'))]

# Predict labels for each image and move them to the appropriate class folder
for img_filename in unlabeled_images:
    image_path = os.path.join(unlabeled_dir, img_filename)

    # Predict the class label
    predicted_class_index = predict_image(image_path, model, data_transforms)

    # Get the class name corresponding to the predicted label
    predicted_class = train_dataset.classes[predicted_class_index]

    # Construct the destination folder path
    dest_folder = os.path.join(data_dir, predicted_class)
    os.makedirs(dest_folder, exist_ok=True)  # Ensure class folder exists

    # Move the image to the predicted class folder
    dest_path = os.path.join(dest_folder, img_filename)
    shutil.move(image_path, dest_path)

print("Labelling complete.")
