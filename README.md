### Baggage Material Classification using Deep Learning

This project aims to classify airport baggage based on material types such as hard plastic, metal, soft plastic, wood, cardboard, and more. Using a preprocessed dataset of baggage images, a deep learning model (ResNet50) is fine-tuned for this task, leveraging PyTorch.

The workflow involves:

- **Data Loading & Augmentation**: The dataset is loaded using `torchvision.datasets.ImageFolder` and enhanced with extensive data augmentation (random cropping, rotation, color jittering, etc.) to improve generalization.
  
- **Model Architecture**: The project employs ResNet50, a pre-trained model from ImageNet. The last few layers are unfrozen for fine-tuning to the specific baggage classification task. The final fully connected layer is replaced with a layer that matches the number of baggage material classes.

- **Training Process**: The model is trained using Adam optimizer with a cosine annealing learning rate scheduler. Early stopping is incorporated to prevent overfitting.

- **Evaluation**: Randomly selected images from the dataset are used to evaluate the model’s performance. The images are displayed with true and predicted labels for easy visualization.

- **Automatic Labeling**: A custom script is provided to classify unlabeled baggage images and move them to their respective class folders.

- **Model Inference**: Trained model weights are saved, and an evaluation function allows for predictions on new images, displaying both the image and the predicted label.

