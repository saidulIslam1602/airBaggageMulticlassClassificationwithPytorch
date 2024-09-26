import os
import shutil
import cv2
import matplotlib.pyplot as plt

# Enable interactive mode for non-blocking display
plt.ion()

# Function to display an image using OpenCV and Matplotlib
def display_image(image_path):
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error loading image {image_path}")
        return False  # Return False if image loading fails
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(img_rgb)
    plt.axis('off')
    plt.draw()  # Draw the plot without blocking
    plt.pause(1)  # Keep the image on screen for 1 second
    return True  # Return True if image was displayed successfully

# Function to label images and move them into corresponding class folders
def label_images(image_folder, output_folder, classes):
    os.makedirs(output_folder, exist_ok=True)

    for class_name in classes:
        class_path = os.path.join(output_folder, class_name)
        os.makedirs(class_path, exist_ok=True)

    for image_file in os.listdir(image_folder):
        if image_file.lower().endswith(('png', 'jpg', 'jpeg')):
            image_path = os.path.join(image_folder, image_file)

            # Check if the image was displayed successfully
            print(f"Displaying image: {image_file}")
            if not display_image(image_path):
                continue  # Skip if image cannot be displayed

            # Debugging statement to ensure program flow
            print("Preparing to ask for label input...")

            # Prompt the user for a label
            label = input(f"Enter label for {image_file} from {classes}: ")

            if label in classes:
                # Move image to corresponding class folder
                dest_folder = os.path.join(output_folder, label)
                shutil.move(image_path, os.path.join(dest_folder, image_file))
                print(f"Moved {image_file} to {label} folder.")
            else:
                print(f"Invalid label! Skipping {image_file}.")

            # Option to break early
            if input("Do you want to continue labeling? (y/n): ").lower() != 'y':
                break

# Define image folder and output folder
image_folder = r"C:\Users\simon\Desktop\baggageData\dataset\data"  # Your unlabelled images path
output_folder = r"C:\Users\simon\Desktop\baggageData\dataset\labeled_data"  # Path for labeled images

# Define class names
classes = ['soft plastic', 'hard plastic', 'metal', 'wood', 'cardboard', 'other material']  # Add any other class names as needed

# Call the labeling function
label_images(image_folder, output_folder, classes)
