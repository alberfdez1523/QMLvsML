from PIL import Image
import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import time
import os
from ImageRead import *  # Import the same image processing functions used in Script.py
from ScriptQNN import *  # Import the quantum neural network functions

# Check for GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define a classical neural network for galaxy classification
class galaxyClassifier(nn.Module):
    def __init__(self, input_size=256):
        super(galaxyClassifier, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 2)
        )
    
    def forward(self, x):
        return self.model(x)

# Load and process images just like in the quantum script
# Define the labels for the images
target_o = [1 for i in range(30)] + [0 for i in range(30)]

# Define the paths to galaxy and no_galaxy images
pathY = r'./dataset_galaxias/galaxy/'
pathN = r'./dataset_galaxias/no_galaxy/'
nameN = ''
nameY = ''

# Load and resize galaxy and no_galaxy images
inputY = [imageResize(callImage(i + 1, pathY, nameY), 16) for i in range(30)]
inputN = [imageResize(callImage(i + 1, pathN, nameN), 16) for i in range(30)]
input_combine = inputY + inputN

# Randomly shuffle the images
np.random.seed(0)
idx = np.array([int(i) for i in range(60)]).flatten()
np.random.shuffle(idx)

# Prepare the input data and labels
dataInput = list(input_combine[i] for i in idx)
dataTarget = list(imageBinarize(input_combine[i]) for i in idx)
data_target_o = list(target_o[i] for i in idx)

# Normalize the input data
X = [normlaizeData(dataInput[i].flatten()) for i in range(60)]
y01 = [data_target_o[i] for i in range(60)]

# Convert to PyTorch tensors
X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
y_tensor = torch.tensor(y01, dtype=torch.long).to(device)

# Show some input images
n_samples_show = 10
fig, axes = plt.subplots(nrows=2, ncols=n_samples_show, figsize=(20, 6))
for i in range(n_samples_show):
    axes[0, i].set_title(f"{'galaxy' if data_target_o[i] == 1 else 'no_galaxy'}")
    axes[0, i].imshow(dataInput[i], cmap='gray')
    axes[0, i].set_xticks([])
    axes[0, i].set_yticks([])
    axes[1, i].set_title(f"{'galaxy' if data_target_o[i + n_samples_show] == 1 else 'no_galaxy'}")
    axes[1, i].imshow(dataInput[i + n_samples_show], cmap='gray')
    axes[1, i].set_xticks([])
    axes[1, i].set_yticks([])
plt.show()

# Load test dataset
target_o_test = [1 for i in range(30)] + [0 for i in range(30)]

# Define the paths for test images
pathY = r'./dataset_galaxias/galaxy1/'
pathN = r'./dataset_galaxias/no_galaxy1/'
inputY = [imageResize(callImage(i + 1, pathY, nameY), 16) for i in range(30)]
inputN = [imageResize(callImage(i + 1, pathN, nameN), 16) for i in range(30)]
input_combine_test = inputY + inputN

# Shuffle test data
np.random.seed(0)
idx_test = np.array([int(i) for i in range(60)]).flatten()
np.random.shuffle(idx_test)

# Prepare test data
dataInput_test = list(input_combine_test[i] for i in idx_test)
data_target_o_test = list(target_o_test[i] for i in idx_test)

# Test with 10, 25, and 50 data points
Xtest = [torch.tensor(normlaizeData(dataInput_test[i].flatten()), dtype=torch.float32, device=device) for i in range(10)]
y01test = [data_target_o_test[i] for i in range(10)]

Xtest25 = [torch.tensor(normlaizeData(dataInput_test[i].flatten()), dtype=torch.float32, device=device) for i in range(25)]
y01test25 = [data_target_o_test[i] for i in range(25)]

Xtest60 = [torch.tensor(normlaizeData(dataInput_test[i].flatten()), dtype=torch.float32, device=device) for i in range(60)]
y01test60 = [data_target_o_test[i] for i in range(60)]

# Convert test data to batched tensors
X_test_tensor = torch.stack(Xtest).to(device)
y_test_tensor = torch.tensor(y01test, dtype=torch.long).to(device)

X_test25_tensor = torch.stack(Xtest25).to(device)
y_test25_tensor = torch.tensor(y01test25, dtype=torch.long).to(device)

X_test60_tensor = torch.stack(Xtest60).to(device)
y_test60_tensor = torch.tensor(y01test60, dtype=torch.long).to(device)

# Define the model path
model_path = "classic_model.pth"

# Check if the model already exists
if os.path.exists(model_path):
    print("Loading existing model...")
    checkpoint = torch.load(model_path, map_location=device)
    model = galaxyClassifier(input_size=256)
    model.to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer = optim.AdamW(model.parameters(), lr=0.001)
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    print("Model loaded successfully.")
else:
    print("No saved model found. Training a new model...")
    model = galaxyClassifier(input_size=256)
    model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=0.001)

# Define loss function and scheduler
criterion = nn.CrossEntropyLoss()
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)

# Calculate initial model accuracy
with torch.no_grad():
    outputs = model(X_tensor)
    _, predicted = torch.max(outputs, 1)
    initial_accuracy = (predicted == y_tensor).sum().item() / len(y_tensor)
    print(f'Initial Accuracy: {initial_accuracy:.4f}')

# Train the model if it doesn't exist
if not os.path.exists(model_path):
    # Training parameters
    num_epochs = 100
    batch_size = 8
    early_stopping_patience = 50
    best_accuracy = 0
    no_improve_epochs = 0
    
    # Lists to track metrics
    train_losses = []
    train_accuracies = []
    test_accuracies = []
    
    print("Training the model...")
    for epoch in range(num_epochs):
        model.train()
        start_time = time.time()
        
        # Shuffle indices for this epoch
        indices = torch.randperm(len(X_tensor))
        
        epoch_loss = 0.0
        correct = 0
        total = 0
        
        # Process in batches
        for i in range(0, len(X_tensor), batch_size):
            batch_indices = indices[i:i+batch_size]
            inputs = X_tensor[batch_indices]
            labels = y_tensor[batch_indices]
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # Backward and optimize
            loss.backward()
            optimizer.step()
            
            # Track metrics
            epoch_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        # Calculate training metrics
        avg_loss = epoch_loss / (len(X_tensor) / batch_size)
        train_accuracy = correct / total
        train_losses.append(avg_loss)
        train_accuracies.append(train_accuracy)
        
        # Evaluate on test data
        model.eval()
        with torch.no_grad():
            test_outputs = model(X_test60_tensor)
            _, test_predicted = torch.max(test_outputs, 1)
            test_accuracy = (test_predicted == y_test60_tensor).sum().item() / len(y_test60_tensor)
            test_accuracies.append(test_accuracy)
        
        # Update learning rate
        scheduler.step(avg_loss)
        
        end_time = time.time()
        print(f'Epoch {epoch+1}/{num_epochs}, '
              f'Loss: {avg_loss:.4f}, '
              f'Train Accuracy: {train_accuracy:.4f}, '
              f'Test Accuracy: {test_accuracy:.4f}, '
              f'Time: {end_time - start_time:.2f}s, '
              f'LR: {optimizer.param_groups[0]["lr"]:.6f}')
        
        # Early stopping check
        if test_accuracy > best_accuracy:
            best_accuracy = test_accuracy
            no_improve_epochs = 0
            # Save the best model
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()
            }, model_path)
            print(f"Saved improved model with accuracy: {best_accuracy:.4f}")
        else:
            no_improve_epochs += 1
            
        if no_improve_epochs >= early_stopping_patience:
            print(f"Early stopping triggered after {epoch+1} epochs")
            break
            
        # Stop if we reach perfect accuracy
        if train_accuracy >= 1 and test_accuracy >= 1:
            print(f"High accuracy reached. Stopping training.")
            break
    
    # Plot training metrics
    plt.figure(figsize=(18, 5))
    plt.subplot(1, 3, 1)
    plt.plot(train_losses)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    
    plt.subplot(1, 3, 2)
    plt.plot(train_accuracies)
    plt.title('Training Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    
    plt.subplot(1, 3, 3)
    plt.plot(test_accuracies)
    plt.title('Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    
    plt.tight_layout()
    plt.savefig('classic_training_metrics.png')
    plt.show()

# Load the best model for evaluation
best_checkpoint = torch.load(model_path, map_location=device)
model.load_state_dict(best_checkpoint['model_state_dict'])
model.eval()

# Evaluate on different test sets
with torch.no_grad():
    # Test on 10 samples
    outputs = model(X_test_tensor)
    _, predicted = torch.max(outputs, 1)
    accuracy_10 = (predicted == y_test_tensor).sum().item() / len(y_test_tensor)
    print(f'Accuracy on 10 test samples: {accuracy_10:.4f}')
    
    # Visualize predictions on 10 samples
    fig, axes = plt.subplots(nrows=2, ncols=5, figsize=(20, 6))
    indices10 = np.random.choice(range(10), 10, replace=False)
    predicted_np = predicted.cpu().numpy()
    
    for i, idx in enumerate(indices10):
        ax_row = i // 5
        ax_col = i % 5
        axes[ax_row, ax_col].imshow(dataInput_test[idx], cmap='gray')
        axes[ax_row, ax_col].set_xticks([])
        axes[ax_row, ax_col].set_yticks([])
        actual = "galaxy" if y01test[idx] == 1 else "no_galaxy"
        prediction = "galaxy" if predicted_np[idx] == 1 else "no_galaxy"
        color = "green" if y01test[idx] == predicted_np[idx] else "red"
        axes[ax_row, ax_col].set_title(f"A: {actual}\nP: {prediction}", color=color)
    
    plt.tight_layout()
    plt.savefig('classic_predictions_10.png')
    plt.show()
    
    # Test on 25 samples
    outputs = model(X_test25_tensor)
    _, predicted = torch.max(outputs, 1)
    accuracy_25 = (predicted == y_test25_tensor).sum().item() / len(y_test25_tensor)
    print(f'Accuracy on 25 test samples: {accuracy_25:.4f}')
    
    # Test on 60 samples
    outputs = model(X_test60_tensor)
    _, predicted = torch.max(outputs, 1)
    accuracy_60 = (predicted == y_test60_tensor).sum().item() / len(y_test60_tensor)
    print(f'Accuracy on 60 test samples: {accuracy_60:.4f}')
    
    # Visualize predictions on 60 samples
    fig, axes = plt.subplots(nrows=2, ncols=5, figsize=(20, 6))
    indices60 = np.random.choice(range(60), 10, replace=False)
    predicted_np = predicted.cpu().numpy()
    
    for i, idx in enumerate(indices60):
        ax_row = i // 5
        ax_col = i % 5
        axes[ax_row, ax_col].imshow(dataInput_test[idx], cmap='gray')
        axes[ax_row, ax_col].set_xticks([])
        axes[ax_row, ax_col].set_yticks([])
        actual = "galaxy" if y01test60[idx] == 1 else "no_galaxy"
        prediction = "galaxy" if predicted_np[idx] == 1 else "no_galaxy"
        color = "green" if y01test60[idx] == predicted_np[idx] else "red"
        axes[ax_row, ax_col].set_title(f"A: {actual}\nP: {prediction}", color=color)
    
    plt.tight_layout()
    plt.savefig('classic_predictions_50.png')
    plt.show()

# GUI for galaxy detection
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk

class galaxyDetectorGUI:
    def __init__(self, model):
        self.root = tk.Tk()
        self.root.title("galaxy Detector - Classical ML")
        self.root.geometry("400x500")
        self.model = model

        # Button to select image
        self.select_button = tk.Button(self.root, text="Select Image", command=self.select_image)
        self.select_button.pack(pady=20)

        # Label to display the image
        self.image_label = tk.Label(self.root)
        self.image_label.pack(pady=10)

        # Label to display the result
        self.result_label = tk.Label(self.root, text="", wraplength=350)
        self.result_label.pack(pady=10)

    def select_image(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("Image files", "*.jpg *.jpeg *.png")]
        )
        if file_path:
            self.process_image(file_path)

    def process_image(self, file_path):
        try:
            # Display the image
            image = Image.open(file_path)
            image.thumbnail((200, 200))
            photo = ImageTk.PhotoImage(image)
            self.image_label.config(image=photo)
            self.image_label.image = photo

            # Process the image for prediction
            image_gray = image.convert('L')
            image_resized = imageResize(np.array(image_gray), 16)
            image_processed = normlaizeData(image_resized.flatten())
            input_tensor = torch.tensor(image_processed, dtype=torch.float32).to(device)

            # Make prediction
            with torch.no_grad():
                output = self.model(input_tensor)
                probabilities = torch.nn.functional.softmax(output, dim=0)
                prediction = torch.argmax(output).item()
                confidence = probabilities[prediction].item() * 100

            # Display result
            result = "galaxy" if prediction == 1 else "no_galaxy"
            self.result_label.config(
                text=f"Prediction: {result}\nConfidence: {confidence:.2f}%"
            )

        except Exception as e:
            self.result_label.config(text=f"Error processing image: {str(e)}")

    def run(self):
        self.root.mainloop()

# Run the GUI if this script is executed directly
if __name__ == "__main__":
    app = galaxyDetectorGUI(model)
    app.run()