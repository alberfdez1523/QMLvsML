import pennylane as qml
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import timeit
import os

# Define paths for saving models
CQ_MODEL_PATH = 'model_cq_lightyears.pt'
QC_MODEL_PATH = 'model_qc_lightyears.pt'
CLASSICAL_MODEL_PATH = 'model_classical_lightyears.pt'

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)

# Define quantum device
# n_qubits = 4
# dev = qml.device("lightning.qubit", wires=n_qubits)

from qiskit_ibm_runtime import QiskitRuntimeService

QiskitRuntimeService.save_account(channel="ibm_quantum", token="5a6fe119a908514054c633bc58c4ee44e3cd0c5794d7e8ff83db14b6ca6497b95dc73721bdfa48430518af50f38fd1599882bb093daf46a41128aa4c469dd7b7", overwrite=True)

# To access saved credentials for the IBM quantum channel and select an instance
service = QiskitRuntimeService(channel="ibm_quantum", instance="ibm-q/open/main")
backend = service.least_busy(operational=True, simulator=False, min_num_qubits=100)
n_qubits = 10
# Specify which qubits to use from the backend
dev = qml.device('qiskit.remote', 
                wires=127,  # Use first n_qubits of backend
                backend=backend, 
                shots=1024)

# Define quantum circuit as a QNode
@qml.qnode(dev)
def quantum_circuit(inputs, weights):
    # Embed classical data into quantum circuit
    qml.templates.AngleEmbedding(inputs, wires=range(n_qubits))

    # Apply parameterized quantum layers
    qml.templates.StronglyEntanglingLayers(weights, wires=range(n_qubits))

    # Return measurements from all qubits
    return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

# Simplified classical preprocessor based on ClassicalModel
class ClassicalPreprocessor(nn.Module):
    def __init__(self, input_size, output_size):
        super(ClassicalPreprocessor, self).__init__()
        self.layer1 = nn.Linear(input_size, output_size)
        
    def forward(self, x):
        return self.layer1(x)

# Simplified classical postprocessor based on ClassicalModel
class ClassicalPostprocessor(nn.Module):
    def __init__(self, input_size, output_size=1):
        super(ClassicalPostprocessor, self).__init__()
        self.layer1 = nn.Linear(input_size, output_size)
        
    def forward(self, x):
        return self.layer1(x)

# Define hybrid model (Classical-Quantum)
class HybridModelCQ(nn.Module):
    def __init__(self, input_size, hidden_size, n_qubits, n_layers):
        super(HybridModelCQ, self).__init__()
        self.n_qubits = n_qubits

        # Classical preprocessor
        self.classical_preprocessor = ClassicalPreprocessor(input_size, n_qubits)

        # Quantum part - modify initialization to avoid broadcasting issues
        self.q_weights = nn.Parameter(0.01 * torch.randn(n_layers, n_qubits, 3))

        # Post-processing
        self.post_processing = ClassicalPostprocessor(n_qubits, 1)

    def forward(self, x):
        # Process one sample at a time to avoid broadcasting issues
        batch_size = x.shape[0]
        outputs = []

        for i in range(batch_size):
            # Classical preprocessing
            x_i = self.classical_preprocessor(x[i:i+1])

            # Quantum processing - handle one sample at a time
            q_out = quantum_circuit(x_i[0], self.q_weights)
            q_out_tensor = torch.tensor(q_out, dtype=torch.float32)

            # Final regression
            out = self.post_processing(q_out_tensor.unsqueeze(0))
            outputs.append(out)

        return torch.cat(outputs, dim=0)

# Define hybrid model (Quantum-Classical)
class HybridModelQC(nn.Module):
    def __init__(self, input_size, hidden_size, n_qubits, n_layers):
        super(HybridModelQC, self).__init__()
        self.input_size = input_size
        self.n_qubits = n_qubits

        # Quantum part - inicialización para evitar problemas de broadcasting
        self.q_weights = nn.Parameter(0.01 * torch.randn(n_layers, n_qubits, 3))

        # Procesador clásico posterior
        self.classical_postprocessor = ClassicalPostprocessor(n_qubits, 1)

    def forward(self, x):
        # Procesar una muestra a la vez para evitar problemas de broadcasting
        batch_size = x.shape[0]
        outputs = []

        for i in range(batch_size):
            # Embedding de datos: tomar solo los primeros n_qubits o rellenar con ceros
            x_embedded = x[i:i+1, :self.n_qubits] if x.shape[1] >= self.n_qubits else \
                torch.cat([x[i:i+1], torch.zeros(1, self.n_qubits - x.shape[1])], dim=1)

            # Procesamiento cuántico - manejar una muestra a la vez
            q_out = quantum_circuit(x_embedded[0], self.q_weights)
            q_out_tensor = torch.tensor(q_out, dtype=torch.float32)

            # Procesamiento clásico posterior
            out = self.classical_postprocessor(q_out_tensor.unsqueeze(0))
            outputs.append(out)

        return torch.cat(outputs, dim=0)
    
# Add classical model for comparison
class ClassicalModel(nn.Module):
    def __init__(self):
        super(ClassicalModel, self).__init__()
        # Initialize with random weights - the model should learn km to light-years conversion
        self.layer1 = nn.Linear(1, 1)
        
    def forward(self, x):
        return self.layer1(x)

# Generate distance conversion data
def generate_distance_data(n_samples, input_features=1):
    # Generate random kilometers between 0 and 1e16
    kilometers = np.random.uniform(0, 1e16, (n_samples, 1))
    
    # Convert to light-years: lightyears = km / 9.461e12
    lightyears = kilometers / 9.461e12
    
    # Create feature vector - include kilometers and other features if needed
    if input_features == 1:
        X = kilometers
    else:
        # Add additional random features for testing model capabilities
        X = np.zeros((n_samples, input_features))
        X[:, 0] = kilometers.flatten()  # First feature is actual distance
        # Add random noise features
        for i in range(1, input_features):
            X[:, i] = np.random.uniform(0, 1, n_samples)
    
    # Normalize features to prevent numerical issues
    X_normalized = X / 1e16
    
    return torch.tensor(X_normalized, dtype=torch.float32), torch.tensor(lightyears, dtype=torch.float32)

def train_and_save_models(force_retrain=False):
    """Train models and save them, or load if they already exist."""
    # Common parameters for all models
    input_size = 8  # Kilometers distance + additional features
    hidden_size_cq = 8  # Not used with simplified models but kept for compatibility
    hidden_size_qc = 8  # Not used with simplified models but kept for compatibility
    n_layers = 2

    epochs = 2
    
    # Initialize models
    model_cq = HybridModelCQ(input_size, hidden_size_cq, n_qubits, n_layers)
    model_qc = HybridModelQC(input_size, hidden_size_qc, n_qubits, n_layers)
    model_classical = ClassicalModel()
    
    # Check if models already exist and load them if they do
    if (not force_retrain and 
        os.path.exists(CQ_MODEL_PATH) and 
        os.path.exists(QC_MODEL_PATH) and
        os.path.exists(CLASSICAL_MODEL_PATH)):
        print("Loading existing models...")
        model_cq.load_state_dict(torch.load(CQ_MODEL_PATH))
        model_qc.load_state_dict(torch.load(QC_MODEL_PATH))
        model_classical.load_state_dict(torch.load(CLASSICAL_MODEL_PATH))
        print("Models loaded successfully!")
    else:
        print("Training new models...")
        # Generate data for training
        X_train, y_train = generate_distance_data(2, input_features=input_size)
        
        # Extract first column for classical model
        X_train_classical = X_train[:, 0:1]  # This keeps it as a 2D tensor
        
        # Use MSE loss for regression
        loss_fn = nn.MSELoss()
        
        # Training Classical-Quantum model
        print("Training Classical-Quantum Model:")
        start_time_cq = timeit.default_timer()
        
        # Use Adam optimizer for better convergence
        optimizer_cq = optim.Adam(model_cq.parameters(), lr=1)
        
        # Add learning rate scheduler
        scheduler_cq = optim.lr_scheduler.ReduceLROnPlateau(optimizer_cq, 'min', patience=50, factor=0.5)
        
        cq_train_losses = []
        for epoch in range(epochs):  
            # Forward pass
            y_pred = model_cq(X_train)
            loss = loss_fn(y_pred, y_train)
            
            # Backward pass and optimization
            optimizer_cq.zero_grad()
            loss.backward()
            optimizer_cq.step()
            
            # Update learning rate based on loss
            scheduler_cq.step(loss)
            
            # Save loss for potential plotting
            cq_train_losses.append(loss.item())
            
            if (epoch + 1) % 1 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.6f}")
                
                # Add early stopping if loss is very small
                if loss.item() < 1e-7:
                    print(f"Early stopping at epoch {epoch+1} with loss {loss.item():.8f}")
                    break
        
        end_time_cq = timeit.default_timer()
        print(f"Training Classical-Quantum Model took {end_time_cq - start_time_cq:.2f} seconds")
        
        # Training Quantum-Classical model
        print("\nTraining Quantum-Classical Model:")
        start_time_qc = timeit.default_timer()
        
        # Use Adam optimizer for better convergence
        optimizer_qc = optim.Adam(model_qc.parameters(), lr=1)
        
        # Add learning rate scheduler
        scheduler_qc = optim.lr_scheduler.ReduceLROnPlateau(optimizer_qc, 'min', patience=50, factor=0.5)
        
        qc_train_losses = []
        
        for epoch in range(epochs):
            # Forward pass
            y_pred = model_qc(X_train)
            loss = loss_fn(y_pred, y_train)
            
            # Backward pass and optimization
            optimizer_qc.zero_grad()
            loss.backward()
            optimizer_qc.step()
            
            # Update learning rate based on loss
            scheduler_qc.step(loss)
            
            # Save loss for potential plotting
            qc_train_losses.append(loss.item())
            
            if (epoch + 1) % 1 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.6f}")
                
                # Add early stopping if loss is very small
                if loss.item() < 1e-7:
                    print(f"Early stopping at epoch {epoch+1} with loss {loss.item():.8f}")
                    break
        
        end_time_qc = timeit.default_timer()
        print(f"Training Quantum-Classical Model took {end_time_qc - start_time_qc:.2f} seconds")
        
        # Training Classical model
        print("\nTraining Classical Model:")
        start_time_classical = timeit.default_timer()
        
        # Use Adam optimizer for better convergence
        optimizer_classical = optim.SGD(model_classical.parameters(), lr=1)
        
        # Add learning rate scheduler with verbose reporting
        scheduler_classical = optim.lr_scheduler.ReduceLROnPlateau(optimizer_classical, 'min', patience=50, factor=0.5)
        
        classical_train_losses = []
        for epoch in range(epochs):  
            # Forward pass - use only the first feature (distance)
            y_pred = model_classical(X_train_classical)
            loss = loss_fn(y_pred, y_train)
            
            # Backward pass and optimization
            optimizer_classical.zero_grad()
            loss.backward()
            optimizer_classical.step()
            
            # Update learning rate based on loss
            scheduler_classical.step(loss)
            
            # Save loss for potential plotting
            classical_train_losses.append(loss.item())
            
            if (epoch + 1) % 1 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.6f}")
                
                # Add early stopping if loss is very small
                if loss.item() < 1e-7:
                    print(f"Early stopping at epoch {epoch+1} with loss {loss.item():.8f}")
                    break
        
        end_time_classical = timeit.default_timer()
        print(f"Training Classical Model took {end_time_classical - start_time_classical:.2f} seconds")
        
        # Save models after training
        print("Saving models...")
        torch.save(model_cq.state_dict(), CQ_MODEL_PATH)
        torch.save(model_qc.state_dict(), QC_MODEL_PATH)
        torch.save(model_classical.state_dict(), CLASSICAL_MODEL_PATH)
        print("Models saved successfully!")
    
    return model_cq, model_qc, model_classical, input_size

def evaluate_models(model_cq, model_qc, model_classical, input_size):
    """Evaluate and compare the models."""
    # Generate test data
    X_test, y_test = generate_distance_data(2, input_features=input_size)
    
    # Extract first column for classical model
    X_test_classical = X_test[:, 0:1]  # Keep as 2D tensor but only first column
    
    with torch.no_grad():
        # Classical-Quantum model
        y_pred_cq = model_cq(X_test)
        mse_cq = nn.functional.mse_loss(y_pred_cq, y_test)
        mae_cq = nn.functional.l1_loss(y_pred_cq, y_test)
        
        # Define accuracy as predictions within ±threshold light-years of true value
        threshold = 1
        accuracy_cq = (torch.abs(y_pred_cq - y_test) < threshold).float().mean()
        
        print(f"\nClassical-Quantum Model Test MSE: {mse_cq.item():.8f}")
        print(f"Classical-Quantum Model Test MAE: {mae_cq.item():.8f}")
        print(f"Classical-Quantum Model Accuracy (+-{threshold} margin): {accuracy_cq.item()*100:.2f}%")
    
        # Quantum-Classical model
        y_pred_qc = model_qc(X_test)
        mse_qc = nn.functional.mse_loss(y_pred_qc, y_test)
        mae_qc = nn.functional.l1_loss(y_pred_qc, y_test)
        
        accuracy_qc = (torch.abs(y_pred_qc - y_test) < threshold).float().mean()
        
        print(f"Quantum-Classical Model Test MSE: {mse_qc.item():.8f}")
        print(f"Quantum-Classical Model Test MAE: {mae_qc.item():.8f}")
        print(f"Quantum-Classical Model Accuracy (+-{threshold} margin): {accuracy_qc.item()*100:.2f}%")
    
        # Classical model - use only distance feature
        y_pred_classical = model_classical(X_test_classical)
        mse_classical = nn.functional.mse_loss(y_pred_classical, y_test)
        mae_classical = nn.functional.l1_loss(y_pred_classical, y_test)
        
        accuracy_classical = (torch.abs(y_pred_classical - y_test) < threshold).float().mean()
        
        print(f"Classical Model Test MSE: {mse_classical.item():.8f}")
        print(f"Classical Model Test MAE: {mae_classical.item():.8f}")
        print(f"Classical Model Accuracy (+-{threshold} margin): {accuracy_classical.item()*100:.2f}%")
    
        print("\nModel Comparison:")
        best_accuracy = max(accuracy_cq.item(), accuracy_qc.item(), accuracy_classical.item())
        if accuracy_cq.item() == best_accuracy:
            print(f"Classical-Quantum model performs best with {accuracy_cq.item()*100:.2f}% accuracy")
        elif accuracy_qc.item() == best_accuracy:
            print(f"Quantum-Classical model performs best with {accuracy_qc.item()*100:.2f}% accuracy")
        else:
            print(f"Classical model performs best with {accuracy_classical.item()*100:.2f}% accuracy")
        
        # Convert predictions and labels to numpy arrays
        y_test_np = y_test.numpy().flatten()
        y_pred_cq_np = y_pred_cq.numpy().flatten()
        y_pred_qc_np = y_pred_qc.numpy().flatten()
        y_pred_classical_np = y_pred_classical.numpy().flatten()
        
        # Get the original kilometer distances (denormalize)
        kilometers_test = X_test[:, 0].numpy() * 1e16
        
        # Create figure with multiple subplots
        plt.figure(figsize=(15, 10))
        
        # Plot 1: Actual vs Predicted Distances
        plt.subplot(2, 2, 1)
        plt.scatter(kilometers_test, y_test_np, color='black', label='True', marker='o')
        plt.scatter(kilometers_test, y_pred_cq_np, color='blue', label='CQ Model', marker='x')
        plt.scatter(kilometers_test, y_pred_qc_np, color='red', label='QC Model', marker='+')
        plt.scatter(kilometers_test, y_pred_classical_np, color='green', label='Classical', marker='*')
        plt.xlabel('Distance (km)')
        plt.ylabel('Distance (light-years)')
        plt.title('Distance Conversion: True vs Predicted')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot 2: Prediction Error Distribution
        plt.subplot(2, 2, 2)
        errors_cq = y_pred_cq_np - y_test_np
        errors_qc = y_pred_qc_np - y_test_np
        errors_classical = y_pred_classical_np - y_test_np
        plt.hist(errors_cq, bins=10, alpha=0.5, label='CQ Error', color='blue')
        plt.hist(errors_qc, bins=10, alpha=0.5, label='QC Error', color='red')
        plt.hist(errors_classical, bins=10, alpha=0.5, label='Classical Error', color='green')
        plt.axvline(x=0, color='black', linestyle='--', label='Zero Error')
        plt.xlabel('Prediction Error (light-years)')
        plt.ylabel('Count')
        plt.title('Error Distribution')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot 3: Data Distribution
        plt.subplot(2, 2, 3)
        plt.hist(kilometers_test, bins=20, alpha=0.5, label='Kilometers', color='black')
        plt.xlabel('Distance (km)')
        plt.ylabel('Count') 
        plt.title('Data Distribution')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Plot 4: Model Comparison
        plt.subplot(2, 2, 4)
        x = np.arange(3)
        heights = [accuracy_cq.item()*100, accuracy_qc.item()*100, accuracy_classical.item()*100]
        plt.bar(x, heights, color=['blue', 'red', 'green'])
        plt.xticks(x, ['CQ Model', 'QC Model', 'Classical'])
        plt.ylabel('Accuracy (%)')
        plt.title(f'Model Accuracy (±{threshold} light-years)')
        for i, v in enumerate(heights):
            plt.text(i, v+1, f"{v:.2f}%", ha='center')
        plt.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig('distance_lightyear_conversion_comparison.png')
        plt.show()
        
        # Print detailed prediction comparison for a few samples
        print("\nDetailed Distance Prediction Comparison:")
        print("{:<5} | {:<15} | {:<15} | {:<15} | {:<15} | {:<15} | {:<10} | {:<10} | {:<10}".format(
            "Index", "km (1e13)", "True LY", "CQ Pred", "QC Pred", "Classical", "CQ Acc?", "QC Acc?", "Class Acc?"))
        print("-" * 125)
        
        for i in range(min(20, len(y_test_np))):
            km_dist = float(kilometers_test[i]) / 1e13  # Display in scientific notation
            true_ly = float(y_test_np[i])
            cq_pred = float(y_pred_cq_np[i])
            qc_pred = float(y_pred_qc_np[i])
            classical_pred = float(y_pred_classical_np[i])
            
            cq_accurate = "Yes" if abs(true_ly - cq_pred) < threshold else "No"
            qc_accurate = "Yes" if abs(true_ly - qc_pred) < threshold else "No"
            classical_accurate = "Yes" if abs(true_ly - classical_pred) < threshold else "No"
            
            print("{:<5} | {:<15.4f} | {:<15.8f} | {:<15.8f} | {:<15.8f} | {:<15.8f} | {:<10} | {:<10} | {:<10}".format(
                i, km_dist, true_ly, cq_pred, qc_pred, classical_pred, cq_accurate, qc_accurate, classical_accurate))
        
        # Calculate and show overall accuracy
        cq_accurate_count = sum(1 for i in range(len(y_test_np)) if abs(y_test_np[i] - y_pred_cq_np[i]) < threshold)
        qc_accurate_count = sum(1 for i in range(len(y_test_np)) if abs(y_test_np[i] - y_pred_qc_np[i]) < threshold)
        classical_accurate_count = sum(1 for i in range(len(y_test_np)) if abs(y_test_np[i] - y_pred_classical_np[i]) < threshold)
        
        print(f"\nCQ Model: {cq_accurate_count}/{len(y_test_np)} predictions within +-{threshold} margin ({accuracy_cq.item()*100:.2f}%)")
        print(f"QC Model: {qc_accurate_count}/{len(y_test_np)} predictions within +-{threshold} margin ({accuracy_qc.item()*100:.2f}%)")
        print(f"Classical Model: {classical_accurate_count}/{len(y_test_np)} predictions within +-{threshold} margin ({accuracy_classical.item()*100:.2f}%)")
    
        # Show exact conversion formula for reference
        print("\nReference: light-years = kilometers / 9.461e12")

if __name__ == "__main__":
    # Parse command-line arguments
    import sys
    force_retrain = "--retrain" in sys.argv
    
    # Train or load models
    model_cq, model_qc, model_classical, input_size = train_and_save_models(force_retrain=force_retrain)
    
    # Evaluate models
    evaluate_models(model_cq, model_qc, model_classical, input_size)