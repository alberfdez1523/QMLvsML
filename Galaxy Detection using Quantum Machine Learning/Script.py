from PIL import Image
import torch
import numpy as np
import matplotlib.pyplot as plt
from qiskit.circuit import Parameter
from qiskit_machine_learning.neural_networks import SamplerQNN
from qiskit_machine_learning.connectors import TorchConnector
from qiskit_machine_learning.gradients import ParamShiftSamplerGradient
from torch import Tensor
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim import SGD
from torch.nn import CrossEntropyLoss
import numpy as np
import matplotlib.pyplot as plt
from ImageRead import *  
from ScriptQNN import *  
import time
import os
from qiskit.primitives import StatevectorSampler


# Define las etiquetas objetivo para las imágenes
target_o = [1 for i in range(30)] + [0 for i in range(30)]

# Define las rutas de las imágenes de numbers y no numbers
pathY = r'./dataset_numbers/one/'
pathN = r'./dataset_numbers/nine/'
nameN = ''
nameY = ''

# Carga y redimensiona las imágenes de numbers y no numbers
inputY = [imageResize(callImage(i + 1, pathY, nameY), 16) for i in range(30)]
inputN = [imageResize(callImage(i + 1, pathN, nameN), 16) for i in range(30)]
input_combine = inputY + inputN

# Mezcla las imágenes de manera aleatoria
np.random.seed(0)
idx = np.array([int(i) for i in range(60)]).flatten()
np.random.shuffle(idx)

# Prepara los datos de entrada y las etiquetas
dataInput = list(input_combine[i] for i in idx)
dataTarget = list(imageBinarize(input_combine[i]) for i in idx)
data_target_o = list(target_o[i] for i in idx)

# Normaliza los datos de entrada
X = [normlaizeData(dataInput[i].flatten()) for i in range(60)]
y01 = [data_target_o[i] for i in range(60)]

# Muestra algunas imágenes de entrada
n_samples_show = 10
fig, axes = plt.subplots(nrows=2, ncols=n_samples_show, figsize=(20, 6))
for i in range(n_samples_show):
    axes[0, i].set_title(f"{'one' if data_target_o[i] == 1 else 'nine'}")
    axes[0, i].imshow(dataInput[i], cmap='gray')
    axes[0, i].set_xticks([])
    axes[0, i].set_yticks([])
    axes[1, i].set_title(f"{'one' if data_target_o[i + n_samples_show] == 1 else 'nine'}")
    axes[1, i].imshow(dataInput[i + n_samples_show], cmap='gray')
    axes[1, i].set_xticks([])
    axes[1, i].set_yticks([])
plt.savefig('Imagenes.png')
plt.show()

# Define las etiquetas objetivo para las imágenes
target_o = [1 for i in range(30)] + [0 for i in range(30)]

# Define las rutas de las imágenes de numbers y no numbers
pathY = r'./dataset_numbers/one1/'
pathN = r'./dataset_numbers/nine1/'
nameN = ''
nameY = ''
inputY = [imageResize(callImage(i + 1, pathY, nameY), 16) for i in range(30)]
inputN = [imageResize(callImage(i + 1, pathN, nameN), 16) for i in range(30)]
input_combine = inputY + inputN

# Mezcla las imágenes de manera aleatoria
np.random.seed(0)
idx = np.array([int(i) for i in range(60)]).flatten()
np.random.shuffle(idx)

# Prepara los datos de entrada y las etiquetas
dataInput = list(input_combine[i] for i in idx)
dataTarget = list(imageBinarize(input_combine[i]) for i in idx)
data_target_o = list(target_o[i] for i in idx)

# Test con 10, 25 y 50 puntos de datos
Xtest = [torch.tensor(normlaizeData(dataInput[i].flatten())) for i in range(10)]
y01test = [data_target_o[i] for i in range(10)]

Xtest25 = [torch.tensor(normlaizeData(dataInput[i].flatten())) for i in range(25)]
y01test25 = [data_target_o[i] for i in range(25)]

Xtest60 = [torch.tensor(normlaizeData(dataInput[i].flatten())) for i in range(60)]
y01test60 = [data_target_o[i] for i in range(60)]

# Configuración del modelo cuántico
np.random.seed(3)
nqubits = 6
num_inputs = 256

qc = QuantumCircuit(nqubits)


# Encoding
param_x=[];
for i in range(num_inputs):
    param_x.append(Parameter('x'+str(i)))
for i in range(8):
    param_x.append(np.pi/2)


feature_map = encoding(qc,param_x,22)


# Optimzing circuit PQC
param_y=[];
for i in range(nqubits*2):
    param_y.append(Parameter('θ'+str(i)))

ansatz=circuit15(qc,param_y)

qc.append(feature_map, range(nqubits))
qc.append(ansatz, range(nqubits))


sampler = StatevectorSampler()

gradient = ParamShiftSamplerGradient(sampler=sampler)

# Modificar la creación de QNN para usar el nuevo sampler
qnn2 = SamplerQNN(
    circuit=qc,
    input_params=feature_map.parameters,
    weight_params=ansatz.parameters,
    interpret=parity,
    output_shape=2,
    sampler=sampler,
    gradient=gradient,
)

# Inicialización del modelo
initial_weights = 0.1 * (2 * np.random.rand(qnn2.num_weights) - 1)

# Definición del nombre del archivo donde se guardará el modelo
model_path = "model.pth"

# Verificación si el modelo ya existe
if os.path.exists(model_path):
    print("Cargando el modelo existente...")
    checkpoint = torch.load(model_path)
    model2 = TorchConnector(qnn2)
    model2.load_state_dict(checkpoint['model_state_dict'])
    optimizer = SGD(model2.parameters(), lr=0.05)
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    print("Modelo cargado exitosamente.")
else:
    print("No se encontró un modelo guardado. Entrenando un nuevo modelo...")
    model2 = TorchConnector(qnn2, initial_weights) # Mover modelo a GPU
    # Inicializar el optimizador Adam
    optimizer = SGD(model2.parameters(), lr=0.05)
    

scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
# Define la función de pérdida
f_loss = CrossEntropyLoss()

best_accuracy_train = 0
best_accuracy_test = 0
best_model_state = None
best_optimizer_state = None

# Calcula la precisión del modelo
y_predict = []
for x in X:
    output = model2(Tensor(x))
    y_predict += [np.argmax(output.cpu().detach().numpy())]

print('Accuracy:', sum(y_predict == np.array(y01)) / len(np.array(y01)))

# Si el modelo no existía, entrena y guarda el modelo
if not os.path.exists(model_path):
    # Entrenamiento del modelo
    Losses = []
    Accuracies_train = []
    Acurracies_test = []

    # Define la función de cierre para el optimizador
    def closure():
        global iteration
        start_time_i = time.time()

        # Asegúrate de reiniciar los gradientes
        optimizer.zero_grad()

        # Calcular la pérdida total para todos los datos
        total_loss = torch.tensor(0.0, requires_grad=True)
        
        with torch.set_grad_enabled(True):
            for x, y_target in zip(X, y01):
                output = model2(Tensor(x)).reshape(1, 2)
                loss = f_loss(output, Tensor([y_target]).long())
                total_loss = total_loss + loss

            # Realizar la retropropagación
            total_loss.backward()

        # Calcular métricas
        with torch.no_grad():
            # Precisión en entrenamiento
            y_predict = []
            for x in X:
                output = model2(Tensor(x))
                y_predict.append(np.argmax(output.cpu().detach().numpy()))
            accuracy_train = sum(np.array(y_predict) == np.array(y01)) / len(y01)

            # # Precisión en prueba
            y_predict_test = []
            for x in Xtest60:
                output = model2(Tensor(x))
                y_predict_test.append(np.argmax(output.cpu().detach().numpy()))
            accuracy_test = sum(np.array(y_predict_test) == np.array(y01test60)) / len(y01test60)

        end_time_i = time.time()
        current_lr = optimizer.param_groups[0]['lr']
        
        scheduler.step(total_loss.item()) # Ajustar la tasa de aprendizaje

        print(f'Iteration: {iteration}, Loss: {total_loss.item():.4f}, '
            f'Time: {end_time_i - start_time_i:.2f}s, '
            f'Accuracy Train: {accuracy_train:.2f}, '
            f'Accuracy Test: {accuracy_test:.2f},'
            f'LearningRate: {current_lr:.6f}')

        iteration += 1
        Losses.append(total_loss.item())
        Accuracies_train.append(accuracy_train)
        Acurracies_test.append(accuracy_test)
        
        global best_accuracy_train, best_accuracy_test, best_model_state, best_optimizer_state
        if accuracy_train > best_accuracy_train and accuracy_test > best_accuracy_test:
            best_accuracy_train = accuracy_train
            best_accuracy_test = accuracy_test
            best_model_state = model2.state_dict()
            best_optimizer_state = optimizer.state_dict()

        return total_loss, accuracy_train, accuracy_test

    # Ejecuta el optimizador
    Epoch = 1
    print('Training the model...')

    for epoch in range(10):  # Número de épocas
        start_time = time.time()
        print(f'Epoch: {epoch + 1}')
        iteration = 1

        for _ in range(10):  # Iteraciones por época
            # Realizar un paso de optimización
            loss, accuracy_tr, acurracy_ts = optimizer.step(closure)

            if accuracy_tr >= 1:
                    print(f"Accuracy reached 100% in epoch {epoch + 1}")
                    break
                
                
        end_time = time.time()
        print(f'Epoch time: {end_time - start_time:.2f} seconds')

        if accuracy_tr >= 1:
            break

        Epoch += 1
        

    # Guarda el modelo entrenado
    print("Guardando el modelo entrenado...")
    torch.save({
        'model_state_dict': best_model_state,
        'optimizer_state_dict': best_optimizer_state
    }, model_path)
    print("Modelo guardado exitosamente.")
    
    #Model2 se convierte en el mejor modelo
    model2.load_state_dict(best_model_state)
    optimizer.load_state_dict(best_optimizer_state)
    
    # Graficar las pérdidas y la precisión
    plt.figure(figsize=(18, 5))  
    plt.subplot(1, 3, 1)
    plt.plot(Losses, label='Loss')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.legend()

    plt.subplot(1, 3, 2)
    plt.plot(Accuracies_train, label='Accuracy')
    plt.xlabel('Iteration')
    plt.ylabel('Accuracy')
    plt.title('Training Accuracy')
    plt.legend()

    plt.subplot(1, 3, 3)
    plt.plot(Acurracies_test, label='Accuracy')
    plt.xlabel('Iteration')
    plt.ylabel('Accuracy')
    plt.title('Test Accuracy')
    plt.legend()
    
    plt.savefig('Training.png')

    plt.show()

with torch.no_grad():  # Configura el modelo en modo de evaluación
    # Predicciones para 10 puntos de datos
    y_predict = []
    incorrect_images_10 = []
    incorrect_labels_10 = []
    incorrect_predictions_10 = []
    for x, y_true in zip(Xtest, y01test):
        output = model2(Tensor(x))
        prediction = np.argmax(output.cpu().detach().numpy())
        y_predict.append(prediction)
        if prediction != y_true:
            incorrect_images_10.append(x)
            incorrect_labels_10.append(y_true)
            incorrect_predictions_10.append(prediction)
    print('Accuracy10data:', sum(np.array(y_predict) == np.array(y01test)) / len(np.array(y01test)))

    # Muestra los resultados para 10 puntos de datos
    fig, axes = plt.subplots(nrows=2, ncols=5, figsize=(20, 6))
    indices10 = np.random.choice(range(10), 10, replace=False)
    for i, idx in enumerate(indices10):
        axes[0, i % 5].imshow(dataInput[idx], cmap='gray')
        axes[0, i % 5].set_xticks([])
        axes[0, i % 5].set_yticks([])
        axes[0, i % 5].set_title(f"Actual: {'one' if y01test[idx] == 1 else 'nine'}")
        axes[1, i % 5].imshow(dataInput[idx], cmap='gray')
        axes[1, i % 5].set_xticks([])
        axes[1, i % 5].set_yticks([])
        axes[1, i % 5].set_title(f"Predicted: {'one' if y_predict[idx] == 1 else 'nine'}")
    plt.savefig('Resultados10.png')
    plt.show()

    # Mostrar imágenes incorrectas para 10 puntos de datos
    if incorrect_images_10:
        fig, axes = plt.subplots(nrows=1, ncols=len(incorrect_images_10), figsize=(20, 6))
        if len(incorrect_images_10) == 1:
            axes = [axes]  # Convertir a lista si solo hay una columna
        for i, (img, label, pred) in enumerate(zip(incorrect_images_10, incorrect_labels_10, incorrect_predictions_10)):
            axes[i].imshow(img.cpu().numpy().reshape(16, 16), cmap='gray')
            axes[i].set_xticks([])
            axes[i].set_yticks([])
            axes[i].set_title(f"Actual: {'one' if label == 1 else 'nine'}\nPredicted: {'one' if pred == 1 else 'nine'}")
        plt.savefig('Incorrectos10.png')
        plt.show()

# Predicciones para 25 puntos de datos
with torch.no_grad():
    y_predict = []
    incorrect_images_25 = []
    incorrect_labels_25 = []
    incorrect_predictions_25 = []
    for x, y_true in zip(Xtest25, y01test25):
        output = model2(Tensor(x))
        prediction = np.argmax(output.cpu().detach().numpy())
        y_predict.append(prediction)
        if prediction != y_true:
            incorrect_images_25.append(x)
            incorrect_labels_25.append(y_true)
            incorrect_predictions_25.append(prediction)
    print('Accuracy25data:', sum(np.array(y_predict) == np.array(y01test25)) / len(np.array(y01test25)))

    # Muestra los resultados para 25 puntos de datos
    fig, axes = plt.subplots(nrows=2, ncols=5, figsize=(20, 6))
    indices25 = np.random.choice(range(25), 10, replace=False)
    for i, idx in enumerate(indices25):
        axes[0, i % 5].imshow(dataInput[idx], cmap='gray')
        axes[0, i % 5].set_xticks([])
        axes[0, i % 5].set_yticks([])
        axes[0, i % 5].set_title(f"Actual: {'one' if y01test25[idx] == 1 else 'nine'}")
        axes[1, i % 5].imshow(dataInput[idx], cmap='gray')
        axes[1, i % 5].set_xticks([])
        axes[1, i % 5].set_yticks([])
        axes[1, i % 5].set_title(f"Predicted: {'one' if y_predict[idx] == 1 else 'nine'}")
    plt.savefig('Resultados25.png')
    plt.show()

    # Mostrar imágenes incorrectas para 25 puntos de datos
    if incorrect_images_25:
        fig, axes = plt.subplots(nrows=1, ncols=len(incorrect_images_25), figsize=(20, 6))
        if len(incorrect_images_25) == 1:
            axes = [axes]  # Convertir a lista si solo hay una columna
        for i, (img, label, pred) in enumerate(zip(incorrect_images_25, incorrect_labels_25, incorrect_predictions_25)):
            axes[i].imshow(img.cpu().numpy().reshape(16, 16), cmap='gray')
            axes[i].set_xticks([])
            axes[i].set_yticks([])
            axes[i].set_title(f"Actual: {'one' if label == 1 else 'nine'}\nPredicted: {'one' if pred == 1 else 'nine'}")
        plt.savefig('Incorrectos25.png')
        plt.show()

# Predicciones para 50 puntos de datos
with torch.no_grad():
    y_predict = []
    incorrect_images_60 = []
    incorrect_labels_60 = []
    incorrect_predictions_60 = []
    for x, y_true in zip(Xtest60, y01test60):
        output = model2(Tensor(x))
        prediction = np.argmax(output.cpu().detach().numpy())
        y_predict.append(prediction)
        if prediction != y_true:
            incorrect_images_60.append(x)
            incorrect_labels_60.append(y_true)
            incorrect_predictions_60.append(prediction)
    print('Accuracy60data:', sum(np.array(y_predict) == np.array(y01test60)) / len(np.array(y01test60)))

    # Muestra los resultados para 50 puntos de datos
    fig, axes = plt.subplots(nrows=2, ncols=5, figsize=(20, 6))
    indices60 = np.random.choice(range(60), 10, replace=False)
    for i, idx in enumerate(indices60):
        axes[0, i % 5].imshow(dataInput[idx], cmap='gray')
        axes[0, i % 5].set_xticks([])
        axes[0, i % 5].set_yticks([])
        axes[0, i % 5].set_title(f"Actual: {'one' if y01test60[idx] == 1 else 'nine'}")
        axes[1, i % 5].imshow(dataInput[idx], cmap='gray')
        axes[1, i % 5].set_xticks([])
        axes[1, i % 5].set_yticks([])
        axes[1, i % 5].set_title(f"Predicted: {'one' if y_predict[idx] == 1 else 'nine'}")
    plt.savefig('Resultados60.png')
    plt.show()

    # Mostrar imágenes incorrectas para 50 puntos de datos
    if incorrect_images_60:
        fig, axes = plt.subplots(nrows=1, ncols=len(incorrect_images_60), figsize=(20, 6))
        for i, (img, label, pred) in enumerate(zip(incorrect_images_60, incorrect_labels_60, incorrect_predictions_60)):
            axes[i].imshow(img.cpu().numpy().reshape(16, 16), cmap='gray')
            axes[i].set_xticks([])
            axes[i].set_yticks([])
            axes[i].set_title(f"Actual: {'one' if label == 1 else 'nine'}\nPredicted: {'one' if pred == 1 else 'nine'}")
        plt.savefig('Incorrectos60.png')
        plt.show()


import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk

class numbersDetectorGUI:
    def __init__(self, model2):
        self.root = tk.Tk()
        self.root.title("Detector de numbers")
        self.root.geometry("400x500")
        self.model = model2

        # Botón para seleccionar imagen
        self.select_button = tk.Button(self.root, text="Seleccionar Imagen", command=self.select_image)
        self.select_button.pack(pady=20)

        # Label para mostrar la imagen
        self.image_label = tk.Label(self.root)
        self.image_label.pack(pady=10)

        # Label para mostrar el resultado
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
            # Mostrar la imagen
            image = Image.open(file_path)
            image.thumbnail((200, 200))
            photo = ImageTk.PhotoImage(image)
            self.image_label.config(image=photo)
            self.image_label.image = photo

            # Procesar la imagen para la predicción
            image_gray = image.convert('L')
            image_resized = imageResize(np.array(image_gray), 16)
            image_processed = normlaizeData(image_resized.flatten())
            input_tensor = torch.tensor(image_processed)

            # Hacer predicción
            with torch.no_grad():
                output = self.model(input_tensor)
                prediction = np.argmax(output.numpy())

            # Mostrar resultado
            result = "finger" if prediction == 1 else "No es finger"
            self.result_label.config(
                text=f"Predicción: {result}\n"
            )

        except Exception as e:
            self.result_label.config(text=f"Error al procesar la imagen: {str(e)}")

    def run(self):
        self.root.mainloop()

# Añadir esto al final del Script.py, después de cargar el modelo
if __name__ == "__main__":
    # Asegurarse de que el modelo está cargado
    if 'model2' in locals():
        app = numbersDetectorGUI(model2)
        app.run()
    else:
        print("Error: El modelo no está cargado correctamente")