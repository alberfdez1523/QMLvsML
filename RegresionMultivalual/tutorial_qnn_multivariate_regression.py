# Importación de bibliotecas necesarias
import matplotlib.pyplot as plt  # Para visualización de datos
import pennylane as qml  # Framework para computación cuántica
from pennylane import numpy as pnp  # NumPy compatible con PennyLane
import jax  # Biblioteca para diferenciación automática y computación acelerada
from jax import numpy as jnp  # NumPy optimizado para JAX
import optax  # Biblioteca de optimización para JAX
from sklearn.metrics import r2_score  # Para evaluar el rendimiento del modelo
from sklearn.neural_network import MLPRegressor  # Perceptrón multicapa clásico
import timeit  # Para medir el tiempo de ejecución
from jax.core import Tracer  # Para compatibilidad con JAX

# Fijar semilla para reproducibilidad
pnp.random.seed(42)

# Parcheamos la función is_abstract de PennyLane para trabajar con versiones más nuevas de JAX
def patched_is_abstract(tensor):
    """
    Función que verifica si un tensor es abstracto, compatible con versiones recientes de JAX.
    
    Args:
        tensor: El tensor a verificar
        
    Returns:
        Boolean indicando si el tensor es abstracto
    """
    if hasattr(jax.core, 'ConcreteArray'):
        return not isinstance(tensor.aval, jax.core.ConcreteArray)
    else:
        # Para versiones más recientes de JAX
        return isinstance(tensor, Tracer)

# Aplicamos el parche
qml.math.is_abstract = patched_is_abstract

# Desactivamos filtrado de trazas para mejor depuración
jax.config.update("jax_traceback_filtering", "off")

# Dispositivo cuántico con 4 qubits (aunque solo usaremos 2)
dev = qml.device('default.qubit', wires=4)

def S(x):
    """
    Función de codificación de datos en el circuito cuántico usando rotación Z.
    
    Args:
        x: Vector de características a codificar
    """
    qml.AngleEmbedding(x, wires=[0, 1], rotation='Z')

def W(params):
    """
    Función de capas entrelazadas fuertemente para el procesamiento cuántico.
    
    Args:
        params: Parámetros entrenables para las capas entrelazadas
    """
    qml.StronglyEntanglingLayers(params, wires=[0, 1])

@qml.qnode(dev, interface="jax")
def quantum_neural_network(params, x):
    """
    Nodo cuántico que define la arquitectura de la red neuronal cuántica.
    
    Args:
        params: Matriz de parámetros entrenables
        x: Vector de características de entrada
        
    Returns:
        Valor esperado del producto de matrices de Pauli-Z en los qubits 0 y 1
    """
    # Determinar dimensiones de la estructura de parámetros
    layers = len(params[:,0,0]) - 1
    n_wires = len(params[0,:,0])
    n_params_rot = len(params[0,0,:])
    
    # Aplicar capas alternadas de entrelazamiento y codificación
    for i in range(layers):
        W(params[i,:,:].reshape(1, n_wires, n_params_rot))
        S(x)
    
    # Capa final de entrelazamiento
    W(params[-1,:,:].reshape(1, n_wires, n_params_rot))

    # Medir el valor esperado del observable Z⊗Z
    return qml.expval(qml.PauliZ(wires=0) @ qml.PauliZ(wires=1))

# Definición de diferentes funciones objetivo para el problema de regresión multivariada
def target_function_1(x):
    """Función cuadrática simple: f(x,y) = 0.5 * (x² + y²)"""
    return 0.5 * (x[0]**2 + x[1]**2)

def target_function_2(x):
    """Función trigonométrica: f(x,y) = sin(x) * cos(y)"""
    return pnp.sin(x[0]) * pnp.cos(x[1])

def target_function_3(x):
    """Función gaussiana: f(x,y) = exp(-x² - y²)"""
    return pnp.exp(-x[0]**2 - x[1]**2)

def target_function_4(x):
    """Función racional: f(x,y) = 1 / (1 + x² + y²)"""
    return 1 / (1 + x[0]**2 + x[1]**2)

def target_function_5(x):
    """Función logarítmica: f(x,y) = log(1 + x² + y²)"""
    return pnp.log(1 + x[0]**2 + x[1]**2)

# Lista con todas las funciones objetivo
target_functions = [target_function_1, target_function_2, target_function_3, target_function_4, target_function_5]

# Definición del rango de los datos de entrada
x1_min = -1
x1_max = 1
x2_min = -1
x2_max = 1
num_samples = 30

# Generación de la rejilla de puntos para el entrenamiento
x1_train = pnp.linspace(x1_min, x1_max, num_samples)
x2_train = pnp.linspace(x2_min, x2_max, num_samples)
x1_mesh, x2_mesh = pnp.meshgrid(x1_train, x2_train)

# Aplanamos la rejilla para obtener pares de características
x_train = pnp.stack((x1_mesh.flatten(), x2_mesh.flatten()), axis=1)

@jax.jit
def mse(params, x, targets):
    """
    Función de error cuadrático medio para un solo punto.
    
    Args:
        params: Parámetros del modelo QNN
        x: Un punto de datos
        targets: Valor objetivo esperado
        
    Returns:
        Error cuadrático para ese punto
    """
    return (quantum_neural_network(params, x) - jnp.array(targets))**2

@jax.jit
def loss_fn(params, x, targets):
    """
    Función de pérdida para todo el conjunto de datos.
    
    Args:
        params: Parámetros del modelo QNN
        x: Conjunto completo de datos
        targets: Valores objetivo esperados
        
    Returns:
        Pérdida media en todo el conjunto de datos
    """
    # Vectorizamos la función mse para aplicarla a todos los puntos
    mse_pred = jax.vmap(mse, in_axes=(None, 0, 0))(params, x, targets)
    loss = jnp.mean(mse_pred)
    return loss

# Configuración del optimizador Adam con tasa de aprendizaje de 0.05
opt = optax.adam(learning_rate=0.05)
max_steps = 3000  # Número máximo de iteraciones de optimización

@jax.jit
def update_step_jit(i, args):
    """
    Función para un solo paso de actualización en el proceso de optimización.
    
    Args:
        i: Índice de la iteración actual
        args: Tupla con (params, opt_state, data, targets, print_training)
        
    Returns:
        Tupla actualizada con los nuevos parámetros y estado del optimizador
    """
    params, opt_state, data, targets, print_training = args
    
    # Cálculo de la pérdida y los gradientes
    loss_val, grads = jax.value_and_grad(loss_fn)(params, data, targets)
    
    # Actualización de parámetros usando el optimizador
    updates, opt_state = opt.update(grads, opt_state)
    params = optax.apply_updates(params, updates)

    # Función para imprimir el progreso cada 300 iteraciones
    def print_fn():
        jax.debug.print("Step: {i}  Loss: {loss_val}", i=i, loss_val=loss_val)
    jax.lax.cond((jnp.mod(i, 300) == 0) & print_training, print_fn, lambda: None)
    
    return (params, opt_state, data, targets, print_training)

@jax.jit
def optimization_jit(params, data, targets, print_training=False):
    """
    Función de optimización completa.
    
    Args:
        params: Parámetros iniciales
        data: Datos de entrenamiento
        targets: Valores objetivo
        print_training: Booleano que indica si imprimir progreso
        
    Returns:
        Parámetros optimizados
    """
    # Inicialización del estado del optimizador
    opt_state = opt.init(params)
    args = (params, opt_state, jnp.asarray(data), targets, print_training)
    
    # Bucle de optimización usando fori_loop de JAX para mejor rendimiento
    (params, opt_state, _, _, _) = jax.lax.fori_loop(0, max_steps + 1, update_step_jit, args)
    return params

# Definición de la forma de los parámetros para las capas entrelazadas
wires = 2  # Número de qubits a utilizar
layers = 4  # Número de capas de entrelazamiento
params_shape = qml.StronglyEntanglingLayers.shape(n_layers=layers + 1, n_wires=wires)

def evaluate(params, data):
    """
    Evalúa el modelo QNN en un conjunto de datos.
    
    Args:
        params: Parámetros optimizados del modelo
        data: Datos de entrada
        
    Returns:
        Predicciones del modelo
    """
    # Vectorizamos la función quantum_neural_network para todo el conjunto de datos
    y_pred = jax.vmap(quantum_neural_network, in_axes=(None, 0))(params, data)
    return y_pred

# Configuración de la visualización: 5 filas (una por función) x 3 columnas (target, QNN, MLP)
fig, axes = plt.subplots(5, 3, figsize=(20, 25), subplot_kw={'projection': '3d'})

# Listas para almacenar métricas de rendimiento
r2_scores_qnn = []  # Puntuaciones R² para QNN
r2_scores_mlp = []  # Puntuaciones R² para MLP
times_qnn = []      # Tiempos de entrenamiento para QNN
times_mlp = []      # Tiempos de entrenamiento para MLP

# Iteramos sobre cada función objetivo
for i, target_function in enumerate(target_functions):
    print(f"Training for target function {i+1}...")
    
    # Calculamos los valores objetivo para la función actual
    y_train = target_function([x1_mesh, x2_mesh]).reshape(-1, 1)
    
    # Medimos el tiempo de entrenamiento para QNN
    start_time = timeit.default_timer()
    # Inicializamos parámetros aleatorios
    params = pnp.random.default_rng().random(size=params_shape)
    # Optimizamos los parámetros
    best_params = optimization_jit(params, x_train, jnp.array(y_train), print_training=True)
    qnn_time = timeit.default_timer() - start_time
    times_qnn.append(qnn_time)
    
    # Evaluamos el modelo QNN y calculamos la métrica R²
    y_predictions_qnn = evaluate(best_params, x_train)
    r2_qnn = round(float(r2_score(y_train, y_predictions_qnn)), 3)
    r2_scores_qnn.append(r2_qnn)
    print(f"R^2 Score for QNN target function {i+1}: {r2_qnn}")

    # Medimos el tiempo de entrenamiento para MLP clásico
    start_time = timeit.default_timer()
    # Configuramos y entrenamos el modelo MLP
    model = MLPRegressor(
        hidden_layer_sizes=(16, 16, 8),  # Arquitectura con 3 capas ocultas
        activation='tanh',                # Función de activación tangente hiperbólica
        solver='adam',                    # Optimizador Adam
        max_iter=3000,                    # Mismo número máximo de iteraciones que QNN
        learning_rate_init=0.05,          # Misma tasa de aprendizaje que QNN
        early_stopping=True,              # Detención temprana
        validation_fraction=0.1,          # 10% de datos para validación
        n_iter_no_change=20,              # Número de iteraciones sin mejora para detenerse
        random_state=42                   # Semilla para reproducibilidad
    )
    model.fit(x_train, y_train.flatten())
    mlp_time = timeit.default_timer() - start_time
    times_mlp.append(mlp_time)
    
    # Evaluamos el modelo MLP y calculamos la métrica R²
    y_predictions_mlp = model.predict(x_train)
    r2_mlp = round(float(r2_score(y_train, y_predictions_mlp)), 3)
    r2_scores_mlp.append(r2_mlp)
    print(f"R^2 Score for MLP target function {i+1}: {r2_mlp}")

    # Visualización de la función objetivo (primera columna)
    ax1 = axes[i, 0]
    ax1.plot_surface(x1_mesh, x2_mesh, y_train.reshape(x1_mesh.shape), cmap='viridis')
    ax1.set_zlim(0, 1)
    ax1.set_xlabel('$x$', fontsize=10)
    ax1.set_ylabel('$y$', fontsize=10)
    ax1.set_zlabel('$f(x,y)$', fontsize=10)
    ax1.set_title(f'Target {i+1}')

    # Visualización de las predicciones QNN (segunda columna)
    ax2 = axes[i, 1]
    ax2.plot_surface(x1_mesh, x2_mesh, y_predictions_qnn.reshape(x1_mesh.shape), cmap='viridis')
    ax2.set_zlim(0, 1)
    ax2.set_xlabel('$x$', fontsize=10)
    ax2.set_ylabel('$y$', fontsize=10)
    ax2.set_zlabel('$f(x,y)$', fontsize=10)
    ax2.set_title(f'QNN Predicted {i+1}\nAccuracy: {round(r2_qnn*100,3)}%')

    # Visualización de las predicciones MLP (tercera columna)
    ax3 = axes[i, 2]
    ax3.plot_surface(x1_mesh, x2_mesh, y_predictions_mlp.reshape(x1_mesh.shape), cmap='viridis')
    ax3.set_zlim(0, 1)
    ax3.set_xlabel('$x$', fontsize=10)
    ax3.set_ylabel('$y$', fontsize=10)
    ax3.set_zlabel('$f(x,y)$', fontsize=10)
    ax3.set_title(f'MLP Predicted {i+1}\nAccuracy: {round(r2_mlp*100,3)}%')

# Ajuste del diseño y guardado de la figura de visualización 3D
plt.tight_layout(pad=3.7)
plt.savefig('qnn_multivariate_regression_results.png')
plt.show()

# Comparación de puntuaciones R² y tiempos de entrenamiento
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))

# Eje X para ambas gráficas (índices de función objetivo)
x = range(1, len(target_functions) + 1)

# Gráfica de comparación de puntuaciones R²
ax1.plot(x, r2_scores_qnn, label='QNN', marker='o')
ax1.plot(x, r2_scores_mlp, label='MLP', marker='o')
ax1.set_xlabel('Target Function')
ax1.set_ylabel('R² Score')
ax1.set_title('R² Score Comparison')
ax1.legend()
ax1.set_xticks(x)
ax1.grid(True)

# Gráfica de comparación de tiempos de entrenamiento
ax2.plot(x, times_qnn, label='QNN', marker='o')
ax2.plot(x, times_mlp, label='MLP', marker='o')
ax2.set_xlabel('Target Function')
ax2.set_ylabel('Training Time (seconds)')
ax2.set_title('Training Time Comparison')
ax2.legend()
ax2.set_xticks(x)
ax2.grid(True)

# Ajuste del diseño y guardado de la figura de comparación
plt.tight_layout()
plt.savefig('qnn_multivariate_regression_acurracies&times.png')
plt.show()