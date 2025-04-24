import pennylane as qml
import numpy as np
import matplotlib.pyplot as plt
import time
from sklearn.neural_network import MLPClassifier
import seaborn as sns
import random
from collections import deque

# Algoritmo de generación de laberinto usando Prim aleatorio
# Basado en: https://programmerclick.com/article/58581420573/
def create_maze_prim(width, height):
    # Inicializar el laberinto con paredes (1)
    maze = np.ones((height, width))

    # Elegir un punto de inicio aleatorio
    start_y, start_x = 0, 0
    maze[start_y, start_x] = 0  # Marcar como camino

    # Lista de paredes frontera
    walls = []

    # Añadir paredes vecinas a la lista
    if start_x > 0:
        walls.append((start_y, start_x - 1))
    if start_y > 0:
        walls.append((start_y - 1, start_x))
    if start_x < width - 1:
        walls.append((start_y, start_x + 1))
    if start_y < height - 1:
        walls.append((start_y + 1, start_x))

    # Mientras haya paredes en la lista
    while walls:
        # Elegir una pared aleatoria
        wall_index = random.randint(0, len(walls) - 1)
        wall_y, wall_x = walls.pop(wall_index)

        # Verificar si solo un lado de la pared es camino
        path_count = 0
        neighbors = []

        # Verificar vecinos
        if wall_x > 0 and maze[wall_y, wall_x - 1] == 0:
            path_count += 1
            neighbors.append((wall_y, wall_x - 1))
        if wall_y > 0 and maze[wall_y - 1, wall_x] == 0:
            path_count += 1
            neighbors.append((wall_y - 1, wall_x))
        if wall_x < width - 1 and maze[wall_y, wall_x + 1] == 0:
            path_count += 1
            neighbors.append((wall_y, wall_x + 1))
        if wall_y < height - 1 and maze[wall_y + 1, wall_x] == 0:
            path_count += 1
            neighbors.append((wall_y + 1, wall_x))

        # Si solo hay un camino adyacente, convertir la pared en camino
        if path_count == 1:
            maze[wall_y, wall_x] = 0

            # Añadir nuevas paredes a la lista
            if wall_x > 0 and maze[wall_y, wall_x - 1] == 1:
                walls.append((wall_y, wall_x - 1))
            if wall_y > 0 and maze[wall_y - 1, wall_x] == 1:
                walls.append((wall_y - 1, wall_x))
            if wall_x < width - 1 and maze[wall_y, wall_x + 1] == 1:
                walls.append((wall_y, wall_x + 1))
            if wall_y < height - 1 and maze[wall_y + 1, wall_x] == 1:
                walls.append((wall_y + 1, wall_x))

    # Asegurar que hay un camino desde el inicio hasta el final
    maze[0, 0] = 0  # Inicio
    maze[height-1, width-1] = 0  # Meta

    # Crear un camino directo desde el final hasta el inicio si no existe
    path = find_path(maze, (0, 0), (height-1, width-1))
    if not path:
        # Inicializar path como conjunto vacío si es None
        path = set()
        
        # Si no hay camino, crear uno
        y, x = height-1, width-1
        while (y, x) != (0, 0):
            if y > 0 and (y-1, x) not in path:
                maze[y-1, x] = 0
                y -= 1
            elif x > 0 and (y, x-1) not in path:
                maze[y, x-1] = 0
                x -= 1
            else:
                # Si no podemos avanzar, romper una pared aleatoria
                directions = []
                if y > 0:
                    directions.append((-1, 0))
                if y < height-1:
                    directions.append((1, 0))
                if x > 0:
                    directions.append((0, -1))
                if x < width-1:
                    directions.append((0, 1))
                
                if directions:
                    dy, dx = random.choice(directions)
                    maze[y+dy, x+dx] = 0
                    y += dy
                    x += dx
                else:
                    break

    return maze

# Función para encontrar un camino en el laberinto usando BFS
def find_path(maze, start, end):
    height, width = maze.shape
    queue = deque([start])
    visited = {start: None}

    while queue:
        y, x = queue.popleft()

        if (y, x) == end:
            # Reconstruir el camino
            path = []
            while (y, x) != start:
                path.append((y, x))
                y, x = visited[(y, x)]
            path.append(start)
            path.reverse()
            return path

        # Explorar vecinos
        for dy, dx in [(-1, 0), (0, 1), (1, 0), (0, -1)]:
            ny, nx = y + dy, x + dx
            if (0 <= ny < height and 0 <= nx < width and
                maze[ny, nx] == 0 and (ny, nx) not in visited):
                queue.append((ny, nx))
                visited[(ny, nx)] = (y, x)

    return None

# Generar datos de entrenamiento a partir del laberinto
def generate_training_data(maze):
    height, width = maze.shape
    X = []
    y = []

    # Para cada celda, determinar si es posible moverse en cada dirección
    for i in range(height):
        for j in range(width):
            if maze[i, j] == 1:  # Si es pared, saltamos
                continue

            # Codificar la posición actual y características del entorno
            # Usamos más características para mejorar el aprendizaje
            current_pos = [i/height, j/width]

            # Verificar movimientos posibles (arriba, derecha, abajo, izquierda)
            # Arriba
            if i > 0:
                features = current_pos + [0]  # Dirección 0 = arriba
                # Añadir características adicionales: ¿hay pared en esa dirección?
                features += [1 if maze[i-1, j] == 1 else 0]
                X.append(features)
                y.append(1 if maze[i-1, j] == 0 else 0)  # 1 si es camino, 0 si es pared

            # Derecha
            if j < width-1:
                features = current_pos + [1]  # Dirección 1 = derecha
                features += [1 if maze[i, j+1] == 1 else 0]
                X.append(features)
                y.append(1 if maze[i, j+1] == 0 else 0)

            # Abajo
            if i < height-1:
                features = current_pos + [2]  # Dirección 2 = abajo
                features += [1 if maze[i+1, j] == 1 else 0]
                X.append(features)
                y.append(1 if maze[i+1, j] == 0 else 0)

            # Izquierda
            if j > 0:
                features = current_pos + [3]  # Dirección 3 = izquierda
                features += [1 if maze[i, j-1] == 1 else 0]
                X.append(features)
                y.append(1 if maze[i, j-1] == 0 else 0)

    return np.array(X), np.array(y)

# Implementación del algoritmo de resolución con ML clásico
def solve_maze_classical(maze, model):
    height, width = maze.shape
    start = (0, 0)
    end = (height-1, width-1)

    # Función para obtener movimientos válidos según el modelo
    def get_valid_moves(pos):
        i, j = pos
        moves = []
        directions = [(i-1, j), (i, j+1), (i+1, j), (i, j-1)]  # arriba, derecha, abajo, izquierda

        for idx, (ni, nj) in enumerate(directions):
            if 0 <= ni < height and 0 <= nj < width:
                # Predecir si el movimiento es válido
                features = np.array([[i/height, j/width, idx, 1 if maze[ni, nj] == 1 else 0]])
                if model.predict(features)[0] == 1:
                    moves.append((ni, nj))

        return moves

    # Algoritmo de búsqueda en anchura (BFS)
    visited = set([start])
    queue = [(start, [start])]

    start_time = time.time()

    while queue:
        (i, j), path = queue.pop(0)

        if (i, j) == end:
            end_time = time.time()
            return path, end_time - start_time

        for next_pos in get_valid_moves((i, j)):
            if next_pos not in visited:
                visited.add(next_pos)
                queue.append((next_pos, path + [next_pos]))

    end_time = time.time()
    return None, end_time - start_time  # No se encontró camino

# Implementación del circuito cuántico para resolver el laberinto
def quantum_circuit(params, features):
    # Codificar las características de entrada en el circuito
    for i in range(4):  # 4 características: posición x, y, dirección, pared
        qml.RY(np.pi * features[i], wires=i)

    # Aplicar capas de entrelazamiento y rotación
    for layer in range(2):  # 2 capas
        # Entrelazamiento
        for i in range(4):
            qml.CNOT(wires=[i, (i+1) % 4])

        # Rotaciones parametrizadas
        for i in range(4):
            qml.RX(params[layer][i][0], wires=i)
            qml.RY(params[layer][i][1], wires=i)
            qml.RZ(params[layer][i][2], wires=i)

    # Medir el qubit de salida
    return qml.expval(qml.PauliZ(0))

# Configurar el dispositivo cuántico
dev = qml.device("lightning.qubit", wires=4)

# Crear el nodo cuántico
@qml.qnode(dev)
def quantum_classifier(params, features):
    return quantum_circuit(params, features)

# Función para entrenar el modelo cuántico
def train_quantum_model(X, y, steps=150):  # Más pasos para mejor aprendizaje
    # Inicializar parámetros con un rango más amplio
    num_layers = 2  # Simplificar para evitar sobreajuste
    num_qubits = 4
    num_params_per_qubit = 3

    # Inicialización controlada de parámetros
    params = np.random.uniform(0, np.pi, (num_layers, num_qubits, num_params_per_qubit))

    # Optimizador con tasa de aprendizaje adaptativa
    opt = qml.AdamOptimizer(stepsize=0.01)

    def cost(params):
        # Usar batch más pequeño para estabilidad
        batch_size = min(32, len(X))
        indices = np.random.choice(len(X), size=batch_size, replace=False)

        X_batch = X[indices]
        y_batch = y[indices]

        # Calcular predicciones
        predictions = [quantum_classifier(params, x) for x in X_batch]

        # Convertir predicciones de [-1,1] a [0,1]
        predictions = [(p + 1) / 2 for p in predictions]

        # Error cuadrático medio
        return np.mean([(pred - label)**2 for pred, label in zip(predictions, y_batch)])

    # Entrenamiento
    start_time = time.time()
    costs = []

    for i in range(steps):
        params, cost_val = opt.step_and_cost(cost, params)
        costs.append(cost_val)

        if (i+1) % 10 == 0:
            print(f"Paso {i+1}, Costo: {cost_val:.4f}")

    training_time = time.time() - start_time
    return params, costs, training_time

# Función para resolver el laberinto con el modelo cuántico
# Versión corregida de la función para resolver el laberinto con el modelo cuántico
def solve_maze_quantum(maze, params):
    """Versión simplificada que combina predicciones cuánticas con BFS clásico"""
    height, width = maze.shape
    start = (0, 0)
    end = (height-1, width-1)

    # Cola para BFS
    queue = deque([(start, [start])])
    visited = set([start])

    start_time = time.time()

    while queue:
        current, path = queue.popleft()
        i, j = current

        # Si llegamos al final
        if current == end:
            end_time = time.time()
            return path, end_time - start_time

        # Explorar vecinos en orden: arriba, derecha, abajo, izquierda
        directions = [(i-1, j), (i, j+1), (i+1, j), (i, j-1)]
        valid_moves = []

        for idx, (ni, nj) in enumerate(directions):
            # Verificar que es una celda válida dentro del laberinto y no es pared
            if 0 <= ni < height and 0 <= nj < width and maze[ni, nj] == 0 and (ni, nj) not in visited:
                # Usar clasificador cuántico como guía adicional
                features = np.array([i/height, j/width, idx/3, 0])  # Normalizar características
                prediction = quantum_classifier(params, features)
                confidence = (prediction + 1) / 2  # Convertir de [-1,1] a [0,1]

                # Aceptar el movimiento si el clasificador da suficiente confianza
                if confidence > 0.3:  # Umbral más bajo para permitir exploración
                    valid_moves.append(((ni, nj), confidence))

        # Ordenar movimientos por confianza y agregar a la cola
        valid_moves.sort(key=lambda x: x[1], reverse=True)
        for move, _ in valid_moves:
            visited.add(move)
            queue.append((move, path + [move]))

    end_time = time.time()
    return None, end_time - start_time

# Visualizar el laberinto y el camino encontrado
def visualize_maze_and_path(maze, path=None, title="Laberinto y Camino"):
    plt.figure(figsize=(10, 10))
    plt.imshow(maze, cmap='binary')

    if path:
        path_y, path_x = zip(*path)
        plt.plot(path_x, path_y, 'r-', linewidth=2)
        plt.plot(path_x[0], path_y[0], 'go', markersize=10)  # Inicio
        plt.plot(path_x[-1], path_y[-1], 'bo', markersize=10)  # Fin

    plt.grid(False)
    plt.title(title)
    plt.xticks([])
    plt.yticks([])
    return plt

# Comparar y visualizar resultados
def compare_and_visualize(classical_path, quantum_path, classical_time, quantum_time,
                          classical_training_time, quantum_training_time, quantum_costs):
    # Visualizar tiempos de entrenamiento
    plt.figure(figsize=(10, 5))
    plt.bar(['ML Clásico', 'ML Cuántico'], [classical_training_time, quantum_training_time])
    plt.title('Tiempo de Entrenamiento')
    plt.ylabel('Tiempo (segundos)')
    plt.savefig('tiempo_entrenamiento.png')
    plt.close()

    # Visualizar tiempos de resolución
    plt.figure(figsize=(10, 5))
    plt.bar(['ML Clásico', 'ML Cuántico'], [classical_time, quantum_time])
    plt.title('Tiempo de Resolución del Laberinto')
    plt.ylabel('Tiempo (segundos)')
    plt.savefig('tiempo_resolucion.png')
    plt.close()

    # Visualizar curva de aprendizaje del modelo cuántico
    plt.figure(figsize=(10, 5))
    plt.plot(quantum_costs)
    plt.title('Curva de Aprendizaje del Modelo Cuántico')
    plt.xlabel('Iteraciones')
    plt.ylabel('Costo')
    plt.savefig('curva_aprendizaje_cuantico.png')
    plt.close()

    # Comparar longitudes de camino
    if classical_path and quantum_path:
        plt.figure(figsize=(10, 5))
        plt.bar(['ML Clásico', 'ML Cuántico'], [len(classical_path), len(quantum_path)])
        plt.title('Longitud del Camino Encontrado')
        plt.ylabel('Número de pasos')
        plt.savefig('longitud_camino.png')
        plt.close()

    # Crear una tabla comparativa
    data = {
        'Métrica': ['Tiempo de Entrenamiento (s)', 'Tiempo de Resolución (s)',
                   'Longitud del Camino', 'Camino Encontrado'],
        'ML Clásico': [classical_training_time, classical_time,
                      len(classical_path) if classical_path else 'N/A',
                      'Si' if classical_path else 'No'],
        'ML Cuántico': [quantum_training_time, quantum_time,
                       len(quantum_path) if quantum_path else 'N/A',
                       'Si' if quantum_path else 'No']
    }

    # Crear una figura para la tabla
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.axis('tight')
    ax.axis('off')
    table = ax.table(cellText=[[data['Métrica'][i], data['ML Clásico'][i], data['ML Cuántico'][i]]
                              for i in range(len(data['Métrica']))],
                    colLabels=['Métrica', 'ML Clásico', 'ML Cuántico'],
                    loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.2, 1.5)
    plt.title('Comparación de Rendimiento', pad=20)
    plt.savefig('tabla_comparativa.png', bbox_inches='tight')
    plt.close()

    return data

# Función principal
def main():
    # Crear laberinto más grande usando el algoritmo de Prim
    maze_size = 200 # Tamaño del laberinto (15x15)
    print(f"Generando laberinto de {maze_size}x{maze_size}...")
    maze = create_maze_prim(maze_size, maze_size)

    # Visualizar el laberinto generado
    plt_maze = visualize_maze_and_path(maze, title=f"Laberinto {maze_size}x{maze_size}")
    plt_maze.savefig('laberinto_generado.png')
    plt_maze.close()

    # Generar datos de entrenamiento
    print("Generando datos de entrenamiento...")
    X, y = generate_training_data(maze)
    print(f"Datos generados: {len(X)} ejemplos")

    print("Entrenando modelo clásico...")
    # Entrenar modelo clásico
    start_time = time.time()
    classical_model = MLPClassifier(hidden_layer_sizes=(32, 16), max_iter=1000, random_state=42)
    classical_model.fit(X, y)
    classical_training_time = time.time() - start_time
    print(f"Tiempo de entrenamiento del modelo clásico: {classical_training_time:.2f} segundos")

    print("Entrenando modelo cuántico...")
    # Entrenar modelo cuántico (con menos pasos para laberintos grandes)
    quantum_params, quantum_costs, quantum_training_time = train_quantum_model(X, y, steps=100)
    print(f"Tiempo de entrenamiento del modelo cuántico: {quantum_training_time:.2f} segundos")

    print("Resolviendo laberinto con modelo clásico...")
    # Resolver laberinto con modelo clásico
    classical_path, classical_time = solve_maze_classical(maze, classical_model)
    print(f"Tiempo de resolución con modelo clásico: {classical_time:.2f} segundos")

    print("Resolviendo laberinto con modelo cuántico...")
    # Resolver laberinto con modelo cuántico
    quantum_path, quantum_time = solve_maze_quantum(maze, quantum_params)
    print(f"Tiempo de resolución con modelo cuántico: {quantum_time:.2f} segundos")

    # Visualizar laberinto con caminos
    if classical_path:
        plt_classical = visualize_maze_and_path(maze, classical_path, "Camino encontrado con ML Clásico")
        plt_classical.savefig('camino_clasico.png')
        plt_classical.close()
        print(f"Longitud del camino clásico: {len(classical_path)}")
    else:
        print("El modelo clásico no encontró un camino")

    if quantum_path:
        plt_quantum = visualize_maze_and_path(maze, quantum_path, "Camino encontrado con ML Cuántico")
        plt_quantum.savefig('camino_cuantico.png')
        plt_quantum.close()
        print(f"Longitud del camino cuántico: {len(quantum_path)}")
    else:
        print("El modelo cuántico no encontró un camino")

    # Comparar y visualizar resultados
    results = compare_and_visualize(
        classical_path, quantum_path,
        classical_time, quantum_time,
        classical_training_time, quantum_training_time,
        quantum_costs
    )

    print("\nResultados:")
    for i in range(len(results['Métrica'])):
        print(f"{results['Métrica'][i]}: Clásico = {results['ML Clásico'][i]}, Cuántico = {results['ML Cuántico'][i]}")

if __name__ == "__main__":
    main()