import networkx as nx
import matplotlib.pyplot as plt
import random
import pennylane as qml
from pennylane import numpy as np
import timeit

# --- Generar el grafo dinamicamente ---
def generar_grafo_ciudad(n_nodos):
    """
    Genera un grafo donde cada nodo representa una interseccion en una ciudad.
    """
    G = nx.Graph()
    G.add_node(0, name="Empresa_Transporte")
    G.add_node(n_nodos - 1, name="SiguienteCiudad")
    for i in range(1, n_nodos - 1):
        G.add_node(i, name=f"{i}")

    # Para cada nodo, generar conexiones aleatorias
    for nodo in range(n_nodos):
        num_conexiones = random.randint(2, 4)
        nodos_disponibles = list(G.nodes() - {nodo})

        if nodos_disponibles:
            nodos_a_conectar = random.sample(nodos_disponibles, num_conexiones)
            for otro_nodo in nodos_a_conectar:
                G.add_edge(nodo, otro_nodo, weight=random.randint(1, 5))

    # Asegurar que el grafo este conectado
    if not nx.is_connected(G):
        componentes = list(nx.connected_components(G))
        for i in range(len(componentes) - 1):
            nodo1 = random.choice(list(componentes[i]))
            nodo2 = random.choice(list(componentes[i + 1]))
            G.add_edge(nodo1, nodo2, weight=random.randint(1, 5))

    return G

# --- Dibujar el grafo ---
def dibujar_grafo(G, pos, title="Grafo de la Ciudad", path=None, must_visit_nodes=None, weight_attribute='weight'):
    plt.figure(figsize=(12, 8))
    node_colors = ['lightblue' if node not in must_visit_nodes else 'orange' for node in G.nodes()]
    edge_colors = ['gray' if path is None or (u, v) not in path and (v, u) not in path else 'red' for u, v in G.edges()]
    nx.draw(G, pos, with_labels=True, labels=nx.get_node_attributes(G, 'name'), node_color=node_colors, edge_color=edge_colors, node_size=500, font_size=10)
    labels = nx.get_edge_attributes(G, weight_attribute)
    nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)
    plt.title(title)
    plt.show()

# Modificar la definicion de parametros y el circuito cuantico
N_QUBITS = 4
N_LAYERS = 3


dev = qml.device("lightning.qubit", wires=N_QUBITS)
@qml.qnode(dev)
def quantum_circuit(params):
    """
    Circuito cuantico mejorado para optimizacion.
    """
    param_idx = 0
    # Preparacion inicial
    for i in range(N_QUBITS):
        qml.Hadamard(wires=i)
        qml.RY(params[param_idx], wires=i)
        param_idx += 1

    # Capas de entrelazamiento y rotaciones
    for layer in range(N_LAYERS):
        # Entrelazamiento basado en la matriz de adyacencia
        for i in range(0, N_QUBITS-1, 2):
            qml.CNOT(wires=[i, i+1])
        for i in range(1, N_QUBITS-1, 2):
            qml.CNOT(wires=[i, i+1])

        # Rotaciones
        for i in range(N_QUBITS):
            qml.RZ(params[param_idx], wires=i)
            param_idx += 1
            qml.RY(params[param_idx], wires=i)
            param_idx += 1
            qml.RX(params[param_idx], wires=i)
            param_idx += 1

    return [qml.expval(qml.PauliZ(i)) for i in range(N_QUBITS)]


# --- Procesar el grafo con optimizacion cuantica ---
def process_graph_quantum(G, start_node, end_node, must_visit_nodes, quantum_times):
    print("\n=== INICIO DE LA OPTIMIZACION CUANTICA ===")
    print(f"Procesando el grafo con {len(G.nodes)} nodos y {len(G.edges)} aristas.")
    start_time_modification = timeit.default_timer()


    # Calcular numero correcto de parametros
    n_params = N_QUBITS + (N_QUBITS * 3 * N_LAYERS)  # Parametros iniciales + (3 rotaciones por qubit por capa)

    print(f"Numero total de parametros: {n_params}")

    def cost_function(params):
        expectation_values = quantum_circuit(params)

        # Penalizar el que lo valores no sean cercanos a 1
        penalty = qml.math.sum((1 - qml.math.abs(expectation_values))**2)
        
        #Penalizar el que no haya diferencias entre los quantum factors
        penalty_diff = qml.math.sum((qml.math.abs(expectation_values[0] - expectation_values[1]))**2)

        # Enfocarse mas en la optimizacion de caminos
        path_coherence_penalty = 0.0
        for u, v, data in G.edges(data=True):
            quantum_factor_u = abs(expectation_values[u % N_QUBITS])
            quantum_factor_v = abs(expectation_values[v % N_QUBITS])
            weight = data['weight']
            # Penalizar mas fuertemente las aristas con pesos altos
            path_coherence_penalty += (quantum_factor_u - quantum_factor_v)**2 * weight

        return  (penalty + (0.2 * path_coherence_penalty) + (0.5 * penalty_diff))

    # Optimizacion con multiples intentos
    best_params = None
    best_cost = float('inf')
    n_attempts = 3
    

    for attempt in range(n_attempts):
        opt = qml.GradientDescentOptimizer(stepsize=0.001)
        params = np.random.uniform(0, 2*np.pi, n_params, requires_grad=True)

        # Implementar early stopping
        patience = 20
        best_iteration_cost = float('inf')
        patience_counter = 0

        for step in range(100):
            params, cost = opt.step_and_cost(cost_function, params)
            
            if cost < best_iteration_cost:
                best_iteration_cost = cost
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= patience:
                break

            if cost < best_cost:
                best_cost = cost
                best_params = params.copy()

    print(f"Mejor costo encontrado: {best_cost:.6f}")
    # Usar los mejores parametros encontrados
    quantum_weights = quantum_circuit(best_params)
    print(f"Factores cuanticos finales: {quantum_weights}")

    # Modificar pesos del grafo
    for (u, v) in G.edges():
        weight = G[u][v]['weight']
        quantum_factor_u = abs(quantum_weights[u % N_QUBITS])
        quantum_factor_v = abs(quantum_weights[v % N_QUBITS])
        modified_weight = weight * (1 + 0.5*(quantum_factor_u + quantum_factor_v))

        G[u][v]['modified_weight'] = modified_weight

    quantum_time_weight = timeit.default_timer() - start_time_modification
    print(f"\nTiempo de ejecucion de la Modificacion Cuantica: {quantum_time_weight:.3f} segundos")

    try:
        start_time_djikstra_quantum = timeit.default_timer()
        path = nx.dijkstra_path(G, start_node, end_node, weight='modified_weight')
        for must_visit_node in must_visit_nodes:
            sub_path = nx.dijkstra_path(G, path[-1], must_visit_node, weight='modified_weight')
            path.extend(sub_path[1:])
        sub_path = nx.dijkstra_path(G, path[-1], end_node, weight='modified_weight')    
        fin_time_djikstra_quantum = timeit.default_timer()
        quantum_time = fin_time_djikstra_quantum - start_time_djikstra_quantum
        print(f"\nTiempo de ejecucion de la Optimizacion Cuantica: {quantum_time:.25f} segundos")
        quantum_times.append(quantum_time)
        path.extend(sub_path[1:])
        path_edges = list(zip(path, path[1:]))
        total_cost = sum(G[u][v]['weight'] for u, v in path_edges)
        return path, path_edges, total_cost, G
    except nx.NetworkXNoPath:
        print(f"No existe un camino entre los nodos {start_node} y {end_node}")
        return None, None, None, None

def main():
    dijkstra_times = []
    quantum_times = []
    dijkstra_costs = []
    quantum_costs = []
    rango = 10
    for i in range(rango):
        print(f"\n=== ITERACION {i+1} ===")
        n_nodos = random.randint(100, 150)
        G = generar_grafo_ciudad(n_nodos)
        pos = nx.spring_layout(G)
        must_visit_nodes = [random.randint(1, n_nodos - 2) for _ in range(10)]
        #dibujar_grafo(G, pos, title="Grafo de la Ciudad Inicial", must_visit_nodes=must_visit_nodes)

        start_node = 0
        end_node = n_nodos - 1

        try:
            # Primero se ejecuta Dijkstra para comprobar el resultado
            print(f"------------------------------------")
            start_time_dijkstra = timeit.default_timer()
            path = nx.dijkstra_path(G, start_node, end_node, weight='weight')
            for must_visit_node in must_visit_nodes:
                sub_path = nx.dijkstra_path(G, path[-1], must_visit_node, weight='weight')
                path.extend(sub_path[1:])
            sub_path = nx.dijkstra_path(G, path[-1], end_node, weight='weight')
            fin_time_dijkstra = timeit.default_timer()
            dijkstra_time = fin_time_dijkstra - start_time_dijkstra
            print(f"\nTiempo de ejecucion de Dijkstra: {dijkstra_time:.25f} segundos")
            dijkstra_times.append(dijkstra_time)
            path.extend(sub_path[1:])
            path_edges = list(zip(path, path[1:]))
            total_cost = sum(G[u][v]['weight'] for u, v in path_edges)
            dijkstra_costs.append(total_cost) 
        
            
            print(f"\nResultados de la optimizacion clasica (Dijkstra) para grafo {i+1}:")
            print("Nodos en el camino optimo:", path)
            print("Suma de los costes del camino optimo:", total_cost)
            print(f"Total de aristas recorridas por Dijkstra: {len(path_edges)}")
            
            # Luego se ejecuta la optimizacion cuantica
            result = process_graph_quantum(G, start_node, end_node, must_visit_nodes, quantum_times)
            if result[0] is not None:
                path, path_edges, total_cost, G = result
                quantum_costs.append(total_cost) 
                print(f"\nResultados de la optimizacion cuantica para grafo {i+1}:")
                print("Nodos en el camino optimo:", path)
                print("Suma de los costes del camino optimo:", total_cost)
                print(f"Total de aristas recorridas por la Optimizacion Cuantica: {len(path_edges)}")
  
                #dibujar_grafo(G, pos, title="Grafo de la Ciudad Optimizado", path=path_edges, must_visit_nodes=must_visit_nodes, weight_attribute='modified_weight')
            else:
                print("No se encontro un camino optimo.")
                quantum_costs.append(None)
        except Exception as e:
            print(f"Error en la optimizacion cuantica: {e}")
            quantum_costs.append(None)

    # Crear grafica para los tiempos de ejecucion
    plt.figure(figsize=(12, 8))
    plt.plot(range(1, rango+1), dijkstra_times, marker='o', label='Dijkstra')
    plt.plot(range(1, rango+1), quantum_times, marker='o', label='Optimizacion Cuantica')
    plt.xlabel('Iteracion')
    plt.ylabel('Tiempo de Ejecucion (segundos)')
    plt.title('Comparacion de Tiempos de Ejecucion')
    plt.legend()
    plt.show()


    # Calcular el promedio de los tiempos de ejecucion
    avg_dijkstra_time = sum(dijkstra_times) / len(dijkstra_times)
    avg_quantum_time = sum(quantum_times) / len(quantum_times)
    print(f"\nTiempo promedio de ejecucion de Dijkstra: {avg_dijkstra_time:.25f} segundos")
    print(f"Tiempo promedio de ejecucion de la Optimizacion Cuantica: {avg_quantum_time:.25f} segundos")
    
    # Mostrar diferencias de costes
    print("\nDiferencias en los costes totales:")
    for i in range(len(dijkstra_costs)):
        if quantum_costs[i] is not None:
            diff = dijkstra_costs[i] - quantum_costs[i]
            print(f"Iteracion {i+1}: Diferencia = {diff:.6f}")

if __name__ == "__main__":
    main()