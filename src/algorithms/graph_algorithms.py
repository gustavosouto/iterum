"""
Implementação de algoritmos de otimização de rotas.
"""

import networkx as nx
import heapq
from typing import Dict, List, Tuple, Optional, Callable
import numpy as np
import logging

logger = logging.getLogger(__name__)


class MultiObjectiveOptimizer:
    """
    Otimizador multi-objetivo para rotas urbanas.
    """
    
    def __init__(self, graph: nx.Graph):
        self.graph = graph
        self.weights = {'tempo': 0.5, 'custo': 0.3, 'impacto_ambiental': 0.2}
    
    def set_weights(self, tempo: float = 0.5, custo: float = 0.3, 
                   impacto_ambiental: float = 0.2):
        """
        Define pesos para função de custo multi-objetivo.
        
        Args:
            tempo: Peso para tempo de viagem
            custo: Peso para custo financeiro
            impacto_ambiental: Peso para impacto ambiental
        """
        total = tempo + custo + impacto_ambiental
        self.weights = {
            'tempo': tempo / total,
            'custo': custo / total,
            'impacto_ambiental': impacto_ambiental / total
        }
    
    def calculate_edge_cost(self, u: str, v: str, edge_data: Dict) -> float:
        """
        Calcula custo ponderado de uma aresta.
        
        Args:
            u: Nó origem
            v: Nó destino
            edge_data: Dados da aresta
            
        Returns:
            Custo ponderado da aresta
        """
        try:
            tempo = edge_data.get('tempo_viagem', 0)
            custo = edge_data.get('custo', 0)
            impacto = edge_data.get('impacto_ambiental', 0)
            
            weighted_cost = (
                self.weights['tempo'] * tempo +
                self.weights['custo'] * custo +
                self.weights['impacto_ambiental'] * impacto
            )
            
            return weighted_cost
            
        except Exception as e:
            logger.error(f"Erro ao calcular custo da aresta: {e}")
            return float('inf')
    
    def dijkstra_multi_objective(self, source: str, target: str) -> Tuple[List[str], float]:
        """
        Implementação do algoritmo de Dijkstra com otimização multi-objetivo.
        
        Args:
            source: Nó de origem
            target: Nó de destino
            
        Returns:
            Tupla com (caminho, custo_total)
        """
        try:
            if source not in self.graph or target not in self.graph:
                return [], float('inf')
            
            # Inicialização
            distances = {node: float('inf') for node in self.graph.nodes()}
            distances[source] = 0
            previous = {}
            visited = set()
            
            # Fila de prioridade: (distância, nó)
            pq = [(0, source)]
            
            while pq:
                current_distance, current_node = heapq.heappop(pq)
                
                if current_node in visited:
                    continue
                
                visited.add(current_node)
                
                if current_node == target:
                    break
                
                # Examinar vizinhos
                for neighbor in self.graph.neighbors(current_node):
                    if neighbor in visited:
                        continue
                    
                    edge_data = self.graph[current_node][neighbor]
                    edge_cost = self.calculate_edge_cost(current_node, neighbor, edge_data)
                    
                    distance = current_distance + edge_cost
                    
                    if distance < distances[neighbor]:
                        distances[neighbor] = distance
                        previous[neighbor] = current_node
                        heapq.heappush(pq, (distance, neighbor))
            
            # Reconstruir caminho
            path = []
            current = target
            while current is not None:
                path.append(current)
                current = previous.get(current)
            
            path.reverse()
            
            if path[0] != source:
                return [], float('inf')
            
            return path, distances[target]
            
        except Exception as e:
            logger.error(f"Erro no algoritmo de Dijkstra: {e}")
            return [], float('inf')
    
    def a_star_multi_objective(self, source: str, target: str, 
                             heuristic_func: Optional[Callable] = None) -> Tuple[List[str], float]:
        """
        Implementação do algoritmo A* com otimização multi-objetivo.
        
        Args:
            source: Nó de origem
            target: Nó de destino
            heuristic_func: Função heurística (se None, usa distância euclidiana)
            
        Returns:
            Tupla com (caminho, custo_total)
        """
        try:
            if source not in self.graph or target not in self.graph:
                return [], float('inf')
            
            if heuristic_func is None:
                heuristic_func = self._euclidean_heuristic
            
            # Inicialização
            g_score = {node: float('inf') for node in self.graph.nodes()}
            g_score[source] = 0
            
            f_score = {node: float('inf') for node in self.graph.nodes()}
            f_score[source] = heuristic_func(source, target)
            
            previous = {}
            open_set = [(f_score[source], source)]
            closed_set = set()
            
            while open_set:
                current_f, current_node = heapq.heappop(open_set)
                
                if current_node in closed_set:
                    continue
                
                closed_set.add(current_node)
                
                if current_node == target:
                    break
                
                # Examinar vizinhos
                for neighbor in self.graph.neighbors(current_node):
                    if neighbor in closed_set:
                        continue
                    
                    edge_data = self.graph[current_node][neighbor]
                    edge_cost = self.calculate_edge_cost(current_node, neighbor, edge_data)
                    
                    tentative_g = g_score[current_node] + edge_cost
                    
                    if tentative_g < g_score[neighbor]:
                        previous[neighbor] = current_node
                        g_score[neighbor] = tentative_g
                        f_score[neighbor] = tentative_g + heuristic_func(neighbor, target)
                        
                        heapq.heappush(open_set, (f_score[neighbor], neighbor))
            
            # Reconstruir caminho
            path = []
            current = target
            while current is not None:
                path.append(current)
                current = previous.get(current)
            
            path.reverse()
            
            if path[0] != source:
                return [], float('inf')
            
            return path, g_score[target]
            
        except Exception as e:
            logger.error(f"Erro no algoritmo A*: {e}")
            return [], float('inf')
    
    def _euclidean_heuristic(self, node1: str, node2: str) -> float:
        """
        Heurística baseada em distância euclidiana.
        
        Args:
            node1: Primeiro nó
            node2: Segundo nó
            
        Returns:
            Distância euclidiana estimada
        """
        try:
            # Assumindo que os nós têm atributos 'lat' e 'lon'
            if 'lat' in self.graph.nodes[node1] and 'lat' in self.graph.nodes[node2]:
                lat1, lon1 = self.graph.nodes[node1]['lat'], self.graph.nodes[node1]['lon']
                lat2, lon2 = self.graph.nodes[node2]['lat'], self.graph.nodes[node2]['lon']
                
                # Distância euclidiana simples (para heurística)
                return np.sqrt((lat2 - lat1)**2 + (lon2 - lon1)**2)
            else:
                return 0
                
        except Exception as e:
            logger.error(f"Erro ao calcular heurística: {e}")
            return 0
    
    def bellman_ford_multi_objective(self, source: str) -> Dict[str, Tuple[float, List[str]]]:
        """
        Implementação do algoritmo Bellman-Ford com otimização multi-objetivo.
        
        Args:
            source: Nó de origem
            
        Returns:
            Dicionário com distâncias e caminhos para todos os nós
        """
        try:
            if source not in self.graph:
                return {}
            
            # Inicialização
            distances = {node: float('inf') for node in self.graph.nodes()}
            distances[source] = 0
            previous = {node: None for node in self.graph.nodes()}
            
            # Relaxamento de arestas (V-1 iterações)
            for _ in range(len(self.graph.nodes()) - 1):
                for u, v, edge_data in self.graph.edges(data=True):
                    edge_cost = self.calculate_edge_cost(u, v, edge_data)
                    
                    if distances[u] + edge_cost < distances[v]:
                        distances[v] = distances[u] + edge_cost
                        previous[v] = u
            
            # Verificar ciclos negativos
            for u, v, edge_data in self.graph.edges(data=True):
                edge_cost = self.calculate_edge_cost(u, v, edge_data)
                if distances[u] + edge_cost < distances[v]:
                    logger.warning("Ciclo negativo detectado no grafo")
            
            # Construir caminhos
            results = {}
            for target in self.graph.nodes():
                if distances[target] != float('inf'):
                    path = []
                    current = target
                    while current is not None:
                        path.append(current)
                        current = previous[current]
                    path.reverse()
                    results[target] = (distances[target], path)
            
            return results
            
        except Exception as e:
            logger.error(f"Erro no algoritmo Bellman-Ford: {e}")
            return {}
    
    def floyd_warshall_multi_objective(self) -> Dict[Tuple[str, str], Tuple[float, List[str]]]:
        """
        Implementação do algoritmo Floyd-Warshall com otimização multi-objetivo.
        
        Returns:
            Dicionário com distâncias e caminhos entre todos os pares de nós
        """
        try:
            nodes = list(self.graph.nodes())
            n = len(nodes)
            
            # Inicialização das matrizes
            dist = {}
            next_node = {}
            
            # Inicializar com infinito
            for i in nodes:
                for j in nodes:
                    if i == j:
                        dist[(i, j)] = 0
                        next_node[(i, j)] = None
                    else:
                        dist[(i, j)] = float('inf')
                        next_node[(i, j)] = None
            
            # Definir distâncias das arestas existentes
            for u, v, edge_data in self.graph.edges(data=True):
                edge_cost = self.calculate_edge_cost(u, v, edge_data)
                dist[(u, v)] = edge_cost
                next_node[(u, v)] = v
            
            # Algoritmo principal
            for k in nodes:
                for i in nodes:
                    for j in nodes:
                        if dist[(i, k)] + dist[(k, j)] < dist[(i, j)]:
                            dist[(i, j)] = dist[(i, k)] + dist[(k, j)]
                            next_node[(i, j)] = next_node[(i, k)]
            
            # Construir caminhos
            results = {}
            for i in nodes:
                for j in nodes:
                    if dist[(i, j)] != float('inf'):
                        path = self._reconstruct_floyd_path(i, j, next_node)
                        results[(i, j)] = (dist[(i, j)], path)
            
            return results
            
        except Exception as e:
            logger.error(f"Erro no algoritmo Floyd-Warshall: {e}")
            return {}
    
    def _reconstruct_floyd_path(self, start: str, end: str, next_node: Dict) -> List[str]:
        """
        Reconstrói caminho a partir da matriz next do Floyd-Warshall.
        
        Args:
            start: Nó inicial
            end: Nó final
            next_node: Matriz de próximos nós
            
        Returns:
            Lista representando o caminho
        """
        if next_node[(start, end)] is None:
            return []
        
        path = [start]
        current = start
        
        while current != end:
            current = next_node[(current, end)]
            path.append(current)
        
        return path


class RouteComparator:
    """
    Comparador de rotas e algoritmos.
    """
    
    def __init__(self):
        self.results = {}
    
    def compare_algorithms(self, graph: nx.Graph, source: str, target: str) -> Dict:
        """
        Compara performance de diferentes algoritmos.
        
        Args:
            graph: Grafo da rede
            source: Nó origem
            target: Nó destino
            
        Returns:
            Dicionário com resultados comparativos
        """
        try:
            optimizer = MultiObjectiveOptimizer(graph)
            results = {}
            
            import time
            
            # Dijkstra
            start_time = time.time()
            dijkstra_path, dijkstra_cost = optimizer.dijkstra_multi_objective(source, target)
            dijkstra_time = time.time() - start_time
            
            results['dijkstra'] = {
                'path': dijkstra_path,
                'cost': dijkstra_cost,
                'execution_time': dijkstra_time,
                'path_length': len(dijkstra_path)
            }
            
            # A*
            start_time = time.time()
            astar_path, astar_cost = optimizer.a_star_multi_objective(source, target)
            astar_time = time.time() - start_time
            
            results['a_star'] = {
                'path': astar_path,
                'cost': astar_cost,
                'execution_time': astar_time,
                'path_length': len(astar_path)
            }
            
            return results
            
        except Exception as e:
            logger.error(f"Erro ao comparar algoritmos: {e}")
            return {}
    
    def analyze_route_quality(self, path: List[str], graph: nx.Graph) -> Dict:
        """
        Analisa qualidade de uma rota.
        
        Args:
            path: Caminho da rota
            graph: Grafo da rede
            
        Returns:
            Dicionário com métricas de qualidade
        """
        try:
            if not path or len(path) < 2:
                return {}
            
            total_distance = 0
            total_time = 0
            total_cost = 0
            safety_scores = []
            
            for i in range(len(path) - 1):
                u, v = path[i], path[i + 1]
                if graph.has_edge(u, v):
                    edge_data = graph[u][v]
                    total_distance += edge_data.get('distancia', 0)
                    total_time += edge_data.get('tempo_viagem', 0)
                    total_cost += edge_data.get('custo', 0)
                    safety_scores.append(edge_data.get('indice_seguranca', 1.0))
            
            return {
                'total_distance': total_distance,
                'total_time': total_time,
                'total_cost': total_cost,
                'average_safety': np.mean(safety_scores) if safety_scores else 0,
                'route_segments': len(path) - 1
            }
            
        except Exception as e:
            logger.error(f"Erro ao analisar qualidade da rota: {e}")
            return {}

