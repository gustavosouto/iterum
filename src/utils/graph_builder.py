"""
Utilitários para construção e manipulação de grafos.
"""

import networkx as nx
import pandas as pd
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class IterumGraphBuilder:
    """
    Construtor de grafos para o sistema Iterum.
    """
    
    def __init__(self):
        self.graph = nx.DiGraph()  # Grafo direcionado
        self.nodes_data = {}
        self.edges_data = {}
    
    def create_recife_network(self) -> nx.DiGraph:
        """
        Cria a rede básica de transporte do Recife para o escopo reduzido.
        
        Returns:
            Grafo NetworkX representando a rede
        """
        try:
            # Definir nós principais baseados no escopo reduzido
            nodes = {
                'FAFIRE': {
                    'lat': -8.0476,
                    'lon': -34.8770,
                    'tipo': 'origem',
                    'nome': 'FAFIRE'
                },
                'FNR': {
                    'lat': -8.0632,
                    'lon': -34.8711,
                    'tipo': 'destino',
                    'nome': 'Faculdade Nova Roma'
                },
                'CBV_INICIO': {
                    'lat': -8.0500,
                    'lon': -34.8800,
                    'tipo': 'intersecao',
                    'nome': 'Início Conde da Boa Vista'
                },
                'CBV_FIM': {
                    'lat': -8.0550,
                    'lon': -34.8750,
                    'tipo': 'intersecao',
                    'nome': 'Fim Conde da Boa Vista'
                },
                'AGAMENON_INICIO': {
                    'lat': -8.0480,
                    'lon': -34.8790,
                    'tipo': 'intersecao',
                    'nome': 'Início Agamenon Magalhães'
                },
                'AGAMENON_FIM': {
                    'lat': -8.0580,
                    'lon': -34.8730,
                    'tipo': 'intersecao',
                    'nome': 'Fim Agamenon Magalhães'
                },
                'CSR_INICIO': {
                    'lat': -8.0490,
                    'lon': -34.8810,
                    'tipo': 'intersecao',
                    'nome': 'Início Cais de Santa Rita'
                },
                'CSR_FIM': {
                    'lat': -8.0570,
                    'lon': -34.8740,
                    'tipo': 'intersecao',
                    'nome': 'Fim Cais de Santa Rita'
                },
                'PINA_ENTRADA': {
                    'lat': -8.0600,
                    'lon': -34.8720,
                    'tipo': 'intersecao',
                    'nome': 'Entrada Via Pina'
                },
                'PINA_SAIDA': {
                    'lat': -8.0620,
                    'lon': -34.8700,
                    'tipo': 'intersecao',
                    'nome': 'Saída Via Pina'
                },
                'MANGUE_ENTRADA': {
                    'lat': -8.0590,
                    'lon': -34.8730,
                    'tipo': 'intersecao',
                    'nome': 'Entrada Via Mangue'
                },
                'MANGUE_SAIDA': {
                    'lat': -8.0610,
                    'lon': -34.8710,
                    'tipo': 'intersecao',
                    'nome': 'Saída Via Mangue'
                }
            }
            
            # Adicionar nós ao grafo
            for node_id, node_data in nodes.items():
                self.graph.add_node(node_id, **node_data)
            
            self.nodes_data = nodes
            
            # Definir arestas (conexões entre nós)
            edges = self._create_edges_data()
            
            # Adicionar arestas ao grafo
            for edge in edges:
                self.graph.add_edge(
                    edge['origem'],
                    edge['destino'],
                    **{k: v for k, v in edge.items() if k not in ['origem', 'destino']}
                )
            
            logger.info(f"Grafo criado com {self.graph.number_of_nodes()} nós e {self.graph.number_of_edges()} arestas")
            
            return self.graph
            
        except Exception as e:
            logger.error(f"Erro ao criar rede do Recife: {e}")
            return nx.DiGraph()
    
    def _create_edges_data(self) -> List[Dict]:
        """
        Cria dados das arestas para as 3 rotas principais.
        
        Returns:
            Lista de dicionários com dados das arestas
        """
        try:
            edges = []
            
            # Rota 1: FAFIRE -> Conde da Boa Vista -> Via Pina -> FNR
            route1_edges = [
                {
                    'origem': 'FAFIRE',
                    'destino': 'CBV_INICIO',
                    'distancia': 2.5,
                    'tempo_viagem': 8.0,  # Tempo base em minutos
                    'custo': 12.50,
                    'modal': 'uber_carro',
                    'indice_seguranca': 0.8,
                    'impacto_ambiental': 0.6,
                    'via': 'acesso_cbv'
                },
                {
                    'origem': 'CBV_INICIO',
                    'destino': 'CBV_FIM',
                    'distancia': 3.2,
                    'tempo_viagem': 12.0,
                    'custo': 16.00,
                    'modal': 'uber_carro',
                    'indice_seguranca': 0.7,
                    'impacto_ambiental': 0.7,
                    'via': 'conde_boa_vista'
                },
                {
                    'origem': 'CBV_FIM',
                    'destino': 'PINA_ENTRADA',
                    'distancia': 1.8,
                    'tempo_viagem': 6.0,
                    'custo': 9.00,
                    'modal': 'uber_carro',
                    'indice_seguranca': 0.9,
                    'impacto_ambiental': 0.5,
                    'via': 'acesso_pina'
                },
                {
                    'origem': 'PINA_ENTRADA',
                    'destino': 'PINA_SAIDA',
                    'distancia': 2.1,
                    'tempo_viagem': 7.0,
                    'custo': 10.50,
                    'modal': 'uber_carro',
                    'indice_seguranca': 0.85,
                    'impacto_ambiental': 0.4,
                    'via': 'via_pina'
                },
                {
                    'origem': 'PINA_SAIDA',
                    'destino': 'FNR',
                    'distancia': 1.4,
                    'tempo_viagem': 5.0,
                    'custo': 7.00,
                    'modal': 'uber_carro',
                    'indice_seguranca': 0.9,
                    'impacto_ambiental': 0.3,
                    'via': 'acesso_fnr'
                }
            ]
            
            # Rota 2: FAFIRE -> Agamenon -> Via Mangue -> FNR
            route2_edges = [
                {
                    'origem': 'FAFIRE',
                    'destino': 'AGAMENON_INICIO',
                    'distancia': 2.2,
                    'tempo_viagem': 7.0,
                    'custo': 11.00,
                    'modal': 'uber_carro',
                    'indice_seguranca': 0.75,
                    'impacto_ambiental': 0.65,
                    'via': 'acesso_agamenon'
                },
                {
                    'origem': 'AGAMENON_INICIO',
                    'destino': 'AGAMENON_FIM',
                    'distancia': 3.5,
                    'tempo_viagem': 15.0,  # Mais lento devido ao tráfego
                    'custo': 17.50,
                    'modal': 'uber_carro',
                    'indice_seguranca': 0.6,
                    'impacto_ambiental': 0.8,
                    'via': 'agamenon_magalhaes'
                },
                {
                    'origem': 'AGAMENON_FIM',
                    'destino': 'MANGUE_ENTRADA',
                    'distancia': 1.6,
                    'tempo_viagem': 5.5,
                    'custo': 8.00,
                    'modal': 'uber_carro',
                    'indice_seguranca': 0.8,
                    'impacto_ambiental': 0.5,
                    'via': 'acesso_mangue'
                },
                {
                    'origem': 'MANGUE_ENTRADA',
                    'destino': 'MANGUE_SAIDA',
                    'distancia': 2.3,
                    'tempo_viagem': 8.0,
                    'custo': 11.50,
                    'modal': 'uber_carro',
                    'indice_seguranca': 0.7,
                    'impacto_ambiental': 0.6,
                    'via': 'via_mangue'
                },
                {
                    'origem': 'MANGUE_SAIDA',
                    'destino': 'FNR',
                    'distancia': 1.2,
                    'tempo_viagem': 4.0,
                    'custo': 6.00,
                    'modal': 'uber_carro',
                    'indice_seguranca': 0.9,
                    'impacto_ambiental': 0.3,
                    'via': 'acesso_fnr'
                }
            ]
            
            # Rota 3: FAFIRE -> Cais de Santa Rita -> Via Pina -> FNR
            route3_edges = [
                {
                    'origem': 'FAFIRE',
                    'destino': 'CSR_INICIO',
                    'distancia': 2.8,
                    'tempo_viagem': 9.0,
                    'custo': 14.00,
                    'modal': 'uber_carro',
                    'indice_seguranca': 0.7,
                    'impacto_ambiental': 0.7,
                    'via': 'acesso_csr'
                },
                {
                    'origem': 'CSR_INICIO',
                    'destino': 'CSR_FIM',
                    'distancia': 3.0,
                    'tempo_viagem': 11.0,
                    'custo': 15.00,
                    'modal': 'uber_carro',
                    'indice_seguranca': 0.65,
                    'impacto_ambiental': 0.75,
                    'via': 'cais_santa_rita'
                },
                {
                    'origem': 'CSR_FIM',
                    'destino': 'PINA_ENTRADA',
                    'distancia': 1.5,
                    'tempo_viagem': 5.0,
                    'custo': 7.50,
                    'modal': 'uber_carro',
                    'indice_seguranca': 0.85,
                    'impacto_ambiental': 0.4,
                    'via': 'acesso_pina'
                }
                # Reutiliza as arestas da Via Pina da Rota 1
            ]
            
            # Combinar todas as arestas
            edges.extend(route1_edges)
            edges.extend(route2_edges)
            edges.extend(route3_edges)
            
            # Adicionar variações para Uber Moto (menor tempo, menor custo, maior risco)
            moto_edges = []
            for edge in edges:
                moto_edge = edge.copy()
                moto_edge['modal'] = 'uber_moto'
                moto_edge['tempo_viagem'] *= 0.7  # 30% mais rápido
                moto_edge['custo'] *= 0.6  # 40% mais barato
                moto_edge['indice_seguranca'] *= 0.8  # 20% menos seguro
                moto_edge['impacto_ambiental'] *= 0.5  # 50% menos impacto
                moto_edges.append(moto_edge)
            
            edges.extend(moto_edges)
            
            return edges
            
        except Exception as e:
            logger.error(f"Erro ao criar dados das arestas: {e}")
            return []
    
    def update_edge_weights_with_predictions(self, predictions_df: pd.DataFrame):
        """
        Atualiza pesos das arestas com previsões de tempo de viagem.
        
        Args:
            predictions_df: DataFrame com previsões de tempo por aresta
        """
        try:
            if predictions_df.empty:
                return
            
            for _, row in predictions_df.iterrows():
                origem = row.get('origem')
                destino = row.get('destino')
                tempo_previsto = row.get('tempo_viagem_previsto')
                
                if (origem and destino and tempo_previsto and 
                    self.graph.has_edge(origem, destino)):
                    
                    # Atualizar tempo de viagem previsto
                    self.graph[origem][destino]['tempo_viagem'] = tempo_previsto
                    
                    logger.debug(f"Atualizado tempo da aresta {origem}->{destino}: {tempo_previsto:.2f} min")
            
            logger.info("Pesos das arestas atualizados com previsões")
            
        except Exception as e:
            logger.error(f"Erro ao atualizar pesos das arestas: {e}")
    
    def get_route_summary(self, path: List[str]) -> Dict:
        """
        Obtém resumo de uma rota.
        
        Args:
            path: Lista de nós representando a rota
            
        Returns:
            Dicionário com resumo da rota
        """
        try:
            if not path or len(path) < 2:
                return {}
            
            total_distance = 0
            total_time = 0
            total_cost = 0
            safety_scores = []
            environmental_impact = []
            route_segments = []
            
            for i in range(len(path) - 1):
                origem, destino = path[i], path[i + 1]
                
                if self.graph.has_edge(origem, destino):
                    edge_data = self.graph[origem][destino]
                    
                    total_distance += edge_data.get('distancia', 0)
                    total_time += edge_data.get('tempo_viagem', 0)
                    total_cost += edge_data.get('custo', 0)
                    safety_scores.append(edge_data.get('indice_seguranca', 1.0))
                    environmental_impact.append(edge_data.get('impacto_ambiental', 0))
                    
                    route_segments.append({
                        'origem': origem,
                        'destino': destino,
                        'via': edge_data.get('via', 'desconhecida'),
                        'distancia': edge_data.get('distancia', 0),
                        'tempo': edge_data.get('tempo_viagem', 0),
                        'custo': edge_data.get('custo', 0)
                    })
            
            return {
                'total_distance_km': round(total_distance, 2),
                'total_time_min': round(total_time, 2),
                'total_cost_brl': round(total_cost, 2),
                'average_safety': round(np.mean(safety_scores) if safety_scores else 0, 3),
                'total_environmental_impact': round(sum(environmental_impact), 3),
                'segments_count': len(route_segments),
                'route_segments': route_segments
            }
            
        except Exception as e:
            logger.error(f"Erro ao obter resumo da rota: {e}")
            return {}
    
    def export_graph(self, filename: str, format: str = 'gexf'):
        """
        Exporta o grafo para arquivo.
        
        Args:
            filename: Nome do arquivo
            format: Formato do arquivo ('gexf', 'graphml', 'gml')
        """
        try:
            if format == 'gexf':
                nx.write_gexf(self.graph, filename)
            elif format == 'graphml':
                nx.write_graphml(self.graph, filename)
            elif format == 'gml':
                nx.write_gml(self.graph, filename)
            else:
                logger.error(f"Formato '{format}' não suportado")
                return
            
            logger.info(f"Grafo exportado para {filename}")
            
        except Exception as e:
            logger.error(f"Erro ao exportar grafo: {e}")
    
    def visualize_graph(self, save_path: Optional[str] = None):
        """
        Visualiza o grafo usando matplotlib.
        
        Args:
            save_path: Caminho para salvar a imagem (opcional)
        """
        try:
            import matplotlib.pyplot as plt
            
            # Usar posições baseadas em coordenadas geográficas
            pos = {}
            for node, data in self.graph.nodes(data=True):
                if 'lat' in data and 'lon' in data:
                    pos[node] = (data['lon'], data['lat'])
            
            plt.figure(figsize=(12, 8))
            
            # Desenhar nós
            node_colors = []
            for node, data in self.graph.nodes(data=True):
                if data.get('tipo') == 'origem':
                    node_colors.append('green')
                elif data.get('tipo') == 'destino':
                    node_colors.append('red')
                else:
                    node_colors.append('lightblue')
            
            nx.draw_networkx_nodes(self.graph, pos, node_color=node_colors, 
                                 node_size=500, alpha=0.8)
            
            # Desenhar arestas
            nx.draw_networkx_edges(self.graph, pos, alpha=0.6, arrows=True, 
                                 arrowsize=20, edge_color='gray')
            
            # Desenhar rótulos
            nx.draw_networkx_labels(self.graph, pos, font_size=8, font_weight='bold')
            
            plt.title('Rede de Transporte Iterum - Recife')
            plt.xlabel('Longitude')
            plt.ylabel('Latitude')
            plt.grid(True, alpha=0.3)
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"Visualização salva em {save_path}")
            
            plt.show()
            
        except Exception as e:
            logger.error(f"Erro ao visualizar grafo: {e}")

