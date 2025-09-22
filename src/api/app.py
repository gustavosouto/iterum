"""
Aplicação Flask principal para a API do Iterum.
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import sys
import os
import logging
from datetime import datetime

# Adicionar o diretório src ao path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from algorithms.graph_algorithms import MultiObjectiveOptimizer, RouteComparator
from utils.graph_builder import IterumGraphBuilder
from models.regression_models import TravelTimeRegressor
from models.time_series_models import ProphetTravelTimeModel

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)  # Permitir CORS para todas as rotas

# Variáveis globais para armazenar modelos e grafo
graph_builder = None
optimizer = None
regression_model = None
prophet_model = None


def initialize_system():
    """Inicializa o sistema Iterum."""
    global graph_builder, optimizer, regression_model, prophet_model
    
    try:
        # Criar grafo da rede
        graph_builder = IterumGraphBuilder()
        graph = graph_builder.create_recife_network()
        
        # Inicializar otimizador
        optimizer = MultiObjectiveOptimizer(graph)
        
        logger.info("Sistema Iterum inicializado com sucesso")
        return True
        
    except Exception as e:
        logger.error(f"Erro ao inicializar sistema: {e}")
        return False


@app.route('/')
def home():
    """Endpoint raiz da API."""
    return jsonify({
        'message': 'API Iterum - Sistema de Otimização de Rotas Urbanas',
        'version': '0.1.0',
        'status': 'ativo',
        'endpoints': {
            'optimize_route': '/api/optimize-route',
            'compare_algorithms': '/api/compare-algorithms',
            'route_summary': '/api/route-summary',
            'graph_info': '/api/graph-info',
            'health': '/api/health'
        }
    })


@app.route('/api/health')
def health_check():
    """Endpoint de verificação de saúde da API."""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'system_initialized': graph_builder is not None
    })


@app.route('/api/graph-info')
def graph_info():
    """Retorna informações sobre o grafo da rede."""
    try:
        if not graph_builder:
            return jsonify({'error': 'Sistema não inicializado'}), 500
        
        graph = graph_builder.graph
        
        return jsonify({
            'nodes_count': graph.number_of_nodes(),
            'edges_count': graph.number_of_edges(),
            'nodes': list(graph.nodes()),
            'is_directed': graph.is_directed(),
            'is_connected': len(list(graph.nodes())) > 0
        })
        
    except Exception as e:
        logger.error(f"Erro ao obter informações do grafo: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/optimize-route', methods=['POST'])
def optimize_route():
    """
    Otimiza rota entre origem e destino.
    
    Payload esperado:
    {
        "origem": "FAFIRE",
        "destino": "FNR",
        "algoritmo": "dijkstra",  // ou "a_star"
        "pesos": {
            "tempo": 0.5,
            "custo": 0.3,
            "impacto_ambiental": 0.2
        }
    }
    """
    try:
        if not optimizer:
            return jsonify({'error': 'Sistema não inicializado'}), 500
        
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'Dados não fornecidos'}), 400
        
        origem = data.get('origem', 'FAFIRE')
        destino = data.get('destino', 'FNR')
        algoritmo = data.get('algoritmo', 'dijkstra')
        pesos = data.get('pesos', {})
        
        # Configurar pesos se fornecidos
        if pesos:
            optimizer.set_weights(
                tempo=pesos.get('tempo', 0.5),
                custo=pesos.get('custo', 0.3),
                impacto_ambiental=pesos.get('impacto_ambiental', 0.2)
            )
        
        # Executar algoritmo
        if algoritmo == 'dijkstra':
            path, cost = optimizer.dijkstra_multi_objective(origem, destino)
        elif algoritmo == 'a_star':
            path, cost = optimizer.a_star_multi_objective(origem, destino)
        else:
            return jsonify({'error': f'Algoritmo "{algoritmo}" não suportado'}), 400
        
        if not path:
            return jsonify({'error': 'Nenhuma rota encontrada'}), 404
        
        # Obter resumo da rota
        route_summary = graph_builder.get_route_summary(path)
        
        return jsonify({
            'algoritmo': algoritmo,
            'origem': origem,
            'destino': destino,
            'path': path,
            'cost': cost,
            'summary': route_summary,
            'pesos_utilizados': optimizer.weights
        })
        
    except Exception as e:
        logger.error(f"Erro ao otimizar rota: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/compare-algorithms', methods=['POST'])
def compare_algorithms():
    """
    Compara performance de diferentes algoritmos.
    
    Payload esperado:
    {
        "origem": "FAFIRE",
        "destino": "FNR",
        "pesos": {
            "tempo": 0.5,
            "custo": 0.3,
            "impacto_ambiental": 0.2
        }
    }
    """
    try:
        if not optimizer or not graph_builder:
            return jsonify({'error': 'Sistema não inicializado'}), 500
        
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'Dados não fornecidos'}), 400
        
        origem = data.get('origem', 'FAFIRE')
        destino = data.get('destino', 'FNR')
        pesos = data.get('pesos', {})
        
        # Configurar pesos se fornecidos
        if pesos:
            optimizer.set_weights(
                tempo=pesos.get('tempo', 0.5),
                custo=pesos.get('custo', 0.3),
                impacto_ambiental=pesos.get('impacto_ambiental', 0.2)
            )
        
        # Comparar algoritmos
        comparator = RouteComparator()
        results = comparator.compare_algorithms(graph_builder.graph, origem, destino)
        
        # Adicionar resumos das rotas
        for algorithm, result in results.items():
            if result['path']:
                result['route_summary'] = graph_builder.get_route_summary(result['path'])
        
        return jsonify({
            'origem': origem,
            'destino': destino,
            'pesos_utilizados': optimizer.weights,
            'results': results
        })
        
    except Exception as e:
        logger.error(f"Erro ao comparar algoritmos: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/route-summary', methods=['POST'])
def route_summary():
    """
    Obtém resumo detalhado de uma rota específica.
    
    Payload esperado:
    {
        "path": ["FAFIRE", "CBV_INICIO", "CBV_FIM", "PINA_ENTRADA", "PINA_SAIDA", "FNR"]
    }
    """
    try:
        if not graph_builder:
            return jsonify({'error': 'Sistema não inicializado'}), 500
        
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'Dados não fornecidos'}), 400
        
        path = data.get('path', [])
        
        if not path:
            return jsonify({'error': 'Caminho não fornecido'}), 400
        
        summary = graph_builder.get_route_summary(path)
        
        return jsonify({
            'path': path,
            'summary': summary
        })
        
    except Exception as e:
        logger.error(f"Erro ao obter resumo da rota: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/update-weights', methods=['POST'])
def update_weights():
    """
    Atualiza pesos da função multi-objetivo.
    
    Payload esperado:
    {
        "tempo": 0.6,
        "custo": 0.2,
        "impacto_ambiental": 0.2
    }
    """
    try:
        if not optimizer:
            return jsonify({'error': 'Sistema não inicializado'}), 500
        
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'Dados não fornecidos'}), 400
        
        tempo = data.get('tempo', 0.5)
        custo = data.get('custo', 0.3)
        impacto_ambiental = data.get('impacto_ambiental', 0.2)
        
        optimizer.set_weights(tempo, custo, impacto_ambiental)
        
        return jsonify({
            'message': 'Pesos atualizados com sucesso',
            'novos_pesos': optimizer.weights
        })
        
    except Exception as e:
        logger.error(f"Erro ao atualizar pesos: {e}")
        return jsonify({'error': str(e)}), 500


@app.errorhandler(404)
def not_found(error):
    """Handler para erro 404."""
    return jsonify({'error': 'Endpoint não encontrado'}), 404


@app.errorhandler(500)
def internal_error(error):
    """Handler para erro 500."""
    return jsonify({'error': 'Erro interno do servidor'}), 500


if __name__ == '__main__':
    # Inicializar sistema
    if initialize_system():
        logger.info("Iniciando servidor Flask...")
        app.run(host='0.0.0.0', port=5000, debug=True)
    else:
        logger.error("Falha ao inicializar sistema. Servidor não será iniciado.")
        sys.exit(1)

