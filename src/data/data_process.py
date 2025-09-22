"""
Módulo para processamento e limpeza de dados.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class DataProcessor:
    """
    Processador principal de dados para o projeto Iterum.
    """
    
    def __init__(self):
        self.processed_data = {}
    
    def process_velocidade_vias(self, df: pd.DataFrame, 
                              horario_inicio: int = 17, 
                              horario_fim: int = 19) -> pd.DataFrame:
        """
        Processa dados de velocidade das vias para o horário de pico.
        
        Args:
            df: DataFrame com dados brutos de velocidade
            horario_inicio: Hora de início do período de interesse
            horario_fim: Hora de fim do período de interesse
            
        Returns:
            DataFrame processado e filtrado
        """
        try:
            if df.empty:
                return df
            
            # Converter coluna de horário se necessário
            if 'horario' in df.columns:
                df['horario'] = pd.to_datetime(df['horario'])
                df['hora'] = df['horario'].dt.hour
                
                # Filtrar horário de pico (17h-19h)
                df_filtered = df[
                    (df['hora'] >= horario_inicio) & 
                    (df['hora'] <= horario_fim)
                ].copy()
                
                return df_filtered
            else:
                logger.warning("Coluna 'horario' não encontrada nos dados")
                return df
                
        except Exception as e:
            logger.error(f"Erro ao processar dados de velocidade: {e}")
            return pd.DataFrame()
    
    def calculate_seasonal_index(self, df: pd.DataFrame, 
                               value_column: str = 'velocidade_media',
                               window_days: int = 7) -> pd.DataFrame:
        """
        Calcula índice sazonal usando método da porcentagem para média móvel.
        
        Args:
            df: DataFrame com dados temporais
            value_column: Nome da coluna com valores a analisar
            window_days: Janela para média móvel (em dias)
            
        Returns:
            DataFrame com índices sazonais
        """
        try:
            if df.empty or value_column not in df.columns:
                return df
            
            # Ordenar por data/hora
            df_sorted = df.sort_values('horario').copy()
            
            # Calcular média móvel
            df_sorted['media_movel'] = df_sorted[value_column].rolling(
                window=window_days * 24,  # 24 horas por dia
                center=True
            ).mean()
            
            # Calcular índice sazonal
            df_sorted['indice_sazonal'] = (
                df_sorted[value_column] / df_sorted['media_movel']
            )
            
            # Agrupar por hora do dia para obter índice médio
            indice_por_hora = df_sorted.groupby('hora').agg({
                'indice_sazonal': 'mean',
                value_column: 'mean'
            }).reset_index()
            
            return indice_por_hora
            
        except Exception as e:
            logger.error(f"Erro ao calcular índice sazonal: {e}")
            return pd.DataFrame()
    
    def prepare_regression_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepara dados para modelo de regressão linear múltipla.
        
        Args:
            df: DataFrame com dados brutos
            
        Returns:
            DataFrame preparado para modelagem
        """
        try:
            if df.empty:
                return df
            
            df_processed = df.copy()
            
            # Criar variáveis dummy para variáveis categóricas
            categorical_columns = ['modal', 'condicao_climatica', 'dia_semana']
            
            for col in categorical_columns:
                if col in df_processed.columns:
                    dummies = pd.get_dummies(df_processed[col], prefix=col)
                    df_processed = pd.concat([df_processed, dummies], axis=1)
                    df_processed.drop(col, axis=1, inplace=True)
            
            # Normalizar variáveis numéricas se necessário
            numeric_columns = ['distancia_trecho', 'indice_seguranca']
            
            for col in numeric_columns:
                if col in df_processed.columns:
                    df_processed[f'{col}_norm'] = (
                        (df_processed[col] - df_processed[col].mean()) / 
                        df_processed[col].std()
                    )
            
            return df_processed
            
        except Exception as e:
            logger.error(f"Erro ao preparar dados para regressão: {e}")
            return pd.DataFrame()
    
    def create_graph_edges(self, route_data: List[Dict]) -> List[Dict]:
        """
        Cria arestas do grafo a partir de dados de rotas.
        
        Args:
            route_data: Lista de dicionários com dados de rotas
            
        Returns:
            Lista de dicionários representando arestas do grafo
        """
        try:
            edges = []
            
            for route in route_data:
                if 'segments' in route:
                    for segment in route['segments']:
                        edge = {
                            'origem': segment.get('origem'),
                            'destino': segment.get('destino'),
                            'distancia': segment.get('distancia', 0),
                            'tempo_viagem': segment.get('tempo_viagem', 0),
                            'custo': segment.get('custo', 0),
                            'modal': segment.get('modal', 'uber_carro'),
                            'indice_seguranca': segment.get('indice_seguranca', 1.0)
                        }
                        edges.append(edge)
            
            return edges
            
        except Exception as e:
            logger.error(f"Erro ao criar arestas do grafo: {e}")
            return []
    
    def aggregate_traffic_data(self, df: pd.DataFrame, 
                             groupby_columns: List[str] = None) -> pd.DataFrame:
        """
        Agrega dados de tráfego por período e localização.
        
        Args:
            df: DataFrame com dados de tráfego
            groupby_columns: Colunas para agrupamento
            
        Returns:
            DataFrame agregado
        """
        try:
            if df.empty:
                return df
            
            if groupby_columns is None:
                groupby_columns = ['hora', 'dia_semana', 'via']
            
            # Verificar se colunas existem
            existing_columns = [col for col in groupby_columns if col in df.columns]
            
            if not existing_columns:
                logger.warning("Nenhuma coluna de agrupamento encontrada")
                return df
            
            # Agregar dados
            aggregated = df.groupby(existing_columns).agg({
                'velocidade_media': ['mean', 'std'],
                'volume_veiculos': 'sum',
                'tempo_viagem': 'mean'
            }).reset_index()
            
            # Achatar nomes das colunas
            aggregated.columns = [
                '_'.join(col).strip() if col[1] else col[0] 
                for col in aggregated.columns.values
            ]
            
            return aggregated
            
        except Exception as e:
            logger.error(f"Erro ao agregar dados de tráfego: {e}")
            return pd.DataFrame()


class TimeSeriesProcessor:
    """
    Processador específico para análise de séries temporais.
    """
    
    def __init__(self):
        self.models = {}
    
    def prepare_prophet_data(self, df: pd.DataFrame, 
                           date_column: str = 'horario',
                           value_column: str = 'tempo_viagem') -> pd.DataFrame:
        """
        Prepara dados para o modelo Prophet.
        
        Args:
            df: DataFrame com dados temporais
            date_column: Nome da coluna de data/hora
            value_column: Nome da coluna com valores a prever
            
        Returns:
            DataFrame formatado para Prophet (colunas 'ds' e 'y')
        """
        try:
            if df.empty or date_column not in df.columns or value_column not in df.columns:
                return pd.DataFrame()
            
            prophet_df = df[[date_column, value_column]].copy()
            prophet_df.columns = ['ds', 'y']
            
            # Remover valores nulos
            prophet_df = prophet_df.dropna()
            
            # Garantir que 'ds' é datetime
            prophet_df['ds'] = pd.to_datetime(prophet_df['ds'])
            
            return prophet_df
            
        except Exception as e:
            logger.error(f"Erro ao preparar dados para Prophet: {e}")
            return pd.DataFrame()
    
    def detect_outliers(self, df: pd.DataFrame, 
                       column: str, method: str = 'iqr') -> pd.DataFrame:
        """
        Detecta e marca outliers nos dados.
        
        Args:
            df: DataFrame com dados
            column: Nome da coluna para análise
            method: Método de detecção ('iqr' ou 'zscore')
            
        Returns:
            DataFrame com coluna adicional 'is_outlier'
        """
        try:
            if df.empty or column not in df.columns:
                return df
            
            df_copy = df.copy()
            
            if method == 'iqr':
                Q1 = df_copy[column].quantile(0.25)
                Q3 = df_copy[column].quantile(0.75)
                IQR = Q3 - Q1
                
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                df_copy['is_outlier'] = (
                    (df_copy[column] < lower_bound) | 
                    (df_copy[column] > upper_bound)
                )
                
            elif method == 'zscore':
                from scipy import stats
                z_scores = np.abs(stats.zscore(df_copy[column]))
                df_copy['is_outlier'] = z_scores > 3
            
            return df_copy
            
        except Exception as e:
            logger.error(f"Erro ao detectar outliers: {e}")
            return df

