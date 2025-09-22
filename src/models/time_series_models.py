"""
Modelos de séries temporais para previsão de tempo de viagem.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class ProphetTravelTimeModel:
    """
    Modelo Prophet para previsão de tempo de viagem.
    """
    
    def __init__(self):
        self.model = None
        self.is_fitted = False
        
        try:
            from prophet import Prophet
            self.Prophet = Prophet
        except ImportError:
            logger.error("Prophet não está instalado. Execute: pip install prophet")
            self.Prophet = None
    
    def fit(self, df: pd.DataFrame, **prophet_kwargs) -> bool:
        """
        Treina o modelo Prophet.
        
        Args:
            df: DataFrame com colunas 'ds' (data) e 'y' (valor)
            **prophet_kwargs: Argumentos adicionais para Prophet
            
        Returns:
            True se treinamento foi bem-sucedido
        """
        try:
            if self.Prophet is None:
                return False
            
            if df.empty or 'ds' not in df.columns or 'y' not in df.columns:
                logger.error("DataFrame deve ter colunas 'ds' e 'y'")
                return False
            
            # Configurações padrão para dados de tráfego
            default_kwargs = {
                'daily_seasonality': True,
                'weekly_seasonality': True,
                'yearly_seasonality': False,
                'seasonality_mode': 'multiplicative',
                'interval_width': 0.8
            }
            
            # Atualizar com argumentos fornecidos
            default_kwargs.update(prophet_kwargs)
            
            self.model = self.Prophet(**default_kwargs)
            self.model.fit(df)
            self.is_fitted = True
            
            logger.info("Modelo Prophet treinado com sucesso")
            return True
            
        except Exception as e:
            logger.error(f"Erro ao treinar modelo Prophet: {e}")
            return False
    
    def predict(self, periods: int = 24, freq: str = 'H') -> pd.DataFrame:
        """
        Faz previsões usando o modelo treinado.
        
        Args:
            periods: Número de períodos para prever
            freq: Frequência das previsões ('H' para horas, 'D' para dias)
            
        Returns:
            DataFrame com previsões
        """
        try:
            if not self.is_fitted or self.model is None:
                logger.error("Modelo não foi treinado")
                return pd.DataFrame()
            
            # Criar dataframe futuro
            future = self.model.make_future_dataframe(periods=periods, freq=freq)
            
            # Fazer previsões
            forecast = self.model.predict(future)
            
            return forecast
            
        except Exception as e:
            logger.error(f"Erro ao fazer previsões: {e}")
            return pd.DataFrame()
    
    def predict_specific_time(self, datetime_list: List[pd.Timestamp]) -> pd.DataFrame:
        """
        Faz previsões para horários específicos.
        
        Args:
            datetime_list: Lista de timestamps para previsão
            
        Returns:
            DataFrame com previsões para os horários especificados
        """
        try:
            if not self.is_fitted or self.model is None:
                logger.error("Modelo não foi treinado")
                return pd.DataFrame()
            
            # Criar dataframe com horários específicos
            future_df = pd.DataFrame({'ds': datetime_list})
            
            # Fazer previsões
            forecast = self.model.predict(future_df)
            
            return forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
            
        except Exception as e:
            logger.error(f"Erro ao fazer previsões específicas: {e}")
            return pd.DataFrame()
    
    def get_components(self) -> pd.DataFrame:
        """
        Obtém componentes da decomposição temporal.
        
        Returns:
            DataFrame com componentes (trend, weekly, daily, etc.)
        """
        try:
            if not self.is_fitted or self.model is None:
                logger.error("Modelo não foi treinado")
                return pd.DataFrame()
            
            # Fazer previsão para obter componentes
            future = self.model.make_future_dataframe(periods=0)
            forecast = self.model.predict(future)
            
            # Retornar componentes relevantes
            component_columns = ['ds', 'trend']
            
            if 'weekly' in forecast.columns:
                component_columns.append('weekly')
            if 'daily' in forecast.columns:
                component_columns.append('daily')
            
            return forecast[component_columns]
            
        except Exception as e:
            logger.error(f"Erro ao obter componentes: {e}")
            return pd.DataFrame()
    
    def evaluate_model(self, test_df: pd.DataFrame) -> Dict:
        """
        Avalia performance do modelo em dados de teste.
        
        Args:
            test_df: DataFrame de teste com colunas 'ds' e 'y'
            
        Returns:
            Dicionário com métricas de avaliação
        """
        try:
            if not self.is_fitted or self.model is None:
                logger.error("Modelo não foi treinado")
                return {}
            
            if test_df.empty or 'ds' not in test_df.columns or 'y' not in test_df.columns:
                logger.error("DataFrame de teste deve ter colunas 'ds' e 'y'")
                return {}
            
            # Fazer previsões para datas de teste
            forecast = self.model.predict(test_df[['ds']])
            
            # Calcular métricas
            y_true = test_df['y'].values
            y_pred = forecast['yhat'].values
            
            mae = np.mean(np.abs(y_true - y_pred))
            mse = np.mean((y_true - y_pred) ** 2)
            rmse = np.sqrt(mse)
            mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
            
            return {
                'mae': mae,
                'mse': mse,
                'rmse': rmse,
                'mape': mape,
                'r2': np.corrcoef(y_true, y_pred)[0, 1] ** 2
            }
            
        except Exception as e:
            logger.error(f"Erro ao avaliar modelo: {e}")
            return {}


class SeasonalDecomposer:
    """
    Decompositor sazonal para análise de padrões temporais.
    """
    
    def __init__(self):
        self.decomposition = None
    
    def decompose(self, df: pd.DataFrame, value_column: str = 'y',
                 date_column: str = 'ds', model: str = 'additive',
                 period: int = 24) -> Dict:
        """
        Realiza decomposição sazonal dos dados.
        
        Args:
            df: DataFrame com dados temporais
            value_column: Nome da coluna com valores
            date_column: Nome da coluna com datas
            model: Tipo de decomposição ('additive' ou 'multiplicative')
            period: Período sazonal (24 para dados horários)
            
        Returns:
            Dicionário com componentes da decomposição
        """
        try:
            from statsmodels.tsa.seasonal import seasonal_decompose
            
            if df.empty or value_column not in df.columns:
                return {}
            
            # Preparar série temporal
            df_sorted = df.sort_values(date_column)
            ts = df_sorted.set_index(date_column)[value_column]
            
            # Realizar decomposição
            self.decomposition = seasonal_decompose(
                ts, model=model, period=period, extrapolate_trend='freq'
            )
            
            return {
                'trend': self.decomposition.trend,
                'seasonal': self.decomposition.seasonal,
                'residual': self.decomposition.resid,
                'observed': self.decomposition.observed
            }
            
        except Exception as e:
            logger.error(f"Erro na decomposição sazonal: {e}")
            return {}
    
    def get_seasonal_indices(self, period: int = 24) -> pd.DataFrame:
        """
        Calcula índices sazonais por período.
        
        Args:
            period: Período sazonal
            
        Returns:
            DataFrame com índices sazonais
        """
        try:
            if self.decomposition is None:
                logger.error("Decomposição não foi realizada")
                return pd.DataFrame()
            
            seasonal_component = self.decomposition.seasonal
            
            # Calcular índices por período
            indices = []
            for i in range(period):
                period_values = seasonal_component.iloc[i::period]
                avg_seasonal = period_values.mean()
                indices.append({'periodo': i, 'indice_sazonal': avg_seasonal})
            
            return pd.DataFrame(indices)
            
        except Exception as e:
            logger.error(f"Erro ao calcular índices sazonais: {e}")
            return pd.DataFrame()


class MovingAverageModel:
    """
    Modelo de média móvel para suavização de dados.
    """
    
    def __init__(self, window: int = 7):
        self.window = window
        self.fitted_values = None
    
    def fit_predict(self, df: pd.DataFrame, value_column: str = 'y') -> pd.DataFrame:
        """
        Aplica média móvel aos dados.
        
        Args:
            df: DataFrame com dados
            value_column: Nome da coluna com valores
            
        Returns:
            DataFrame com valores suavizados
        """
        try:
            if df.empty or value_column not in df.columns:
                return df
            
            df_result = df.copy()
            
            # Calcular média móvel
            df_result['moving_average'] = df_result[value_column].rolling(
                window=self.window, center=True
            ).mean()
            
            # Calcular índices (valor / média móvel)
            df_result['seasonal_index'] = (
                df_result[value_column] / df_result['moving_average']
            )
            
            self.fitted_values = df_result
            
            return df_result
            
        except Exception as e:
            logger.error(f"Erro no modelo de média móvel: {e}")
            return df
    
    def get_seasonal_pattern(self, df: pd.DataFrame, 
                           time_column: str = 'hora') -> pd.DataFrame:
        """
        Extrai padrão sazonal dos dados.
        
        Args:
            df: DataFrame com dados processados
            time_column: Nome da coluna de tempo (hora, dia, etc.)
            
        Returns:
            DataFrame com padrão sazonal médio
        """
        try:
            if self.fitted_values is None:
                logger.error("Modelo não foi ajustado")
                return pd.DataFrame()
            
            if time_column not in self.fitted_values.columns:
                logger.error(f"Coluna '{time_column}' não encontrada")
                return pd.DataFrame()
            
            # Agrupar por período e calcular índice médio
            seasonal_pattern = self.fitted_values.groupby(time_column).agg({
                'seasonal_index': 'mean'
            }).reset_index()
            
            return seasonal_pattern
            
        except Exception as e:
            logger.error(f"Erro ao extrair padrão sazonal: {e}")
            return pd.DataFrame()

