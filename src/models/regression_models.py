"""
Modelos de regressão para previsão de tempo de viagem.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
import logging

logger = logging.getLogger(__name__)


class TravelTimeRegressor:
    """
    Modelo de regressão para previsão de tempo de viagem.
    """
    
    def __init__(self, model_type: str = 'linear'):
        self.model_type = model_type
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_names = []
        self.is_fitted = False
        
        # Inicializar modelo baseado no tipo
        self._initialize_model()
    
    def _initialize_model(self):
        """Inicializa o modelo baseado no tipo especificado."""
        try:
            if self.model_type == 'linear':
                self.model = LinearRegression()
            elif self.model_type == 'ridge':
                self.model = Ridge(alpha=1.0)
            elif self.model_type == 'lasso':
                self.model = Lasso(alpha=1.0)
            elif self.model_type == 'random_forest':
                self.model = RandomForestRegressor(n_estimators=100, random_state=42)
            elif self.model_type == 'gradient_boosting':
                self.model = GradientBoostingRegressor(n_estimators=100, random_state=42)
            else:
                logger.warning(f"Tipo de modelo '{self.model_type}' não reconhecido. Usando LinearRegression.")
                self.model = LinearRegression()
                self.model_type = 'linear'
                
        except Exception as e:
            logger.error(f"Erro ao inicializar modelo: {e}")
            self.model = LinearRegression()
    
    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepara features para o modelo de regressão.
        
        Args:
            df: DataFrame com dados brutos
            
        Returns:
            DataFrame com features preparadas
        """
        try:
            if df.empty:
                return df
            
            df_processed = df.copy()
            
            # Variáveis categóricas esperadas
            categorical_columns = ['modal', 'condicao_climatica', 'dia_semana']
            
            # Codificar variáveis categóricas
            for col in categorical_columns:
                if col in df_processed.columns:
                    if col not in self.label_encoders:
                        self.label_encoders[col] = LabelEncoder()
                        df_processed[f'{col}_encoded'] = self.label_encoders[col].fit_transform(
                            df_processed[col].astype(str)
                        )
                    else:
                        # Usar encoder já treinado
                        try:
                            df_processed[f'{col}_encoded'] = self.label_encoders[col].transform(
                                df_processed[col].astype(str)
                            )
                        except ValueError:
                            # Lidar com categorias não vistas
                            logger.warning(f"Categorias não vistas em '{col}'. Usando valor padrão.")
                            df_processed[f'{col}_encoded'] = 0
                    
                    # Remover coluna original
                    df_processed.drop(col, axis=1, inplace=True)
            
            # Criar features de interação se necessário
            if 'distancia_trecho' in df_processed.columns and 'horario' in df_processed.columns:
                df_processed['distancia_x_horario'] = (
                    df_processed['distancia_trecho'] * df_processed['horario']
                )
            
            return df_processed
            
        except Exception as e:
            logger.error(f"Erro ao preparar features: {e}")
            return df
    
    def fit(self, X: pd.DataFrame, y: pd.Series, test_size: float = 0.2) -> Dict:
        """
        Treina o modelo de regressão.
        
        Args:
            X: DataFrame com features
            y: Series com variável target (tempo_viagem)
            test_size: Proporção dos dados para teste
            
        Returns:
            Dicionário com métricas de treinamento
        """
        try:
            if X.empty or y.empty:
                logger.error("Dados de entrada estão vazios")
                return {}
            
            # Preparar features
            X_processed = self.prepare_features(X)
            
            # Selecionar apenas colunas numéricas
            numeric_columns = X_processed.select_dtypes(include=[np.number]).columns
            X_numeric = X_processed[numeric_columns]
            
            self.feature_names = list(X_numeric.columns)
            
            # Dividir dados
            X_train, X_test, y_train, y_test = train_test_split(
                X_numeric, y, test_size=test_size, random_state=42
            )
            
            # Normalizar features (exceto para árvores)
            if self.model_type in ['linear', 'ridge', 'lasso']:
                X_train_scaled = self.scaler.fit_transform(X_train)
                X_test_scaled = self.scaler.transform(X_test)
            else:
                X_train_scaled = X_train
                X_test_scaled = X_test
            
            # Treinar modelo
            self.model.fit(X_train_scaled, y_train)
            self.is_fitted = True
            
            # Fazer previsões
            y_train_pred = self.model.predict(X_train_scaled)
            y_test_pred = self.model.predict(X_test_scaled)
            
            # Calcular métricas
            train_metrics = self._calculate_metrics(y_train, y_train_pred)
            test_metrics = self._calculate_metrics(y_test, y_test_pred)
            
            # Cross-validation
            cv_scores = cross_val_score(
                self.model, X_train_scaled, y_train, cv=5, scoring='neg_mean_squared_error'
            )
            
            results = {
                'train_metrics': train_metrics,
                'test_metrics': test_metrics,
                'cv_rmse_mean': np.sqrt(-cv_scores.mean()),
                'cv_rmse_std': np.sqrt(cv_scores.std()),
                'feature_names': self.feature_names
            }
            
            # Adicionar importância das features para modelos baseados em árvores
            if hasattr(self.model, 'feature_importances_'):
                results['feature_importance'] = dict(
                    zip(self.feature_names, self.model.feature_importances_)
                )
            
            # Adicionar coeficientes para modelos lineares
            if hasattr(self.model, 'coef_'):
                results['coefficients'] = dict(
                    zip(self.feature_names, self.model.coef_)
                )
                if hasattr(self.model, 'intercept_'):
                    results['intercept'] = self.model.intercept_
            
            logger.info(f"Modelo {self.model_type} treinado com sucesso")
            logger.info(f"RMSE teste: {test_metrics['rmse']:.4f}")
            logger.info(f"R² teste: {test_metrics['r2']:.4f}")
            
            return results
            
        except Exception as e:
            logger.error(f"Erro ao treinar modelo: {e}")
            return {}
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Faz previsões usando o modelo treinado.
        
        Args:
            X: DataFrame com features
            
        Returns:
            Array com previsões
        """
        try:
            if not self.is_fitted:
                logger.error("Modelo não foi treinado")
                return np.array([])
            
            # Preparar features
            X_processed = self.prepare_features(X)
            
            # Selecionar apenas features usadas no treinamento
            X_features = X_processed[self.feature_names]
            
            # Normalizar se necessário
            if self.model_type in ['linear', 'ridge', 'lasso']:
                X_scaled = self.scaler.transform(X_features)
            else:
                X_scaled = X_features
            
            # Fazer previsões
            predictions = self.model.predict(X_scaled)
            
            return predictions
            
        except Exception as e:
            logger.error(f"Erro ao fazer previsões: {e}")
            return np.array([])
    
    def _calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
        """
        Calcula métricas de avaliação.
        
        Args:
            y_true: Valores reais
            y_pred: Valores preditos
            
        Returns:
            Dicionário com métricas
        """
        try:
            mae = mean_absolute_error(y_true, y_pred)
            mse = mean_squared_error(y_true, y_pred)
            rmse = np.sqrt(mse)
            r2 = r2_score(y_true, y_pred)
            
            # MAPE (Mean Absolute Percentage Error)
            mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
            
            return {
                'mae': mae,
                'mse': mse,
                'rmse': rmse,
                'r2': r2,
                'mape': mape
            }
            
        except Exception as e:
            logger.error(f"Erro ao calcular métricas: {e}")
            return {}
    
    def get_model_equation(self) -> str:
        """
        Retorna equação do modelo (para modelos lineares).
        
        Returns:
            String com equação do modelo
        """
        try:
            if not self.is_fitted or not hasattr(self.model, 'coef_'):
                return "Equação não disponível para este tipo de modelo"
            
            equation = "TempoViagem = "
            
            if hasattr(self.model, 'intercept_'):
                equation += f"{self.model.intercept_:.4f}"
            
            for i, (feature, coef) in enumerate(zip(self.feature_names, self.model.coef_)):
                if coef >= 0 and i > 0:
                    equation += " + "
                elif coef < 0:
                    equation += " - " if i > 0 else "-"
                    coef = abs(coef)
                
                equation += f"{coef:.4f} * {feature}"
            
            return equation
            
        except Exception as e:
            logger.error(f"Erro ao gerar equação: {e}")
            return "Erro ao gerar equação"


class ModelComparator:
    """
    Comparador de diferentes modelos de regressão.
    """
    
    def __init__(self):
        self.models = {}
        self.results = {}
    
    def compare_models(self, X: pd.DataFrame, y: pd.Series, 
                      model_types: List[str] = None) -> Dict:
        """
        Compara performance de diferentes modelos.
        
        Args:
            X: DataFrame com features
            y: Series com target
            model_types: Lista de tipos de modelos para comparar
            
        Returns:
            Dicionário com resultados comparativos
        """
        try:
            if model_types is None:
                model_types = ['linear', 'ridge', 'lasso', 'random_forest', 'gradient_boosting']
            
            results = {}
            
            for model_type in model_types:
                logger.info(f"Treinando modelo: {model_type}")
                
                model = TravelTimeRegressor(model_type=model_type)
                model_results = model.fit(X, y)
                
                if model_results:
                    results[model_type] = {
                        'model': model,
                        'test_rmse': model_results['test_metrics']['rmse'],
                        'test_r2': model_results['test_metrics']['r2'],
                        'test_mae': model_results['test_metrics']['mae'],
                        'cv_rmse': model_results['cv_rmse_mean'],
                        'full_results': model_results
                    }
            
            # Ordenar por RMSE (menor é melhor)
            sorted_results = sorted(
                results.items(), 
                key=lambda x: x[1]['test_rmse']
            )
            
            self.results = dict(sorted_results)
            
            # Log dos resultados
            logger.info("Comparação de modelos:")
            for model_name, result in sorted_results:
                logger.info(f"{model_name}: RMSE={result['test_rmse']:.4f}, R²={result['test_r2']:.4f}")
            
            return self.results
            
        except Exception as e:
            logger.error(f"Erro ao comparar modelos: {e}")
            return {}
    
    def get_best_model(self) -> Optional[TravelTimeRegressor]:
        """
        Retorna o melhor modelo baseado no RMSE.
        
        Returns:
            Melhor modelo treinado
        """
        try:
            if not self.results:
                logger.error("Nenhum modelo foi comparado ainda")
                return None
            
            best_model_name = list(self.results.keys())[0]
            best_model = self.results[best_model_name]['model']
            
            logger.info(f"Melhor modelo: {best_model_name}")
            
            return best_model
            
        except Exception as e:
            logger.error(f"Erro ao obter melhor modelo: {e}")
            return None

