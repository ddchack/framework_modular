# learning/continuous_learner.py
"""
Módulo de Aprendizaje Continuo
==============================

Este módulo implementa capacidades de aprendizaje continuo para el asistente de IA.
Permite al sistema aprender de las interacciones y mejorar con el tiempo.

#ddchack - Sistema de aprendizaje adaptativo
"""

import json
import pickle
import logging
import asyncio
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from collections import defaultdict, deque
from abc import ABC, abstractmethod
import os

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    logging.warning("NumPy no disponible. Funcionalidad de ML limitada.")

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics.pairwise import cosine_similarity
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logging.warning("Scikit-learn no disponible. Usando aprendizaje básico.")

logger = logging.getLogger(__name__)

@dataclass
class Interaction:
    """
    Representa una interacción usuario-asistente
    #ddchack - Estructura de datos para interacciones
    """
    user_message: str
    assistant_response: str
    timestamp: datetime
    user_id: str
    nlp_result: Dict[str, Any]
    feedback: Optional[float] = None  # Rating 0-1
    context: Optional[Dict[str, Any]] = None

@dataclass
class LearningMetrics:
    """
    Métricas del sistema de aprendizaje
    """
    total_interactions: int
    feedback_count: int
    average_feedback: float
    last_update: datetime
    model_version: str

class BaseLearningStrategy(ABC):
    """
    Estrategia base para aprendizaje
    """
    
    @abstractmethod
    async def learn(self, interactions: List[Interaction]) -> Dict[str, Any]:
        """Aprende de las interacciones"""
        pass
    
    @abstractmethod
    async def predict(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Hace predicciones basadas en el aprendizaje"""
        pass

class PatternLearningStrategy(BaseLearningStrategy):
    """
    Estrategia de aprendizaje basada en patrones
    #ddchack - Aprendizaje por detección de patrones
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.patterns = defaultdict(list)
        self.response_templates = {}
        self.intent_responses = defaultdict(list)
        
    async def learn(self, interactions: List[Interaction]) -> Dict[str, Any]:
        """
        Aprende patrones de las interacciones
        """
        learned_patterns = 0
        
        for interaction in interactions:
            intent = interaction.nlp_result.get('intent', 'general')
            user_msg = interaction.user_message.lower()
            response = interaction.assistant_response
            
            # Aprender patrones de respuesta por intención
            if intent not in self.intent_responses:
                self.intent_responses[intent] = []
            
            # Solo guardar respuestas con feedback positivo
            if interaction.feedback and interaction.feedback > 0.7:
                self.intent_responses[intent].append({
                    'response': response,
                    'feedback': interaction.feedback,
                    'context': interaction.context or {}
                })
                learned_patterns += 1
            
            # Identificar patrones en mensajes de usuario
            words = user_msg.split()
            for i in range(len(words) - 1):
                bigram = f"{words[i]} {words[i+1]}"
                self.patterns[intent].append(bigram)
        
        return {
            'patterns_learned': learned_patterns,
            'intents_updated': len(self.intent_responses),
            'total_patterns': sum(len(patterns) for patterns in self.patterns.values())
        }
    
    async def predict(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Predice respuesta basada en patrones aprendidos
        """
        intent = input_data.get('intent', 'general')
        user_message = input_data.get('message', '').lower()
        
        # Buscar mejores respuestas para la intención
        if intent in self.intent_responses and self.intent_responses[intent]:
            responses = self.intent_responses[intent]
            # Ordenar por feedback y tomar las mejores
            best_responses = sorted(responses, key=lambda x: x['feedback'], reverse=True)[:3]
            
            return {
                'suggested_responses': [r['response'] for r in best_responses],
                'confidence': len(best_responses) / max(len(responses), 1),
                'based_on_patterns': True
            }
        
        return {
            'suggested_responses': [],
            'confidence': 0.0,
            'based_on_patterns': False
        }

class FeedbackLearningStrategy(BaseLearningStrategy):
    """
    Estrategia de aprendizaje basada en feedback del usuario
    #ddchack - Aprendizaje por refuerzo con feedback
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.feedback_history = deque(maxlen=config.get('memory_limit', 1000))
        self.response_scores = defaultdict(list)
        self.learning_rate = config.get('learning_rate', 0.1)
        
    async def learn(self, interactions: List[Interaction]) -> Dict[str, Any]:
        """
        Aprende de feedback positivo y negativo
        """
        feedback_processed = 0
        
        for interaction in interactions:
            if interaction.feedback is not None:
                self.feedback_history.append(interaction)
                
                # Analizar qué hizo que la respuesta fuera buena o mala
                response_features = self._extract_response_features(interaction)
                
                # Actualizar scores de características de respuesta
                for feature, value in response_features.items():
                    self.response_scores[feature].append({
                        'value': value,
                        'feedback': interaction.feedback,
                        'timestamp': interaction.timestamp
                    })
                
                feedback_processed += 1
        
        return {
            'feedback_processed': feedback_processed,
            'total_feedback_history': len(self.feedback_history),
            'features_tracked': len(self.response_scores)
        }
    
    def _extract_response_features(self, interaction: Interaction) -> Dict[str, Any]:
        """
        Extrae características de la respuesta para análisis
        """
        response = interaction.assistant_response
        
        return {
            'response_length': len(response),
            'word_count': len(response.split()),
            'has_question': '?' in response,
            'has_greeting': any(word in response.lower() for word in ['hola', 'buenos', 'saludos']),
            'has_thanks': any(word in response.lower() for word in ['gracias', 'de nada']),
            'sentiment_positive': interaction.nlp_result.get('sentiment', {}).get('positive', 0),
            'intent': interaction.nlp_result.get('intent', 'general'),
            'entities_mentioned': len(interaction.nlp_result.get('entities', [])),
            'response_formality': self._assess_formality(response)
        }
    
    def _assess_formality(self, text: str) -> float:
        """
        Evalúa el nivel de formalidad del texto (0-1)
        """
        formal_indicators = ['usted', 'por favor', 'estimado', 'cordialmente', 'atentamente']
        informal_indicators = ['tú', 'te', 'hey', 'ok', 'vale', 'genial']
        
        formal_count = sum(1 for word in formal_indicators if word in text.lower())
        informal_count = sum(1 for word in informal_indicators if word in text.lower())
        
        if formal_count + informal_count == 0:
            return 0.5  # Neutral
        
        return formal_count / (formal_count + informal_count)
    
    async def predict(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Predice características óptimas para la respuesta
        """
        if not self.feedback_history:
            return {'recommendations': {}, 'confidence': 0.0}
        
        # Analizar qué características correlacionan con buen feedback
        recommendations = {}
        
        for feature, scores in self.response_scores.items():
            if len(scores) >= 5:  # Mínimo de datos para hacer recomendaciones
                positive_feedback = [s for s in scores if s['feedback'] > 0.7]
                negative_feedback = [s for s in scores if s['feedback'] < 0.3]
                
                if positive_feedback and negative_feedback:
                    avg_positive = np.mean([s['value'] for s in positive_feedback]) if NUMPY_AVAILABLE else sum(s['value'] for s in positive_feedback) / len(positive_feedback)
                    avg_negative = np.mean([s['value'] for s in negative_feedback]) if NUMPY_AVAILABLE else sum(s['value'] for s in negative_feedback) / len(negative_feedback)
                    
                    if abs(avg_positive - avg_negative) > 0.1:  # Diferencia significativa
                        recommendations[feature] = {
                            'target_value': avg_positive,
                            'confidence': len(positive_feedback) / len(scores)
                        }
        
        return {
            'recommendations': recommendations,
            'confidence': len(recommendations) / max(len(self.response_scores), 1)
        }

class ContextualLearningStrategy(BaseLearningStrategy):
    """
    Estrategia de aprendizaje contextual
    #ddchack - Aprendizaje consciente del contexto
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.context_patterns = defaultdict(lambda: defaultdict(list))
        self.temporal_patterns = defaultdict(list)
        
    async def learn(self, interactions: List[Interaction]) -> Dict[str, Any]:
        """
        Aprende patrones contextuales
        """
        context_patterns_learned = 0
        
        for interaction in interactions:
            if interaction.context:
                # Aprender patrones por hora del día
                hour = interaction.timestamp.hour
                intent = interaction.nlp_result.get('intent', 'general')
                
                self.temporal_patterns[hour].append({
                    'intent': intent,
                    'feedback': interaction.feedback,
                    'response_length': len(interaction.assistant_response)
                })
                
                # Aprender patrones por contexto de conversación
                if 'conversation_stage' in interaction.context:
                    stage = interaction.context['conversation_stage']
                    self.context_patterns[stage][intent].append({
                        'response': interaction.assistant_response,
                        'feedback': interaction.feedback,
                        'timestamp': interaction.timestamp
                    })
                    
                context_patterns_learned += 1
        
        return {
            'context_patterns_learned': context_patterns_learned,
            'temporal_patterns': len(self.temporal_patterns),
            'context_stages': len(self.context_patterns)
        }
    
    async def predict(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Predice basado en contexto temporal y conversacional
        """
        current_hour = datetime.now().hour
        intent = input_data.get('intent', 'general')
        context = input_data.get('context', {})
        
        predictions = {}
        
        # Predicciones temporales
        if current_hour in self.temporal_patterns:
            hour_data = self.temporal_patterns[current_hour]
            intent_matches = [d for d in hour_data if d['intent'] == intent]
            
            if intent_matches:
                avg_response_length = sum(d['response_length'] for d in intent_matches) / len(intent_matches)
                predictions['recommended_response_length'] = avg_response_length
        
        # Predicciones contextuales
        if 'conversation_stage' in context:
            stage = context['conversation_stage']
            if stage in self.context_patterns and intent in self.context_patterns[stage]:
                stage_responses = self.context_patterns[stage][intent]
                best_responses = [r for r in stage_responses if r.get('feedback', 0) > 0.7]
                
                if best_responses:
                    predictions['context_optimized_responses'] = [r['response'] for r in best_responses[-3:]]
        
        return {
            'predictions': predictions,
            'confidence': len(predictions) / 3  # Máximo 3 tipos de predicción
        }

class ContinuousLearner:
    """
    Sistema principal de aprendizaje continuo
    #ddchack - Coordinador de aprendizaje inteligente
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.enabled = config.get('enabled', True)
        self.memory_limit = config.get('memory_limit', 1000)
        self.save_interval = config.get('save_interval', 300)  # segundos
        
        # Almacenamiento de interacciones
        self.interactions = deque(maxlen=self.memory_limit)
        self.metrics = LearningMetrics(
            total_interactions=0,
            feedback_count=0,
            average_feedback=0.0,
            last_update=datetime.now(),
            model_version="1.0.0"
        )
        
        # Estrategias de aprendizaje
        self.strategies = {
            'pattern': PatternLearningStrategy(config),
            'feedback': FeedbackLearningStrategy(config),
            'contextual': ContextualLearningStrategy(config)
        }
        
        # Estado de aprendizaje
        self.learning_active = False
        self.last_save = datetime.now()
        
        # Archivos de persistencia
        self.data_dir = config.get('data_dir', './data/learning')
        os.makedirs(self.data_dir, exist_ok=True)
        
        # Cargar datos previos
        asyncio.create_task(self._load_previous_data())
        
        logger.info("Sistema de aprendizaje continuo inicializado")
    
    async def learn_from_interaction(self, user_message: str, assistant_response: str, 
                                   nlp_result: Dict[str, Any], user_id: str = "default",
                                   feedback: Optional[float] = None, 
                                   context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Aprende de una nueva interacción
        
        Args:
            user_message (str): Mensaje del usuario
            assistant_response (str): Respuesta del asistente
            nlp_result (Dict): Resultado del análisis NLP
            user_id (str): ID del usuario
            feedback (Optional[float]): Feedback del usuario (0-1)
            context (Optional[Dict]): Contexto adicional
            
        Returns:
            Dict: Resultado del aprendizaje
        """
        if not self.enabled:
            return {'status': 'disabled'}
        
        # Crear interacción
        interaction = Interaction(
            user_message=user_message,
            assistant_response=assistant_response,
            timestamp=datetime.now(),
            user_id=user_id,
            nlp_result=nlp_result,
            feedback=feedback,
            context=context
        )
        
        # Agregar a memoria
        self.interactions.append(interaction)
        self.metrics.total_interactions += 1
        
        if feedback is not None:
            self.metrics.feedback_count += 1
            # Actualizar promedio de feedback
            total_feedback = (self.metrics.average_feedback * (self.metrics.feedback_count - 1) + feedback)
            self.metrics.average_feedback = total_feedback / self.metrics.feedback_count
        
        # Aprendizaje periódico
        learning_results = {}
        if len(self.interactions) % 10 == 0:  # Aprender cada 10 interacciones
            learning_results = await self._perform_learning()
        
        # Guardar periódicamente
        if (datetime.now() - self.last_save).seconds > self.save_interval:
            await self._save_data()
        
        return {
            'status': 'learned',
            'interaction_id': len(self.interactions),
            'learning_results': learning_results,
            'metrics': asdict(self.metrics)
        }
    
    async def _perform_learning(self) -> Dict[str, Any]:
        """
        Ejecuta el proceso de aprendizaje con todas las estrategias
        """
        if self.learning_active:
            return {'status': 'already_learning'}
        
        self.learning_active = True
        results = {}
        
        try:
            # Obtener interacciones recientes para aprendizaje
            recent_interactions = list(self.interactions)[-100:]  # Últimas 100
            
            # Aplicar cada estrategia de aprendizaje
            for strategy_name, strategy in self.strategies.items():
                try:
                    strategy_result = await strategy.learn(recent_interactions)
                    results[strategy_name] = strategy_result
                    logger.info(f"Aprendizaje completado para estrategia: {strategy_name}")
                except Exception as e:
                    logger.error(f"Error en estrategia {strategy_name}: {e}")
                    results[strategy_name] = {'error': str(e)}
            
            self.metrics.last_update = datetime.now()
            
        finally:
            self.learning_active = False
        
        return results
    
    async def get_learning_insights(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Obtiene insights y recomendaciones basadas en el aprendizaje
        
        Args:
            input_data (Dict): Datos de entrada (mensaje, intención, contexto)
            
        Returns:
            Dict: Insights y recomendaciones
        """
        if not self.enabled or not self.interactions:
            return {'insights': {}, 'recommendations': {}}
        
        insights = {}
        
        # Obtener predicciones de cada estrategia
        for strategy_name, strategy in self.strategies.items():
            try:
                prediction = await strategy.predict(input_data)
                insights[strategy_name] = prediction
            except Exception as e:
                logger.error(f"Error obteniendo insights de {strategy_name}: {e}")
                insights[strategy_name] = {'error': str(e)}
        
        # Combinar recomendaciones
        combined_recommendations = self._combine_insights(insights)
        
        return {
            'insights': insights,
            'recommendations': combined_recommendations,
            'confidence': self._calculate_overall_confidence(insights)
        }
    
    def _combine_insights(self, insights: Dict[str, Any]) -> Dict[str, Any]:
        """
        Combina insights de diferentes estrategias
        """
        recommendations = {}
        
        # Combinar respuestas sugeridas
        all_suggested_responses = []
        for strategy_insights in insights.values():
            if 'suggested_responses' in strategy_insights:
                all_suggested_responses.extend(strategy_insights['suggested_responses'])
        
        if all_suggested_responses:
            # Eliminar duplicados y tomar las mejores
            unique_responses = list(set(all_suggested_responses))
            recommendations['suggested_responses'] = unique_responses[:3]
        
        # Combinar recomendaciones de características
        feature_recommendations = {}
        for strategy_insights in insights.values():
            if 'recommendations' in strategy_insights:
                feature_recommendations.update(strategy_insights['recommendations'])
        
        if feature_recommendations:
            recommendations['response_features'] = feature_recommendations
        
        return recommendations
    
    def _calculate_overall_confidence(self, insights: Dict[str, Any]) -> float:
        """
        Calcula la confianza general basada en todos los insights
        """
        confidences = []
        
        for strategy_insights in insights.values():
            if 'confidence' in strategy_insights and isinstance(strategy_insights['confidence'], (int, float)):
                confidences.append(strategy_insights['confidence'])
        
        if not confidences:
            return 0.0
        
        # Promedio ponderado (dar más peso a estrategias con más datos)
        return sum(confidences) / len(confidences)
    
    async def _save_data(self) -> None:
        """
        Guarda datos de aprendizaje de forma persistente
        """
        try:
            # Guardar interacciones
            interactions_file = os.path.join(self.data_dir, 'interactions.pkl')
            with open(interactions_file, 'wb') as f:
                pickle.dump(list(self.interactions), f)
            
            # Guardar métricas
            metrics_file = os.path.join(self.data_dir, 'metrics.json')
            with open(metrics_file, 'w') as f:
                json.dump(asdict(self.metrics), f, default=str, indent=2)
            
            # Guardar estado de estrategias
            strategies_file = os.path.join(self.data_dir, 'strategies.pkl')
            with open(strategies_file, 'wb') as f:
                pickle.dump(self.strategies, f)
            
            self.last_save = datetime.now()
            logger.info("Datos de aprendizaje guardados correctamente")
            
        except Exception as e:
            logger.error(f"Error guardando datos de aprendizaje: {e}")
    
    async def _load_previous_data(self) -> None:
        """
        Carga datos de aprendizaje previos
        """
        try:
            # Cargar interacciones
            interactions_file = os.path.join(self.data_dir, 'interactions.pkl')
            if os.path.exists(interactions_file):
                with open(interactions_file, 'rb') as f:
                    loaded_interactions = pickle.load(f)
                    self.interactions.extend(loaded_interactions)
                logger.info(f"Cargadas {len(loaded_interactions)} interacciones previas")
            
            # Cargar métricas
            metrics_file = os.path.join(self.data_dir, 'metrics.json')
            if os.path.exists(metrics_file):
                with open(metrics_file, 'r') as f:
                    metrics_data = json.load(f)
                    self.metrics = LearningMetrics(
                        total_interactions=metrics_data.get('total_interactions', 0),
                        feedback_count=metrics_data.get('feedback_count', 0),
                        average_feedback=metrics_data.get('average_feedback', 0.0),
                        last_update=datetime.fromisoformat(metrics_data.get('last_update', datetime.now().isoformat())),
                        model_version=metrics_data.get('model_version', '1.0.0')
                    )
                logger.info("Métricas de aprendizaje cargadas")
            
            # Cargar estado de estrategias
            strategies_file = os.path.join(self.data_dir, 'strategies.pkl')
            if os.path.exists(strategies_file):
                with open(strategies_file, 'rb') as f:
                    loaded_strategies = pickle.load(f)
                    self.strategies.update(loaded_strategies)
                logger.info("Estados de estrategias cargados")
                
        except Exception as e:
            logger.error(f"Error cargando datos previos: {e}")
            logger.info("Iniciando con datos en blanco")
    
    def add_feedback(self, interaction_index: int, feedback: float) -> bool:
        """
        Agrega feedback a una interacción específica
        
        Args:
            interaction_index (int): Índice de la interacción
            feedback (float): Valor del feedback (0-1)
            
        Returns:
            bool: True si se agregó correctamente
        """
        try:
            if 0 <= interaction_index < len(self.interactions):
                interaction = list(self.interactions)[interaction_index]
                interaction.feedback = feedback
                
                # Actualizar métricas
                if self.metrics.feedback_count == 0:
                    self.metrics.average_feedback = feedback
                else:
                    total_feedback = (self.metrics.average_feedback * self.metrics.feedback_count + feedback)
                    self.metrics.feedback_count += 1
                    self.metrics.average_feedback = total_feedback / self.metrics.feedback_count
                
                logger.info(f"Feedback {feedback} agregado a interacción {interaction_index}")
                return True
            else:
                logger.warning(f"Índice de interacción inválido: {interaction_index}")
                return False
                
        except Exception as e:
            logger.error(f"Error agregando feedback: {e}")
            return False
    
    def get_learning_statistics(self) -> Dict[str, Any]:
        """
        Retorna estadísticas completas del sistema de aprendizaje
        #ddchack - Métricas y estadísticas del aprendizaje
        """
        if not self.interactions:
            return {
                'status': 'no_data',
                'message': 'No hay datos de aprendizaje disponibles'
            }
        
        interactions_list = list(self.interactions)
        
        # Estadísticas básicas
        stats = {
            'total_interactions': len(interactions_list),
            'feedback_percentage': (self.metrics.feedback_count / len(interactions_list)) * 100,
            'average_feedback': round(self.metrics.average_feedback, 3),
            'last_update': self.metrics.last_update.isoformat(),
            'learning_enabled': self.enabled
        }
        
        # Análisis temporal
        if len(interactions_list) > 1:
            first_interaction = interactions_list[0].timestamp
            last_interaction = interactions_list[-1].timestamp
            time_span = last_interaction - first_interaction
            
            stats['time_span_days'] = time_span.days
            stats['interactions_per_day'] = len(interactions_list) / max(time_span.days, 1)
        
        # Análisis por intención
        intent_stats = defaultdict(int)
        feedback_by_intent = defaultdict(list)
        
        for interaction in interactions_list:
            intent = interaction.nlp_result.get('intent', 'unknown')
            intent_stats[intent] += 1
            
            if interaction.feedback is not None:
                feedback_by_intent[intent].append(interaction.feedback)
        
        stats['intent_distribution'] = dict(intent_stats)
        stats['intent_feedback'] = {
            intent: {
                'count': len(feedbacks),
                'average': sum(feedbacks) / len(feedbacks) if feedbacks else 0
            }
            for intent, feedbacks in feedback_by_intent.items()
        }
        
        # Análisis de usuarios
        user_stats = defaultdict(int)
        for interaction in interactions_list:
            user_stats[interaction.user_id] += 1
        
        stats['user_distribution'] = dict(user_stats)
        stats['active_users'] = len(user_stats)
        
        return stats
    
    async def export_learning_data(self, format: str = 'json') -> str:
        """
        Exporta datos de aprendizaje en el formato especificado
        
        Args:
            format (str): Formato de exportación ('json', 'csv')
            
        Returns:
            str: Ruta del archivo exportado
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        if format.lower() == 'json':
            filename = f"learning_export_{timestamp}.json"
            filepath = os.path.join(self.data_dir, filename)
            
            export_data = {
                'metadata': {
                    'export_timestamp': datetime.now().isoformat(),
                    'total_interactions': len(self.interactions),
                    'metrics': asdict(self.metrics)
                },
                'interactions': [
                    {
                        'user_message': interaction.user_message,
                        'assistant_response': interaction.assistant_response,
                        'timestamp': interaction.timestamp.isoformat(),
                        'user_id': interaction.user_id,
                        'nlp_result': interaction.nlp_result,
                        'feedback': interaction.feedback,
                        'context': interaction.context
                    }
                    for interaction in self.interactions
                ],
                'statistics': self.get_learning_statistics()
            }
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False)
                
        elif format.lower() == 'csv':
            filename = f"learning_export_{timestamp}.csv"
            filepath = os.path.join(self.data_dir, filename)
            
            # Crear CSV básico (requeriría pandas para implementación completa)
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write("timestamp,user_id,user_message,assistant_response,intent,feedback\n")
                for interaction in self.interactions:
                    intent = interaction.nlp_result.get('intent', 'unknown')
                    feedback = interaction.feedback or ''
                    # Escapar comillas en CSV
                    user_msg = interaction.user_message.replace('"', '""')
                    assistant_resp = interaction.assistant_response.replace('"', '""')
                    
                    f.write(f'"{interaction.timestamp.isoformat()}","{interaction.user_id}","{user_msg}","{assistant_resp}","{intent}","{feedback}"\n')
        
        else:
            raise ValueError(f"Formato no soportado: {format}")
        
        logger.info(f"Datos exportados a: {filepath}")
        return filepath
    
    def reset_learning_data(self, confirm: bool = False) -> bool:
        """
        Reinicia todos los datos de aprendizaje
        
        Args:
            confirm (bool): Confirmación para la acción destructiva
            
        Returns:
            bool: True si se reinició correctamente
        """
        if not confirm:
            logger.warning("reset_learning_data requiere confirmación explícita")
            return False
        
        try:
            # Limpiar memoria
            self.interactions.clear()
            
            # Reiniciar métricas
            self.metrics = LearningMetrics(
                total_interactions=0,
                feedback_count=0,
                average_feedback=0.0,
                last_update=datetime.now(),
                model_version="1.0.0"
            )
            
            # Reiniciar estrategias
            self.strategies = {
                'pattern': PatternLearningStrategy(self.config),
                'feedback': FeedbackLearningStrategy(self.config),
                'contextual': ContextualLearningStrategy(self.config)
            }
            
            # Eliminar archivos de datos
            for filename in ['interactions.pkl', 'metrics.json', 'strategies.pkl']:
                filepath = os.path.join(self.data_dir, filename)
                if os.path.exists(filepath):
                    os.remove(filepath)
            
            logger.info("Datos de aprendizaje reiniciados completamente")
            return True
            
        except Exception as e:
            logger.error(f"Error reiniciando datos de aprendizaje: {e}")
            return False

# Funciones auxiliares para análisis de aprendizaje

def analyze_learning_trends(learner: ContinuousLearner, days: int = 7) -> Dict[str, Any]:
    """
    Analiza tendencias de aprendizaje en los últimos días
    #ddchack - Análisis de tendencias temporales
    """
    if not learner.interactions:
        return {'error': 'No hay datos disponibles'}
    
    cutoff_date = datetime.now() - timedelta(days=days)
    recent_interactions = [
        interaction for interaction in learner.interactions 
        if interaction.timestamp >= cutoff_date
    ]
    
    if not recent_interactions:
        return {'error': f'No hay interacciones en los últimos {days} días'}
    
    # Análisis por día
    daily_stats = defaultdict(lambda: {'interactions': 0, 'feedback_count': 0, 'avg_feedback': 0})
    
    for interaction in recent_interactions:
        day_key = interaction.timestamp.date().isoformat()
        daily_stats[day_key]['interactions'] += 1
        
        if interaction.feedback is not None:
            current_count = daily_stats[day_key]['feedback_count']
            current_avg = daily_stats[day_key]['avg_feedback']
            
            daily_stats[day_key]['feedback_count'] += 1
            new_count = daily_stats[day_key]['feedback_count']
            daily_stats[day_key]['avg_feedback'] = (current_avg * current_count + interaction.feedback) / new_count
    
    return {
        'period_days': days,
        'total_interactions': len(recent_interactions),
        'daily_breakdown': dict(daily_stats),
        'trend_summary': {
            'most_active_day': max(daily_stats.items(), key=lambda x: x[1]['interactions'])[0] if daily_stats else None,
            'best_feedback_day': max(daily_stats.items(), key=lambda x: x[1]['avg_feedback'])[0] if daily_stats else None
        }
    }

def generate_learning_report(learner: ContinuousLearner) -> str:
    """
    Genera un reporte completo del sistema de aprendizaje
    #ddchack - Generador de reportes inteligentes
    """
    stats = learner.get_learning_statistics()
    trends = analyze_learning_trends(learner)
    
    report_lines = [
        "=== REPORTE DE APRENDIZAJE DEL ASISTENTE IA ===",
        f"Generado: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "📊 ESTADÍSTICAS GENERALES:",
        f"  • Total de interacciones: {stats.get('total_interactions', 0)}",
        f"  • Porcentaje con feedback: {stats.get('feedback_percentage', 0):.1f}%",
        f"  • Feedback promedio: {stats.get('average_feedback', 0):.3f}/1.0",
        f"  • Usuarios activos: {stats.get('active_users', 0)}",
        "",
        "🎯 DISTRIBUCIÓN POR INTENCIONES:"
    ]
    
    intent_dist = stats.get('intent_distribution', {})
    for intent, count in sorted(intent_dist.items(), key=lambda x: x[1], reverse=True)[:5]:
        percentage = (count / stats.get('total_interactions', 1)) * 100
        report_lines.append(f"  • {intent}: {count} ({percentage:.1f}%)")
    
    if trends.get('daily_breakdown'):
        report_lines.extend([
            "",
            "📈 TENDENCIAS RECIENTES (últimos 7 días):",
            f"  • Total interacciones: {trends.get('total_interactions', 0)}",
            f"  • Día más activo: {trends.get('trend_summary', {}).get('most_active_day', 'N/A')}",
            f"  • Mejor día (feedback): {trends.get('trend_summary', {}).get('best_feedback_day', 'N/A')}"
        ])
    
    report_lines.extend([
        "",
        "🔧 ESTADO DEL SISTEMA:",
        f"  • Aprendizaje activo: {'✅ Sí' if learner.enabled else '❌ No'}",
        f"  • Última actualización: {stats.get('last_update', 'N/A')}",
        f"  • Memoria utilizada: {len(learner.interactions)}/{learner.memory_limit}",
        "",
        "--- Fin del reporte ---"
    ])
    
    return "\n".join(report_lines)