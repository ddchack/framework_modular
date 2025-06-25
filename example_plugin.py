# plugins/example_plugin.py
"""
Plugin de Ejemplo para el Framework de IA
==========================================

Este plugin demuestra cómo crear extensiones personalizadas para el framework.
Incluye funcionalidades de ejemplo como análisis de clima emocional y respuestas contextuales.

#ddchack - Plugin de demostración para extensiones del framework
"""

import logging
import asyncio
from typing import Dict, List, Any, Optional
from datetime import datetime
import json
import re

logger = logging.getLogger(__name__)

class Plugin:
    """
    Plugin de ejemplo que extiende las capacidades del framework
    #ddchack - Extensión modular de ejemplo
    """
    
    def __init__(self):
        self.name = "example_plugin"
        self.version = "1.0.0"
        self.description = "Plugin de ejemplo con funcionalidades de análisis avanzado"
        self.author = "Framework AI Team"
        
        # Configuración del plugin
        self.config = {
            'enabled': True,
            'priority': 5,  # Prioridad de ejecución (1-10)
            'features': {
                'emotional_climate': True,
                'context_enhancement': True,
                'custom_responses': True,
                'analytics': True
            }
        }
        
        # Estado interno del plugin
        self.interaction_count = 0
        self.emotional_history = []
        self.context_cache = {}
        
        logger.info(f"Plugin {self.name} v{self.version} inicializado")
    
    async def initialize(self, framework_config: Dict[str, Any]) -> bool:
        """
        Inicializa el plugin con la configuración del framework
        
        Args:
            framework_config (Dict): Configuración del framework principal
            
        Returns:
            bool: True si la inicialización fue exitosa
        """
        try:
            # Configurar el plugin basado en la config del framework
            if 'plugins' in framework_config and self.name in framework_config['plugins']:
                plugin_config = framework_config['plugins'][self.name]
                self.config.update(plugin_config)
            
            # Verificar dependencias
            if not self._check_dependencies():
                logger.warning(f"Plugin {self.name}: Algunas dependencias no están disponibles")
                return False
            
            # Inicializar componentes específicos
            await self._initialize_components()
            
            logger.info(f"Plugin {self.name} inicializado correctamente")
            return True
            
        except Exception as e:
            logger.error(f"Error inicializando plugin {self.name}: {e}")
            return False
    
    def _check_dependencies(self) -> bool:
        """
        Verifica que las dependencias del plugin estén disponibles
        """
        required_modules = ['datetime', 'json', 're']
        
        for module in required_modules:
            try:
                __import__(module)
            except ImportError:
                logger.error(f"Plugin {self.name}: Módulo requerido '{module}' no disponible")
                return False
        
        return True
    
    async def _initialize_components(self) -> None:
        """
        Inicializa componentes específicos del plugin
        """
        # Cargar datos previos si existen
        await self._load_persistent_data()
        
        # Configurar patrones de respuesta personalizados
        self.custom_patterns = {
            'greeting': [
                "¡Hola! Es un placer verte de nuevo. ¿En qué puedo ayudarte hoy?",
                "¡Saludos! Estoy aquí para asistirte. ¿Qué necesitas?",
                "¡Bienvenido! ¿Cómo puedo hacer tu día mejor?"
            ],
            'encouragement': [
                "¡Excelente pregunta! Me encanta tu curiosidad.",
                "Esa es una perspectiva muy interesante.",
                "¡Perfecto! Vamos a explorar eso juntos."
            ],
            'empathy': [
                "Entiendo cómo te sientes. Es completamente normal.",
                "Puedo percibir que esto es importante para ti.",
                "Tu preocupación es válida y comprensible."
            ]
        }
    
    async def _load_persistent_data(self) -> None:
        """
        Carga datos persistentes del plugin
        """
        try:
            # En un caso real, esto cargaría desde archivo o base de datos
            # Por ahora, inicializamos con datos de ejemplo
            self.emotional_history = []
            self.context_cache = {}
            
        except Exception as e:
            logger.warning(f"No se pudieron cargar datos persistentes: {e}")
    
    async def process_pre_nlp(self, user_message: str, user_id: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Procesa el mensaje antes del análisis NLP
        
        Args:
            user_message (str): Mensaje del usuario
            user_id (str): ID del usuario
            context (Dict): Contexto de la conversación
            
        Returns:
            Dict: Datos procesados y enriquecidos
        """
        if not self.config['enabled']:
            return {'modified': False}
        
        enhanced_data = {
            'original_message': user_message,
            'user_id': user_id,
            'context': context,
            'modified': False,
            'enhancements': {}
        }
        
        try:
            # Análisis de patrones emocionales
            if self.config['features']['emotional_climate']:
                emotional_indicators = await self._analyze_emotional_patterns(user_message)
                enhanced_data['enhancements']['emotional_indicators'] = emotional_indicators
                enhanced_data['modified'] = True
            
            # Enriquecimiento contextual
            if self.config['features']['context_enhancement']:
                context_enhancements = await self._enhance_context(user_message, user_id, context)
                enhanced_data['enhancements']['context'] = context_enhancements
                enhanced_data['modified'] = True
            
            self.interaction_count += 1
            
        except Exception as e:
            logger.error(f"Error en process_pre_nlp: {e}")
        
        return enhanced_data
    
    async def _analyze_emotional_patterns(self, message: str) -> Dict[str, Any]:
        """
        Analiza patrones emocionales en el mensaje
        #ddchack - Análisis emocional avanzado
        """
        # Patrones emocionales básicos
        emotion_patterns = {
            'joy': ['feliz', 'contento', 'alegre', 'genial', 'excelente', 'fantástico'],
            'sadness': ['triste', 'deprimido', 'melancólico', 'desanimado'],
            'anger': ['enojado', 'molesto', 'furioso', 'irritado', 'frustrado'],
            'fear': ['miedo', 'nervioso', 'ansioso', 'preocupado', 'asustado'],
            'surprise': ['sorprendido', 'asombrado', 'impresionado'],
            'excitement': ['emocionado', 'entusiasmado', 'eufórico']
        }
        
        detected_emotions = {}
        message_lower = message.lower()
        
        for emotion, keywords in emotion_patterns.items():
            matches = [word for word in keywords if word in message_lower]
            if matches:
                detected_emotions[emotion] = {
                    'confidence': len(matches) / len(keywords),
                    'keywords_found': matches
                }
        
        # Análisis de intensidad emocional
        intensity_markers = ['muy', 'súper', 'extremadamente', 'increíblemente', 'bastante']
        intensity = len([marker for marker in intensity_markers if marker in message_lower])
        
        return {
            'detected_emotions': detected_emotions,
            'emotional_intensity': min(intensity / len(intensity_markers), 1.0),
            'overall_tone': self._determine_overall_tone(detected_emotions),
            'timestamp': datetime.now().isoformat()
        }
    
    def _determine_overall_tone(self, emotions: Dict[str, Any]) -> str:
        """
        Determina el tono general basado en las emociones detectadas
        """
        if not emotions:
            return 'neutral'
        
        # Mapear emociones a tonos
        positive_emotions = ['joy', 'excitement', 'surprise']
        negative_emotions = ['sadness', 'anger', 'fear']
        
        positive_score = sum(
            emotions[emotion]['confidence'] 
            for emotion in positive_emotions 
            if emotion in emotions
        )
        
        negative_score = sum(
            emotions[emotion]['confidence'] 
            for emotion in negative_emotions 
            if emotion in emotions
        )
        
        if positive_score > negative_score:
            return 'positive'
        elif negative_score > positive_score:
            return 'negative'
        else:
            return 'neutral'
    
    async def _enhance_context(self, message: str, user_id: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Enriquece el contexto con información adicional
        """
        enhancements = {
            'message_patterns': {},
            'user_insights': {},
            'contextual_cues': {}
        }
        
        # Análisis de patrones de mensaje
        enhancements['message_patterns'] = {
            'is_question': '?' in message,
            'is_request': any(word in message.lower() for word in ['por favor', 'puedes', 'ayuda']),
            'is_greeting': any(word in message.lower() for word in ['hola', 'buenos', 'saludos']),
            'is_farewell': any(word in message.lower() for word in ['adiós', 'chao', 'hasta']),
            'message_length': len(message),
            'word_count': len(message.split())
        }
        
        # Insights del usuario basados en historial
        if user_id in self.context_cache:
            user_history = self.context_cache[user_id]
            enhancements['user_insights'] = {
                'previous_interactions': len(user_history.get('messages', [])),
                'average_message_length': user_history.get('avg_length', 0),
                'preferred_topics': user_history.get('topics', []),
                'interaction_frequency': user_history.get('frequency', 'unknown')
            }
        else:
            # Inicializar cache para nuevo usuario
            self.context_cache[user_id] = {
                'messages': [],
                'avg_length': 0,
                'topics': [],
                'frequency': 'new_user'
            }
        
        # Actualizar cache del usuario
        self._update_user_cache(user_id, message)
        
        return enhancements
    
    def _update_user_cache(self, user_id: str, message: str) -> None:
        """
        Actualiza el cache de información del usuario
        """
        if user_id not in self.context_cache:
            self.context_cache[user_id] = {'messages': [], 'avg_length': 0, 'topics': []}
        
        user_cache = self.context_cache[user_id]
        user_cache['messages'].append({
            'content': message,
            'timestamp': datetime.now().isoformat(),
            'length': len(message)
        })
        
        # Mantener solo los últimos 50 mensajes
        if len(user_cache['messages']) > 50:
            user_cache['messages'] = user_cache['messages'][-50:]
        
        # Calcular longitud promedio
        lengths = [msg['length'] for msg in user_cache['messages']]
        user_cache['avg_length'] = sum(lengths) / len(lengths) if lengths else 0
    
    async def process_post_nlp(self, nlp_result: Dict[str, Any], enhanced_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Procesa los resultados después del análisis NLP
        
        Args:
            nlp_result (Dict): Resultado del procesamiento NLP
            enhanced_data (Dict): Datos enriquecidos del pre-procesamiento
            
        Returns:
            Dict: Resultado procesado y enriquecido
        """
        if not self.config['enabled']:
            return nlp_result
        
        try:
            # Combinar insights emocionales con análisis NLP
            if 'enhancements' in enhanced_data and 'emotional_indicators' in enhanced_data['enhancements']:
                emotional_data = enhanced_data['enhancements']['emotional_indicators']
                
                # Enriquecer análisis de sentimientos
                if 'sentiment' in nlp_result:
                    nlp_result['sentiment']['plugin_emotional_analysis'] = emotional_data
                    nlp_result['sentiment']['enhanced_tone'] = emotional_data.get('overall_tone', 'neutral')
            
            # Agregar recomendaciones de respuesta
            if self.config['features']['custom_responses']:
                response_recommendations = await self._generate_response_recommendations(nlp_result, enhanced_data)
                nlp_result['plugin_recommendations'] = response_recommendations
            
            # Métricas del plugin
            if self.config['features']['analytics']:
                nlp_result['plugin_analytics'] = {
                    'processed_by': self.name,
                    'version': self.version,
                    'interaction_count': self.interaction_count,
                    'processing_timestamp': datetime.now().isoformat()
                }
            
        except Exception as e:
            logger.error(f"Error en process_post_nlp: {e}")
        
        return nlp_result
    
    async def _generate_response_recommendations(self, nlp_result: Dict[str, Any], enhanced_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Genera recomendaciones para la respuesta basadas en el análisis
        """
        recommendations = {
            'response_style': 'neutral',
            'suggested_patterns': [],
            'emotional_adaptation': {},
            'context_considerations': []
        }
        
        # Determinar estilo de respuesta basado en el análisis emocional
        if 'enhancements' in enhanced_data and 'emotional_indicators' in enhanced_data['enhancements']:
            emotional_data = enhanced_data['enhancements']['emotional_indicators']
            overall_tone = emotional_data.get('overall_tone', 'neutral')
            
            if overall_tone == 'positive':
                recommendations['response_style'] = 'enthusiastic'
                recommendations['suggested_patterns'] = self.custom_patterns['encouragement']
            elif overall_tone == 'negative':
                recommendations['response_style'] = 'empathetic'
                recommendations['suggested_patterns'] = self.custom_patterns['empathy']
            else:
                recommendations['response_style'] = 'balanced'
                recommendations['suggested_patterns'] = self.custom_patterns['greeting']
            
            recommendations['emotional_adaptation'] = {
                'detected_tone': overall_tone,
                'suggested_approach': recommendations['response_style'],
                'emotional_intensity': emotional_data.get('emotional_intensity', 0)
            }
        
        # Consideraciones contextuales
        intent = nlp_result.get('intent', 'general')
        if intent == 'greeting':
            recommendations['context_considerations'].append('Use warm, welcoming language')
        elif intent == 'help':
            recommendations['context_considerations'].append('Provide clear, structured assistance')
        elif intent == 'information':
            recommendations['context_considerations'].append('Be informative and precise')
        
        return recommendations
    
    async def process_response(self, generated_response: str, context: Dict[str, Any]) -> str:
        """
        Procesa y potencialmente modifica la respuesta generada
        
        Args:
            generated_response (str): Respuesta generada por el framework
            context (Dict): Contexto completo de la conversación
            
        Returns:
            str: Respuesta procesada (potencialmente modificada)
        """
        if not self.config['enabled'] or not self.config['features']['custom_responses']:
            return generated_response
        
        try:
            # Aplicar mejoras específicas del plugin
            enhanced_response = generated_response
            
            # Agregar elementos emocionales si es apropiado
            if 'plugin_recommendations' in context:
                recommendations = context['plugin_recommendations']
                response_style = recommendations.get('response_style', 'neutral')
                
                if response_style == 'enthusiastic' and not any(marker in enhanced_response.lower() for marker in ['!', 'genial', 'excelente']):
                    enhanced_response = self._add_enthusiasm(enhanced_response)
                elif response_style == 'empathetic' and not any(marker in enhanced_response.lower() for marker in ['entiendo', 'comprendo']):
                    enhanced_response = self._add_empathy(enhanced_response)
            
            return enhanced_response
            
        except Exception as e:
            logger.error(f"Error procesando respuesta: {e}")
            return generated_response
    
    def _add_enthusiasm(self, response: str) -> str:
        """
        Agrega elementos de entusiasmo a la respuesta
        """
        enthusiasm_prefixes = [
            "¡Excelente! ",
            "¡Fantástico! ",
            "¡Me alegra que preguntes! "
        ]
        
        # Seleccionar prefijo aleatoriamente
        import random
        prefix = random.choice(enthusiasm_prefixes)
        
        return prefix + response
    
    def _add_empathy(self, response: str) -> str:
        """
        Agrega elementos empáticos a la respuesta
        """
        empathy_prefixes = [
            "Entiendo tu preocupación. ",
            "Comprendo cómo te sientes. ",
            "Es natural que te preguntes esto. "
        ]
        
        import random
        prefix = random.choice(empathy_prefixes)
        
        return prefix + response
    
    async def get_analytics(self) -> Dict[str, Any]:
        """
        Retorna analíticas del plugin
        #ddchack - Métricas y estadísticas del plugin
        """
        return {
            'plugin_info': {
                'name': self.name,
                'version': self.version,
                'description': self.description,
                'author': self.author
            },
            'statistics': {
                'total_interactions': self.interaction_count,
                'emotional_analyses': len(self.emotional_history),
                'cached_users': len(self.context_cache),
                'enabled_features': [feature for feature, enabled in self.config['features'].items() if enabled]
            },
            'performance': {
                'average_processing_time': 'N/A',  # Se podría implementar medición real
                'success_rate': 100.0,  # Se podría calcular basado en errores
                'last_update': datetime.now().isoformat()
            },
            'config': self.config
        }
    
    async def cleanup(self) -> None:
        """
        Limpieza al finalizar el plugin
        """
        try:
            # Guardar datos persistentes
            await self._save_persistent_data()
            
            # Limpiar cache
            self.context_cache.clear()
            self.emotional_history.clear()
            
            logger.info(f"Plugin {self.name} limpiado correctamente")
            
        except Exception as e:
            logger.error(f"Error durante limpieza del plugin {self.name}: {e}")
    
    async def _save_persistent_data(self) -> None:
        """
        Guarda datos persistentes del plugin
        """
        try:
            # En un caso real, esto guardaría en archivo o base de datos
            # Por ahora, simplemente registramos que se intentó guardar
            logger.info(f"Datos del plugin {self.name} guardados")
            
        except Exception as e:
            logger.warning(f"No se pudieron guardar datos persistentes: {e}")

# Funciones auxiliares del plugin

def validate_plugin_config(config: Dict[str, Any]) -> bool:
    """
    Valida la configuración del plugin
    """
    required_keys = ['enabled', 'features']
    
    for key in required_keys:
        if key not in config:
            return False
    
    if not isinstance(config['features'], dict):
        return False
    
    return True

def get_plugin_info() -> Dict[str, Any]:
    """
    Retorna información básica del plugin sin instanciarlo
    """
    return {
        'name': 'example_plugin',
        'version': '1.0.0',
        'description': 'Plugin de ejemplo con funcionalidades de análisis avanzado',
        'author': 'Framework AI Team',
        'requirements': ['datetime', 'json', 're'],
        'capabilities': [
            'emotional_analysis',
            'context_enhancement', 
            'response_customization',
            'analytics'
        ]
    }