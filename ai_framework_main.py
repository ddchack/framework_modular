# AI Assistant Framework
# Framework modular para construir asistentes de IA con capacidades de procesamiento de lenguaje natural
# #ddchack - Framework base para desarrollo de asistentes inteligentes

"""
Framework Modular para Asistentes de IA
======================================

Este framework proporciona una base extensible para crear asistentes de IA
con capacidades de procesamiento de lenguaje natural y aprendizaje continuo.

Características principales:
- Arquitectura modular y extensible
- Procesamiento de lenguaje natural
- Sistema de aprendizaje continuo
- Gestión de conversaciones
- Integración con múltiples APIs de IA
- Sistema de plugins
"""

import os
import json
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
from abc import ABC, abstractmethod

# Configuración de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class AIFramework:
    """
    Clase principal del framework de IA
    #ddchack - Núcleo del sistema de asistente
    """
    
    def __init__(self, config_path: str = "config/config.json"):
        """
        Inicializa el framework con la configuración especificada
        
        Args:
            config_path (str): Ruta al archivo de configuración
        """
        self.config = self._load_config(config_path)
        self.nlp_processor = None
        self.learning_module = None
        self.conversation_manager = None
        self.plugins = {}
        
        # #ddchack - Inicialización de componentes
        self._initialize_components()
        
        logger.info("Framework de IA inicializado correctamente")
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """
        Carga la configuración desde archivo JSON
        
        Args:
            config_path (str): Ruta al archivo de configuración
            
        Returns:
            Dict[str, Any]: Configuración cargada
        """
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
                
            # Validar que no hay claves sensibles expuestas
            self._validate_config_security(config)
            return config
            
        except FileNotFoundError:
            logger.warning(f"Archivo de configuración no encontrado: {config_path}")
            return self._get_default_config()
        except json.JSONDecodeError as e:
            logger.error(f"Error al parsear configuración JSON: {e}")
            return self._get_default_config()
    
    def _validate_config_security(self, config: Dict[str, Any]) -> None:
        """
        Valida que la configuración no contenga claves sensibles expuestas
        #ddchack - Seguridad de configuración
        """
        sensitive_keys = ['api_key', 'secret', 'password', 'token', 'private_key']
        
        def check_nested_dict(d: Dict[str, Any], path: str = "") -> None:
            for key, value in d.items():
                current_path = f"{path}.{key}" if path else key
                
                if any(sensitive in key.lower() for sensitive in sensitive_keys):
                    if isinstance(value, str) and value and not value.startswith("${"):
                        logger.warning(
                            f"Posible clave sensible detectada en {current_path}. "
                            f"Asegúrate de usar variables de entorno."
                        )
                
                if isinstance(value, dict):
                    check_nested_dict(value, current_path)
        
        check_nested_dict(config)
    
    def _get_default_config(self) -> Dict[str, Any]:
        """
        Retorna configuración por defecto
        #ddchack - Configuración segura por defecto
        """
        return {
            "ai_providers": {
                "openai": {
                    "api_key": "${OPENAI_API_KEY}",  # Variable de entorno
                    "model": "gpt-3.5-turbo",
                    "max_tokens": 1000
                },
                "anthropic": {
                    "api_key": "${ANTHROPIC_API_KEY}",  # Variable de entorno
                    "model": "claude-3-sonnet",
                    "max_tokens": 1000
                }
            },
            "nlp": {
                "language": "es",
                "sentiment_analysis": True,
                "entity_recognition": True,
                "intent_classification": True
            },
            "learning": {
                "enabled": True,
                "feedback_weight": 0.1,
                "memory_limit": 1000,
                "save_interval": 300  # segundos
            },
            "conversation": {
                "max_history": 50,
                "context_window": 10,
                "response_timeout": 30
            }
        }
    
    def _initialize_components(self) -> None:
        """
        Inicializa todos los componentes del framework
        """
        # Importar e inicializar componentes principales
        try:
            from .nlp.processor import NLPProcessor
            from .learning.continuous_learner import ContinuousLearner
            from .conversation.manager import ConversationManager
            
            self.nlp_processor = NLPProcessor(self.config.get('nlp', {}))
            self.learning_module = ContinuousLearner(self.config.get('learning', {}))
            self.conversation_manager = ConversationManager(self.config.get('conversation', {}))
            
            # #ddchack - Cargar plugins disponibles
            self._load_plugins()
            
        except ImportError as e:
            logger.error(f"Error al importar componentes: {e}")
            logger.info("Algunos componentes pueden no estar disponibles")
    
    def _load_plugins(self) -> None:
        """
        Carga plugins disponibles del directorio plugins/
        """
        plugins_dir = "plugins"
        if not os.path.exists(plugins_dir):
            logger.info("Directorio de plugins no encontrado")
            return
        
        for file in os.listdir(plugins_dir):
            if file.endswith('.py') and not file.startswith('__'):
                plugin_name = file[:-3]
                try:
                    # Importación dinámica de plugins
                    module = __import__(f"plugins.{plugin_name}", fromlist=[plugin_name])
                    if hasattr(module, 'Plugin'):
                        self.plugins[plugin_name] = module.Plugin()
                        logger.info(f"Plugin '{plugin_name}' cargado correctamente")
                except Exception as e:
                    logger.error(f"Error al cargar plugin '{plugin_name}': {e}")
    
    async def process_message(self, message: str, user_id: str = "default") -> str:
        """
        Procesa un mensaje del usuario y genera una respuesta
        
        Args:
            message (str): Mensaje del usuario
            user_id (str): ID del usuario
            
        Returns:
            str: Respuesta generada por el asistente
        """
        try:
            # #ddchack - Pipeline de procesamiento de mensajes
            
            # 1. Procesar lenguaje natural
            nlp_result = await self.nlp_processor.process(message) if self.nlp_processor else {}
            
            # 2. Gestionar contexto de conversación
            context = await self.conversation_manager.get_context(user_id) if self.conversation_manager else {}
            
            # 3. Generar respuesta usando IA
            response = await self._generate_response(message, nlp_result, context)
            
            # 4. Guardar en historial
            if self.conversation_manager:
                await self.conversation_manager.add_interaction(user_id, message, response)
            
            # 5. Aprendizaje continuo
            if self.learning_module:
                await self.learning_module.learn_from_interaction(message, response, nlp_result)
            
            return response
            
        except Exception as e:
            logger.error(f"Error al procesar mensaje: {e}")
            return "Lo siento, hubo un error al procesar tu mensaje. Por favor, inténtalo de nuevo."
    
    async def _generate_response(self, message: str, nlp_result: Dict, context: Dict) -> str:
        """
        Genera respuesta usando el proveedor de IA configurado
        
        Args:
            message (str): Mensaje original
            nlp_result (Dict): Resultado del análisis NLP
            context (Dict): Contexto de la conversación
            
        Returns:
            str: Respuesta generada
        """
        # Aquí se integraría con APIs de IA (OpenAI, Anthropic, etc.)
        # Las claves API se obtienen de variables de entorno
        
        # Ejemplo de estructura (sin implementación real de API)
        prompt = self._build_prompt(message, nlp_result, context)
        
        # TODO: Implementar llamada a API de IA
        # api_key = os.getenv('OPENAI_API_KEY')  # Desde variable de entorno
        # response = await ai_provider.generate(prompt, api_key)
        
        # Respuesta de ejemplo para demostración
        return f"Procesando: '{message}' - Intención detectada: {nlp_result.get('intent', 'general')}"
    
    def _build_prompt(self, message: str, nlp_result: Dict, context: Dict) -> str:
        """
        Construye el prompt para el modelo de IA
        #ddchack - Construcción inteligente de prompts
        """
        prompt_parts = []
        
        # Contexto del sistema
        prompt_parts.append("Eres un asistente de IA útil y amigable.")
        
        # Información del análisis NLP
        if nlp_result.get('intent'):
            prompt_parts.append(f"Intención detectada: {nlp_result['intent']}")
        
        if nlp_result.get('entities'):
            prompt_parts.append(f"Entidades: {nlp_result['entities']}")
        
        # Contexto de conversación
        if context.get('history'):
            prompt_parts.append("Historial reciente:")
            for interaction in context['history'][-3:]:  # Últimas 3 interacciones
                prompt_parts.append(f"Usuario: {interaction['user']}")
                prompt_parts.append(f"Asistente: {interaction['assistant']}")
        
        # Mensaje actual
        prompt_parts.append(f"Usuario: {message}")
        prompt_parts.append("Asistente:")
        
        return "\n".join(prompt_parts)
    
    def get_status(self) -> Dict[str, Any]:
        """
        Retorna el estado actual del framework
        """
        return {
            "status": "active",
            "components": {
                "nlp_processor": self.nlp_processor is not None,
                "learning_module": self.learning_module is not None,
                "conversation_manager": self.conversation_manager is not None
            },
            "plugins_loaded": list(self.plugins.keys()),
            "timestamp": datetime.now().isoformat()
        }

# Función principal para pruebas
async def main():
    """
    Función principal para demostrar el uso del framework
    #ddchack - Punto de entrada principal
    """
    # Inicializar framework
    ai_framework = AIFramework()
    
    print("=== Framework de IA Inicializado ===")
    print(f"Estado: {ai_framework.get_status()}")
    
    # Ejemplo de uso
    test_messages = [
        "Hola, ¿cómo estás?",
        "¿Puedes ayudarme con programación?",
        "Explícame qué es machine learning"
    ]
    
    for message in test_messages:
        print(f"\nUsuario: {message}")
        response = await ai_framework.process_message(message)
        print(f"Asistente: {response}")

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())