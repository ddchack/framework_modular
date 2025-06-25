# examples/basic_usage.py
"""
Ejemplos básicos de uso del Framework de IA
===========================================

Este archivo demuestra cómo usar las diferentes características del framework.
Incluye ejemplos prácticos para desarrolladores que quieren integrar el sistema.

#ddchack - Ejemplos prácticos del framework
"""

import asyncio
import logging
import json
from datetime import datetime
import os
import sys

# Agregar el directorio padre al path para importar el framework
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main import AIFramework

# Configurar logging para ejemplos
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def example_basic_conversation():
    """
    Ejemplo básico de conversación con el asistente
    #ddchack - Conversación básica de demostración
    """
    print("=== EJEMPLO: Conversación Básica ===")
    
    # Inicializar framework
    ai_framework = AIFramework()
    
    # Mensajes de ejemplo
    test_messages = [
        "Hola, ¿cómo estás?",
        "¿Puedes explicarme qué es la inteligencia artificial?",
        "Gracias por la información",
        "¿Qué puedes hacer para ayudarme?",
        "Adiós"
    ]
    
    user_id = "demo_user_001"
    
    print(f"Iniciando conversación con usuario: {user_id}")
    print("-" * 50)
    
    for i, message in enumerate(test_messages, 1):
        print(f"\n[Turno {i}]")
        print(f"Usuario: {message}")
        
        # Procesar mensaje
        response = await ai_framework.process_message(message, user_id)
        print(f"Asistente: {response}")
        
        # Pequeña pausa para simular conversación natural
        await asyncio.sleep(0.5)
    
    # Mostrar estado final
    print("\n" + "="*50)
    print("Estado final del framework:")
    status = ai_framework.get_status()
    print(json.dumps(status, indent=2, default=str))

async def example_with_feedback_learning():
    """
    Ejemplo de aprendizaje con feedback del usuario
    """
    print("\n=== EJEMPLO: Aprendizaje con Feedback ===")
    
    ai_framework = AIFramework()
    user_id = "learning_user_001"
    
    # Simulación de interacciones con diferentes niveles de feedback
    interactions = [
        {
            "message": "¿Cómo programar un loop en Python?",
            "expected_quality": 0.9,  # Respuesta que esperamos sea buena
            "feedback": 0.9
        },
        {
            "message": "Explícame machine learning",
            "expected_quality": 0.8,
            "feedback": 0.8
        },
        {
            "message": "¿Qué hora es?",
            "expected_quality": 0.3,  # Respuesta que esperamos sea limitada
            "feedback": 0.3
        },
        {
            "message": "Ayúdame con estructuras de datos",
            "expected_quality": 0.9,
            "feedback": 0.95
        }
    ]
    
    print("Procesando interacciones con feedback...")
    
    for i, interaction in enumerate(interactions, 1):
        print(f"\n--- Interacción {i} ---")
        print(f"Usuario: {interaction['message']}")
        
        # Procesar mensaje
        response = await ai_framework.process_message(interaction['message'], user_id)
        print(f"Asistente: {response}")
        
        # Simular feedback del usuario
        if ai_framework.learning_module:
            learning_result = await ai_framework.learning_module.learn_from_interaction(
                user_message=interaction['message'],
                assistant_response=response,
                nlp_result={'intent': 'information'},  # Simplificado para el ejemplo
                user_id=user_id,
                feedback=interaction['feedback']
            )
            print(f"Feedback: {interaction['feedback']}/1.0")
            print(f"Aprendizaje: {learning_result.get('status', 'N/A')}")
    
    # Mostrar estadísticas de aprendizaje
    if ai_framework.learning_module:
        print("\n--- Estadísticas de Aprendizaje ---")
        stats = ai_framework.learning_module.get_learning_statistics()
        print(f"Total interacciones: {stats.get('total_interactions', 0)}")
        print(f"Feedback promedio: {stats.get('average_feedback', 0):.3f}")
        print(f"Porcentaje con feedback: {stats.get('feedback_percentage', 0):.1f}%")

async def example_nlp_analysis():
    """
    Ejemplo detallado de análisis NLP
    """
    print("\n=== EJEMPLO: Análisis NLP Detallado ===")
    
    ai_framework = AIFramework()
    
    # Textos de ejemplo para análisis
    test_texts = [
        "¡Estoy súper emocionado por aprender sobre inteligencia artificial!",
        "Me siento un poco confundido con este problema de programación.",
        "¿Podrías ayudarme a entender mejor los algoritmos de machine learning?",
        "Gracias, esa explicación fue muy clara y útil.",
        "No estoy seguro de que eso sea correcto, me parece confuso."
    ]
    
    for i, text in enumerate(test_texts, 1):
        print(f"\n--- Análisis {i} ---")
        print(f"Texto: \"{text}\"")
        
        if ai_framework.nlp_processor:
            # Realizar análisis NLP completo
            nlp_result = await ai_framework.nlp_processor.process(text)
            
            print(f"Idioma: {nlp_result.language}")
            print(f"Intención: {nlp_result.intent} (confianza: {nlp_result.confidence:.3f})")
            
            # Análisis de sentimientos
            sentiment = nlp_result.sentiment
            print(f"Sentimiento:")
            print(f"  - Positivo: {sentiment.get('positive', 0):.3f}")
            print(f"  - Negativo: {sentiment.get('negative', 0):.3f}")
            print(f"  - Neutral: {sentiment.get('neutral', 0):.3f}")
            print(f"  - Compuesto: {sentiment.get('compound', 0):.3f}")
            
            # Entidades encontradas
            if nlp_result.entities:
                print(f"Entidades encontradas:")
                for entity in nlp_result.entities:
                    print(f"  - {entity['text']} ({entity['label']})")
            else:
                print("No se encontraron entidades específicas")
            
            # Palabras clave
            if nlp_result.keywords:
                print(f"Palabras clave: {', '.join(nlp_result.keywords[:5])}")

async def example_conversation_management():
    """
    Ejemplo de gestión avanzada de conversaciones
    """
    print("\n=== EJEMPLO: Gestión de Conversaciones ===")
    
    ai_framework = AIFramework()
    
    if not ai_framework.conversation_manager:
        print("Gestor de conversaciones no disponible")
        return
    
    # Simular múltiples usuarios
    users = ["alice_123", "bob_456", "charlie_789"]
    
    for user_id in users:
        print(f"\n--- Conversación con {user_id} ---")
        
        # Iniciar conversación
        context = await ai_framework.conversation_manager.start_conversation(user_id, "Hola")
        print(f"Conversación iniciada: {context.conversation_id}")
        
        # Simular intercambio de mensajes
        messages = [
            f"Hola, soy {user_id}",
            "¿Puedes ayudarme con programación?",
            "Gracias por tu ayuda"
        ]
        
        for message in messages:
            await ai_framework.conversation_manager.add_interaction(
                user_id, message, f"Respuesta para: {message}"
            )
        
        # Obtener contexto actual
        current_context = await ai_framework.conversation_manager.get_context(user_id)
        print(f"Mensajes en historial: {len(current_context.get('history', []))}")
        print(f"Estado: {current_context.get('conversation_state', 'unknown')}")
    
    # Estadísticas generales de conversaciones
    print("\n--- Estadísticas de Conversaciones ---")
    stats = ai_framework.conversation_manager.get_conversation_statistics()
    print(f"Conversaciones activas: {stats.get('active_conversations', 0)}")
    print(f"Usuarios únicos: {stats.get('unique_users', 0)}")
    print(f"Total de mensajes: {stats.get('total_messages', 0)}")
    print(f"Promedio mensajes/conversación: {stats.get('average_messages_per_conversation', 0):.1f}")

async def example_plugin_usage():
    """
    Ejemplo de uso de plugins
    """
    print("\n=== EJEMPLO: Uso de Plugins ===")
    
    ai_framework = AIFramework()
    
    # Mostrar plugins cargados
    if ai_framework.plugins:
        print("Plugins cargados:")
        for plugin_name, plugin in ai_framework.plugins.items():
            print(f"  - {plugin_name}")
            
            # Si el plugin tiene método de analytics, mostrarlo
            if hasattr(plugin, 'get_analytics'):
                try:
                    analytics = await plugin.get_analytics()
                    print(f"    Versión: {analytics.get('plugin_info', {}).get('version', 'N/A')}")
                    print(f"    Interacciones: {analytics.get('statistics', {}).get('total_interactions', 0)}")
                except Exception as e:
                    print(f"    Error obteniendo analytics: {e}")
    else:
        print("No hay plugins cargados")
    
    # Ejemplo de procesamiento con plugins
    test_message = "¡Estoy muy emocionado por aprender sobre IA!"
    print(f"\nProcesando mensaje con plugins: \"{test_message}\"")
    
    response = await ai_framework.process_message(test_message, "plugin_test_user")
    print(f"Respuesta: {response}")

async def example_error_handling():
    """
    Ejemplo de manejo de errores y casos edge
    """
    print("\n=== EJEMPLO: Manejo de Errores ===")
    
    ai_framework = AIFramework()
    
    # Casos edge para testing
    edge_cases = [
        "",  # Mensaje vacío
        "   ",  # Solo espacios
        "a" * 5000,  # Mensaje muy largo
        "🚀🤖💻",  # Solo emojis
        "SELECT * FROM users; DROP TABLE users;",  # Posible SQL injection
        "<script>alert('xss')</script>",  # Posible XSS
    ]
    
    print("Probando casos edge y manejo de errores...")
    
    for i, test_input in enumerate(edge_cases, 1):
        print(f"\n--- Test {i} ---")
        display_input = test_input[:50] + "..." if len(test_input) > 50 else test_input
        print(f"Input: \"{display_input}\"")
        
        try:
            response = await ai_framework.process_message(test_input, f"edge_test_{i}")
            print(f"Respuesta: {response[:100]}{'...' if len(response) > 100 else ''}")
            print("✅ Procesado sin errores")
        except Exception as e:
            print(f"❌ Error: {e}")

async def example_performance_monitoring():
    """
    Ejemplo de monitoreo de rendimiento
    """
    print("\n=== EJEMPLO: Monitoreo de Rendimiento ===")
    
    ai_framework = AIFramework()
    
    # Simular carga de trabajo
    print("Simulando carga de trabajo...")
    
    start_time = datetime.now()
    message_count = 10
    
    for i in range(message_count):
        message = f"Mensaje de prueba número {i+1}"
        await ai_framework.process_message(message, f"perf_user_{i % 3}")  # 3 usuarios
        
        if i % 3 == 0:
            print(f"Procesados {i+1}/{message_count} mensajes...")
    
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    
    print(f"\nRendimiento:")
    print(f"Tiempo total: {duration:.2f} segundos")
    print(f"Mensajes por segundo: {message_count/duration:.2f}")
    print(f"Tiempo promedio por mensaje: {duration/message_count:.3f} segundos")
    
    # Estadísticas del sistema
    status = ai_framework.get_status()
    print(f"\nEstado del sistema:")
    print(f"Componentes activos: {status.get('components', {})}")
    print(f"Plugins cargados: {len(status.get('plugins_loaded', []))}")

async def main():
    """
    Función principal que ejecuta todos los ejemplos
    #ddchack - Demostración completa del framework
    """
    print("🤖 FRAMEWORK DE IA - EJEMPLOS DE USO")
    print("=" * 60)
    
    examples = [
        ("Conversación Básica", example_basic_conversation),
        ("Aprendizaje con Feedback", example_with_feedback_learning),
        ("Análisis NLP", example_nlp_analysis),
        ("Gestión de Conversaciones", example_conversation_management),
        ("Uso de Plugins", example_plugin_usage),
        ("Manejo de Errores", example_error_handling),
        ("Monitoreo de Rendimiento", example_performance_monitoring)
    ]
    
    for name, example_func in examples:
        try:
            print(f"\n🔄 Ejecutando: {name}")
            await example_func()
            print(f"✅ {name} completado")
        except Exception as e:
            print(f"❌ Error en {name}: {e}")
            logger.error(f"Error ejecutando {name}", exc_info=True)
        
        # Pausa entre ejemplos
        await asyncio.sleep(1)
    
    print("\n🎉 Todos los ejemplos completados!")
    print("Para más información, consulta la documentación en docs/")

if __name__ == "__main__":
    # Ejecutar ejemplos
    asyncio.run(main())

---

# tests/test_framework.py
"""
Tests básicos para el Framework de IA
====================================

Tests unitarios y de integración para verificar el funcionamiento correcto
del framework principal y sus componentes.

#ddchack - Suite de tests para validación del framework
"""

import pytest
import asyncio
import json
import os
import sys
from unittest.mock import Mock, patch, AsyncMock

# Agregar directorio padre para importar framework
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main import AIFramework

class TestAIFramework:
    """
    Tests para la clase principal AIFramework
    """
    
    @pytest.fixture
    async def framework(self):
        """
        Fixture que crea una instancia del framework para testing
        """
        # Configuración mínima para tests
        test_config = {
            "ai_providers": {
                "test_provider": {
                    "api_key": "${TEST_API_KEY}",
                    "model": "test-model"
                }
            },
            "nlp": {
                "language": "es",
                "sentiment_analysis": {"enabled": True}
            },
            "learning": {
                "enabled": True,
                "memory_limit": 10
            },
            "conversation": {
                "max_history": 5,
                "context_window": 3
            }
        }
        
        # Crear framework con configuración de test
        framework = AIFramework()
        framework.config = test_config
        return framework
    
    @pytest.mark.asyncio
    async def test_framework_initialization(self, framework):
        """
        Test básico de inicialización del framework
        """
        assert framework is not None
        assert hasattr(framework, 'config')
        assert hasattr(framework, 'nlp_processor')
        assert hasattr(framework, 'learning_module')
        assert hasattr(framework, 'conversation_manager')
    
    @pytest.mark.asyncio
    async def test_process_message_basic(self, framework):
        """
        Test básico de procesamiento de mensajes
        """
        message = "Hola, ¿cómo estás?"
        user_id = "test_user_001"
        
        response = await framework.process_message(message, user_id)
        
        assert response is not None
        assert isinstance(response, str)
        assert len(response) > 0
    
    @pytest.mark.asyncio
    async def test_process_empty_message(self, framework):
        """
        Test de manejo de mensajes vacíos
        """
        empty_messages = ["", "   ", None]
        
        for message in empty_messages:
            response = await framework.process_message(message or "", "test_user")
            assert response is not None
            assert isinstance(response, str)
    
    @pytest.mark.asyncio
    async def test_multiple_users(self, framework):
        """
        Test de manejo de múltiples usuarios
        """
        users = ["user_1", "user_2", "user_3"]
        message = "Test message"
        
        responses = []
        for user_id in users:
            response = await framework.process_message(message, user_id)
            responses.append(response)
        
        # Todos los usuarios deben recibir respuesta
        assert len(responses) == len(users)
        assert all(isinstance(r, str) and len(r) > 0 for r in responses)
    
    def test_get_status(self, framework):
        """
        Test de obtención de estado del framework
        """
        status = framework.get_status()
        
        assert isinstance(status, dict)
        assert 'status' in status
        assert 'components' in status
        assert 'timestamp' in status
    
    def test_config_validation(self, framework):
        """
        Test de validación de configuración
        """
        # Test con configuración válida
        valid_config = {
            "ai_providers": {
                "openai": {"api_key": "${OPENAI_API_KEY}"}
            }
        }
        
        # No debe lanzar excepciones
        framework._validate_config_security(valid_config)
        
        # Test con configuración potencialmente insegura
        unsafe_config = {
            "ai_providers": {
                "openai": {"api_key": "sk-real-key-exposed"}
            }
        }
        
        # Debe detectar la clave expuesta (verificar logs)
        framework._validate_config_security(unsafe_config)

class TestNLPProcessor:
    """
    Tests para el procesador NLP
    """
    
    @pytest.fixture
    def nlp_config(self):
        return {
            "language": "es",
            "sentiment_analysis": {"enabled": True, "threshold": 0.6},
            "entity_recognition": {"enabled": True},
            "intent_classification": {"enabled": True, "confidence_threshold": 0.7}
        }
    
    @pytest.mark.asyncio
    async def test_basic_processing(self, nlp_config):
        """
        Test básico de procesamiento NLP
        """
        try:
            from nlp.processor import NLPProcessor
            
            processor = NLPProcessor(nlp_config)
            text = "Hola, estoy muy feliz hoy"
            
            result = await processor.process(text)
            
            assert result is not None
            assert result.text == text
            assert result.language == "es"
            assert 'sentiment' in result.__dict__
            assert 'intent' in result.__dict__
            
        except ImportError:
            pytest.skip("Módulo NLP no disponible")
    
    @pytest.mark.asyncio
    async def test_sentiment_analysis(self, nlp_config):
        """
        Test específico de análisis de sentimientos
        """
        try:
            from nlp.processor import NLPProcessor
            
            processor = NLPProcessor(nlp_config)
            
            # Texto positivo
            positive_text = "¡Estoy súper feliz y emocionado!"
            result_pos = await processor.process(positive_text)
            
            # Texto negativo
            negative_text = "Estoy muy triste y deprimido"
            result_neg = await processor.process(negative_text)
            
            # Verificar que detecta sentimientos opuestos
            assert result_pos.sentiment['positive'] > result_pos.sentiment['negative']
            assert result_neg.sentiment['negative'] > result_neg.sentiment['positive']
            
        except ImportError:
            pytest.skip("Módulo NLP no disponible")

class TestContinuousLearner:
    """
    Tests para el sistema de aprendizaje continuo
    """
    
    @pytest.fixture
    def learning_config(self):
        return {
            "enabled": True,
            "memory_limit": 10,
            "feedback_weight": 0.1,
            "save_interval": 300
        }
    
    @pytest.mark.asyncio
    async def test_learn_from_interaction(self, learning_config):
        """
        Test de aprendizaje desde interacciones
        """
        try:
            from learning.continuous_learner import ContinuousLearner
            
            learner = ContinuousLearner(learning_config)
            
            result = await learner.learn_from_interaction(
                user_message="¿Qué es Python?",
                assistant_response="Python es un lenguaje de programación...",
                nlp_result={'intent': 'information'},
                user_id="test_user",
                feedback=0.9
            )
            
            assert result is not None
            assert result['status'] == 'learned'
            assert 'interaction_id' in result
            
        except ImportError:
            pytest.skip("Módulo de aprendizaje no disponible")
    
    @pytest.mark.asyncio
    async def test_learning_statistics(self, learning_config):
        """
        Test de estadísticas de aprendizaje
        """
        try:
            from learning.continuous_learner import ContinuousLearner
            
            learner = ContinuousLearner(learning_config)
            
            # Agregar algunas interacciones
            for i in range(3):
                await learner.learn_from_interaction(
                    user_message=f"Pregunta {i}",
                    assistant_response=f"Respuesta {i}",
                    nlp_result={'intent': 'general'},
                    feedback=0.8
                )
            
            stats = learner.get_learning_statistics()
            
            assert stats['total_interactions'] == 3
            assert stats['feedback_percentage'] == 100.0
            assert stats['average_feedback'] == 0.8
            
        except ImportError:
            pytest.skip("Módulo de aprendizaje no disponible")

class TestConversationManager:
    """
    Tests para el gestor de conversaciones
    """
    
    @pytest.fixture
    def conv_config(self):
        return {
            "max_history": 10,
            "context_window": 5,
            "session_timeout": 3600,
            "enable_memory": True
        }
    
    @pytest.mark.asyncio
    async def test_start_conversation(self, conv_config):
        """
        Test de inicio de conversación
        """
        try:
            from conversation.manager import ConversationManager
            
            manager = ConversationManager(conv_config)
            user_id = "test_user_conv"
            
            context = await manager.start_conversation(user_id)
            
            assert context is not None
            assert context.user_id == user_id
            assert context.conversation_id is not None
            
        except ImportError:
            pytest.skip("Módulo de conversación no disponible")
    
    @pytest.mark.asyncio
    async def test_add_interaction(self, conv_config):
        """
        Test de agregar interacciones
        """
        try:
            from conversation.manager import ConversationManager
            
            manager = ConversationManager(conv_config)
            user_id = "test_user_interaction"
            
            success = await manager.add_interaction(
                user_id=user_id,
                user_message="Hola",
                assistant_response="¡Hola! ¿Cómo estás?"
            )
            
            assert success is True
            
            # Verificar contexto
            context = await manager.get_context(user_id)
            assert context['has_active_conversation'] is True
            assert len(context['history']) > 0
            
        except ImportError:
            pytest.skip("Módulo de conversación no disponible")

class TestPluginSystem:
    """
    Tests para el sistema de plugins
    """
    
    @pytest.mark.asyncio
    async def test_plugin_loading(self):
        """
        Test de carga de plugins
        """
        framework = AIFramework()
        
        # Verificar que se intenta cargar plugins
        assert hasattr(framework, 'plugins')
        assert isinstance(framework.plugins, dict)
    
    @pytest.mark.asyncio
    async def test_example_plugin(self):
        """
        Test del plugin de ejemplo
        """
        try:
            from plugins.example_plugin import Plugin
            
            plugin = Plugin()
            
            assert plugin.name == "example_plugin"
            assert plugin.version == "1.0.0"
            
            # Test de inicialización
            init_result = await plugin.initialize({})
            assert isinstance(init_result, bool)
            
        except ImportError:
            pytest.skip("Plugin de ejemplo no disponible")

class TestSecurity:
    """
    Tests de seguridad
    """
    
    def test_config_security_validation(self):
        """
        Test de validación de seguridad en configuración
        """
        framework = AIFramework()
        
        # Configuración segura (con variables de entorno)
        secure_config = {
            "api_key": "${API_KEY}",
            "database_url": "${DATABASE_URL}"
        }
        
        # No debe lanzar excepciones
        framework._validate_config_security(secure_config)
        
        # Configuración insegura (con claves hardcodeadas)
        insecure_config = {
            "api_key": "sk-1234567890abcdef",
            "secret": "my-secret-password"
        }
        
        # Debe detectar problemas (verificar en logs)
        framework._validate_config_security(insecure_config)
    
    @pytest.mark.asyncio
    async def test_input_sanitization(self):
        """
        Test de sanitización de entrada
        """
        framework = AIFramework()
        
        # Inputs potencialmente maliciosos
        malicious_inputs = [
            "<script>alert('xss')</script>",
            "'; DROP TABLE users; --",
            "javascript:alert('xss')",
            "data:text/html,<script>alert('xss')</script>"
        ]
        
        for malicious_input in malicious_inputs:
            response = await framework.process_message(malicious_input, "security_test")
            
            # La respuesta no debe contener el input malicioso sin sanitizar
            assert response is not None
            assert isinstance(response, str)

# Configuración de pytest
def pytest_configure(config):
    """
    Configuración global de pytest
    """
    # Configurar logging para tests
    import logging
    logging.basicConfig(level=logging.WARNING)

# Fixtures globales
@pytest.fixture(scope="session")
def event_loop():
    """
    Fixture para manejar el event loop en tests async
    """
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()

if __name__ == "__main__":
    # Ejecutar tests si se corre directamente
    pytest.main([__file__, "-v"])

---

# tests/conftest.py
"""
Configuración compartida para tests
===================================

Fixtures y configuraciones comunes para todos los tests del framework.

#ddchack - Configuración central de testing
"""

import pytest
import asyncio
import tempfile
import os
import shutil
from unittest.mock import Mock, AsyncMock

@pytest.fixture(scope="session")
def event_loop():
    """
    Event loop compartido para toda la sesión de tests
    """
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    yield loop
    loop.close()

@pytest.fixture
def temp_dir():
    """
    Directorio temporal para tests que necesitan archivos
    """
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)

@pytest.fixture
def mock_env_vars(monkeypatch):
    """
    Variables de entorno mock para tests
    """
    test_env = {
        'OPENAI_API_KEY': 'test-openai-key',
        'ANTHROPIC_API_KEY': 'test-anthropic-key',
        'DATABASE_URL': 'sqlite:///test.db',
        'ENVIRONMENT': 'test'
    }
    
    for key, value in test_env.items():
        monkeypatch.setenv(key, value)
    
    return test_env

---

# Makefile - Comandos de desarrollo
# #ddchack - Automatización de tareas de desarrollo

.PHONY: help install install-dev test test-cov lint format clean docker docs

# Variables
PYTHON := python3
PIP := pip3
PROJECT_NAME := ai-assistant-framework

help: ## Mostrar esta ayuda
	@echo "Comandos disponibles:"
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  \033[36m%-15s\033[0m %s\n", $1, $2}' $(MAKEFILE_LIST)

install: ## Instalar dependencias básicas
	$(PIP) install -r requirements.txt

install-dev: ## Instalar dependencias de desarrollo
	$(PIP) install -r requirements-full.txt
	$(PIP) install -e .

install-full: ## Instalar todas las dependencias incluyendo opcionales
	$(PIP) install -r requirements-full.txt
	$(PYTHON) -m spacy download es_core_news_sm
	$(PYTHON) -m spacy download en_core_web_sm
	$(PYTHON) -m nltk.downloader punkt stopwords wordnet omw-1.4 vader_lexicon

test: ## Ejecutar tests básicos
	$(PYTHON) -m pytest tests/ -v

test-cov: ## Ejecutar tests con cobertura
	$(PYTHON) -m pytest tests/ --cov=./ --cov-report=html --cov-report=term-missing

test-integration: ## Ejecutar tests de integración
	$(PYTHON) -m pytest tests/ -m integration -v

lint: ## Verificar código con linters
	$(PYTHON) -m flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics
	$(PYTHON) -m flake8 . --count --exit-zero --max-complexity=10 --max-line-length=127 --statistics

format: ## Formatear código con black
	$(PYTHON) -m black . --line-length=100
	$(PYTHON) -m isort . --profile black

type-check: ## Verificar tipos con mypy
	$(PYTHON) -m mypy . --ignore-missing-imports

clean: ## Limpiar archivos temporales
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf build/
	rm -rf dist/
	rm -rf .coverage
	rm -rf htmlcov/
	rm -rf .pytest_cache/
	rm -rf logs/*.log

docker-build: ## Construir imagen Docker
	docker build -t $(PROJECT_NAME):latest .

docker-run: ## Ejecutar contenedor Docker
	docker run -p 8000:8000 --env-file .env $(PROJECT_NAME):latest

docker-compose-up: ## Levantar stack completo con docker-compose
	docker-compose up -d

docker-compose-down: ## Bajar stack de docker-compose
	docker-compose down

docs: ## Generar documentación
	@echo "Generando documentación..."
	@mkdir -p docs/build
	@echo "Documentación disponible en docs/"

run-example: ## Ejecutar ejemplos básicos
	$(PYTHON) examples/basic_usage.py

setup-dev: ## Configuración inicial para desarrollo
	cp .env.example .env
	@echo "⚠️  Edita el archivo .env con tus claves API reales"
	@echo "📁 Creando directorios necesarios..."
	mkdir -p data/learning data/conversations logs config
	@echo "✅ Configuración de desarrollo completada"

security-check: ## Verificar seguridad del código
	$(PIP) install bandit safety
	$(PYTHON) -m bandit -r . -f json -o security-report.json
	$(PYTHON) -m safety check

performance-test: ## Test de rendimiento básico
	$(PYTHON) -c "import asyncio; from examples.basic_usage import example_performance_monitoring; asyncio.run(example_performance_monitoring())"

---

# .github/workflows/ci.yml
# GitHub Actions para CI/CD
# #ddchack - Pipeline de integración continua

name: CI/CD Pipeline

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: [3.8, 3.9, '3.10', 3.11]

    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python ${{ matrix.python-version }}
      uses: actions/setup-python@v4
      with:
        python-version: ${{ matrix.python-version }}
    
    - name: Cache pip dependencies
      uses: actions/cache@v3
      with:
        path: ~/.cache/pip
        key: ${{ runner.os }}-pip-${{ hashFiles('**/requirements.txt') }}
        restore-keys: |
          ${{ runner.os }}-pip-
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
        pip install pytest pytest-cov pytest-asyncio
    
    - name: Lint with flake8
      run: |
        pip install flake8
        flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics
        flake8 . --count --exit-zero --max-complexity=10 --max-line-length=127 --statistics
    
    - name: Test with pytest
      run: |
        python -m pytest tests/ -v --cov=./ --cov-report=xml
    
    - name: Upload coverage to Codecov
      uses: codecov/codecov-action@v3
      with:
        file: ./coverage.xml
        flags: unittests
        name: codecov-umbrella

  security:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: 3.9
    
    - name: Install security tools
      run: |
        python -m pip install --upgrade pip
        pip install bandit safety
    
    - name: Run Bandit security check
      run: bandit -r . -f json -o bandit-report.json
    
    - name: Run Safety check
      run: safety check --json --output safety-report.json
    
    - name: Upload security reports
      uses: actions/upload-artifact@v3
      with:
        name: security-reports
        path: |
          bandit-report.json
          safety-report.json

  docker:
    runs-on: ubuntu-latest
    needs: test
    if: github.ref == 'refs/heads/main'
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Build Docker image
      run: docker build -t ai-assistant-framework:latest .
    
    - name: Test Docker image
      run: |
        docker run --rm ai-assistant-framework:latest python -c "from main import AIFramework; print('Docker image OK')"

---

# docs/api.md
# Documentación de API

## Framework Principal

### AIFramework

Clase principal del framework que coordina todos los componentes.

#### Constructor
```python
AIFramework(config_path: str = "config/config.json")
```

#### Métodos Principales

##### process_message()
```python
async def process_message(
    message: str, 
    user_id: str = "default"
) -> str
```

Procesa un mensaje del usuario y genera una respuesta.

**Parámetros:**
- `message` (str): Mensaje del usuario a procesar
- `user_id` (str): Identificador único del usuario

**Retorna:**
- `str`: Respuesta generada por el asistente

**Ejemplo:**
```python
framework = AIFramework()
response = await framework.process_message("Hola", "usuario_123")
print(response)  # "¡Hola! ¿En qué puedo ayudarte?"
```

##### get_status()
```python
def get_status() -> Dict[str, Any]
```

Obtiene el estado actual del framework.

**Retorna:**
- `Dict[str, Any]`: Estado del framework incluyendo componentes activos

## Módulo NLP

### NLPProcessor

Procesador principal de lenguaje natural.

##### process()
```python
async def process(text: str) -> NLPResult
```

**Parámetros:**
- `text` (str): Texto a analizar

**Retorna:**
- `NLPResult`: Objeto con resultados del análisis

### NLPResult

Estructura de datos que contiene los resultados del análisis NLP.

**Atributos:**
- `text` (str): Texto original
- `language` (str): Idioma detectado
- `sentiment` (Dict): Análisis de sentimientos
- `entities` (List): Entidades reconocidas
- `intent` (str): Intención clasificada
- `confidence` (float): Confianza de la clasificación
- `tokens` (List[str]): Tokens extraídos
- `keywords` (List[str]): Palabras clave

## Módulo de Aprendizaje

### ContinuousLearner

Sistema de aprendizaje continuo.

##### learn_from_interaction()
```python
async def learn_from_interaction(
    user_message: str,
    assistant_response: str,
    nlp_result: Dict[str, Any],
    user_id: str = "default",
    feedback: Optional[float] = None,
    context: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]
```

Aprende de una interacción usuario-asistente.

**Parámetros:**
- `user_message` (str): Mensaje del usuario
- `assistant_response` (str): Respuesta del asistente
- `nlp_result` (Dict): Resultado del análisis NLP
- `user_id` (str): ID del usuario
- `feedback` (Optional[float]): Feedback del usuario (0-1)
- `context` (Optional[Dict]): Contexto adicional

## Módulo de Conversación

### ConversationManager

Gestor de conversaciones y contexto.

##### start_conversation()
```python
async def start_conversation(
    user_id: str, 
    initial_message: str = None
) -> ConversationContext
```

##### add_interaction()
```python
async def add_interaction(
    user_id: str,
    user_message: str,
    assistant_response: str,
    metadata: Optional[Dict[str, Any]] = None
) -> bool
```

##### get_context()
```python
async def get_context(user_id: str) -> Dict[str, Any]
```

## Sistema de Plugins

### Plugin Base

Estructura base para crear plugins personalizados.

```python
class Plugin:
    def __init__(self):
        self.name = "plugin_name"
        self.version = "1.0.0"
    
    async def initialize(self, framework_config: Dict[str, Any]) -> bool:
        # Inicialización del plugin
        pass
    
    async def process_pre_nlp(
        self, 
        user_message: str, 
        user_id: str, 
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        # Procesamiento antes del NLP
        pass
    
    async def process_post_nlp(
        self, 
        nlp_result: Dict[str, Any], 
        enhanced_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        # Procesamiento después del NLP
        pass
    
    async def process_response(
        self, 
        generated_response: str, 
        context: Dict[str, Any]
    ) -> str:
        # Procesamiento de la respuesta final
        pass
```

---

# docs/deployment.md
# Guía de Despliegue

## Despliegue Local

### Requisitos Previos
- Python 3.8+
- Git
- Variables de entorno configuradas

### Pasos de Instalación

1. **Clonar el repositorio:**
```bash
git clone https://github.com/tu-usuario/ai-assistant-framework.git
cd ai-assistant-framework
```

2. **Crear entorno virtual:**
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# o
venv\Scripts\activate     # Windows
```

3. **Instalar dependencias:**
```bash
pip install -r requirements.txt
```

4. **Configurar variables de entorno:**
```bash
cp .env.example .env
# Editar .env con tus claves API
```

5. **Ejecutar el framework:**
```bash
python main.py
```

## Despliegue con Docker

### Construcción de la Imagen

```bash
docker build -t ai-assistant-framework .
```

### Ejecución del Contenedor

```bash
docker run -p 8000:8000 --env-file .env ai-assistant-framework
```

### Docker Compose (Recomendado)

```bash
# Levantar todo el stack
docker-compose up -d

# Ver logs
docker-compose logs -f

# Parar servicios
docker-compose down
```

## Despliegue en Producción

### Consideraciones de Seguridad

1. **Variables de Entorno:**
   - Usar servicios de gestión de secretos
   - Nunca hardcodear claves API
   - Rotar claves regularmente

2. **Base de Datos:**
   - Usar PostgreSQL en producción
   - Configurar backups automáticos
   - Habilitar SSL/TLS

3. **Monitoreo:**
   - Configurar Prometheus/Grafana
   - Habilitar logging estructurado
   - Configurar alertas

### Ejemplo de Configuración Nginx

```nginx
server {
    listen 80;
    server_name tu-dominio.com;

    location / {
        proxy_pass http://localhost:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

### Variables de Entorno para Producción

```bash
# Aplicación
ENVIRONMENT=production
PORT=8000
WORKERS=4

# Base de datos
DATABASE_URL=postgresql://user:pass@localhost:5432/aiframework
REDIS_URL=redis://localhost:6379

# APIs de IA
OPENAI_API_KEY=sk-your-real-key
ANTHROPIC_API_KEY=sk-ant-your-real-key

# Seguridad
SECRET_KEY=your-super-secret-key
ALLOWED_ORIGINS=https://tu-dominio.com

# Monitoreo
SENTRY_DSN=https://your-sentry-dsn
```

### Escalabilidad

Para entornos de alto tráfico:

1. **Load Balancer:** Usar múltiples instancias
2. **Cache Distribuido:** Redis Cluster
3. **Base de Datos:** PostgreSQL con réplicas de lectura
4. **CDN:** Para archivos estáticos
5. **Auto-scaling:** Kubernetes o similar

---

# scripts/install.sh
#!/bin/bash
# Script de instalación automatizada
# #ddchack - Instalador del framework

set -e

echo "🤖 Instalando Framework de IA..."

# Verificar Python
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 no está instalado"
    exit 1
fi

# Verificar versión de Python
python_version=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
required_version="3.8"

if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" != "$required_version" ]; then
    echo "❌ Se requiere Python 3.8 o superior. Versión actual: $python_version"
    exit 1
fi

echo "✅ Python $python_version detectado"

# Crear entorno virtual
if [ ! -d "venv" ]; then
    echo "📦 Creando entorno virtual..."
    python3 -m venv venv
fi

# Activar entorno virtual
echo "🔧 Activando entorno virtual..."
source venv/bin/activate

# Actualizar pip
echo "⬆️  Actualizando pip..."
pip install --upgrade pip

# Instalar dependencias
echo "📚 Instalando dependencias..."
if [ "$1" = "--full" ]; then
    pip install -r requirements-full.txt
    echo "🌍 Descargando modelos de spaCy..."
    python -m spacy download es_core_news_sm || echo "⚠️ Modelo español no disponible"
    python -m spacy download en_core_web_sm || echo "⚠️ Modelo inglés no disponible"
else
    pip install -r requirements.txt
fi

# Crear directorios necesarios
echo "📁 Creando directorios..."
mkdir -p data/learning data/conversations logs config plugins

# Configurar variables de entorno
if [ ! -f ".env" ]; then
    echo "🔑 Configurando variables de entorno..."
    cp .env.example .env
    echo "⚠️  Edita el archivo .env con tus claves API reales"
fi

# Verificar instalación
echo "🧪 Verificando instalación..."
python -c "from main import AIFramework; print('✅ Framework importado correctamente')"

echo "🎉 Instalación completada!"
echo ""
echo "Próximos pasos:"
echo "1. Editar .env con tus claves API"
echo "2. Ejecutar: python main.py"
echo "3. O ejecutar ejemplos: python examples/basic_usage.py"
echo ""
echo "Para activar el entorno virtual en el futuro:"
echo "source venv/bin/activate"