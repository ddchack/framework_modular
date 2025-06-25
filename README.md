# Framework Modular para Asistentes de IA

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code Style: Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

Un framework extensible para construir asistentes de IA con capacidades avanzadas de procesamiento de lenguaje natural y aprendizaje continuo.

<!-- #ddchack - Framework base para desarrollo de asistentes inteligentes -->

## 🚀 Características Principales

- **🧠 Procesamiento NLP Avanzado**: Análisis de sentimientos, reconocimiento de entidades, clasificación de intenciones
- **📚 Aprendizaje Continuo**: Sistema adaptativo que mejora con cada interacción
- **💬 Gestión Conversacional**: Manejo inteligente del contexto y flujo de conversaciones
- **🔌 Arquitectura Modular**: Sistema de plugins extensible
- **🔒 Seguridad Integrada**: Gestión segura de APIs y configuraciones
- **📊 Monitoreo y Métricas**: Sistema completo de análisis y reportes
- **🌐 Multi-proveedor**: Soporte para OpenAI, Anthropic, Hugging Face y más

## 📋 Requisitos del Sistema

- Python 3.8 o superior
- 4GB RAM mínimo (8GB recomendado)
- Conexión a internet para APIs de IA

## 🛠️ Instalación

### Instalación Básica

```bash
# Clonar el repositorio
git clone https://github.com/tu-usuario/ai-assistant-framework.git
cd ai-assistant-framework

# Crear entorno virtual
python -m venv venv
source venv/bin/activate  # Linux/Mac
# o
venv\Scripts\activate     # Windows

# Instalar dependencias básicas
pip install -r requirements.txt
```

### Instalación Completa (con todas las dependencias opcionales)

```bash
# Instalar dependencias completas para NLP avanzado
pip install -r requirements-full.txt

# Descargar modelos de spaCy (opcional)
python -m spacy download es_core_news_sm
python -m spacy download en_core_web_sm
```

## ⚙️ Configuración

### 1. Variables de Entorno

```bash
# Copiar archivo de ejemplo
cp .env.example .env

# Editar .env con tus claves API
nano .env
```

**⚠️ IMPORTANTE**: Nunca subas tu archivo `.env` a repositorios públicos. Las claves API deben mantenerse privadas.

### 2. Configuración del Framework

Edita `config/config.json` para personalizar el comportamiento:

```json
{
  "ai_providers": {
    "openai": {
      "api_key": "${OPENAI_API_KEY}",
      "model": "gpt-3.5-turbo"
    }
  },
  "nlp": {
    "language": "es",
    "sentiment_analysis": true
  },
  "learning": {
    "enabled": true,
    "feedback_weight": 0.1
  }
}
```

## 🚀 Uso Rápido

### Ejemplo Básico

```python
import asyncio
from ai_framework import AIFramework

async def main():
    # Inicializar framework
    ai = AIFramework()
    
    # Procesar mensaje
    response = await ai.process_message(
        message="Hola, ¿cómo estás?",
        user_id="usuario_123"
    )
    
    print(f"Respuesta: {response}")
    
    # Obtener estadísticas
    stats = ai.get_status()
    print(f"Estado: {stats}")

# Ejecutar
asyncio.run(main())
```

### Ejemplo con Feedback y Aprendizaje

```python
async def example_with_learning():
    ai = AIFramework()
    
    # Interacción 1
    response1 = await ai.process_message(
        message="¿Qué es machine learning?",
        user_id="student_001"
    )
    
    # Simular feedback positivo
    await ai.learning_module.learn_from_interaction(
        user_message="¿Qué es machine learning?",
        assistant_response=response1,
        nlp_result={},
        feedback=0.9  # Feedback muy positivo
    )
    
    # Obtener insights de aprendizaje
    insights = await ai.learning_module.get_learning_insights({
        'message': '¿Qué es deep learning?',
        'intent': 'information'
    })
    
    print(f"Insights: {insights}")
```

## 📁 Estructura del Proyecto

```
ai-assistant-framework/
├── README.md                 # Este archivo
├── requirements.txt          # Dependencias básicas
├── requirements-full.txt     # Dependencias completas
├── .env.example             # Plantilla de variables de entorno
├── .gitignore               # Archivos a ignorar en Git
├── main.py                  # Archivo principal del framework
├── config/
│   ├── config.json          # Configuración principal
│   └── logging.conf         # Configuración de logs
├── nlp/
│   ├── __init__.py
│   ├── processor.py         # Procesador NLP principal
│   ├── sentiment.py         # Análisis de sentimientos
│   ├── entities.py          # Reconocimiento de entidades
│   └── intents.py           # Clasificación de intenciones
├── learning/
│   ├── __init__.py
│   ├── continuous_learner.py # Aprendizaje continuo
│   ├── strategies.py        # Estrategias de aprendizaje
│   └── feedback.py          # Gestión de feedback
├── conversation/
│   ├── __init__.py
│   ├── manager.py           # Gestor de conversaciones
│   ├── memory.py            # Sistema de memoria
│   └── flow.py              # Control de flujo
├── plugins/
│   ├── __init__.py
│   ├── example_plugin.py    # Plugin de ejemplo
│   └── README.md            # Documentación de plugins
├── data/
│   ├── learning/            # Datos de aprendizaje
│   ├── conversations/       # Historial de conversaciones
│   └── models/              # Modelos entrenados
├── logs/                    # Archivos de log
├── tests/
│   ├── __init__.py
│   ├── test_framework.py    # Tests del framework principal
│   ├── test_nlp.py          # Tests de NLP
│   ├── test_learning.py     # Tests de aprendizaje
│   └── test_conversation.py # Tests de conversaciones
└── docs/
    ├── api.md               # Documentación de API
    ├── plugins.md           # Guía de plugins
    ├── deployment.md        # Guía de despliegue
    └── examples/            # Ejemplos adicionales
```

## 🧩 Componentes Principales

### 1. Procesador NLP (`nlp/processor.py`)

Maneja el análisis de texto con:
- **Análisis de sentimientos**: Detecta emociones en el texto
- **Reconocimiento de entidades**: Identifica nombres, fechas, ubicaciones
- **Clasificación de intenciones**: Determina qué quiere hacer el usuario
- **Tokenización y preprocesamiento**: Limpia y estructura el texto

```python
from nlp.processor import NLPProcessor

nlp = NLPProcessor(config)
result = await nlp.process("Estoy muy feliz hoy")
print(result.sentiment)  # {'positive': 0.8, 'negative': 0.1, ...}
```

### 2. Aprendizaje Continuo (`learning/continuous_learner.py`)

Sistema que mejora automáticamente:
- **Aprendizaje por patrones**: Detecta respuestas exitosas
- **Feedback del usuario**: Incorpora valoraciones
- **Memoria contextual**: Recuerda conversaciones anteriores
- **Métricas de rendimiento**: Monitorea mejoras

```python
from learning.continuous_learner import ContinuousLearner

learner = ContinuousLearner(config)
await learner.learn_from_interaction(
    user_message="¿Cómo programar en Python?",
    assistant_response="Python es un lenguaje...",
    nlp_result=nlp_data,
    feedback=0.95
)
```

### 3. Gestor de Conversaciones (`conversation/manager.py`)

Mantiene el contexto conversacional:
- **Historial de mensajes**: Recuerda conversaciones
- **Estados de conversación**: Maneja flujos complejos
- **Memoria de usuario**: Personaliza respuestas
- **Gestión de sesiones**: Controla timeouts y límites

```python
from conversation.manager import ConversationManager

conv_mgr = ConversationManager(config)
context = await conv_mgr.get_context("usuario_123")
await conv_mgr.add_interaction("usuario_123", "Hola", "¡Hola! ¿Cómo estás?")
```

## 🔌 Sistema de Plugins

El framework soporta plugins personalizados para extender funcionalidad:

```python
# plugins/mi_plugin.py
class Plugin:
    def __init__(self):
        self.name = "mi_plugin"
        self.version = "1.0.0"
    
    async def process(self, data):
        # Tu lógica personalizada aquí
        return {"processed": True, "data": data}
```

Ver [documentación de plugins](docs/plugins.md) para más detalles.

## 📊 Monitoreo y Métricas

### Estadísticas del Framework

```python
# Obtener estado general
stats = framework.get_status()

# Estadísticas de aprendizaje
learning_stats = framework.learning_module.get_learning_statistics()

# Estadísticas de conversaciones
conv_stats = framework.conversation_manager.get_conversation_statistics()
```

### Exportar Datos

```python
# Exportar datos de aprendizaje
report_path = await learner.export_learning_data(format='json')

# Generar reporte completo
from learning.continuous_learner import generate_learning_report
report = generate_learning_report(learner)
print(report)
```

## 🔒 Seguridad y Mejores Prácticas

### ✅ Configuración Segura

- **Variables de entorno**: Todas las claves API se almacenan en `.env`
- **Validación de entrada**: Sanitización automática de inputs
- **Rate limiting**: Control de límites de uso
- **Logging seguro**: Los logs no contienen información sensible

### ✅ Desarrollo Responsable

```python
# ✅ CORRECTO - Usar variables de entorno
api_key = os.getenv('OPENAI_API_KEY')

# ❌ INCORRECTO - Nunca hardcodear claves
api_key = "sk-real-api-key-here"  # ¡NO HACER ESTO!
```

### ✅ Validación de Datos

El framework incluye validación automática:

```python
# Validación automática de configuración
framework._validate_config_security(config)

# Límites de entrada
max_message_length = 2000
forbidden_patterns = ["<script>", "javascript:"]
```

## 🧪 Testing

Ejecutar tests:

```bash
# Tests básicos
python -m pytest tests/

# Tests con cobertura
python -m pytest tests/ --cov=./ --cov-report=html

# Tests específicos
python -m pytest tests/test_nlp.py -v
```

Crear nuevos tests:

```python
# tests/test_mi_modulo.py
import pytest
from ai_framework import AIFramework

@pytest.mark.asyncio
async def test_process_message():
    framework = AIFramework()
    response = await framework.process_message("Test message")
    assert response is not None
    assert len(response) > 0
```

## 📚 Documentación Adicional

- [📖 API Reference](docs/api.md) - Documentación completa de la API
- [🔌 Plugin Development](docs/plugins.md) - Crear plugins personalizados
- [🚀 Deployment Guide](docs/deployment.md) - Despliegue en producción
- [💡 Examples](docs/examples/) - Ejemplos prácticos adicionales

## 🤝 Contribución

¡Las contribuciones son bienvenidas! Por favor:

1. Fork el proyecto
2. Crea una rama para tu feature (`git checkout -b feature/nueva-funcionalidad`)
3. Commit tus cambios (`git commit -am 'Agregar nueva funcionalidad'`)
4. Push a la rama (`git push origin feature/nueva-funcionalidad`)
5. Abre un Pull Request

### Guías de Contribución

- Sigue el estilo de código existente
- Agrega tests para nuevas funcionalidades
- Actualiza la documentación
- Asegúrate de que los tests pasen

## 📝 Changelog

### v3.0.1 (2025-06-25)
- ✨ Framework base con NLP, aprendizaje y conversaciones
- 🔒 Sistema de seguridad integrado
- 🔌 Soporte para plugins
- 📊 Sistema de métricas y monitoreo
- 📚 Documentación completa

## 📄 Licencia

Este proyecto está bajo la Licencia MIT. Ver [LICENSE](LICENSE) para más detalles.

## 🙋‍♂️ Soporte

- **Issues**: [GitHub Issues](https://github.com/tu-usuario/ai-assistant-framework/issues)
- **Discusiones**: [GitHub Discussions](https://github.com/tu-usuario/ai-assistant-framework/discussions)
- **Email**: info@carlos-donoso.com

## 🏆 Créditos

Desarrollado con ❤️ por [Carlos Donoso]

### Dependencias Principales

- [NLTK](https://www.nltk.org/) - Procesamiento de lenguaje natural
- [spaCy](https://spacy.io/) - NLP avanzado
- [scikit-learn](https://scikit-learn.org/) - Machine learning
- [asyncio](https://docs.python.org/3/library/asyncio.html) - Programación asíncrona

---

**#ddchack** - Framework modular para asistentes de IA inteligentes 🤖✨
