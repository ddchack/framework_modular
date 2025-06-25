# .env.example
# Copia este archivo como .env y completa con tus claves reales
# #ddchack - Configuración segura de variables de entorno

# === CLAVES DE APIs DE IA ===
# NUNCA subas estas claves a repositorios públicos

# OpenAI API Key
# Obtén tu clave en: https://platform.openai.com/api-keys
OPENAI_API_KEY=sk-your-openai-api-key-here

# Anthropic API Key  
# Obtén tu clave en: https://console.anthropic.com/
ANTHROPIC_API_KEY=sk-ant-your-anthropic-api-key-here

# Hugging Face API Key
# Obtén tu clave en: https://huggingface.co/settings/tokens
HUGGINGFACE_API_KEY=hf_your-huggingface-token-here

# Google Cloud API Key (opcional)
# Para servicios de Google Cloud AI
GOOGLE_CLOUD_API_KEY=your-google-cloud-key-here

# === CONFIGURACIÓN DE BASE DE DATOS ===
# Si usas base de datos para almacenamiento persistente

# Base de datos principal
DATABASE_URL=postgresql://username:password@localhost:5432/ai_framework_db

# Redis para cache (opcional)
REDIS_URL=redis://localhost:6379

# === CONFIGURACIÓN DE APLICACIÓN ===

# Entorno de ejecución
ENVIRONMENT=development

# Puerto de la aplicación
PORT=8000

# Clave secreta para sesiones
SECRET_KEY=your-very-secret-key-change-this-in-production

# === CONFIGURACIÓN DE LOGGING ===

# Nivel de logging (DEBUG, INFO, WARNING, ERROR)
LOG_LEVEL=INFO

# Directorio de logs
LOG_DIR=./logs

# === CONFIGURACIÓN DE SEGURIDAD ===

# CORS origins permitidos (separados por coma)
ALLOWED_ORIGINS=http://localhost:3000,http://localhost:8080

# Rate limiting
RATE_LIMIT_REQUESTS_PER_MINUTE=60
RATE_LIMIT_REQUESTS_PER_HOUR=1000

# === CONFIGURACIÓN DE MODELOS ===

# Directorio para modelos locales
MODELS_DIR=./models

# Directorio para datos de entrenamiento
DATA_DIR=./data

# === WEBHOOKS Y NOTIFICACIONES ===

# Slack webhook para notificaciones (opcional)
SLACK_WEBHOOK_URL=https://hooks.slack.com/services/YOUR/SLACK/WEBHOOK

# Discord webhook (opcional)
DISCORD_WEBHOOK_URL=https://discord.com/api/webhooks/YOUR/DISCORD/WEBHOOK

# === MONITOREO Y ANALYTICS ===

# Sentry DSN para monitoreo de errores (opcional)
SENTRY_DSN=https://your-sentry-dsn-here

# Google Analytics ID (opcional)
GOOGLE_ANALYTICS_ID=GA-XXXXXXXXX-X

# === SERVICIOS EXTERNOS ===

# Servicio de traducción (opcional)
TRANSLATION_API_KEY=your-translation-api-key

# Servicio de síntesis de voz (opcional)
TTS_API_KEY=your-text-to-speech-api-key

# === INSTRUCCIONES DE USO ===
# 1. Copia este archivo como .env: cp .env.example .env
# 2. Completa las variables con tus valores reales
# 3. Asegúrate de que .env esté en tu .gitignore
# 4. Nunca compartas tu archivo .env
# 5. En producción, usa variables de entorno del sistema o servicios seguros

# === NOTAS DE SEGURIDAD ===
# - Cambia todas las claves por defecto
# - Usa claves fuertes y únicas
# - Rota las claves regularmente
# - Monitorea el uso de las APIs
# - Revisa los logs de acceso regularmente