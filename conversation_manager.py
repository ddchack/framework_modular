# conversation/manager.py
"""
Gestor de Conversaciones
========================

Este módulo maneja el contexto y flujo de las conversaciones del asistente de IA.
Mantiene el historial, estado y memoria de las interacciones con usuarios.

#ddchack - Sistema inteligente de gestión conversacional
"""

import json
import logging
import asyncio
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from collections import defaultdict, deque
from enum import Enum
import uuid

logger = logging.getLogger(__name__)

class ConversationState(Enum):
    """
    Estados posibles de una conversación
    """
    STARTED = "started"
    ACTIVE = "active"
    WAITING = "waiting"
    PAUSED = "paused"
    ENDED = "ended"

@dataclass
class Message:
    """
    Representa un mensaje en la conversación
    #ddchack - Estructura de mensaje conversacional
    """
    id: str
    sender: str  # 'user' o 'assistant'
    content: str
    timestamp: datetime
    metadata: Optional[Dict[str, Any]] = None

@dataclass
class ConversationContext:
    """
    Contexto de una conversación
    """
    user_id: str
    conversation_id: str
    state: ConversationState
    created_at: datetime
    last_activity: datetime
    messages: List[Message]
    user_profile: Dict[str, Any]
    session_data: Dict[str, Any]
    topic: Optional[str] = None

class ConversationMemory:
    """
    Sistema de memoria conversacional
    #ddchack - Memoria inteligente de conversaciones
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.max_history = config.get('max_history', 50)
        self.context_window = config.get('context_window', 10)
        
        # Almacenamiento en memoria
        self.short_term_memory = {}  # Memoria de la sesión actual
        self.long_term_memory = defaultdict(dict)  # Memoria persistente por usuario
        self.semantic_memory = defaultdict(list)  # Memoria semántica por temas
        
    def store_short_term(self, user_id: str, key: str, value: Any) -> None:
        """
        Almacena información en memoria a corto plazo
        """
        if user_id not in self.short_term_memory:
            self.short_term_memory[user_id] = {}
        
        self.short_term_memory[user_id][key] = {
            'value': value,
            'timestamp': datetime.now()
        }
    
    def get_short_term(self, user_id: str, key: str) -> Any:
        """
        Recupera información de memoria a corto plazo
        """
        user_memory = self.short_term_memory.get(user_id, {})
        item = user_memory.get(key)
        
        if item:
            # Verificar que no haya expirado (30 minutos por defecto)
            expiry = item['timestamp'] + timedelta(minutes=30)
            if datetime.now() < expiry:
                return item['value']
            else:
                # Limpiar memoria expirada
                del user_memory[key]
        
        return None
    
    def store_long_term(self, user_id: str, key: str, value: Any) -> None:
        """
        Almacena información en memoria a largo plazo
        """
        self.long_term_memory[user_id][key] = {
            'value': value,
            'created': datetime.now(),
            'access_count': 1,
            'last_access': datetime.now()
        }
    
    def get_long_term(self, user_id: str, key: str) -> Any:
        """
        Recupera información de memoria a largo plazo
        """
        user_memory = self.long_term_memory.get(user_id, {})
        item = user_memory.get(key)
        
        if item:
            # Actualizar estadísticas de acceso
            item['access_count'] += 1
            item['last_access'] = datetime.now()
            return item['value']
        
        return None
    
    def store_semantic(self, topic: str, content: Dict[str, Any]) -> None:
        """
        Almacena información semántica por tema
        """
        self.semantic_memory[topic].append({
            'content': content,
            'timestamp': datetime.now(),
            'relevance_score': 1.0
        })
        
        # Mantener solo los más relevantes (máximo 20 por tema)
        if len(self.semantic_memory[topic]) > 20:
            # Ordenar por relevancia y timestamp
            self.semantic_memory[topic].sort(
                key=lambda x: (x['relevance_score'], x['timestamp']),
                reverse=True
            )
            self.semantic_memory[topic] = self.semantic_memory[topic][:20]
    
    def get_semantic(self, topic: str, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Recupera información semántica relacionada con un tema
        """
        topic_memory = self.semantic_memory.get(topic, [])
        
        # Ordenar por relevancia y retornar los más relevantes
        sorted_memory = sorted(
            topic_memory,
            key=lambda x: (x['relevance_score'], x['timestamp']),
            reverse=True
        )
        
        return [item['content'] for item in sorted_memory[:limit]]
    
    def cleanup_expired(self) -> None:
        """
        Limpia memoria expirada
        """
        current_time = datetime.now()
        
        # Limpiar memoria a corto plazo (más de 1 hora)
        for user_id in list(self.short_term_memory.keys()):
            user_memory = self.short_term_memory[user_id]
            for key in list(user_memory.keys()):
                if current_time - user_memory[key]['timestamp'] > timedelta(hours=1):
                    del user_memory[key]
            
            # Eliminar usuarios sin memoria
            if not user_memory:
                del self.short_term_memory[user_id]
        
        # Limpiar memoria semántica antigua (más de 30 días)
        for topic in list(self.semantic_memory.keys()):
            self.semantic_memory[topic] = [
                item for item in self.semantic_memory[topic]
                if current_time - item['timestamp'] < timedelta(days=30)
            ]
            
            # Eliminar temas vacíos
            if not self.semantic_memory[topic]:
                del self.semantic_memory[topic]

class ConversationFlow:
    """
    Gestor del flujo conversacional
    #ddchack - Controlador de flujo inteligente
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.flow_rules = self._load_flow_rules()
        self.conversation_templates = self._load_conversation_templates()
    
    def _load_flow_rules(self) -> Dict[str, Any]:
        """
        Carga reglas de flujo conversacional
        """
        return {
            'greeting_responses': [
                "¡Hola! ¿En qué puedo ayudarte hoy?",
                "¡Buenos días! Estoy aquí para asistirte.",
                "¡Hola! Me alegra verte de nuevo. ¿Qué necesitas?"
            ],
            'clarification_prompts': [
                "¿Podrías darme más detalles sobre eso?",
                "No estoy seguro de entender. ¿Puedes explicarlo de otra manera?",
                "¿Te refieres a...?"
            ],
            'continuation_prompts': [
                "¿Hay algo más en lo que pueda ayudarte?",
                "¿Te gustaría saber algo más sobre este tema?",
                "¿Tienes alguna otra pregunta?"
            ],
            'closing_responses': [
                "¡Ha sido un placer ayudarte! Que tengas un excelente día.",
                "Perfecto. Si necesitas algo más, no dudes en preguntar.",
                "¡Hasta luego! Estaré aquí cuando me necesites."
            ]
        }
    
    def _load_conversation_templates(self) -> Dict[str, Any]:
        """
        Carga plantillas de conversación para diferentes escenarios
        """
        return {
            'technical_support': {
                'opening': "Entiendo que tienes un problema técnico. Vamos a resolverlo paso a paso.",
                'information_gathering': [
                    "¿Puedes describir exactamente qué está pasando?",
                    "¿Cuándo comenzó este problema?",
                    "¿Has intentado alguna solución?"
                ],
                'solution_steps': "Te voy a guiar a través de la solución:",
                'verification': "¿Esto resolvió tu problema?",
                'closing': "¡Excelente! Tu problema ha sido resuelto."
            },
            'information_request': {
                'acknowledgment': "Perfecto, te ayudo con esa información.",
                'clarification': "Para darte la información más precisa, ¿podrías especificar...?",
                'response': "Aquí está la información que solicitaste:",
                'additional_help': "¿Te gustaría saber algo más sobre este tema?"
            },
            'casual_conversation': {
                'engagement': "¡Me gusta charlar contigo!",
                'topic_exploration': "Eso es interesante. Cuéntame más sobre...",
                'empathy': "Entiendo cómo te sientes.",
                'encouragement': "¡Eso suena genial!"
            }
        }
    
    def determine_conversation_type(self, message: str, context: ConversationContext) -> str:
        """
        Determina el tipo de conversación basado en el mensaje y contexto
        """
        message_lower = message.lower()
        
        # Palabras clave para diferentes tipos
        technical_keywords = ['problema', 'error', 'no funciona', 'ayuda', 'arreglar', 'solucionar']
        information_keywords = ['qué es', 'cómo', 'cuándo', 'dónde', 'explica', 'información']
        casual_keywords = ['hola', 'cómo estás', 'charlar', 'contar', 'opinas']
        
        if any(keyword in message_lower for keyword in technical_keywords):
            return 'technical_support'
        elif any(keyword in message_lower for keyword in information_keywords):
            return 'information_request'
        elif any(keyword in message_lower for keyword in casual_keywords):
            return 'casual_conversation'
        else:
            return 'general'
    
    def get_flow_suggestion(self, conversation_type: str, stage: str) -> str:
        """
        Obtiene sugerencia de respuesta basada en el tipo y etapa de conversación
        """
        template = self.conversation_templates.get(conversation_type, {})
        suggestion = template.get(stage, "")
        
        if isinstance(suggestion, list):
            # Seleccionar aleatoriamente de las opciones
            import random
            return random.choice(suggestion)
        
        return suggestion
    
    def advance_conversation_stage(self, context: ConversationContext, current_response: str) -> str:
        """
        Determina la siguiente etapa de la conversación
        """
        message_count = len(context.messages)
        last_user_message = None
        
        # Encontrar el último mensaje del usuario
        for message in reversed(context.messages):
            if message.sender == 'user':
                last_user_message = message.content.lower()
                break
        
        # Lógica simple para determinar etapa
        if message_count <= 2:
            return 'opening'
        elif any(word in last_user_message for word in ['gracias', 'perfecto', 'resuelto', 'listo']):
            return 'closing'
        elif '?' in last_user_message or any(word in last_user_message for word in ['qué', 'cómo', 'cuándo']):
            return 'information_gathering'
        else:
            return 'continuation'

class ConversationManager:
    """
    Gestor principal de conversaciones
    #ddchack - Orquestador maestro de conversaciones
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.max_history = config.get('max_history', 50)
        self.context_window = config.get('context_window', 10)
        self.session_timeout = config.get('session_timeout', 3600)  # 1 hora
        self.enable_memory = config.get('enable_memory', True)
        
        # Componentes
        self.memory = ConversationMemory(config) if self.enable_memory else None
        self.flow_manager = ConversationFlow(config)
        
        # Almacenamiento de conversaciones activas
        self.active_conversations: Dict[str, ConversationContext] = {}
        self.user_sessions: Dict[str, str] = {}  # user_id -> conversation_id
        
        # Tareas de limpieza
        self._cleanup_task = None
        self._start_cleanup_task()
        
        logger.info("Gestor de conversaciones inicializado")
    
    def _start_cleanup_task(self) -> None:
        """
        Inicia tarea de limpieza periódica
        """
        async def cleanup_loop():
            while True:
                try:
                    await asyncio.sleep(300)  # Cada 5 minutos
                    await self._cleanup_expired_conversations()
                    if self.memory:
                        self.memory.cleanup_expired()
                except Exception as e:
                    logger.error(f"Error en limpieza periódica: {e}")
        
        self._cleanup_task = asyncio.create_task(cleanup_loop())
    
    async def _cleanup_expired_conversations(self) -> None:
        """
        Limpia conversaciones expiradas
        """
        current_time = datetime.now()
        expired_conversations = []
        
        for conv_id, context in self.active_conversations.items():
            time_since_activity = current_time - context.last_activity
            if time_since_activity.seconds > self.session_timeout:
                expired_conversations.append(conv_id)
        
        for conv_id in expired_conversations:
            context = self.active_conversations[conv_id]
            context.state = ConversationState.ENDED
            
            # Remover de sesiones activas
            if context.user_id in self.user_sessions:
                del self.user_sessions[context.user_id]
            
            # Archivar conversación (aquí se podría guardar en BD)
            logger.info(f"Conversación {conv_id} archivada por inactividad")
            del self.active_conversations[conv_id]
    
    async def start_conversation(self, user_id: str, initial_message: str = None) -> ConversationContext:
        """
        Inicia una nueva conversación o recupera la existente
        
        Args:
            user_id (str): ID del usuario
            initial_message (str): Mensaje inicial opcional
            
        Returns:
            ConversationContext: Contexto de la conversación
        """
        # Verificar si ya existe una conversación activa
        if user_id in self.user_sessions:
            conv_id = self.user_sessions[user_id]
            if conv_id in self.active_conversations:
                context = self.active_conversations[conv_id]
                context.last_activity = datetime.now()
                context.state = ConversationState.ACTIVE
                return context
        
        # Crear nueva conversación
        conversation_id = str(uuid.uuid4())
        
        # Cargar perfil de usuario de memoria a largo plazo
        user_profile = {}
        if self.memory:
            stored_profile = self.memory.get_long_term(user_id, 'profile')
            if stored_profile:
                user_profile = stored_profile
            else:
                # Crear perfil básico
                user_profile = {
                    'first_interaction': datetime.now().isoformat(),
                    'interaction_count': 0,
                    'preferences': {},
                    'conversation_style': 'neutral'
                }
                self.memory.store_long_term(user_id, 'profile', user_profile)
        
        context = ConversationContext(
            user_id=user_id,
            conversation_id=conversation_id,
            state=ConversationState.STARTED,
            created_at=datetime.now(),
            last_activity=datetime.now(),
            messages=[],
            user_profile=user_profile,
            session_data={}
        )
        
        # Agregar mensaje inicial si se proporciona
        if initial_message:
            await self.add_message(context, 'user', initial_message)
        
        # Registrar conversación
        self.active_conversations[conversation_id] = context
        self.user_sessions[user_id] = conversation_id
        
        logger.info(f"Nueva conversación iniciada: {conversation_id} para usuario {user_id}")
        return context
    
    async def add_message(self, context: ConversationContext, sender: str, content: str, 
                         metadata: Optional[Dict[str, Any]] = None) -> Message:
        """
        Agrega un mensaje a la conversación
        
        Args:
            context (ConversationContext): Contexto de la conversación
            sender (str): 'user' o 'assistant'
            content (str): Contenido del mensaje
            metadata (Optional[Dict]): Metadatos adicionales
            
        Returns:
            Message: Mensaje creado
        """
        message = Message(
            id=str(uuid.uuid4()),
            sender=sender,
            content=content,
            timestamp=datetime.now(),
            metadata=metadata or {}
        )
        
        context.messages.append(message)
        context.last_activity = datetime.now()
        
        # Mantener límite de historial
        if len(context.messages) > self.max_history:
            # Remover mensajes más antiguos pero mantener estructura de conversación
            excess = len(context.messages) - self.max_history
            context.messages = context.messages[excess:]
        
        # Actualizar estado de conversación
        if context.state == ConversationState.STARTED:
            context.state = ConversationState.ACTIVE
        
        # Actualizar memoria si está habilitada
        if self.memory and sender == 'user':
            # Almacenar en memoria a corto plazo
            self.memory.store_short_term(context.user_id, 'last_message', content)
            
            # Actualizar perfil de usuario
            context.user_profile['interaction_count'] += 1
            self.memory.store_long_term(context.user_id, 'profile', context.user_profile)
        
        return message
    
    async def get_context(self, user_id: str) -> Dict[str, Any]:
        """
        Obtiene el contexto actual para un usuario
        
        Args:
            user_id (str): ID del usuario
            
        Returns:
            Dict: Contexto de conversación
        """
        # Verificar conversación activa
        if user_id not in self.user_sessions:
            return {
                'has_active_conversation': False,
                'history': [],
                'user_profile': {},
                'session_data': {}
            }
        
        conv_id = self.user_sessions[user_id]
        context = self.active_conversations.get(conv_id)
        
        if not context:
            return {
                'has_active_conversation': False,
                'history': [],
                'user_profile': {},
                'session_data': {}
            }
        
        # Preparar historial para ventana de contexto
        recent_messages = context.messages[-self.context_window:]
        history = [
            {
                'sender': msg.sender,
                'content': msg.content,
                'timestamp': msg.timestamp.isoformat(),
                'metadata': msg.metadata
            }
            for msg in recent_messages
        ]
        
        # Obtener información adicional de memoria
        memory_data = {}
        if self.memory:
            memory_data = {
                'short_term': self.memory.short_term_memory.get(user_id, {}),
                'semantic_topics': list(self.memory.semantic_memory.keys())
            }
        
        return {
            'has_active_conversation': True,
            'conversation_id': context.conversation_id,
            'conversation_state': context.state.value,
            'history': history,
            'user_profile': context.user_profile,
            'session_data': context.session_data,
            'memory': memory_data,
            'conversation_stage': self._determine_conversation_stage(context),
            'topic': context.topic
        }
    
    def _determine_conversation_stage(self, context: ConversationContext) -> str:
        """
        Determina la etapa actual de la conversación
        """
        message_count = len(context.messages)
        
        if message_count == 0:
            return 'initial'
        elif message_count <= 2:
            return 'opening'
        elif message_count <= 10:
            return 'development'
        else:
            return 'extended'
    
    async def add_interaction(self, user_id: str, user_message: str, assistant_response: str,
                            metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        Agrega una interacción completa (pregunta del usuario + respuesta del asistente)
        
        Args:
            user_id (str): ID del usuario
            user_message (str): Mensaje del usuario
            assistant_response (str): Respuesta del asistente
            metadata (Optional[Dict]): Metadatos adicionales
            
        Returns:
            bool: True si se agregó correctamente
        """
        try:
            # Obtener o crear conversación
            context = await self.start_conversation(user_id)
            
            # Agregar mensaje del usuario
            await self.add_message(context, 'user', user_message, metadata)
            
            # Agregar respuesta del asistente
            await self.add_message(context, 'assistant', assistant_response, metadata)
            
            # Detectar y almacenar tema si es relevante
            topic = self._extract_topic(user_message, assistant_response)
            if topic and self.memory:
                self.memory.store_semantic(topic, {
                    'user_message': user_message,
                    'assistant_response': assistant_response,
                    'user_id': user_id,
                    'conversation_id': context.conversation_id
                })
                context.topic = topic
            
            return True
            
        except Exception as e:
            logger.error(f"Error agregando interacción: {e}")
            return False
    
    def _extract_topic(self, user_message: str, assistant_response: str) -> Optional[str]:
        """
        Extrae el tema principal de la conversación
        """
        # Palabras clave para identificar temas
        topic_keywords = {
            'tecnología': ['programación', 'código', 'software', 'computadora', 'tecnología'],
            'salud': ['medicina', 'salud', 'enfermedad', 'síntoma', 'doctor'],
            'educación': ['estudiar', 'aprender', 'escuela', 'universidad', 'curso'],
            'trabajo': ['trabajo', 'empleo', 'oficina', 'carrera', 'profesional'],
            'viaje': ['viaje', 'vacaciones', 'turismo', 'ciudad', 'país'],
            'comida': ['comida', 'receta', 'cocinar', 'restaurante', 'comer']
        }
        
        combined_text = (user_message + ' ' + assistant_response).lower()
        
        for topic, keywords in topic_keywords.items():
            if any(keyword in combined_text for keyword in keywords):
                return topic
        
        return None
    
    async def end_conversation(self, user_id: str, reason: str = "user_request") -> bool:
        """
        Termina una conversación activa
        
        Args:
            user_id (str): ID del usuario
            reason (str): Razón del cierre
            
        Returns:
            bool: True si se cerró correctamente
        """
        if user_id not in self.user_sessions:
            return False
        
        conv_id = self.user_sessions[user_id]
        context = self.active_conversations.get(conv_id)
        
        if context:
            context.state = ConversationState.ENDED
            context.session_data['end_reason'] = reason
            context.session_data['ended_at'] = datetime.now().isoformat()
            
            # Limpiar sesión activa
            del self.user_sessions[user_id]
            
            logger.info(f"Conversación {conv_id} terminada: {reason}")
            return True
        
        return False
    
    def get_conversation_statistics(self) -> Dict[str, Any]:
        """
        Obtiene estadísticas de las conversaciones
        #ddchack - Métricas conversacionales
        """
        active_count = len(self.active_conversations)
        total_messages = sum(len(ctx.messages) for ctx in self.active_conversations.values())
        
        # Estadísticas por estado
        state_counts = defaultdict(int)
        for context in self.active_conversations.values():
            state_counts[context.state.value] += 1
        
        # Usuarios únicos
        unique_users = len(set(ctx.user_id for ctx in self.active_conversations.values()))
        
        # Duración promedio de conversaciones
        active_durations = []
        for context in self.active_conversations.values():
            duration = (datetime.now() - context.created_at).total_seconds() / 60  # minutos
            active_durations.append(duration)
        
        avg_duration = sum(active_durations) / len(active_durations) if active_durations else 0
        
        return {
            'active_conversations': active_count,
            'total_messages': total_messages,
            'unique_users': unique_users,
            'average_duration_minutes': round(avg_duration, 2),
            'conversations_by_state': dict(state_counts),
            'average_messages_per_conversation': round(total_messages / max(active_count, 1), 2),
            'memory_enabled': self.enable_memory,
            'memory_stats': self._get_memory_stats() if self.memory else {}
        }
    
    def _get_memory_stats(self) -> Dict[str, Any]:
        """
        Obtiene estadísticas de memoria
        """
        if not self.memory:
            return {}
        
        return {
            'short_term_users': len(self.memory.short_term_memory),
            'long_term_users': len(self.memory.long_term_memory),
            'semantic_topics': len(self.memory.semantic_memory),
            'total_semantic_entries': sum(len(entries) for entries in self.memory.semantic_memory.values())
        }
    
    def __del__(self):
        """
        Limpieza al destruir el objeto
        """
        if self._cleanup_task and not self._cleanup_task.done():
            self._cleanup_task.cancel()