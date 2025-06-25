# nlp/processor.py
"""
Módulo de Procesamiento de Lenguaje Natural
===========================================

Este módulo maneja el análisis y procesamiento de texto usando técnicas de NLP.
Incluye análisis de sentimientos, reconocimiento de entidades y clasificación de intenciones.

#ddchack - Procesador NLP avanzado
"""

import re
import json
import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass
from abc import ABC, abstractmethod

try:
    # Importaciones opcionales para NLP avanzado
    import nltk
    from nltk.sentiment import SentimentIntensityAnalyzer
    from nltk.tokenize import word_tokenize, sent_tokenize
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False
    logging.warning("NLTK no disponible. Usando procesamiento básico de texto.")

try:
    import spacy
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False
    logging.warning("SpaCy no disponible. Funcionalidad limitada de NER.")

logger = logging.getLogger(__name__)

@dataclass
class NLPResult:
    """
    Resultado del procesamiento NLP
    #ddchack - Estructura de datos para resultados NLP
    """
    text: str
    language: str
    sentiment: Dict[str, float]
    entities: List[Dict[str, Any]]
    intent: str
    confidence: float
    tokens: List[str]
    keywords: List[str]
    processed_at: datetime

class BaseNLPAnalyzer(ABC):
    """
    Clase base para analizadores NLP
    """
    
    @abstractmethod
    async def analyze(self, text: str) -> Dict[str, Any]:
        """Analiza el texto y retorna resultados"""
        pass

class SentimentAnalyzer(BaseNLPAnalyzer):
    """
    Analizador de sentimientos
    #ddchack - Análisis de emociones en texto
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.analyzer = None
        
        if NLTK_AVAILABLE:
            try:
                self.analyzer = SentimentIntensityAnalyzer()
            except LookupError:
                logger.info("Descargando recursos NLTK para análisis de sentimientos...")
                nltk.download('vader_lexicon', quiet=True)
                self.analyzer = SentimentIntensityAnalyzer()
    
    async def analyze(self, text: str) -> Dict[str, Any]:
        """
        Analiza el sentimiento del texto
        
        Args:
            text (str): Texto a analizar
            
        Returns:
            Dict: Scores de sentimiento
        """
        if self.analyzer:
            # Análisis con NLTK VADER
            scores = self.analyzer.polarity_scores(text)
            return {
                'positive': scores['pos'],
                'negative': scores['neg'],
                'neutral': scores['neu'],
                'compound': scores['compound'],
                'method': 'vader'
            }
        else:
            # Análisis básico por palabras clave
            return await self._basic_sentiment_analysis(text)
    
    async def _basic_sentiment_analysis(self, text: str) -> Dict[str, Any]:
        """
        Análisis básico de sentimientos usando palabras clave
        """
        positive_words = {
            'bueno', 'excelente', 'fantástico', 'genial', 'perfecto', 'maravilloso',
            'increíble', 'satisfecho', 'feliz', 'contento', 'bien', 'positivo',
            'correcto', 'acertado', 'exitoso', 'favorable'
        }
        
        negative_words = {
            'malo', 'terrible', 'horrible', 'pésimo', 'deficiente', 'insatisfecho',
            'triste', 'enojado', 'molesto', 'frustrado', 'decepcionado', 'negativo',
            'incorrecto', 'equivocado', 'fallido', 'desfavorable'
        }
        
        words = text.lower().split()
        pos_count = sum(1 for word in words if word in positive_words)
        neg_count = sum(1 for word in words if word in negative_words)
        total_sentiment_words = pos_count + neg_count
        
        if total_sentiment_words == 0:
            return {
                'positive': 0.0,
                'negative': 0.0,
                'neutral': 1.0,
                'compound': 0.0,
                'method': 'keyword_basic'
            }
        
        pos_score = pos_count / len(words)
        neg_score = neg_count / len(words)
        neutral_score = 1.0 - (pos_score + neg_score)
        compound = pos_score - neg_score
        
        return {
            'positive': pos_score,
            'negative': neg_score,
            'neutral': max(0.0, neutral_score),
            'compound': compound,
            'method': 'keyword_basic'
        }

class EntityRecognizer(BaseNLPAnalyzer):
    """
    Reconocedor de entidades nombradas
    #ddchack - Extracción de entidades del texto
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.nlp = None
        
        if SPACY_AVAILABLE:
            try:
                self.nlp = spacy.load("es_core_news_sm")
            except OSError:
                logger.warning("Modelo español de SpaCy no encontrado. Usando extracción básica.")
    
    async def analyze(self, text: str) -> List[Dict[str, Any]]:
        """
        Extrae entidades nombradas del texto
        
        Args:
            text (str): Texto a analizar
            
        Returns:
            List[Dict]: Lista de entidades encontradas
        """
        if self.nlp:
            return await self._spacy_entity_extraction(text)
        else:
            return await self._basic_entity_extraction(text)
    
    async def _spacy_entity_extraction(self, text: str) -> List[Dict[str, Any]]:
        """
        Extracción de entidades usando SpaCy
        """
        doc = self.nlp(text)
        entities = []
        
        for ent in doc.ents:
            entities.append({
                'text': ent.text,
                'label': ent.label_,
                'start': ent.start_char,
                'end': ent.end_char,
                'confidence': 1.0,  # SpaCy no proporciona scores directamente
                'method': 'spacy'
            })
        
        return entities
    
    async def _basic_entity_extraction(self, text: str) -> List[Dict[str, Any]]:
        """
        Extracción básica usando patrones regex
        """
        entities = []
        
        # Patrones para diferentes tipos de entidades
        patterns = {
            'EMAIL': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            'PHONE': r'\b(?:\+?34)?[6789]\d{8}\b',
            'URL': r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+',
            'DATE': r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b',
            'MONEY': r'\b\d+[.,]?\d*\s?(?:€|EUR|dollars?|USD)\b'
        }
        
        for entity_type, pattern in patterns.items():
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                entities.append({
                    'text': match.group(),
                    'label': entity_type,
                    'start': match.start(),
                    'end': match.end(),
                    'confidence': 0.8,
                    'method': 'regex'
                })
        
        # Entidades personalizadas del config
        custom_entities = self.config.get('custom_entities', [])
        for entity in custom_entities:
            pattern = rf'\b{re.escape(entity)}\b'
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                entities.append({
                    'text': match.group(),
                    'label': 'CUSTOM',
                    'start': match.start(),
                    'end': match.end(),
                    'confidence': 0.9,
                    'method': 'custom_regex'
                })
        
        return entities

class IntentClassifier(BaseNLPAnalyzer):
    """
    Clasificador de intenciones
    #ddchack - Clasificación inteligente de intenciones
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.intent_patterns = self._load_intent_patterns()
    
    def _load_intent_patterns(self) -> Dict[str, List[str]]:
        """
        Carga patrones de intenciones predefinidos
        """
        return {
            'saludo': [
                'hola', 'buenos días', 'buenas tardes', 'buenas noches', 'saludos',
                'qué tal', 'cómo estás', 'hey', 'hi', 'hello'
            ],
            'despedida': [
                'adiós', 'hasta luego', 'nos vemos', 'chao', 'bye', 'hasta pronto',
                'que tengas buen día', 'hasta la vista'
            ],
            'pregunta': [
                'qué', 'cómo', 'cuándo', 'dónde', 'por qué', 'para qué', 'cuál',
                'puedes', 'podrías', 'me ayudas', 'explica', 'dime'
            ],
            'ayuda': [
                'ayuda', 'help', 'socorro', 'asistencia', 'soporte', 'apoyo',
                'no entiendo', 'no sé', 'estoy perdido'
            ],
            'queja': [
                'problema', 'error', 'fallo', 'no funciona', 'mal', 'incorrecto',
                'molesto', 'frustrated', 'insatisfecho'
            ],
            'agradecimiento': [
                'gracias', 'thanks', 'te agradezco', 'muchas gracias',
                'perfecto', 'excelente', 'genial'
            ],
            'información': [
                'información', 'datos', 'detalles', 'características', 'especificaciones',
                'describe', 'cuéntame', 'háblame'
            ]
        }
    
    async def analyze(self, text: str) -> Tuple[str, float]:
        """
        Clasifica la intención del texto
        
        Args:
            text (str): Texto a clasificar
            
        Returns:
            Tuple[str, float]: Intención y confianza
        """
        text_lower = text.lower()
        intent_scores = {}
        
        # Calcular scores para cada intención
        for intent, patterns in self.intent_patterns.items():
            score = 0.0
            matches = 0
            
            for pattern in patterns:
                if pattern in text_lower:
                    score += 1.0
                    matches += 1
                # Bonus por coincidencias al inicio
                if text_lower.startswith(pattern):
                    score += 0.5
            
            if matches > 0:
                # Normalizar por longitud del texto y número de patrones
                normalized_score = (score / len(text.split())) * (matches / len(patterns))
                intent_scores[intent] = min(normalized_score, 1.0)
        
        if not intent_scores:
            return self.config.get('fallback_intent', 'general'), 0.1
        
        # Obtener la intención con mayor score
        best_intent = max(intent_scores.items(), key=lambda x: x[1])
        confidence_threshold = self.config.get('confidence_threshold', 0.7)
        
        if best_intent[1] >= confidence_threshold:
            return best_intent[0], best_intent[1]
        else:
            return self.config.get('fallback_intent', 'general'), best_intent[1]

class NLPProcessor:
    """
    Procesador principal de NLP que coordina todos los analizadores
    #ddchack - Orquestador principal de NLP
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.language = config.get('language', 'es')
        
        # Inicializar analizadores
        self.sentiment_analyzer = SentimentAnalyzer(config.get('sentiment_analysis', {}))
        self.entity_recognizer = EntityRecognizer(config.get('entity_recognition', {}))
        self.intent_classifier = IntentClassifier(config.get('intent_classification', {}))
        
        # Configurar tokenizador
        self.lemmatizer = None
        self.stop_words = set()
        
        if NLTK_AVAILABLE:
            try:
                self.lemmatizer = WordNetLemmatizer()
                if self.language == 'es':
                    self.stop_words = set(stopwords.words('spanish'))
                else:
                    self.stop_words = set(stopwords.words('english'))
            except LookupError:
                logger.info("Descargando recursos NLTK...")
                nltk.download('punkt', quiet=True)
                nltk.download('stopwords', quiet=True)
                nltk.download('wordnet', quiet=True)
                nltk.download('omw-1.4', quiet=True)
                
                self.lemmatizer = WordNetLemmatizer()
                if self.language == 'es':
                    self.stop_words = set(stopwords.words('spanish'))
                else:
                    self.stop_words = set(stopwords.words('english'))
        
        logger.info("Procesador NLP inicializado correctamente")
    
    async def process(self, text: str) -> NLPResult:
        """
        Procesa texto completo con todos los analizadores
        
        Args:
            text (str): Texto a procesar
            
        Returns:
            NLPResult: Resultado completo del análisis
        """
        if not text or not text.strip():
            return self._empty_result()
        
        # Preprocesamiento
        cleaned_text = self._preprocess_text(text)
        
        # Análisis paralelo de componentes
        sentiment = await self.sentiment_analyzer.analyze(cleaned_text)
        entities = await self.entity_recognizer.analyze(cleaned_text)
        intent, confidence = await self.intent_classifier.analyze(cleaned_text)
        
        # Tokenización y extracción de palabras clave
        tokens = self._tokenize(cleaned_text)
        keywords = self._extract_keywords(tokens)
        
        return NLPResult(
            text=text,
            language=self.language,
            sentiment=sentiment,
            entities=entities,
            intent=intent,
            confidence=confidence,
            tokens=tokens,
            keywords=keywords,
            processed_at=datetime.now()
        )
    
    def _preprocess_text(self, text: str) -> str:
        """
        Preprocesa el texto según la configuración
        """
        processed = text
        
        preprocessing_config = self.config.get('preprocessing', {})
        
        if preprocessing_config.get('lowercase', True):
            processed = processed.lower()
        
        # Limpiar caracteres especiales pero mantener puntuación básica
        processed = re.sub(r'[^\w\s.,!?¿¡]', ' ', processed)
        processed = re.sub(r'\s+', ' ', processed).strip()
        
        return processed
    
    def _tokenize(self, text: str) -> List[str]:
        """
        Tokeniza el texto en palabras
        """
        if NLTK_AVAILABLE:
            tokens = word_tokenize(text, language='spanish' if self.language == 'es' else 'english')
        else:
            tokens = text.split()
        
        # Filtrar stopwords si está configurado
        if self.config.get('preprocessing', {}).get('remove_stopwords', True):
            tokens = [token for token in tokens if token.lower() not in self.stop_words]
        
        # Lematización si está disponible y configurada
        if (self.lemmatizer and 
            self.config.get('preprocessing', {}).get('lemmatization', True)):
            tokens = [self.lemmatizer.lemmatize(token) for token in tokens]
        
        return tokens
    
    def _extract_keywords(self, tokens: List[str]) -> List[str]:
        """
        Extrae palabras clave del texto tokenizado
        """
        # Filtrar tokens cortos y que no sean solo números
        keywords = [
            token for token in tokens 
            if len(token) > 2 and not token.isdigit()
        ]
        
        # Obtener frecuencias y seleccionar las más importantes
        from collections import Counter
        token_freq = Counter(keywords)
        
        # Retornar las 10 palabras más frecuentes
        return [word for word, _ in token_freq.most_common(10)]
    
    def _empty_result(self) -> NLPResult:
        """
        Retorna resultado vacío para texto inválido
        """
        return NLPResult(
            text="",
            language=self.language,
            sentiment={'positive': 0.0, 'negative': 0.0, 'neutral': 1.0, 'compound': 0.0},
            entities=[],
            intent='general',
            confidence=0.0,
            tokens=[],
            keywords=[],
            processed_at=datetime.now()
        )
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Retorna estadísticas del procesador
        #ddchack - Métricas del procesador NLP
        """
        return {
            'language': self.language,
            'nltk_available': NLTK_AVAILABLE,
            'spacy_available': SPACY_AVAILABLE,
            'components': {
                'sentiment_analyzer': self.sentiment_analyzer is not None,
                'entity_recognizer': self.entity_recognizer is not None,
                'intent_classifier': self.intent_classifier is not None
            },
            'stop_words_count': len(self.stop_words),
            'intent_patterns': len(self.intent_classifier.intent_patterns) if self.intent_classifier else 0
        }