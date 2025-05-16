import os
import joblib
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
import string
import re
from django.conf import settings

# Download required NLTK resources
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

# Define model paths
MODELS_DIR = os.path.join(settings.BASE_DIR, 'media', 'models')
os.makedirs(MODELS_DIR, exist_ok=True)

TOPIC_MODEL_PATH = os.path.join(MODELS_DIR, 'topic_classifier.joblib')
DOC_TYPE_MODEL_PATH = os.path.join(MODELS_DIR, 'doctype_classifier.joblib')

class DocumentClassifier:
    """Document classifier for both topic and document type classification."""
    
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        
        # Initialize or load classifiers
        self.topic_classifier = self._load_model(TOPIC_MODEL_PATH)
        self.doc_type_classifier = self._load_model(DOC_TYPE_MODEL_PATH)
    
    def _load_model(self, model_path):
        """Load a pre-trained model or return None if it doesn't exist."""
        if os.path.exists(model_path):
            return joblib.load(model_path)
        return None
    
    def preprocess_text(self, text):
        """Clean and preprocess text for classification."""
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters and numbers
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # Remove stopwords and lemmatize
        cleaned_tokens = [
            self.lemmatizer.lemmatize(token)
            for token in tokens
            if token not in self.stop_words and len(token) > 2
        ]
        
        return ' '.join(cleaned_tokens)
    
    def train_topic_classifier(self, documents, labels):
        """Train the topic classifier with provided documents and labels."""
        # Preprocess documents
        processed_docs = [self.preprocess_text(doc) for doc in documents]
        
        # Create and train the pipeline
        pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(max_features=5000)),
            ('classifier', MultinomialNB())
        ])
        
        # Train the model
        pipeline.fit(processed_docs, labels)
        
        # Save the model
        joblib.dump(pipeline, TOPIC_MODEL_PATH)
        self.topic_classifier = pipeline
        
        return pipeline
    
    def train_doc_type_classifier(self, documents, labels):
        """Train the document type classifier with provided documents and labels."""
        # Preprocess documents
        processed_docs = [self.preprocess_text(doc) for doc in documents]
        
        # Create and train the pipeline
        pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(max_features=5000)),
            ('classifier', MultinomialNB())
        ])
        
        # Train the model
        pipeline.fit(processed_docs, labels)
        
        # Save the model
        joblib.dump(pipeline, DOC_TYPE_MODEL_PATH)
        self.doc_type_classifier = pipeline
        
        return pipeline
    
    def classify_document(self, text, classification_type='topic'):
        """Classify a document by topic or document type."""
        # Preprocess the text
        processed_text = self.preprocess_text(text)
        
        if classification_type == 'topic':
            if self.topic_classifier is None:
                return {'error': 'Topic classifier has not been trained yet.'}
            
            # Predict topic
            prediction = self.topic_classifier.predict([processed_text])[0]
            probabilities = self.topic_classifier.predict_proba([processed_text])[0]
            confidence = float(max(probabilities))
            
            return {
                'classification_type': 'topic',
                'result': prediction,
                'confidence': confidence
            }
            
        elif classification_type == 'document_type':
            if self.doc_type_classifier is None:
                return {'error': 'Document type classifier has not been trained yet.'}
            
            # Predict document type
            prediction = self.doc_type_classifier.predict([processed_text])[0]
            probabilities = self.doc_type_classifier.predict_proba([processed_text])[0]
            confidence = float(max(probabilities))
            
            return {
                'classification_type': 'document_type',
                'result': prediction,
                'confidence': confidence
            }
            
        else:
            return {'error': 'Invalid classification type.'}

# Create a single instance to be used by views
classifier = DocumentClassifier() 