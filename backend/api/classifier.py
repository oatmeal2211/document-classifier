import os
import joblib
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
import string
import re
from django.conf import settings
import csv
import django
import sys
import requests
import PyPDF2
from io import BytesIO
from bs4 import BeautifulSoup
import time
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry
from sklearn.naive_bayes import MultinomialNB

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
    
    def train_topic_classifier(self, documents, labels, model_type='LinearSVC'):
        """Train the topic classifier with provided documents and labels."""
        # Preprocess documents
        processed_docs = [self.preprocess_text(doc) for doc in documents]
        
        # Split data into training and testing sets
        docs_train, docs_test, labels_train, labels_test = train_test_split(
            processed_docs, labels, test_size=0.2, random_state=42
        )
        
        # Define the classifier based on model_type
        if model_type == 'LinearSVC':
            classifier = LinearSVC(C=1.5)
        elif model_type == 'MultinomialNB':
            classifier = MultinomialNB()
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        # Create and train the pipeline
        pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(max_features=5000, ngram_range=(1, 3))),
            ('classifier', classifier)
        ])
        
        # Train the model
        pipeline.fit(docs_train, labels_train)
        
        # Evaluate the model (optional, but good for debugging)
        accuracy = pipeline.score(docs_test, labels_test)
        print(f"Topic classifier test accuracy ({model_type}): {accuracy:.2f}")
        
        # Save the model
        joblib.dump(pipeline, TOPIC_MODEL_PATH)
        self.topic_classifier = pipeline
        
        return pipeline
    
    def train_doc_type_classifier(self, documents, labels, model_type='LinearSVC'):
        """Train the document type classifier with provided documents and labels."""
        # Preprocess documents
        processed_docs = [self.preprocess_text(doc) for doc in documents]
        
        # Split data into training and testing sets
        docs_train, docs_test, labels_train, labels_test = train_test_split(
            processed_docs, labels, test_size=0.2, random_state=42
        )
        
        # Define the classifier based on model_type
        if model_type == 'LinearSVC':
            classifier = LinearSVC(C=1.5)
        elif model_type == 'MultinomialNB':
            classifier = MultinomialNB()
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        # Create and train the pipeline
        pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(max_features=5000, ngram_range=(1, 3))),
            ('classifier', classifier)
        ])
        
        # Train the model
        pipeline.fit(docs_train, labels_train)
        
        # Evaluate the model (optional, but good for debugging)
        accuracy = pipeline.score(docs_test, labels_test)
        print(f"Document type classifier test accuracy ({model_type}): {accuracy:.2f}")
        
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

            # Calculate confidence based on classifier type
            # Corrected step name from 'classifier' to 'clf'
            if isinstance(self.topic_classifier.named_steps['classifier'], LinearSVC):
                # Use decision_function for LinearSVC
                scores = self.topic_classifier.decision_function([processed_text])[0]
                # Get the index of the predicted class
                pred_idx = self.topic_classifier.classes_.tolist().index(prediction)
                # Confidence is the absolute value of the score for the predicted class
                confidence = float(abs(scores[pred_idx]))
            # Keep the predict_proba logic just in case a different classifier type is used later
            else:
                # Use predict_proba for other classifiers (like MultinomialNB)
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

            # Calculate confidence based on classifier type
            # Corrected step name from 'classifier' to 'clf'
            if isinstance(self.doc_type_classifier.named_steps['classifier'], LinearSVC):
                # Use decision_function for LinearSVC
                scores = self.doc_type_classifier.decision_function([processed_text])[0]
                # Get the index of the predicted class
                pred_idx = self.doc_type_classifier.classes_.tolist().index(prediction)
                # Confidence is the absolute value of the score for the predicted class
                confidence = float(abs(scores[pred_idx]))
            # Keep the predict_proba logic just in case a different classifier type is used later
            else:
                # Use predict_proba for other classifiers (like MultinomialNB)
                probabilities = self.doc_type_classifier.predict_proba([processed_text])[0]
                confidence = float(max(probabilities))

            return {
                'classification_type': 'document_type',
                'result': prediction,
                'confidence': confidence
            }

        else:
            return {'error': 'Invalid classification type.'}


def fetch_document_content(url):
    """Fetches content from a URL and extracts text based on content type, limiting large documents."""
    # Clean the URL: remove trailing text like (Source Name) and handle potential extra spaces/newlines
    cleaned_url = re.sub(r'\s*\([^)]*\)\s*$', '', url).strip()
    # Corrected regex for removing quotes/apostrophes and backslashes
    cleaned_url = re.sub(r'[\'"\\]', '', cleaned_url)

    # Add a scheme if missing (basic check)
    if not re.match(r'^https?://', cleaned_url):
        cleaned_url = 'https://' + cleaned_url

    # Add a User-Agent header and set up retry strategy
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    
    # Configure requests to retry failed connections and statuses
    retry_strategy = Retry(
        total=3, # Number of retries
        backoff_factor=1, # Factor by which wait time increases
        status_forcelist=[401, 403, 404, 429, 500, 502, 503, 504], # Status codes to retry
        allowed_methods=["HEAD", "GET", "OPTIONS"]
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    http = requests.Session()
    http.mount("https://", adapter)
    http.mount("http://", adapter)

    try:
        # Use the session with retry strategy
        # Set stream=True to avoid downloading the entire content at once for large files
        response = http.get(cleaned_url, timeout=15, headers=headers, stream=True)
        response.raise_for_status() # Raise an HTTPError for bad responses (4xx or 5xx)
        
        content_type = response.headers.get('Content-Type', '').lower()
        
        if 'application/pdf' in content_type:
            print(f"  Processing as PDF (first page only)...")
            # Read only the first page
            with BytesIO(response.content) as f:
                reader = PyPDF2.PdfReader(f)
                if len(reader.pages) > 0:
                    page = reader.pages[0]
                    text = page.extract_text() or ''
                    return text.strip()
                else:
                    return ""
        
        elif 'text/html' in content_type:
            print(f"  Processing as HTML (approx. first page)...")
            # Read a limited chunk of HTML content
            chunk_size = 8192 # Read first 8KB as an approximation of the initial view
            html_content = ""
            try:
                # Read the first chunk
                html_content = next(response.iter_content(chunk_size=chunk_size)).decode('utf-8', errors='ignore')
            except StopIteration:
                # Handle cases where content is smaller than chunk size
                pass
            finally:
                # Ensure the response body is closed
                response.close()

            soup = BeautifulSoup(html_content, 'lxml')
            
            # Extract text from key elements in the initial chunk - no fallback to full body
            text = ''
            # Strictly extract from the initial chunk by finding all relevant elements in the parsed limited HTML
            for element in soup.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'li', 'article', 'main']):
                 text += element.get_text() + '\n' # Add newline for better separation
            
            # Basic cleaning of extracted HTML text
            text = re.sub(r'\s+', ' ', text).strip() # Replace multiple spaces with a single space
            text = re.sub(r'(\n)\s+', '\n', text).strip() # Clean up spaces after newlines

            return text

        else:
            print(f"Warning: Unhandled content type {content_type} for URL {cleaned_url}.")
            return ""

    except requests.exceptions.Timeout:
        print(f"Error fetching {cleaned_url}: Request timed out.")
        return ""
    except requests.exceptions.RequestException as e:
        print(f"Error fetching {cleaned_url}: {e}")
        return ""
    except Exception as e:
        print(f"Error processing content from {cleaned_url}: {e}")
        return ""

# Create a single instance to be used by views
classifier = DocumentClassifier() 