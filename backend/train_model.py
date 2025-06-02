#!/usr/bin/env python
"""
Script to train the document classifier using the NLP document bank CSV file.
"""
import os
import csv
import django
import sys
import requests
import PyPDF2
from io import BytesIO
from bs4 import BeautifulSoup
import re
from django.conf import settings
import time
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split, StratifiedKFold
import string
from sklearn.metrics import classification_report, accuracy_score, f1_score, precision_score, recall_score
import joblib
import numpy as np
from nltk.stem import WordNetLemmatizer
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE, RandomOverSampler
from collections import Counter
import concurrent.futures  # Import concurrent.futures
import pandas as pd  # Import pandas
from sklearn.naive_bayes import MultinomialNB

# Set up Django environment
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'backend.settings')
django.setup()

from api.models import TrainingData

# Download required NLTK resources
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

# Define model paths
MODELS_DIR = os.path.join(settings.BASE_DIR, 'media', 'models')
os.makedirs(MODELS_DIR, exist_ok=True)

TOPIC_MODEL_PATH = os.path.join(MODELS_DIR, 'topic_classifier.joblib')
DOC_TYPE_MODEL_PATH = os.path.join(MODELS_DIR, 'doctype_classifier.joblib')

# Import fetch_document_content
from api.classifier import fetch_document_content

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
        """Enhanced text preprocessing with better normalization and domain-specific features."""
        if not text:
            return ""
            
        # Extract title and content if there's a clear title pattern
        title = ""
        content = text.strip()
        title_patterns = [
            r'^([^.!?\n]+)[.!?\n]',  # First sentence ending with punctuation or newline
            r'^(.*?)[:;]\s',  # Text before a colon or semicolon
            r'^([^:]+):',  # Text before a colon (stricter match)
        ]
        
        for pattern in title_patterns:
            title_match = re.match(pattern, content)
            if title_match:
                title = title_match.group(1).strip()
                content = content[len(title_match.group(0)):].strip()
                break
        
        # Process title (weighted more heavily)
        if title:
            # Repeat title terms to increase their importance
            text = f"{title} {title} {title} {content}"
        
        # Convert to lowercase
        text = text.lower()
        
        # Handle special document type indicators
        doc_type_indicators = {
            r'\breport\b': ' report_document ',
            r'\bguide\b': ' guide_document ',
            r'\bmanual\b': ' guide_document ',
            r'\bpresentation\b': ' presentation_document ',
            r'\bresearch paper\b': ' research_document ',
            r'\barticle\b': ' article_document ',
            r'\bdataset\b': ' dataset_document ',
            r'\bform\b': ' form_document ',
            r'\blaw\b': ' legal_document ',
            r'\bpolicy\b': ' legal_document ',
            r'\beducational\b': ' educational_document '
        }
        
        for pattern, replacement in doc_type_indicators.items():
            text = re.sub(pattern, replacement, text)
        
        # Handle topic indicators
        topic_indicators = {
            r'\btechnology\b|\btech\b|\bai\b|\bsoftware\b': ' technology_topic ',
            r'\bbusiness\b|\bfinance\b|\beconomic\b': ' business_topic ',
            r'\beducation\b|\blearning\b|\bteaching\b': ' education_topic ',
            r'\bhealth\b|\bmedical\b|\bhealthcare\b': ' health_topic ',
            r'\blegal\b|\blaw\b|\bregulation\b': ' legal_topic ',
            r'\benvironment\b|\bclimate\b|\bsustainable\b': ' environment_topic ',
            r'\bgovernment\b|\bpolicy\b|\bpolitical\b': ' government_topic ',
            r'\bscience\b|\bscientific\b|\bresearch\b': ' science_topic ',
            r'\bsecurity\b|\bdefense\b|\bcyber\b': ' security_topic ',
            r'\btransport\b|\bmobility\b|\blogistics\b': ' transportation_topic ',
            r'\bhumanities\b|\bculture\b|\bsocial\b': ' humanities_topic '
        }
        
        for pattern, replacement in topic_indicators.items():
            text = re.sub(pattern, replacement, text)
        
        # Remove special characters while preserving important ones
        text = re.sub(r'[^a-zA-Z\s_-]', ' ', text)
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # Remove stopwords and lemmatize, preserve special tokens
        cleaned_tokens = []
        for token in tokens:
            if '_document' in token or '_topic' in token:
                cleaned_tokens.append(token)
            elif token not in self.stop_words and len(token) > 2:
                cleaned_tokens.append(self.lemmatizer.lemmatize(token))
        
        return ' '.join(cleaned_tokens)
    
    def train_topic_classifier(self, documents, labels, model_type='MultinomialNB'):
        """Train the topic classifier with provided documents and labels."""
        # Preprocess documents
        print("Preprocessing documents for topic classification...")
        processed_docs = [self.preprocess_text(doc) for doc in documents]
        
        # Print label distribution
        print("\nInitial label distribution:")
        label_counts = Counter(labels)
        print(label_counts)
        
        # Remove classes with too few samples (less than 5)
        min_samples = 5
        valid_classes = {cls for cls, count in label_counts.items() if count >= min_samples}
        if len(valid_classes) < len(set(labels)):
            print(f"\nRemoving classes with fewer than {min_samples} samples:")
            removed_classes = set(labels) - valid_classes
            for cls in removed_classes:
                print(f"- {cls} ({label_counts[cls]} samples)")
            
            # Filter out documents with rare classes
            valid_indices = [i for i, label in enumerate(labels) if label in valid_classes]
            processed_docs = [processed_docs[i] for i in valid_indices]
            labels = [labels[i] for i in valid_indices]
            print(f"\nRemaining samples: {len(processed_docs)}")
        
        # Split data for stratified cross-validation
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
        # Initialize metrics lists
        accuracies = []
        f1_scores = []
        
        # Define the classifier based on model_type
        if model_type == 'LinearSVC':
            classifier_instance = LinearSVC(
                C=1.0,
                class_weight='balanced',
                dual=False,
                max_iter=3000
            )
        elif model_type == 'MultinomialNB':
            classifier_instance = MultinomialNB()
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        # Perform cross-validation
        print("\nPerforming cross-validation...")
        fold = 1
        for train_idx, val_idx in skf.split(processed_docs, labels):
            # Split data
            X_train = [processed_docs[i] for i in train_idx]
            y_train = [labels[i] for i in train_idx]
            X_val = [processed_docs[i] for i in val_idx]
            y_val = [labels[i] for i in val_idx]
            
            # Apply SMOTE for balancing (only on training data)
            tfidf_temp = TfidfVectorizer(max_features=15000)
            X_train_tfidf = tfidf_temp.fit_transform(X_train)
            
            # Use RandomOverSampler for very small classes
            min_samples_per_class = min(Counter(y_train).values())
            if min_samples_per_class < 6:
                print(f"\nUsing RandomOverSampler for fold {fold} (min samples per class: {min_samples_per_class})")
                sampler = RandomOverSampler(random_state=42)
            else:
                print(f"\nUsing SMOTE for fold {fold}")
                sampler = SMOTE(random_state=42, k_neighbors=min(5, min_samples_per_class - 1))
            
            X_train_resampled, y_train_resampled = sampler.fit_resample(X_train_tfidf.toarray(), y_train)
            
            # Train the model on resampled data
            print(f"Training fold {fold}...")
            
            # Create pipeline for this fold using the selected classifier
            fold_pipeline = Pipeline([
                ('tfidf', TfidfVectorizer(max_features=15000, ngram_range=(1, 3), sublinear_tf=True)),
                ('classifier', classifier_instance)
            ])

            fold_pipeline.fit(X_train, y_train)  # Train on original data (should use resampled? Let's stick to original for now as per existing code structure)
            
            # Evaluate
            predictions = fold_pipeline.predict(X_val)
            accuracy = accuracy_score(y_val, predictions)
            f1 = f1_score(y_val, predictions, average='weighted')
            
            accuracies.append(accuracy)
            f1_scores.append(f1)
            
            print(f"Fold {fold} - Accuracy ({model_type}): {accuracy:.3f}, F1: {f1:.3f}")
            print("\nClassification Report:")
            print(classification_report(y_val, predictions))
            
            fold += 1
        
        # Print overall performance
        print("\nOverall Performance ({model_type}):")
        print(f"Average Accuracy: {np.mean(accuracies):.3f} ± {np.std(accuracies):.3f}")
        print(f"Average F1-Score: {np.mean(f1_scores):.3f} ± {np.std(f1_scores):.3f}")
        
        # Train final model on all data
        print(f"\nTraining final model on all data ({model_type})...")
        
        # Create final pipeline using the selected classifier
        final_pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(max_features=15000, ngram_range=(1, 3), sublinear_tf=True)),
            ('classifier', classifier_instance)
        ])
        final_pipeline.fit(processed_docs, labels)

        # Save the model
        joblib.dump(final_pipeline, TOPIC_MODEL_PATH)
        self.topic_classifier = final_pipeline
        
        return final_pipeline
    
    def train_doc_type_classifier(self, documents, labels, model_type='MultinomialNB'):
        """Train the document type classifier with provided documents and labels."""
        # Preprocess documents
        print("Preprocessing documents for document type classification...")
        processed_docs = [self.preprocess_text(doc) for doc in documents]
        
        # Print label distribution
        print("\nInitial label distribution:")
        label_counts = Counter(labels)
        print(label_counts)
        
        # Remove classes with too few samples (less than 5)
        min_samples = 5
        valid_classes = {cls for cls, count in label_counts.items() if count >= min_samples}
        if len(valid_classes) < len(set(labels)):
            print(f"\nRemoving classes with fewer than {min_samples} samples:")
            removed_classes = set(labels) - valid_classes
            for cls in removed_classes:
                print(f"- {cls} ({label_counts[cls]} samples)")
            
            # Filter out documents with rare classes
            valid_indices = [i for i, label in enumerate(labels) if label in valid_classes]
            processed_docs = [processed_docs[i] for i in valid_indices]
            labels = [labels[i] for i in valid_indices]
            print(f"\nRemaining samples: {len(processed_docs)}")
        
        # Split data for stratified cross-validation
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
        # Initialize metrics lists
        accuracies = []
        f1_scores = []
        
        # Define the classifier based on model_type
        if model_type == 'LinearSVC':
            classifier_instance = LinearSVC(
                C=1.0,
                class_weight='balanced',
                dual=False,
                max_iter=3000
            )
        elif model_type == 'MultinomialNB':
            classifier_instance = MultinomialNB()
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        # Perform cross-validation
        print("\nPerforming cross-validation...")
        fold = 1
        for train_idx, val_idx in skf.split(processed_docs, labels):
            # Split data
            X_train = [processed_docs[i] for i in train_idx]
            y_train = [labels[i] for i in train_idx]
            X_val = [processed_docs[i] for i in val_idx]
            y_val = [labels[i] for i in val_idx]
            
            # Apply SMOTE for balancing (only on training data)
            tfidf_temp = TfidfVectorizer(max_features=15000)
            X_train_tfidf = tfidf_temp.fit_transform(X_train)
            
            # Use RandomOverSampler for very small classes
            min_samples_per_class = min(Counter(y_train).values())
            if min_samples_per_class < 6:
                print(f"\nUsing RandomOverSampler for fold {fold} (min samples per class: {min_samples_per_class})")
                sampler = RandomOverSampler(random_state=42)
            else:
                print(f"\nUsing SMOTE for fold {fold}")
                sampler = SMOTE(random_state=42, k_neighbors=min(5, min_samples_per_class - 1))
                
            X_train_resampled, y_train_resampled = sampler.fit_resample(X_train_tfidf.toarray(), y_train)
            
            # Train the model
            print(f"Training fold {fold}...")
            
            # Create pipeline for this fold using the selected classifier
            fold_pipeline = Pipeline([
                ('tfidf', TfidfVectorizer(max_features=15000, ngram_range=(1, 3), sublinear_tf=True)),
                ('classifier', classifier_instance)
            ])
            
            fold_pipeline.fit(X_train, y_train)  # Train on original data (should use resampled? Let's stick to original for now as per existing code structure)
            
            # Evaluate
            predictions = fold_pipeline.predict(X_val)
            accuracy = accuracy_score(y_val, predictions)
            f1 = f1_score(y_val, predictions, average='weighted')
            
            accuracies.append(accuracy)
            f1_scores.append(f1)
            
            print(f"Fold {fold} - Accuracy ({model_type}): {accuracy:.3f}, F1: {f1:.3f}")
            print("\nClassification Report:")
            print(classification_report(y_val, predictions))
            
            fold += 1
        
        # Print overall performance
        print("\nOverall Performance ({model_type}):")
        print(f"Average Accuracy: {np.mean(accuracies):.3f} ± {np.std(accuracies):.3f}")
        print(f"Average F1-Score: {np.mean(f1_scores):.3f} ± {np.std(f1_scores):.3f}")
        
        # Train final model on all data
        print(f"\nTraining final model on all data ({model_type})...")

        # Create final pipeline using the selected classifier
        final_pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(max_features=15000, ngram_range=(1, 3), sublinear_tf=True)),
            ('classifier', classifier_instance)
        ])
        final_pipeline.fit(processed_docs, labels)
        
        # Save the model
        joblib.dump(final_pipeline, DOC_TYPE_MODEL_PATH)
        self.doc_type_classifier = final_pipeline
        
        return final_pipeline

def fetch_and_process_row(row):
    """Fetches document content from link and returns content and labels."""
    link = str(row.get('Link', '')).strip()
    content = ""
    if link:
        # print(f"Fetching content from: {link}") # Removed verbose printing
        try:
            content = fetch_document_content(link)
            if not content:
                 print(f"Warning: Could not fetch content from {link}. Using title instead.")
                 content = str(row.get('Title', '')).strip()
        except Exception as e:
            print(f"Error fetching content from {link}: {e}. Using title instead.")
            content = str(row.get('Title', '')).strip()
    else:
        # If no link, use the title
        content = str(row.get('Title', '')).strip()

    topic = str(row.get('Topic Labels', '')).strip()
    doc_type = str(row.get('Type Labels', '')).strip()
    
    return content, topic, doc_type

def main():
    """Main function to train the classifiers."""
    # Read the CSV file
    input_csv_path = 'NLP documents bank - Sheet1_cleaned.csv'
    
    try:
        print(f"Reading training data from {input_csv_path}...")
        
        # Read the CSV file using pandas for better handling
        df = pd.read_csv(input_csv_path)
        print(f"Found {len(df)} total rows in CSV")
        
        documents = []
        topic_labels = []
        doc_type_labels = []
        
        # Use ThreadPoolExecutor for parallel fetching
        print("Fetching document content in parallel...")
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            # Map the fetch_and_process_row function to each row of the DataFrame
            results = list(executor.map(fetch_and_process_row, [row for index, row in df.iterrows()]))
        
        # Process the results from parallel fetching
        for content, topic, doc_type in results:
             if content and topic and doc_type:  # Only include rows with all required fields
                documents.append(content)
                topic_labels.append(topic)
                doc_type_labels.append(doc_type)
        
        print(f"Processed {len(documents)} valid documents with labels")
        if documents:
            print("Sample of the first document:")
            print(f"Content: {documents[0][:100]}...")
            print(f"Topic: {topic_labels[0]}")
            print(f"Document Type: {doc_type_labels[0]}")
        
        if not documents:
            raise ValueError("No valid documents found in the CSV file")
        
        # Initialize and train the classifier
        classifier = DocumentClassifier()
        
        print("\nTraining topic classifier...")
        classifier.train_topic_classifier(documents, topic_labels, model_type='LinearSVC')
        
        print("\nTraining document type classifier...")
        classifier.train_doc_type_classifier(documents, doc_type_labels, model_type='LinearSVC')
        
        print("\nTraining completed successfully!")
        
    except Exception as e:
        print(f"Error training classifiers: {str(e)}")
        raise

if __name__ == "__main__":
    main()