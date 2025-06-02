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
from sklearn.model_selection import train_test_split
import joblib
from nltk.stem import WordNetLemmatizer
from sklearn.pipeline import Pipeline
import concurrent.futures # Import concurrent.futures

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

        # Split data into training and testing sets
        docs_train, docs_test, labels_train, labels_test = train_test_split(
            processed_docs, labels, test_size=0.2, random_state=42
        )

        # Create a pipeline with TF-IDF vectorizer and Linear SVC classifier
        topic_classifier_pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(max_features=15000, ngram_range=(1, 4), min_df=2)),
            ('clf', LinearSVC(C=0.1, random_state=42, class_weight='balanced', loss='squared_hinge'))
        ])
        
        # Train the model
        topic_classifier_pipeline.fit(docs_train, labels_train)
        
        # Evaluate the model (optional, but good for debugging)
        accuracy = topic_classifier_pipeline.score(docs_test, labels_test)
        print(f"Topic classifier test accuracy: {accuracy:.2f}")
        
        # Save the model
        joblib.dump(topic_classifier_pipeline, TOPIC_MODEL_PATH)
        self.topic_classifier = topic_classifier_pipeline
        
        return topic_classifier_pipeline
    
    def train_doc_type_classifier(self, documents, labels):
        """Train the document type classifier with provided documents and labels."""
        # Preprocess documents
        processed_docs = [self.preprocess_text(doc) for doc in documents]

        # Split data into training and testing sets
        docs_train, docs_test, labels_train, labels_test = train_test_split(
            processed_docs, labels, test_size=0.2, random_state=42
        )

        # Create a pipeline with TF-IDF vectorizer and Linear SVC classifier
        document_type_classifier_pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(max_features=7000, ngram_range=(1, 3), min_df=2)),
            ('clf', LinearSVC(C=0.5, random_state=42, class_weight='balanced')) # Add class_weight='balanced' here
        ])
        
        # Train the model
        document_type_classifier_pipeline.fit(docs_train, labels_train)
        
        # Evaluate the model (optional, but good for debugging)
        accuracy = document_type_classifier_pipeline.score(docs_test, labels_test)
        print(f"Document type classifier test accuracy: {accuracy:.2f}")
        
        # Save the model
        joblib.dump(document_type_classifier_pipeline, DOC_TYPE_MODEL_PATH)
        self.doc_type_classifier = document_type_classifier_pipeline
        
        return document_type_classifier_pipeline
    
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
            if isinstance(self.topic_classifier.named_steps['clf'], LinearSVC):
                # Use decision_function for LinearSVC
                scores = self.topic_classifier.decision_function([processed_text])[0]
                # Get the index of the predicted class
                pred_idx = self.topic_classifier.classes_.tolist().index(prediction)
                # Confidence is the absolute value of the score for the predicted class
                confidence = float(abs(scores[pred_idx]))
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
            if isinstance(self.doc_type_classifier.named_steps['clf'], LinearSVC):
                # Use decision_function for LinearSVC
                scores = self.doc_type_classifier.decision_function([processed_text])[0]
                # Get the index of the predicted class
                pred_idx = self.doc_type_classifier.classes_.tolist().index(prediction)
                # Confidence is the absolute value of the score for the predicted class
                confidence = float(abs(scores[pred_idx]))
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

# Create a single instance to be used by views
classifier = DocumentClassifier()

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
            # Ensure response body is closed after reading chunk
            try:
                html_content = response.iter_content(chunk_size=chunk_size).__next__().decode('utf-8', errors='ignore')
            finally:
                response.close()

            soup = BeautifulSoup(html_content, 'lxml')
            
            # More comprehensive HTML text extraction
            text = ''
            for element in soup.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'li', 'article', 'main']):
                 text += element.get_text() + '\n' # Add newline for better separation

            # If still not much text, try body or other common containers
            if len(text.strip()) < 100: # Arbitrary threshold, adjust if needed
                 body_text = soup.body.get_text() if soup.body else ''
                 # Simple cleaning for fallback text
                 body_text = re.sub(r'\s+', ' ', body_text).strip()
                 # Prioritize more structured text if available
                 text = text if len(text.strip()) > len(body_text) else body_text
            
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

def clean_csv_data(csv_path):
    """Extract training data from the CSV file, fetching content concurrently."""
    documents_to_fetch = []
    documents_no_url = []
    
    # Read the CSV and filter documents first
    with open(csv_path, 'r', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            title = row.get('Title', '')
            url = row.get('Link')
            doc_id = row.get('Document ID', 'N/A')

            # Split topic and document type labels and filter out empty ones
            topic_label = row.get('Topic Labels', '').strip()
            doc_type_labels = [label.strip() for label in row.get('Type Labels', '').split(',') if label.strip()]

            # We should only skip rows with absolutely no content (neither URL nor title) AND no labels
            if not (url or title) and not (topic_label or doc_type_labels):
                 print(f"Skipping row {doc_id} due to missing link, title, and labels in cleaned CSV.")
                 continue

            if url:
                documents_to_fetch.append(row)
            elif title:
                # Documents with no URL but a title are processed immediately with the title
                documents_no_url.append(row)
            else:
                # This case should ideally be caught by the check above, but as a safeguard
                print(f"Warning: Document {doc_id} has no link or title, but has labels. Processing with empty content for now.")
                # Process with empty content, labels will still be added if present
                documents_no_url.append(row) # Treat as no-url case, will use empty content

    training_data = []

    # Process documents with no URL (using title as content or empty if no title)
    for row in documents_no_url:
         content = row.get('Title', '').strip()
         topic_label = row.get('Topic Labels', '').strip() # Single topic label
         doc_type_labels = [label.strip() for label in row.get('Type Labels', '').split(',') if label.strip()] # List of types

         # Create training entries based on available labels and content
         # Ensure we only add if at least one label is present
         if topic_label or doc_type_labels:
             # If there's a topic label, create entries for each document type with this topic
             if topic_label:
                 if doc_type_labels:
                     for doc_type in doc_type_labels:
                         training_data.append({
                             'content': content,
                             'topic_label': topic_label,
                             'document_type_label': doc_type
                         })
                 else:
                      # If no document types but has topic, add with empty doc type
                      training_data.append({
                          'content': content,
                          'topic_label': topic_label,
                          'document_type_label': '' # Empty doc type label
                      })

             # If no topic label but has document types, create entries with empty topic
             elif doc_type_labels:
                  for doc_type in doc_type_labels:
                      training_data.append({
                          'content': content,
                          'topic_label': '', # Empty topic label
                          'document_type_label': doc_type
                      })

             print(f"Processed document {row.get('Document ID', 'N/A')}.")

    # Fetch content for documents with URLs concurrently
    if documents_to_fetch:
        print(f"Fetching content concurrently for {len(documents_to_fetch)} documents...")
        # Use a thread pool executor
        # Adjust max_workers based on your system and network capabilities
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            # Submit fetching tasks and map futures to original rows
            future_to_row = {executor.submit(fetch_document_content, doc['Link']): doc for doc in documents_to_fetch}

            for future in concurrent.futures.as_completed(future_to_row):
                row = future_to_row[future]
                doc_id = row.get('Document ID', 'N/A')
                url = row.get('Link')

                try:
                    content = future.result()
                    print(f"Finished fetching content for {doc_id}.")

                    # If content fetching failed or returned empty, use the title as content
                    if not content:
                         title = row.get('Title', '')
                         if title:
                             content = title
                             print(f"Using title as content for {doc_id} due to empty fetched content.")
                         else:
                             print(f"Warning: Document {doc_id} has empty fetched content and missing title.")
                             content = '' # Use empty content if no title fallback

                    # Get cleaned labels from the row
                    topic_label = row.get('Topic Labels', '').strip() # Single topic label
                    doc_type_labels = [label.strip() for label in row.get('Type Labels', '').split(',') if label.strip()] # List of types

                    # Create training entries based on available labels and fetched content
                    # Ensure we only add if at least one label is present
                    if topic_label or doc_type_labels:
                        # If there's a topic label, create entries for each document type with this topic
                        if topic_label:
                            if doc_type_labels:
                                for doc_type in doc_type_labels:
                                    training_data.append({
                                        'content': content,
                                        'topic_label': topic_label,
                                        'document_type_label': doc_type
                                    })

                            else:
                                 # If no document types but has topic, add with empty doc type
                                  training_data.append({
                                      'content': content,
                                      'topic_label': topic_label,
                                      'document_type_label': '' # Empty doc type label
                                  })

                        # If no topic label but has document types, create entries with empty topic
                        elif doc_type_labels:
                             for doc_type in doc_type_labels:
                                  training_data.append({
                                      'content': content,
                                      'topic_label': '', # Empty topic label
                                      'document_type_label': doc_type
                                  })

                        print(f"Processed document {doc_id} with title fallback.")
                    else:
                         print(f"Skipping document {doc_id} due to fetch exception and missing title.")
                         continue # Skip if fetch failed and no title to fallback on

                except Exception as exc:
                    print(f'Error fetching or processing document {doc_id} ({url}): {exc}')
                    # Fallback to using title if fetching failed due to an exception
                    title = row.get('Title', '')
                    if title:
                         content = title
                         print(f"Using title as content for {doc_id} due to fetch exception.")

                         # Get cleaned labels from the row
                         topic_label = row.get('Topic Labels', '').strip() # Single topic label
                         doc_type_labels = [label.strip() for label in row.get('Type Labels', '').split(',') if label.strip()] # List of types

                         # Create training entries based on available labels and title content
                         # Ensure we only add if at least one label is present
                         if topic_label or doc_type_labels:
                              # If there's a topic label, create entries for each document type with this topic
                             if topic_label:
                                 if doc_type_labels:
                                     for doc_type in doc_type_labels:
                                         training_data.append({
                                             'content': content,
                                             'topic_label': topic_label,
                                             'document_type_label': doc_type
                                         })
                                 else:
                                      # If no document types but has topic, add with empty doc type
                                       training_data.append({
                                           'content': content,
                                           'topic_label': topic_label,
                                           'document_type_label': '' # Empty doc type label
                                       })

                              # If no topic label but has document types, create entries with empty topic
                             elif doc_type_labels:
                                for doc_type in doc_type_labels:
                                    training_data.append({
                                        'content': content,
                                        'topic_label': '', # Empty topic label
                                        'document_type_label': doc_type
                                    })
    
                         print(f"Processed document {doc_id} with title fallback.")
                    else:
                         print(f"Skipping document {doc_id} due to fetch exception and missing title.")
                         continue # Skip if fetch failed and no title to fallback on
    
    # --- Basic Oversampling for Topic Labels ---
    # Count occurrences of each topic label
    topic_counts = {}
    for item in training_data:
        topic_counts[item['topic_label']] = topic_counts.get(item['topic_label'], 0) + 1

    # Identify minority topics and determine how many duplicates are needed
    oversample_target = 40 # Target number of documents for minority classes
    minority_topics_to_duplicate = {}
    for label, count in topic_counts.items():
        # Define minority threshold (e.g., less than 10% of the average count, or a fixed number)
        # Using a fixed threshold of 10 for simplicity here.
        if count < 10 and count > 0: # Only oversample if count is between 1 and 9
             minority_topics_to_duplicate[label] = oversample_target - count

    # Duplicate entries for minority topics
    additional_training_data = []
    for item in training_data:
        label = item['topic_label']
        if label in minority_topics_to_duplicate:
            num_duplicates = minority_topics_to_duplicate[label]
            for _ in range(num_duplicates):
                additional_training_data.append(item) # Append the original item
            # Remove the label from the dict after duplicating to avoid re-duplicating
            del minority_topics_to_duplicate[label]

    # Add the duplicated entries to the training data
    training_data.extend(additional_training_data)
    # --- End Oversampling Logic ---
    
    return training_data

def import_training_data(training_data):
    """Import training data into the database."""
    # First, clear existing training data
    TrainingData.objects.all().delete()
    
    # Add new training data
    count = 0
    for item in training_data:
        TrainingData.objects.create(
            content=item['content'],
            topic_label=item['topic_label'],
            document_type_label=item['document_type_label']
        )
        count += 1
    
    return count

def train_models():
    """Train both classifiers using the training data."""
    training_data = TrainingData.objects.all()
    
    if not training_data:
        print("No training data available.")
        return False
    
    # Extract content and labels
    documents = [td.content for td in training_data]
    topic_labels = [td.topic_label for td in training_data]
    doc_type_labels = [td.document_type_label for td in training_data]
    
    # Train both classifiers
    print("Training topic classifier...")
    classifier.train_topic_classifier(documents, topic_labels)
    
    print("Training document type classifier...")
    classifier.train_doc_type_classifier(documents, doc_type_labels)
    
    return True

def main():
    # Path to the CSV file
    csv_path = 'NLP documents bank - Sheet1_cleaned.csv'
    
    # Check if the file exists
    if not os.path.exists(csv_path):
        print(f"Error: File {csv_path} not found.")
        sys.exit(1)
    
    print(f"Reading training data from {csv_path}...")
    training_data = clean_csv_data(csv_path)
    
    print(f"Extracted {len(training_data)} training examples.")
    
    print("Importing training data into database...")
    imported_count = import_training_data(training_data)
    
    print(f"Imported {imported_count} training examples.")
    
    # Print unique topic and document type labels in the training data
    unique_topic_labels = sorted(list(set([item['topic_label'] for item in training_data])))
    unique_doctype_labels = sorted(list(set([item['document_type_label'] for item in training_data])))
    print("\nUnique Topic Labels in Training Data:")
    print(unique_topic_labels)
    print("\nUnique Document Type Labels in Training Data:")
    print(unique_doctype_labels)

    # Print distribution of topic labels
    topic_label_counts = {}
    for item in training_data:
        label = item['topic_label']
        topic_label_counts[label] = topic_label_counts.get(label, 0) + 1

    print("\nTopic Label Distribution in Training Data:")
    for label, count in sorted(topic_label_counts.items()):
        print(f"  {label}: {count}")

    print("\nTraining models...")
    success = train_models()
    
    if success:
        print("Models trained successfully!")
    else:
        print("Failed to train models.")

if __name__ == "__main__":
    main() 