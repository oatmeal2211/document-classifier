#!/usr/bin/env python
"""
Script to train the document classifier using the NLP document bank CSV file.
"""
import os
import csv
import django
import sys

# Set up Django environment
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'backend.settings')
django.setup()

from api.models import TrainingData
from api.classifier import classifier

def clean_csv_data(csv_path):
    """Extract training data from the CSV file."""
    training_data = []
    
    with open(csv_path, 'r', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            # Use title as content for training purposes
            # In a real scenario, you might want to fetch the actual content from the URLs
            content = row['Title']
            
            # Split topic labels by comma if there are multiple
            topic_labels = [label.strip() for label in row['Topic Labels'].split(',')]
            
            # Split document type labels by comma if there are multiple
            doc_type_labels = [label.strip() for label in row['Type Labels'].split(',')]
            
            # For each topic and document type combination, create a training entry
            for topic in topic_labels:
                for doc_type in doc_type_labels:
                    training_data.append({
                        'content': content,
                        'topic_label': topic,
                        'document_type_label': doc_type
                    })
    
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
    csv_path = 'NLP documents bank - Sheet1.csv'
    
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
    
    print("Training models...")
    success = train_models()
    
    if success:
        print("Models trained successfully!")
    else:
        print("Failed to train models.")

if __name__ == "__main__":
    main() 