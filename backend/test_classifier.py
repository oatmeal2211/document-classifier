#!/usr/bin/env python
"""
Script to test the trained document classifier.
"""
import os
import django

# Set up Django environment
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'backend.settings')
django.setup()

from api.classifier import classifier

def test_classifier():
    """Test the classifier with sample texts."""
    
    test_documents = [
    # Modified Existing Documents
    "Quarterly financial analysis report: Review of Q3 2023 market performance, focusing on technology sector investments and projected growth based on recent economic indicators.", # Modified Finance
    "Hands-on tutorial: Implementing a basic sentiment analysis model using NLTK and scikit-learn in Python for classifying movie reviews.", # Modified Tech/Education
    "Latest research on Alzheimer's disease: Exploring potential biomarkers and early diagnostic methods through neuroimaging and genetic studies.", # Modified Health
    "Press Release: Landmark court decision on property rights sets new precedent for land ownership disputes involving historical claims.", # Modified Legal
    "Higher Education Admission Application: Please provide your academic history, personal statement, and contact information to apply.", # Modified University Application
    "Employment Application Form: Complete all sections including work experience, educational background, and references for consideration.", # Modified Job Application
    "Explore the enchanting landscapes of Kyoto, Japan. Visit ancient temples, serene gardens, and vibrant markets in this cultural heartland.", # Modified Tourism

    # New Documents
    "A new report highlights the impact of deforestation on climate change and proposes sustainable forestry practices to mitigate environmental damage.", # Environmental Conservation
    "Classic Chocolate Chip Cookie Recipe: Ingredients and step-by-step instructions for baking perfect chewy cookies.", # Recipe
    "Product Review: The new XYZ noise-canceling headphones offer superb audio quality and comfort, though the battery life could be improved.", # Product Review
    "Travel Guide Snippet: Best places to eat street food in Bangkok – a culinary journey through local markets and hidden gems.", # Travel Guide
    "Chapter One: The old house stood silhouetted against the stormy sky, its windows like vacant eyes staring out over the choppy sea. A perfect place for secrets to hide.", # Creative Writing
    ]
    
    print("Testing Topic Classification:")
    print("-" * 50)
    for i, doc in enumerate(test_documents):
        print(f"Document {i+1}: {doc[:50]}...")
        result = classifier.classify_document(doc, classification_type='topic')
        
        if 'error' in result:
            print(f"  Error: {result['error']}")
        else:
            print(f"  Topic: {result['result']} (Confidence: {result['confidence']*100:.1f}%)")
    
    print("\nTesting Document Type Classification:")
    print("-" * 50)
    for i, doc in enumerate(test_documents):
        print(f"Document {i+1}: {doc[:50]}...")
        result = classifier.classify_document(doc, classification_type='document_type')
        
        if 'error' in result:
            print(f"  Error: {result['error']}")
        else:
            print(f"  Document Type: {result['result']} (Confidence: {result['confidence']*100:.1f}%)")

if __name__ == "__main__":
    test_classifier() 