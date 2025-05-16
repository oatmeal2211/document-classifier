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
        # Finance/Economics document
        "Analysis of market trends and economic indicators for Q3 2023 shows steady growth in emerging markets. The report examines inflation rates and GDP growth across different regions.",
        
        # Technology/Education document (like a tutorial)
        "A comprehensive guide to natural language processing with Python using spaCy. Learn how to tokenize text, perform entity recognition, and build custom pipelines for text analysis.",
        
        # Sports news
        "China ends Malaysia's Sudirman Cup hopes in quarter-finals. The national badminton team fought hard but couldn't overcome the defending champions.",
        
        # Health research document
        "Machine Learning applications in medical diagnosis: A comparative study of ML techniques for early detection of osteoporosis and risk assessment in patients.",
        
        # Legal document (court case)
        "He-Con Sdn Bhd v Bulyah bt Ishak & Anor: Court ruling on land dispute case establishes new precedent for property rights in Malaysia.",
        
        # University application form
        "University of Malaysia Application Form for Undergraduate Studies. Please complete all sections to apply for admission to the university program.",
        
        # Tourism brochure
        "Discover the beautiful islands of Langkawi, the Jewel of Kedah. Explore pristine beaches, lush rainforests, and experience the rich cultural heritage.",
        
        # Job application form
        "Malaysia Airlines Job Application Form. Please fill in your personal details, qualifications, and work experience to apply for available positions."
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