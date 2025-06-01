#!/usr/bin/env python
"""
Script to test the trained document classifier.
"""
import os
import django
import sys

# Add the parent directory of 'backend' to sys.path so 'api' is importable
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Set up Django environment
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'backend.settings')
try:
    django.setup()
except Exception as e:
    print(f"Django setup warning: {e}")
    print("Continuing without Django setup...")

# Import classifier (no fallback needed)
try:
    from api.classifier import classify_document
except ImportError as e:
    print(f"Failed to import classifier: {e}")
    sys.exit(1)

def clean_text(text):
    """Basic text cleaning for better model input"""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = " ".join(text.split())
    return text

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
    
    print("="*70)
    print("TESTING DOCUMENT CLASSIFIER")
    print("="*70)
    
    print(f"Testing with {len(test_documents)} sample documents...")
    print()
    
    print("TOPIC CLASSIFICATION RESULTS:")
    print("-" * 50)
    
    topic_results = []
    for i, doc in enumerate(test_documents):
        print(f"\nDocument {i+1}:")
        print(f"Text: {doc[:80]}...")
        cleaned_doc = clean_text(doc)
        result = classify_document(cleaned_doc, classification_type='topic')
        
        if 'error' in result:
            print(f"  ❌ Error: {result['error']}")
            topic_results.append(None)
        else:
            print(f"  📂 Topic: {result['result']}")
            print(f"  📊 Confidence: {result['confidence']*100:.1f}%")
            topic_results.append(result)
    
    print("\n" + "="*70)
    print("DOCUMENT TYPE CLASSIFICATION RESULTS:")
    print("-" * 50)
    
    doctype_results = []
    for i, doc in enumerate(test_documents):
        print(f"\nDocument {i+1}:")
        print(f"Text: {doc[:80]}...")
        cleaned_doc = clean_text(doc)
        result = classify_document(cleaned_doc, classification_type='document_type')
        
        if 'error' in result:
            print(f"  ❌ Error: {result['error']}")
            doctype_results.append(None)
        else:
            print(f"  📄 Document Type: {result['result']}")
            print(f"  📊 Confidence: {result['confidence']*100:.1f}%")
            doctype_results.append(result)
    
    # Summary
    print("\n" + "="*70)
    print("CLASSIFICATION SUMMARY")
    print("="*70)
    
    successful_topic = sum(1 for r in topic_results if r is not None)
    successful_doctype = sum(1 for r in doctype_results if r is not None)
    
    print(f"Topic Classification Success Rate: {successful_topic}/{len(test_documents)} ({successful_topic/len(test_documents)*100:.1f}%)")
    print(f"Document Type Classification Success Rate: {successful_doctype}/{len(test_documents)} ({successful_doctype/len(test_documents)*100:.1f}%)")
    
    if successful_topic > 0:
        avg_topic_confidence = sum(r['confidence'] for r in topic_results if r is not None) / successful_topic
        print(f"Average Topic Classification Confidence: {avg_topic_confidence*100:.1f}%")
    
    if successful_doctype > 0:
        avg_doctype_confidence = sum(r['confidence'] for r in doctype_results if r is not None) / successful_doctype
        print(f"Average Document Type Classification Confidence: {avg_doctype_confidence*100:.1f}%")
    
    # Detailed results table
    print("\n" + "="*70)
    print("DETAILED RESULTS TABLE")
    print("="*70)
    print(f"{'Doc':<3} {'Topic':<20} {'Conf%':<6} {'Doc Type':<20} {'Conf%':<6}")
    print("-" * 70)
    
    for i in range(len(test_documents)):
        topic_label = topic_results[i]['result'][:18] if topic_results[i] else "ERROR"
        topic_conf = f"{topic_results[i]['confidence']*100:.1f}" if topic_results[i] else "N/A"
        
        doctype_label = doctype_results[i]['result'][:18] if doctype_results[i] else "ERROR"
        doctype_conf = f"{doctype_results[i]['confidence']*100:.1f}" if doctype_results[i] else "N/A"
        
        print(f"{i+1:<3} {topic_label:<20} {topic_conf:<6} {doctype_label:<20} {doctype_conf:<6}")

def check_models_exist():
    """Check if trained models exist"""
    topic_model_path = os.path.join("media", "models", "topic_classifier")
    doctype_model_path = os.path.join("media", "models", "doctype_classifier")
    
    topic_exists = os.path.exists(topic_model_path) and os.path.exists(os.path.join(topic_model_path, "config.json"))
    doctype_exists = os.path.exists(doctype_model_path) and os.path.exists(os.path.join(doctype_model_path, "config.json"))
    
    print("MODEL STATUS CHECK:")
    print("-" * 30)
    print(f"Topic Classifier: {'✅ Found' if topic_exists else '❌ Missing'} at {topic_model_path}")
    print(f"Document Type Classifier: {'✅ Found' if doctype_exists else '❌ Missing'} at {doctype_model_path}")
    
    if not topic_exists or not doctype_exists:
        print("\n⚠️  Some models are missing. Please run train_model.py first to train the models.")
        return False
    
    return True

if __name__ == "__main__":
    print("Document Classifier Test Suite")
    print("=" * 70)
    
    # Check if models exist
    if check_models_exist():
        print("\n✅ All models found. Starting tests...\n")
        test_classifier()
    else:
        print("\n❌ Cannot run tests without trained models.")
        print("Please run: python train_model.py")
        sys.exit(1)