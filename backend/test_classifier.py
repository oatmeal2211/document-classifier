#!/usr/bin/env python
"""
Script to test the trained document classifier.
"""
import os
import django
import csv

# Set up Django environment
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'backend.settings')
django.setup()

from api.classifier import classifier

def test_classifier_new_cases():
    """Test the classifier with new, hardcoded test cases."""

    new_test_cases = [
        {
            "content": "Recent breakthroughs in compact fusion reactor technology could revolutionize energy production.",
            "expected_topic": "Technology",
            "expected_types": ["Research/Academic", "Article/News"],
        },
        {
            "content": "Summary of the new corporate tax regulations effective from next fiscal year.",
            "expected_topic": "Politics/Law",
            "expected_types": ["Legal/Policy", "Article/News"],
        },
        {
            "content": "A delicious and easy-to-follow recipe for a classic vegan lasagna.",
            "expected_topic": "Lifestyle/Culture",
            "expected_types": ["Recipe"],
        },
        {
            "content": "The impact of prolonged social media use on adolescent mental health: A research perspective.",
            "expected_topic": "Health",
            "expected_types": ["Article/News", "Research/Academic"],
        },
        {
            "content": "Press release announcing a strategic partnership between innovaTech and quantiCorp.",
            "expected_topic": "Business/Finance",
            "expected_types": ["Article/News"],
        },
        {
            "content": "A guide to implementing robust security measures for your cloud infrastructure.",
            "expected_topic": "Technology",
            "expected_types": ["Guide/Manual"],
        },
        # Added new test cases for better coverage
        {
            "content": "Analyzing the Q3 earnings report for major tech companies in Silicon Valley.",
            "expected_topic": "Business/Finance",
            "expected_types": ["Report/Study"],
        },
        {
            "content": "The history and evolution of parliamentary systems in Western democracies.",
            "expected_topic": "Politics/Law",
            "expected_types": ["Research/Academic"],
        },
        {
            "content": "Latest research findings on the effectiveness of cognitive behavioral therapy for anxiety.",
            "expected_topic": "Health",
            "expected_types": ["Research/Academic", "Article/News"],
        },
        {
            "content": "A comprehensive manual for setting up and configuring your new wireless router.",
            "expected_topic": "Technology",
            "expected_types": ["Guide/Manual"],
        },
         {
            "content": "New study reveals significant decline in polar ice caps due to global warming.",
            "expected_topic": "Environment/Sustainability",
            "expected_types": ["Report/Study", "Article/News"],
        },
        {
            "content": "Exploring traditional cuisine and dining etiquette in Japan.",
            "expected_topic": "Lifestyle/Culture",
            "expected_types": ["Article/News"],
        },
        {
            "content": "Match analysis and highlights from yesterday's football game between Manchester United and Liverpool.",
            "expected_topic": "Sports",
            "expected_types": ["Article/News"],
        },
        {
            "content": "A guide for tourists visiting the historical sites in Rome, Italy.",
            "expected_topic": "Tourism",
            "expected_types": ["Guide/Manual"],
        },
        {
            "content": "Report on the feasibility of implementing high-speed rail across the country.",
            "expected_topic": "Transportation",
            "expected_types": ["Report/Study"],
        },
        {
            "content": "Discussion on current foreign policy challenges facing the European Union.",
            "expected_topic": "Foreign Policy",
            "expected_types": ["Report/Study", "Article/News"],
        },
        {
            "content": "An overview of the basic principles of quantum mechanics for beginners.",
            "expected_topic": "Research/Academia",
            "expected_types": ["Informational Text", "Educational Resource"], # Mapping 'Educational Resource' to Education
        },
        {
            "content": "Best practices for securing enterprise networks against cyber threats.",
            "expected_topic": "Security",
            "expected_types": ["Guide/Manual"],
        },
        {
            "content": "The latest developments in space exploration and potential human missions to Mars.",
            "expected_topic": "Space",
            "expected_types": ["Article/News"],
        },
        {
            "content": "Filling out the application form for a postgraduate program at a UK university.",
            "expected_topic": "Education", # Could also be General
            "expected_types": ["Application Form"],
        },
        {
            "content": "Summary of key legal arguments in the recent Supreme Court ruling on privacy rights.",
            "expected_topic": "Legal/Policy",
            "expected_types": ["Legal/Policy", "Report/Study"],
        },
        {
            "content": "A brochure detailing the services offered by a financial consulting firm.",
            "expected_topic": "Business/Finance",
            "expected_types": ["Brochure"],
        },
        {
            "content": "Presentation slides from a conference on the future of work and automation.",
            "expected_topic": "Business/Finance",
            "expected_types": ["Presentation"],
        },
        {
            "content": "A questionnaire designed to collect feedback on customer satisfaction with a new product.",
            "expected_topic": "Business/Finance", # Could be General
            "expected_types": ["Questionnaire"],
        },
         {
            "content": "An informational text explaining the process of photosynthesis for high school students.",
            "expected_topic": "Education",
            "expected_types": ["Informational Text", "Educational Resource"], # Mapping 'Educational Resource' to Education
        },
         {
            "content": "A review of the latest smartphone model released by major tech company.",
            "expected_topic": "Technology",
            "expected_types": ["Review"],
        },
         {
            "content": "Guidelines for conducting ethical research involving human subjects.",
            "expected_topic": "Research/Academia",
            "expected_types": ["Guide/Manual", "Legal/Policy"], # Could be Policy too
        },
    ]

    print(f"\n{'='*60}")
    print(f"Testing classifier with {len(new_test_cases)} new hardcoded test cases")
    print(f"{'='*60}")

    print("\nTesting Topic Classification (New Cases):")
    print("-" * 50)
    correct_topic_predictions = 0
    total_topic_tests = 0

    for i, test_case in enumerate(new_test_cases):
        document_content = test_case['content']
        expected_topic = test_case['expected_topic']

        if not expected_topic:
            continue

        total_topic_tests += 1

        print(f"\nDocument {i+1}: {document_content[:70]}...")
        result = classifier.classify_document(document_content, classification_type='topic')

        if 'error' in result:
            print(f"  Error: {result['error']}")
        else:
            prediction = result['result']
            confidence = result['confidence']
            print(f"  Predicted Topic: {prediction} (Confidence: {confidence*100:.1f}%)")
            print(f"  Expected Topic: {expected_topic}")

            if prediction == expected_topic:
                print("  Result: CORRECT")
                correct_topic_predictions += 1
            else:
                print("  Result: INCORRECT")

    topic_accuracy = (correct_topic_predictions / total_topic_tests) * 100 if total_topic_tests > 0 else 0
    print("\n" + "-" * 50)
    print(f"Topic Classification Accuracy (New Cases): {topic_accuracy:.1f}% (Tested on {total_topic_tests} documents)")

    print("\nTesting Document Type Classification (New Cases):")
    print("-" * 50)
    correct_doctype_predictions = 0
    total_doctype_tests = 0

    for i, test_case in enumerate(new_test_cases):
        document_content = test_case['content']
        expected_doctypes = test_case['expected_types']

        if not expected_doctypes:
            continue

        total_doctype_tests += 1

        print(f"\nDocument {i+1}: {document_content[:70]}...")
        result = classifier.classify_document(document_content, classification_type='document_type')

        if 'error' in result:
            print(f"  Error: {result['error']}")
        else:
            prediction = result['result']
            confidence = result['confidence']
            print(f"  Predicted Type: {prediction} (Confidence: {confidence*100:.1f}%)")
            print(f"  Expected Types: {', '.join(expected_doctypes)}")

            if prediction in expected_doctypes:
                print("  Result: CORRECT")
                correct_doctype_predictions += 1
            else:
                print("  Result: INCORRECT")

    doctype_accuracy = (correct_doctype_predictions / total_doctype_tests) * 100 if total_doctype_tests > 0 else 0
    print("\n" + "-" * 50)
    print(f"Document Type Classification Accuracy (New Cases): {doctype_accuracy:.1f}% (Tested on {total_doctype_tests} documents)")


if __name__ == "__main__":
    # Only run tests on the new, hardcoded test cases
    test_classifier_new_cases()