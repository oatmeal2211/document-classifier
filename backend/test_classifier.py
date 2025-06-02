#!/usr/bin/env python
"""
Script to test the trained document classifier.
"""
import os
import django
import csv
from clean_labels import PRIMARY_TOPICS, PRIMARY_DOC_TYPES

# Set up Django environment
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'backend.settings')
django.setup()

# Import the classifier after setting up the Django environment
from api.classifier import classifier

# Define test cases at module level
test_cases = [
    {
        "content": "Recent breakthroughs in compact fusion reactor technology could revolutionize energy production.",
        "expected_topic": "Technology",
        "expected_type": "Research",
    },
    {
        "content": "Summary of the new corporate tax regulations effective from next fiscal year.",
        "expected_topic": "Legal/Policy",
        "expected_type": "Legal",
    },
    {
        "content": "A comprehensive review of sustainable tourism practices in coastal regions.",
        "expected_topic": "Humanities",
        "expected_type": "Research",
    },
    {
        "content": "The impact of social media use on adolescent mental health: A research perspective.",
        "expected_topic": "Health",
        "expected_type": "Research",
    },
    {
        "content": "Press release: Tech startup secures $50M funding for AI development.",
        "expected_topic": "Business",
        "expected_type": "Article",
    },
    {
        "content": "Step-by-step guide to implementing cloud security measures.",
        "expected_topic": "Technology",
        "expected_type": "Guide",
    },
    {
        "content": "Q3 2025 Financial Performance Analysis of Major Tech Companies",
        "expected_topic": "Business",
        "expected_type": "Report",
    },
    {
        "content": "Evolution of Democratic Systems: A Historical Analysis",
        "expected_topic": "Government",
        "expected_type": "Research",
    },
    {
        "content": "Latest findings on cognitive behavioral therapy effectiveness",
        "expected_topic": "Health",
        "expected_type": "Research",
    },
    {
        "content": "User Manual: Advanced Network Configuration Guide",
        "expected_topic": "Technology",
        "expected_type": "Guide",
    },
    {
        "content": "Climate Change Impact Report: Arctic Ice Reduction Analysis",
        "expected_topic": "Environment",
        "expected_type": "Report",
    },
    {
        "content": "Application form for research grant funding in renewable energy",
        "expected_topic": "Science",
        "expected_type": "Form",
    },
    {
        "content": "Educational curriculum for advanced mathematics courses",
        "expected_topic": "Education",
        "expected_type": "Educational",
    },
    {
        "content": "Cybersecurity threat assessment report for financial institutions",
        "expected_topic": "Security",
        "expected_type": "Report",
    },
    {
        "content": "Urban transportation infrastructure development plan",
        "expected_topic": "Transportation",
        "expected_type": "Report",
    },
    {
        "content": "Dataset of global climate measurements from 2020-2025",
        "expected_topic": "Environment",
        "expected_type": "Dataset",
    },
    {
        "content": "Conference presentation on quantum computing applications",
        "expected_topic": "Technology",
        "expected_type": "Presentation",
    },
    {
        "content": "Reference guide to international trade regulations",
        "expected_topic": "Legal/Policy",
        "expected_type": "Reference",
    }
]

# Added new test cases for better coverage
test_cases.extend([
    {
        "content": "Detailed minutes from the recent board meeting discussing Q1 performance.",
        "expected_topic": "Business",
        "expected_type": "Report"
    },
    {
        "content": "Proposed legislative changes for environmental protection in urban areas.",
        "expected_topic": "Environment",
        "expected_type": "Report"
    },
    {
        "content": "A study on the effectiveness of online learning platforms in higher education.",
        "expected_topic": "Education",
        "expected_type": "Research"
    },
    {
        "content": "Guidelines for secure software development practices.",
        "expected_topic": "Security",
        "expected_type": "Guide"
    },
    {
        "content": "Research findings on the impact of diet on mental health.",
        "expected_topic": "Health",
        "expected_type": "Research"
    },
    {
        "content": "Analysis of traffic flow patterns in metropolitan transportation networks.",
        "expected_topic": "Transportation",
        "expected_type": "Report"
    },
    {
        "content": "A historical essay on the philosophical underpinnings of democracy.",
        "expected_topic": "Humanities",
        "expected_type": "Research"
    },
    {
        "content": "Summary of recent discoveries in quantum physics.",
        "expected_topic": "Science",
        "expected_type": "Report"
    },
    {
        "content": "Official government statement regarding new immigration policies.",
        "expected_topic": "Government",
        "expected_type": "Legal"
    },
    {
        "content": "Step-by-step form for applying for a small business grant.",
        "expected_topic": "Business",
        "expected_type": "Form"
    },
    {
        "content": "Educational material for teaching basic programming concepts to children.",
        "expected_topic": "Education",
        "expected_type": "Educational"
    },
    {
        "content": "Reference manual for troubleshooting network connectivity issues.",
        "expected_topic": "Technology",
        "expected_type": "Reference"
    }
])

def verify_categories(test_cases):
    """Verify that all test cases use valid primary categories."""
    invalid_topics = set()
    invalid_types = set()
    
    for test_case in test_cases:
        if test_case["expected_topic"] not in PRIMARY_TOPICS:
            invalid_topics.add(test_case["expected_topic"])
        if test_case["expected_type"] not in PRIMARY_DOC_TYPES:
            invalid_types.add(test_case["expected_type"])
    
    if invalid_topics or invalid_types:
        print("\nWARNING: Found invalid categories in test cases:")
        if invalid_topics:
            print("\nInvalid topics:", ", ".join(sorted(invalid_topics)))
        if invalid_types:
            print("\nInvalid document types:", ", ".join(sorted(invalid_types)))
        return False
    return True

def run_tests():
    """Run the classifier tests on the test cases."""
    if not verify_categories(test_cases):
        print("\nPlease fix the invalid categories in the test cases before running the tests.")
        return

    total_tests = len(test_cases)
    topic_correct = 0
    type_correct = 0
    
    print("\nRunning classification tests...")
    print("=" * 80)
    
    for i, test_case in enumerate(test_cases, 1):
        content = test_case["content"]
        expected_topic = test_case["expected_topic"]
        expected_type = test_case["expected_type"]
        
        # Get predictions for topic
        topic_predictions = classifier.classify_document(content, classification_type='topic')
        predicted_topic = topic_predictions["result"]
        
        # Get predictions for document type
        type_predictions = classifier.classify_document(content, classification_type='document_type')
        predicted_type = type_predictions["result"]
        
        # Check if predictions are correct
        topic_matches = predicted_topic == expected_topic
        type_matches = predicted_type == expected_type
        
        if topic_matches:
            topic_correct += 1
        if type_matches:
            type_correct += 1
            
        # Print result for this test case
        print(f"\nTest Case {i}:")
        print(f"Content: {content}")
        print(f"Topic: {predicted_topic} ({'✓' if topic_matches else '✗'}, Expected: {expected_topic})")
        print(f"Type: {predicted_type} ({'✓' if type_matches else '✗'}, Expected: {expected_type})")
        
    # Calculate and print accuracy
    topic_accuracy = (topic_correct / total_tests) * 100
    type_accuracy = (type_correct / total_tests) * 100
    
    print("\n" + "=" * 80)
    print(f"Testing completed. Results:")
    print(f"Topic Classification Accuracy: {topic_accuracy:.1f}%")
    print(f"Document Type Classification Accuracy: {type_accuracy:.1f}%")
    print(f"Total test cases: {total_tests}")

if __name__ == "__main__":
    run_tests()