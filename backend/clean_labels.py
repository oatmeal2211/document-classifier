#!/usr/bin/env python

import csv
import re
import os
import sys
import pandas as pd
import numpy as np

# Define the input and output file paths
# Assuming the script is run from the backend/ directory where the CSV is located
input_csv_path = 'NLP documents bank - Sheet1.csv'
output_csv_path = 'NLP documents bank - Sheet1_cleaned.csv'

# Define primary topic categories
PRIMARY_TOPICS = {
    'Technology',           # Technology, AI, Computing, Digital
    'Business',            # Business, Finance, Economics, Management
    'Research/Academic',   # General Research, Academic content
    'Education',          # Education, Teaching, Learning
    'Health',             # Health, Medical, Psychology, Wellness
    'Legal/Policy',       # Legal, Policy, Compliance
    'Environment',        # Environment, Sustainability, Climate
    'Government',         # Politics, Law, Public Policy
    'Science',            # General Science, Space, Engineering
    'Security',           # Security, Defense, Cybersecurity
    'Transportation',     # Transportation, Mobility, Logistics
    'Humanities'          # Culture, Arts, Social Sciences
}

# Define primary document types
PRIMARY_DOC_TYPES = {
    'Article',           # Articles, News, Blog Posts
    'Research',          # Research Papers, Academic Papers
    'Report',            # Reports, Studies, Whitepapers
    'Guide',            # Guides, Manuals, Instructions
    'Legal',            # Legal Documents, Policies
    'Educational',       # Educational Materials, Textbooks
    'Form',             # Forms, Applications, Questionnaires
    'Presentation',     # Presentations, Slides
    'Reference',        # References, Documentation
    'Dataset'           # Data, Datasets, Collections
}

# Define topic label mappings for standardization
TOPIC_LABEL_MAPPING = {
    # Technology & Computing
    'AI': 'Technology',
    'Artificial Intelligence': 'Technology',
    'Machine Learning': 'Technology',
    'Deep Learning': 'Technology',
    'EdTech': 'Technology',
    'ICT4E': 'Technology',
    'Blockchain': 'Technology',
    'Cryptography': 'Technology',
    'Quantum Computing': 'Technology',
    'Cybersecurity': 'Technology',
    'Data Science': 'Technology',
    'Cloud Computing': 'Technology',
    'IoT': 'Technology',
    'Robotics': 'Technology',
    'AR': 'Technology',
    'VR': 'Technology',
    'Digital Technology': 'Technology',
    'Technology/Innovation': 'Technology',
    'Tech': 'Technology',
    'IT': 'Technology',
    'Computer Vision': 'Technology',
    'NLP': 'Technology',
    'Developer Tools': 'Technology',
    'Software': 'Technology',
    
    # Business & Finance
    'Finance': 'Business',
    'Economics': 'Business',
    'Business': 'Business',
    'Marketing': 'Business',
    'Real Estate': 'Business',
    'HR': 'Business',
    'Human Resources': 'Business',
    'Management': 'Business',
    'Workforce': 'Business',
    'Strategic Planning': 'Business',
    'Analytics': 'Business',
    'CRM': 'Business',
    'M&A': 'Business',
    'Financial': 'Business',
    'Labor Economics': 'Business',
    'International Trade': 'Business',
    
    # Research & Academic
    'Research': 'Research/Academic',
    'Academic': 'Research/Academic',
    'Scientific Research': 'Research/Academic',
    'Academic Research': 'Research/Academic',
    'Research Paper': 'Research/Academic',
    'Academic Paper': 'Research/Academic',
    'Academia': 'Research/Academic',
    
    # Education
    'Education/Learning': 'Education',
    'Learning': 'Education',
    'Training': 'Education',
    'Teaching': 'Education',
    'Student': 'Education',
    'Teacher': 'Education',
    'Educational': 'Education',
    'Higher Ed': 'Education',
    'MOOC': 'Education',
    'OER': 'Education',
    'Class': 'Education',
    'Homework': 'Education',
    
    # Health & Medical
    'Healthcare': 'Health',
    'Medical': 'Health',
    'Wellness': 'Health',
    'Psychology': 'Health',
    'Mental Health': 'Health',
    'Clinical': 'Health',
    'Biology': 'Health',
    'Medicine': 'Health',
    'Epidemiology': 'Health',
    'Disease': 'Health',
    
    # Legal & Policy
    'Law': 'Legal/Policy',
    'Policy': 'Legal/Policy',
    'Legal': 'Legal/Policy',
    'Compliance': 'Legal/Policy',
    'Regulation': 'Legal/Policy',
    'Risk': 'Legal/Policy',
    'Governance': 'Legal/Policy',
    
    # Environment
    'Environmental': 'Environment',
    'Sustainability': 'Environment',
    'Climate': 'Environment',
    'Energy': 'Environment',
    
    # Government & Politics
    'Politics': 'Government',
    'Political': 'Government',
    'Government': 'Government',
    'Public Policy': 'Government',
    'Foreign Policy': 'Government',
    'International Relations': 'Government',
    'Diplomacy': 'Government',
    'Global Relations': 'Government',
    
    # Science & Engineering
    'Space': 'Science',
    'Astronomy': 'Science',
    'Engineering': 'Science',
    'Physics': 'Science',
    'Chemistry': 'Science',
    'Astrobiology': 'Science',
    'Scientific': 'Science',
    
    # Security & Defense
    'Security/Defense': 'Security',
    'Defense': 'Security',
    'Cybersecurity': 'Security',
    'Information Security': 'Security',
    'Network Security': 'Security',
    
    # Transportation & Logistics
    'Transportation/Mobility': 'Transportation',
    'Mobility': 'Transportation',
    'Transit': 'Transportation',
    'Logistics': 'Transportation',
    'Supply Chain': 'Transportation',
    'Freight': 'Transportation',
    
    # Humanities & Culture
    'Culture': 'Humanities',
    'Lifestyle': 'Humanities',
    'Arts': 'Humanities',
    'Social Sciences': 'Humanities',
    'Tourism': 'Humanities',
    'Travel': 'Humanities',
    'Sports': 'Humanities',
    'Athletics': 'Humanities'
}
def normalize_topic_label(label, visited=None):
    """Normalize a topic label by mapping to predefined primary categories with cycle detection."""
    if not label:
        return ""
    
    # Initialize visited set for cycle detection
    if visited is None:
        visited = set()
    
    # Convert to standard format and split multiple topics
    topics = [t.strip() for t in str(label).split(',')]
    normalized_topics = set()
    
    for topic in topics:
        if not topic:
            continue
            
        if topic in visited:
            continue
            
        visited.add(topic)
        
        # Check for exact matches in mapping
        if topic in TOPIC_LABEL_MAPPING:
            mapped = TOPIC_LABEL_MAPPING[topic]
            if mapped in PRIMARY_TOPICS:
                normalized_topics.add(mapped)
                continue
        
        # Convert to lowercase for remaining checks
        topic_lower = topic.lower()
        
        # First try direct mapping with lowercase
        if topic_lower in [t.lower() for t in PRIMARY_TOPICS]:
            for primary in PRIMARY_TOPICS:
                if primary.lower() == topic_lower:
                    normalized_topics.add(primary)
                    break
            continue
                    
        # Try substring matching with primary categories
        matched = False
        for primary in PRIMARY_TOPICS:
            primary_lower = primary.lower()
            if primary_lower in topic_lower or topic_lower in primary_lower:
                normalized_topics.add(primary)
                matched = True
                break
                
        if matched:
            continue
            
        # Apply domain-specific rules
        if any(tech in topic_lower for tech in ['ai', 'software', 'computing', 'digital', 'data', 'cyber']):
            normalized_topics.add('Technology')
        elif any(biz in topic_lower for biz in ['finance', 'economic', 'market', 'business', 'trade']):
            normalized_topics.add('Business')
        elif any(edu in topic_lower for edu in ['education', 'learning', 'teaching', 'student', 'academic']):
            normalized_topics.add('Education')
        elif any(health in topic_lower for health in ['health', 'medical', 'clinical', 'disease']):
            normalized_topics.add('Health')
        elif any(gov in topic_lower for gov in ['government', 'policy', 'regulation', 'law']):
            normalized_topics.add('Legal/Policy')
    
    # Return the first normalized topic if any found
    normalized_list = sorted(normalized_topics)
    return normalized_list[0] if normalized_list else "Technology"  # Default to Technology if no match

# Additional normalization for document type labels
def normalize_doc_type_label(label, visited=None):
    """Normalize a document type label by mapping to predefined document type categories."""
    if not label:
        return ""
    
    # Initialize visited set for cycle detection
    if visited is None:
        visited = set()
    
    # Convert to standard format and split multiple types
    types = [t.strip() for t in str(label).split(',')]
    normalized_types = set()
    
    for doc_type in types:
        if not doc_type:
            continue
            
        if doc_type in visited:
            continue
            
        visited.add(doc_type)
        
        # Check for exact matches in mapping
        if doc_type in DOC_TYPE_LABEL_MAPPING:
            mapped = DOC_TYPE_LABEL_MAPPING[doc_type]
            if mapped in PRIMARY_DOC_TYPES:
                normalized_types.add(mapped)
                continue
        
        # Convert to lowercase for remaining checks
        type_lower = doc_type.lower()
        
        # First try direct mapping with lowercase
        if type_lower in [t.lower() for t in PRIMARY_DOC_TYPES]:
            for primary in PRIMARY_DOC_TYPES:
                if primary.lower() == type_lower:
                    normalized_types.add(primary)
                    break
            continue
        
        # Special case mappings for common types
        if 'white paper' in type_lower or 'whitepaper' in type_lower:
            normalized_types.add('Report')
        elif 'technical' in type_lower:
            if 'guide' in type_lower:
                normalized_types.add('Guide')
            else:
                normalized_types.add('Report')
        elif any(blog in type_lower for blog in ['blog', 'news', 'article', 'post']):
            normalized_types.add('Article')
        elif any(report in type_lower for report in ['report', 'study', 'analysis', 'assessment']):
            normalized_types.add('Report')
        elif any(research in type_lower for research in ['research', 'academic', 'scientific', 'paper']):
            normalized_types.add('Research')
        elif any(guide in type_lower for guide in ['guide', 'manual', 'instruction', 'tutorial']):
            normalized_types.add('Guide')
        elif any(legal in type_lower for legal in ['legal', 'law', 'regulation', 'policy']):
            normalized_types.add('Legal')
        elif any(edu in type_lower for edu in ['course', 'educational', 'learning', 'teaching']):
            normalized_types.add('Educational')
    
    # Return the first normalized type if any found
    normalized_list = sorted(normalized_types)
    return normalized_list[0] if normalized_list else "Article"  # Default to Article if no match
label_mapping = {
    # Generalized Topic Labels
    'finance': 'Business/Finance',
    'economics': 'Business/Finance',
    'business': 'Business/Finance',
    'marketing': 'Business/Finance',
    'international trade': 'Business/Finance',
    'real estate': 'Business/Finance',
    'financial oversight': 'Business/Finance',
    'private equity': 'Business/Finance',
    'banking sector': 'Business/Finance',
    'crypto trends': 'Technology',
    'blockchain infrastructure': 'Technology',
    'agile cryptography': 'Technology',
    'industry challenges': 'Business/Finance',
    'market analysis': 'Business/Finance',
    'market performance': 'Business/Finance',
    'sales forecasting': 'Business/Finance',
    'sales management': 'Business/Finance',
    'sales strategy': 'Business/Finance',
    'financial crimes': 'Politics/Law',
    'money laundering': 'Politics/Law',
    'valuation methods': 'Business/Finance',
    'venture capital': 'Business/Finance',
    'm&a': 'Business/Finance',
    'hr alignment': 'Business/Finance',
    'human resources': 'Business/Finance',
    'hr': 'Business/Finance',
    'training': 'Business/Finance',
    'workforce development': 'Business/Finance',
    'employment': 'Business/Finance',
    'hr theory': 'Business/Finance',
    'strategic hr': 'Business/Finance',
    'hr practices': 'Business/Finance',
    'work psychology': 'Health',
    'labor economics': 'Business/Finance',
    'workforce trends': 'Business/Finance',
    'performance': 'Business/Finance',
    'workplace communication': 'Business/Finance',
    'change management': 'Business/Finance',
    'leadership': 'Business/Finance',
    'future of work': 'Business/Finance',
    'talent strategy': 'Business/Finance',
    'retention': 'Business/Finance',
    'people analytics': 'Business/Finance',
    'talent retention': 'Business/Finance',
    'hr tech': 'Technology',
    'human capital': 'Business/Finance',
    'workforce strategy': 'Business/Finance',
    'hiring': 'Business/Finance',
    'skills gap': 'Business/Finance',
    'hr design': 'Business/Finance',
    'behavioral science': 'Health',
    'metrics': 'Business/Finance',
    'alignment': 'Business/Finance',
    'workforce modeling': 'Business/Finance',
    'engagement': 'Business/Finance',
    'experience metrics': 'Business/Finance',
    'shrm theory': 'Business/Finance',
    'hr compliance': 'Politics/Law',
    'policy development': 'Politics/Law',
    'ethics': 'Politics/Law',
    'hr capstone': 'Business/Finance',
    'practical application': 'Business/Finance',
    'hr trends': 'Business/Finance',
    'global talent': 'Business/Finance',
    'future workforce': 'Business/Finance',
    'strategy': 'Business/Finance',
    'talent management': 'Business/Finance',
    'lifecycle': 'Business/Finance',
    'immigration': 'Politics/Law',
    'global hr': 'Business/Finance',
    'dei': 'Business/Finance',
    'inclusion strategies': 'Business/Finance',
    'risk': 'Business/Finance',
    'public sector hr': 'Politics/Law',
    'tech tools': 'Technology',
    'automation': 'Technology',
    'analytics': 'Business/Finance',
    'insights': 'Business/Finance',
    'esg': 'Environment/Sustainability',
    'sustainability': 'Environment/Sustainability',
    'pandemic': 'Health',
    'remote work': 'Business/Finance',
    'pay strategy': 'Business/Finance',
    'benchmarking': 'Business/Finance',
    'labor': 'Business/Finance',
    'labor market trends': 'Business/Finance',
    'workforce': 'Business/Finance',

    'technology': 'Technology',
    'ai': 'Technology',
    'artificial intelligence': 'Technology',
    'machine learning': 'Technology',
    'nlp': 'Technology',
    'deep learning': 'Technology',
    'computer science': 'Technology',
    'mobile devices': 'Technology',
    'semiconductors': 'Technology',
    'ai hardware': 'Technology',
    'social chat': 'Technology',
    'translation ai': 'Technology',
    'networking': 'Technology',
    'ai integration': 'Technology',
    'enterprise ai': 'Business/Finance',
    'data platforms': 'Technology',
    'computing research': 'Research/Academia',
    'human-computer interaction': 'Technology',
    'scientific discovery': 'Research/Academia',
    'cryptography': 'Technology',
    'lattice theory': 'Technology',
    'data science': 'Technology',
    'text analysis': 'Technology',
    'digital humanities': 'Research/Academia',
    'cybersecurity': 'Security',
    'risk analysis': 'Business/Finance',
    'information technology': 'Technology',
    'software development': 'Technology',
    'information systems': 'Technology',
    'text analytics': 'Technology',
    'developer tools': 'Technology',
    'user experience': 'Technology',
    'model evaluation': 'Technology',
    'computer vision': 'Technology',
    'ai platforms': 'Technology',
    'cloud': 'Technology',
    'ai assistants': 'Technology',
    'collaboration': 'Business/Finance',
    'crm': 'Business/Finance',
    'ai acceleration': 'Technology',
    'creative tools': 'Technology',
    'music tech': 'Lifestyle/Culture',
    'speech ai': 'Technology',
    'enterprise software': 'Business/Finance',
    'multimodal models': 'Technology',
    'chatgpt': 'Technology',
    'chat interfaces': 'Technology',
    'model lifecycle': 'Technology',
    'api': 'Technology',
    'surveys': 'Research/Academia',
    'generative ai': 'Technology',
    'multimodal ai': 'Technology',
    'model availability': 'Technology',
    'model analysis': 'Technology',
    'startups': 'Business/Finance',
    'hardware': 'Technology',
    'social media': 'Lifestyle/Culture',

    'research': 'Research/Academia',
    'science': 'Research/Academia',
    'research trends': 'Research/Academia',
    'research analysis': 'Research/Academia',
    'academia': 'Research/Academia',
    'research methodology': 'Research/Academia',
    'research policy': 'Research/Academia',
    'open science': 'Research/Academia',
    'library science': 'Research/Academia',
    'academic publications': 'Research/Academia',
    'legal research': 'Research/Academia',
    'legal studies': 'Politics/Law',
    'scientific research': 'Research/Academia',
    'research evaluation': 'Research/Academia',
    'research metrics': 'Research/Academia',
    'cryptology': 'Technology',
    'formal verification': 'Technology',
    'cryptography topics': 'Technology',
    'applied cryptology': 'Technology',
    '2024 publications': 'Research/Academia',
    'journal articles': 'Research/Academia',
    'complexity measures': 'Technology',
    'block cipher cryptanalysis': 'Technology',
    'arxiv preprints': 'Research/Academia',
    'legal scholarship': 'Research/Academia',

    'sports': 'Sports',

    'health': 'Health',
    'healthcare': 'Health',
    'mental health': 'Health',
    'public health': 'Health',
    'epidemiology': 'Health',
    'global health': 'Health',
    'healthcare workforce': 'Health',
    'medicine': 'Health',
    'clinical research': 'Research/Academia',
    'infectious diseases': 'Health',
    'digital health': 'Technology',
    'global health funding': 'Business/Finance',
    'global health policy': 'Politics/Law',
    'psychology': 'Health',
    'personal development': 'Lifestyle/Culture',
    'biology': 'Research/Academia',

    'law': 'Politics/Law',
    'land law': 'Politics/Law',
    'politics': 'Politics/Law',
    'public policy': 'Politics/Law',
    'international relations': 'Politics/Law',
    'geopolitics': 'Politics/Law',
    'tax policy': 'Politics/Law',
    'transportation policy': 'Politics/Law',
    'national security': 'Politics/Law',
    'government operations': 'Politics/Law',
    'military cooperation': 'Politics/Law',
    'counterterrorism': 'Politics/Law',
    'international justice': 'Politics/Law',
    'political science': 'Politics/Law',
    'regulatory standards': 'Politics/Law',
    'copyright law': 'Politics/Law',
    'patent law': 'Politics/Law',
    'appropriations': 'Politics/Law',
    'federal law': 'Politics/Law',
    'legal language models': 'Technology',
    'case law prediction': 'Technology',
    'corporate governance': 'Business/Finance',
    'constitutional law': 'Politics/Law',
    'legal profession': 'Politics/Law',
    'global legal developments': 'Politics/Law',
    'case law': 'Legal/Policy',
    'sentencing guidelines': 'Politics/Law',
    'legal writing': 'Research/Academia',
    'citation standards': 'Research/Academia',
    'local legislation': 'Politics/Law',
    'emergency management': 'Politics/Law',
    'legal industry trends': 'Business/Finance',
    'law firm growth': 'Business/Finance',
    'legal technology': 'Technology',
    'general counsel': 'Politics/Law',
    'legal strategy': 'Politics/Law',
    'employment law': 'Politics/Law',
    'california labor laws': 'Politics/Law',
    'legal ai': 'Technology',
    'legal operations': 'Business/Finance',
    'corporate legal': 'Politics/Law',
    'intellectual property': 'Politics/Law',
    'privacy': 'Politics/Law',
    'access to justice': 'Politics/Law',
    'legal data': 'Politics/Law',
    'llms': 'Technology',
    'legal accuracy': 'Politics/Law',
    'legal text analytics': 'Technology',
    'india': 'Politics/Law',
    'jurisprudence': 'Politics/Law',
    'legal theory': 'Politics/Law',
    'democratic ideals': 'Politics/Law',
    'public law': 'Politics/Law',
    'various legal topics': 'Politics/Law',
    'political economy': 'Business/Finance',
    'global politics': 'Politics/Law',
    'various ir topics': 'Politics/Law',
    'un sanctions': 'Politics/Law',
    'u.s.-china relations': 'Politics/Law',
    'economic strategy': 'Business/Finance',
    'u.s. foreign policy': 'Politics/Law',
    'global conflicts': 'Politics/Law',
    'southeast asia': 'Politics/Law',
    'u.s. military policy': 'Politics/Law',
    'engineering': 'Technology',
    'information retrieval': 'Technology',
    'information extraction': 'Technology',
    'language documentation': 'Research/Academia',
    'authorship analysis': 'Research/Academia',
    'document processing': 'Technology',
    'speech processing': 'Technology',
    'neuro-symbolic ai': 'Technology',
    'interpretability': 'Technology',
    'brain science': 'Health',
    'accessibility': 'General',
    'molecular dynamics': 'Research/Academia',
    'reinforcement learning': 'Technology',
    'materials science': 'Research/Academia',
    'media studies': 'Lifestyle/Culture',
    'manufacturing': 'Business/Finance',
    'content marketing': 'Business/Finance',
    'education policy': 'Education',
    'student assessment': 'Education',
    'hr analytics': 'Business/Finance',
    'workforce transformation': 'Business/Finance',
    'asymmetric encryption': 'Technology',
    'cryptographic techniques': 'Technology',
    'data security': 'Security',
    'cryptographic methods': 'Technology',
    'authentication': 'Security',
    'information theory': 'Research/Academia',
    'cryptographic applications': 'Technology',
    'practical implementations': 'Technology',
    'high-dimensional systems': 'Research/Academia',
    'applied cryptography': 'Technology',
    'textbook': 'Education',
    'lfsr': 'Technology',
    'hash functions': 'Technology',
    'mathematics': 'Research/Academia',
    'reference book': 'Research/Academia',
    'classical cryptography': 'Technology',
    'quantum computing': 'Technology',
    'formal verification': 'Technology',
    'post-quantum cryptography': 'Technology',
    'nist standards': 'Security',
    'organizational readiness': 'Business/Finance',
    'iot security': 'Security',
    'data protection': 'Security',
    'applied cryptology': 'Technology',
    'complexity measures': 'Research/Academia',
    'block cipher cryptanalysis': 'Technology',
    'encryption risks': 'Security',
    'bfsi sector': 'Business/Finance',
    'data analysis': 'Business/Finance',
    'ai in transportation': 'Technology',
    'budgeting': 'Business/Finance',
    'freight market': 'Business/Finance',
    'logistics': 'Business/Finance',
    'future outlook': 'Business/Finance',
    'industry trends': 'Business/Finance',
    'federated learning': 'Technology',
    'intelligent transportation': 'Transportation',
    'graph neural networks': 'Technology',
    'object detection': 'Technology',
    'planning': 'General',
    'interdisciplinary transportation research': 'Research/Academia',
    'technical papers': 'Research/Academia',
    'systems': 'Technology',
    'research digest': 'Report/Study',
    'emergency transportation': 'Transportation',
    'accident report': 'Transportation',
    'traffic congestion': 'Transportation',
    'road travel': 'Transportation',
    'public transportation': 'Transportation',
    'autonomous vehicles': 'Technology',
    'public health crisis': 'Health',
    'driver shortage': 'Business/Finance',
    'air travel': 'Tourism',
    'policy': 'Politics/Law',
    'circulars': 'Research/Academia',
    'research circulars': 'Research/Academia',
    'whitepapers': 'Report/Study',
    'back office efficiency': 'Business/Finance',
    'innovation': 'General',
    'infrastructure': 'Politics/Law',
    'strategic research': 'Research/Academia',
    'asphalt mixtures': 'General',
    'federal judiciary': 'Politics/Law',
    'caseload statistics': 'Politics/Law',
    'data management': 'Technology',
    'digital products': 'Business/Finance',
    'legislation': 'Politics/Law',
    'case law prediction': 'Technology',
    'legal industry trends': 'Business/Finance',
    'law firm growth': 'Business/Finance',
    'legal ai': 'Technology',
    'contract review': 'Legal/Policy',
    'legal operations': 'Business/Finance',
    'corporate legal': 'Business/Finance',
    'intellectual property': 'Politics/Law',
    'privacy': 'Politics/Law',
    'access to justice': 'Politics/Law',
    'legal data': 'Politics/Law',
    'legal accuracy': 'Politics/Law',
    'legal text analytics': 'Technology',
    'india': 'Politics/Law',
    'jurisprudence': 'Politics/Law',
    'legal theory': 'Politics/Law',
    'democratic ideals': 'Politics/Law',
    'public law': 'Politics/Law',
    'various legal topics': 'Politics/Law',
    'faculty rankings': 'Research/Academia',
    'research study': 'Research/Academia',
    'political economy': 'Business/Finance',
    'global politics': 'Politics/Law',
    'various ir topics': 'Politics/Law',
    'un sanctions': 'Politics/Law',
    'u.s.-china relations': 'Politics/Law',
    'economic strategy': 'Business/Finance',
    'u.s. foreign policy': 'Politics/Law',
    'global conflicts': 'Politics/Law',
    'southeast asia': 'Politics/Law',
    'u.s. military policy': 'Politics/Law',
    'development': 'General',
    'open access': 'Research/Academia',
    'global climate policy': 'Environment/Sustainability',
    'latin american perspectives': 'Politics/Law',


    # Business and Finance
    'Business': 'Business/Finance',
    'Finance': 'Business/Finance',
    'Financial': 'Business/Finance',
    'Economics': 'Business/Finance',
    'Market Analysis': 'Business/Finance',
    'Investment': 'Business/Finance',

    # Education
    'Online Learning': 'Education',
    'E-learning': 'Education',
    'Higher Education': 'Education',
    'K-12': 'Education',
    'STEM Education': 'Education',
    'Educational': 'Education',
    'Teaching': 'Education',
    'Learning': 'Education',
    'Pedagogy': 'Education',
    'MOOCs': 'Education',
    'Distance Learning': 'Education',
    'Remote Learning': 'Education',

    # Health
    'Healthcare': 'Health',
    'Medical': 'Health',
    'Public Health': 'Health',
    'Mental Health': 'Health',
    'Clinical': 'Health',
    'Wellness': 'Health',
    'COVID': 'Health',
    'Pandemic': 'Health',

    # Legal and Policy
    'Law': 'Legal/Policy',
    'Legal': 'Legal/Policy',
    'Policy': 'Legal/Policy',
    'Regulation': 'Legal/Policy',
    'Compliance': 'Legal/Policy',
    'Governance': 'Legal/Policy',

    # Politics
    'Political': 'Politics/Law',
    'Government': 'Politics/Law',
    'Public Policy': 'Politics/Law',
    'Legislation': 'Politics/Law',

    # Environment
    'Environmental': 'Environment/Sustainability',
    'Sustainability': 'Environment/Sustainability',
    'Climate': 'Environment/Sustainability',
    'Green': 'Environment/Sustainability',
    'Renewable': 'Environment/Sustainability',

    # Culture and Lifestyle
    'Culture': 'Lifestyle/Culture',
    'Lifestyle': 'Lifestyle/Culture',
    'Social': 'Lifestyle/Culture',
    'Entertainment': 'Lifestyle/Culture',

    # Tourism and Travel
    'Travel': 'Tourism',
    'Tourism/Travel': 'Tourism',
    'Hospitality': 'Tourism',
    'Adventure Tourism': 'Tourism',

    # Transportation
    'Transport': 'Transportation',
    'Mobility': 'Transportation',
    'Transit': 'Transportation',
    'Automotive': 'Transportation',
    'Aviation': 'Transportation',

    # International Relations
    'International Relations': 'Foreign Policy',
    'Global Relations': 'Foreign Policy',
    'Diplomacy': 'Foreign Policy',
    'International Affairs': 'Foreign Policy',

    # Space and Astronomy
    'Astronomy': 'Space',
    'Aerospace': 'Space',
    'Space Exploration': 'Space',
    'Astrobiology': 'Space',

    # Security
    'Defense': 'Security',
    'Cybersecurity': 'Security',
    'Information Security': 'Security',
    'Network Security': 'Security',

    # Sports and Athletics
    'Athletics': 'Sports',
    'Physical Education': 'Sports',
    'Sport Science': 'Sports',

    # Global Collaboration
    'International Collaboration': 'Global Collaboration',
    'Global Partnership': 'Global Collaboration',
    'Cross-border Cooperation': 'Global Collaboration'
}

# Define document type label mappings
DOC_TYPE_LABEL_MAPPING = {
    # Articles & News
    'Article': 'Article',
    'News Article': 'Article',
    'News': 'Article',
    'Blog Post': 'Article',
    'Press Release': 'Article',
    'Newsletter': 'Article',
    'Magazine Article': 'Article',
    'Opinion Piece': 'Article',
    'Editorial': 'Article',
    'Daily Update': 'Article',
    'Announcement': 'Article',
    
    # Research Documents
    'Research Paper': 'Research',
    'Academic Paper': 'Research',
    'Scientific Paper': 'Research',
    'Research Article': 'Research',
    'Conference Paper': 'Research',
    'Dissertation': 'Research',
    'Thesis': 'Research',
    'Journal Article': 'Research',
    'Academic Journal': 'Research',
    'Peer-reviewed Article': 'Research',
    'Academic Publications': 'Research',
    'Research/Academic': 'Research',
    'Workshop Paper': 'Research',
    'Conference Proceedings': 'Research',
    'Systematic Review': 'Research',
    'Literature Review': 'Research',
    'Research Collection': 'Research',
    'Review Paper': 'Research',
    'Foundational Paper': 'Research',
    'Seminal Paper': 'Research',
    
    # Reports & Studies
    'Report': 'Report',
    'Study': 'Report',
    'Analysis': 'Report',
    'Research Report': 'Report',
    'Technical Report': 'Report',
    'Market Report': 'Report',
    'Annual Report': 'Report',
    'Survey Report': 'Report',
    'Case Study': 'Report',
    'White Paper': 'Report',
    'Whitepaper': 'Report',
    'Policy Brief': 'Report',
    'Briefing': 'Report',
    'Risk Assessment': 'Report',
    'Research Digest': 'Report',
    'Research Circular': 'Report',
    'Disease Update': 'Report',
    
    # Guides & Manuals
    'Manual': 'Guide',
    'Guide': 'Guide',
    'Handbook': 'Guide',
    'Tutorial': 'Guide',
    'Instructions': 'Guide',
    'Documentation': 'Guide',
    'Reference Guide': 'Guide',
    'User Guide': 'Guide',
    'Guidelines': 'Guide',
    'Best Practices': 'Guide',
    'Guide/Manual': 'Guide',
    'Guidance Document': 'Guide',
    'Framework': 'Guide',
    'Toolkit': 'Guide',
    
    # Legal & Policy Documents
    'Policy Document': 'Legal',
    'Legal Document': 'Legal',
    'Regulation': 'Legal',
    'Law': 'Legal',
    'Legislation': 'Legal',
    'Contract': 'Legal',
    'Agreement': 'Legal',
    'Legal/Policy': 'Legal',
    'Court Judgment': 'Legal',
    'Court Opinions': 'Legal',
    'Legal Update': 'Legal',
    
    # Educational Materials
    'Educational Resource': 'Educational',
    'Textbook': 'Educational',
    'Course Material': 'Educational',
    'Curriculum': 'Educational',
    'Lesson Plan': 'Educational',
    'Online Course': 'Educational',
    'Training Material': 'Educational',
    'Educational Text': 'Educational',
    'Learning Resource': 'Educational',
    'Teaching Material': 'Educational',
    
    # Forms & Applications
    'Form': 'Form',
    'Application': 'Form',
    'Registration Form': 'Form',
    'Survey': 'Form',
    'Quiz': 'Form',
    'Feedback Form': 'Form',
    'Assessment Form': 'Form',
    'Questionnaire': 'Form',
    'Application Form': 'Form',
    'Template': 'Form',
    
    # Presentations
    'Slides': 'Presentation',
    'Presentation': 'Presentation',
    'Slideshow': 'Presentation',
    'Deck': 'Presentation',
    'Conference Presentation': 'Presentation',
    'Webinar': 'Presentation',
    'Pitch Deck': 'Presentation',
    
    # References
    'Reference': 'Reference',
    'Index': 'Reference',
    'Encyclopedia Entry': 'Reference',
    'Reference Book': 'Reference',
    'Glossary': 'Reference',
    'Dictionary': 'Reference',
    'Bibliography': 'Reference',
    'Catalog': 'Reference',
    'Directory': 'Reference',
    'Web Resource': 'Reference',
    'Online Repository': 'Reference',
    'Database': 'Reference',
    'Archive': 'Reference',
    'Journal': 'Reference',
    'Journal Issue': 'Reference',
    
    # Datasets & Collections
    'Dataset': 'Dataset',
    'Data Collection': 'Dataset',
    'Database': 'Dataset',
    'Statistics': 'Dataset',
    'Metrics': 'Dataset',
    'Research Data': 'Dataset',
    'Data Repository': 'Dataset',
    'Data Archive': 'Dataset'
}
def clean_csv():
    """Clean and standardize the labels in the CSV file."""
    print(f"Reading input CSV from {input_csv_path}...")
    
    try:
        # Read the CSV file
        df = pd.read_csv(input_csv_path)
        initial_rows = len(df)
        print(f"Initial number of rows: {initial_rows}")
        
        # Clean topic labels
        print("\nCleaning topic labels...")
        df['Topic Labels'] = df['Topic Labels'].fillna('')
        df['Topic Labels'] = df['Topic Labels'].apply(normalize_topic_label)
        
        # Clean document type labels
        print("\nCleaning document type labels...")
        df['Type Labels'] = df['Type Labels'].fillna('')
        # Split multiple document types and clean each one
        df['Type Labels'] = df['Type Labels'].apply(
            lambda x: ','.join(sorted(set(
                normalize_doc_type_label(t.strip()) 
                for t in str(x).split(',')
                if t.strip()
            )))
        )
        
        # Remove rows where both labels are empty
        df = df[~((df['Topic Labels'] == '') & (df['Type Labels'] == ''))]
        final_rows = len(df)
        
        # Save the cleaned CSV
        df.to_csv(output_csv_path, index=False)
        print(f"\nCleaned CSV saved to {output_csv_path}")
        print(f"Rows processed: {initial_rows} -> {final_rows}")
        
        # Print statistics
        print("\nTopic Label Distribution:")
        topic_counts = df['Topic Labels'].value_counts()
        for topic, count in topic_counts.items():
            print(f"{topic:30} {count:5d}")
        
        print("\nDocument Type Label Distribution:")
        type_counts = pd.Series([
            t.strip() 
            for types in df['Type Labels'].str.split(',') 
            for t in types 
            if t.strip()
        ]).value_counts()
        
        for doc_type, count in type_counts.items():
            print(f"{doc_type:30} {count:5d}")
        
        # Validate primary categories
        unknown_topics = set(topic_counts.index) - PRIMARY_TOPICS - {''}
        unknown_doc_types = set(type_counts.index) - PRIMARY_DOC_TYPES - {''}
        
        if unknown_topics:
            print("\nWARNING: Found topics not in primary categories:")
            for topic in sorted(unknown_topics):
                print(f"  - {topic}")
        
        if unknown_doc_types:
            print("\nWARNING: Found document types not in primary categories:")
            for doc_type in sorted(unknown_doc_types):
                print(f"  - {doc_type}")
                
    except Exception as e:
        print(f"Error processing CSV: {str(e)}")
        raise

def filter_csv(df):
    """Filter out rows where either topic or document type is not in primary categories."""
    # First normalize all labels
    df['Topic Labels'] = df['Topic Labels'].apply(lambda x: normalize_topic_label(x) if pd.notna(x) else x)
    df['Type Labels'] = df['Type Labels'].apply(lambda x: normalize_doc_type_label(x) if pd.notna(x) else x)
    
    # Filter rows where both topic and doc_type are in primary categories
    valid_rows = (df['Topic Labels'].isin(PRIMARY_TOPICS)) & (df['Type Labels'].isin(PRIMARY_DOC_TYPES))
    return df[valid_rows].copy()

def main():
    try:
        print("Starting the label cleaning process...")
        # Read the CSV file with pandas
        print(f"Reading input CSV from {input_csv_path}...")
        df = pd.read_csv(input_csv_path)
        initial_rows = len(df)
        print(f"Initial number of rows: {initial_rows}")
        print(f"CSV columns: {', '.join(df.columns)}\n")

        # Debug print of first few rows
        print("Sample of first few rows before cleaning:")
        print(df[['Title', 'Topic Labels', 'Type Labels']].head())
        print("\n")

        # Clean topic labels
        print("Cleaning topic labels...")
        df['Topic Labels'] = df['Topic Labels'].fillna('')
        df['Topic Labels'] = df['Topic Labels'].apply(lambda x: normalize_topic_label(x) if pd.notna(x) else '')
        
        # Clean document type labels
        print("\nCleaning document type labels...")
        df['Type Labels'] = df['Type Labels'].fillna('')
        df['Type Labels'] = df['Type Labels'].apply(lambda x: normalize_doc_type_label(x) if pd.notna(x) else '')

        # Remove rows with empty or invalid labels
        print("\nRemoving invalid entries...")
        valid_topics = df['Topic Labels'].str.strip() != ''
        valid_types = df['Type Labels'].str.strip() != ''
        df = df[valid_topics & valid_types].copy()

        # Save the cleaned CSV
        print("\nSaving cleaned data...")
        df.to_csv(output_csv_path, index=False)
        final_rows = len(df)
        
        print(f"\nCleaned CSV saved to {output_csv_path}")
        print(f"Rows processed: {initial_rows} -> {final_rows}")
        
        # Print final statistics
        print("\nFinal Topic Label Distribution:")
        topic_counts = df['Topic Labels'].value_counts()
        print(topic_counts)
        
        print("\nFinal Document Type Label Distribution:")
        type_counts = df['Type Labels'].value_counts()
        print(type_counts)
        
        print("\nSample of first few rows after cleaning:")
        print(df[['Title', 'Topic Labels', 'Type Labels']].head())
        
    except Exception as e:
        print(f"Error processing the CSV file: {str(e)}")
        raise

if __name__ == "__main__":
    main()