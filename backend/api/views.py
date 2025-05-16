from django.shortcuts import render
from rest_framework import viewsets, status
from rest_framework.decorators import action
from rest_framework.response import Response
from rest_framework.parsers import MultiPartParser, FormParser, JSONParser
import os

from .models import Document, ClassificationResult, TrainingData
from .serializers import DocumentSerializer, ClassificationResultSerializer, TrainingDataSerializer
from .classifier import classifier
from .file_processor import extract_text_from_file, clean_text

class DocumentViewSet(viewsets.ModelViewSet):
    """API endpoint for documents."""
    queryset = Document.objects.all()
    serializer_class = DocumentSerializer
    parser_classes = (MultiPartParser, FormParser, JSONParser)
    
    def perform_create(self, serializer):
        """Process the document when it's uploaded."""
        document = serializer.save()
        
        # Get the file path
        file_path = document.file.path
        
        # Extract text from the document
        extracted_text = extract_text_from_file(file_path)
        
        # Clean the extracted text
        cleaned_text = clean_text(extracted_text)
        
        # Update the document with the extracted content
        document.content = cleaned_text
        document.save()
    
    @action(detail=True, methods=['post'])
    def classify(self, request, pk=None):
        """Classify a document by topic or document type."""
        document = self.get_object()
        
        # Check if we have a document with content
        if not document.content:
            return Response(
                {"error": "No content available for classification."},
                status=status.HTTP_400_BAD_REQUEST
            )
        
        # Get classification type from request
        classification_type = request.data.get('classification_type', 'topic')
        if classification_type not in ['topic', 'document_type']:
            return Response(
                {"error": "Invalid classification type. Must be 'topic' or 'document_type'."},
                status=status.HTTP_400_BAD_REQUEST
            )
        
        # Classify the document
        classification_result = classifier.classify_document(
            document.content, 
            classification_type=classification_type
        )
        
        # Handle error in classification
        if 'error' in classification_result:
            return Response(
                classification_result,
                status=status.HTTP_400_BAD_REQUEST
            )
        
        # Save the classification result
        result = ClassificationResult.objects.create(
            document=document,
            classification_type=classification_type,
            result=classification_result['result'],
            confidence=classification_result['confidence']
        )
        
        # Return the classification result
        return Response(ClassificationResultSerializer(result).data)

class ClassificationResultViewSet(viewsets.ReadOnlyModelViewSet):
    """API endpoint for classification results."""
    queryset = ClassificationResult.objects.all()
    serializer_class = ClassificationResultSerializer
    
    def get_queryset(self):
        """Filter results by document if specified."""
        queryset = ClassificationResult.objects.all()
        document_id = self.request.query_params.get('document_id', None)
        
        if document_id is not None:
            queryset = queryset.filter(document_id=document_id)
            
        return queryset

class TrainingDataViewSet(viewsets.ModelViewSet):
    """API endpoint for managing training data."""
    queryset = TrainingData.objects.all()
    serializer_class = TrainingDataSerializer
    
    @action(detail=False, methods=['post'])
    def train(self, request):
        """Train the classifiers using the available training data."""
        # Get all training data
        training_data = TrainingData.objects.all()
        
        if not training_data:
            return Response(
                {"error": "No training data available."},
                status=status.HTTP_400_BAD_REQUEST
            )
        
        # Extract content and labels
        documents = [td.content for td in training_data]
        topic_labels = [td.topic_label for td in training_data]
        doc_type_labels = [td.document_type_label for td in training_data]
        
        # Train both classifiers
        classifier.train_topic_classifier(documents, topic_labels)
        classifier.train_doc_type_classifier(documents, doc_type_labels)
        
        return Response({"status": "Classifiers trained successfully."})
    
    @action(detail=False, methods=['post'])
    def bulk_upload(self, request):
        """Bulk upload training data from a CSV or JSON file."""
        if 'file' not in request.FILES:
            return Response(
                {"error": "No file provided."},
                status=status.HTTP_400_BAD_REQUEST
            )
            
        file = request.FILES['file']
        
        # Process based on file extension
        filename = file.name.lower()
        
        if filename.endswith('.csv'):
            # Process CSV file
            import csv
            import io
            
            # Read CSV
            csv_data = io.StringIO(file.read().decode('utf-8'))
            reader = csv.DictReader(csv_data)
            
            # Check required fields
            required_fields = ['content', 'topic_label', 'document_type_label']
            first_row = next(reader, None)
            
            if not first_row or not all(field in first_row for field in required_fields):
                return Response(
                    {"error": f"CSV must contain these columns: {', '.join(required_fields)}"},
                    status=status.HTTP_400_BAD_REQUEST
                )
            
            # Process rows
            created_count = 0
            csv_data.seek(0)  # Reset to beginning, skipping header
            next(reader)  # Skip header row
            
            for row in reader:
                TrainingData.objects.create(
                    content=row['content'],
                    topic_label=row['topic_label'],
                    document_type_label=row['document_type_label']
                )
                created_count += 1
            
            return Response({"status": f"Successfully uploaded {created_count} training samples."})
            
        elif filename.endswith('.json'):
            # Process JSON file
            import json
            
            try:
                json_data = json.loads(file.read().decode('utf-8'))
                
                if not isinstance(json_data, list):
                    return Response(
                        {"error": "JSON file must contain a list of training data objects."},
                        status=status.HTTP_400_BAD_REQUEST
                    )
                
                # Process JSON objects
                created_count = 0
                for item in json_data:
                    if all(field in item for field in ['content', 'topic_label', 'document_type_label']):
                        TrainingData.objects.create(
                            content=item['content'],
                            topic_label=item['topic_label'],
                            document_type_label=item['document_type_label']
                        )
                        created_count += 1
                
                return Response({"status": f"Successfully uploaded {created_count} training samples."})
                
            except json.JSONDecodeError:
                return Response(
                    {"error": "Invalid JSON format."},
                    status=status.HTTP_400_BAD_REQUEST
                )
        else:
            return Response(
                {"error": "Unsupported file format. Please upload CSV or JSON."},
                status=status.HTTP_400_BAD_REQUEST
            )
