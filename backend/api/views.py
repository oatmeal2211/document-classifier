from django.shortcuts import render
from rest_framework import viewsets, status
from rest_framework.decorators import action
from rest_framework.response import Response
from rest_framework.parsers import MultiPartParser, FormParser, JSONParser
import os
import subprocess
import sys

from .models import Document, ClassificationResult, TrainingData
from .serializers import DocumentSerializer, ClassificationResultSerializer, TrainingDataSerializer
from .import classifier
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
    
    @action(detail=False, methods=['get'])
    def model_status(self, request):
        """Get the status of loaded models."""
        model_info = classifier.get_model_info()
        return Response(model_info)
    
    @action(detail=False, methods=['post'])
    def reload_models(self, request):
        """Reload the classification models."""
        result = classifier.reload_models()
        return Response(result)

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
        """
        Train the classifiers using the available training data.
        This triggers the external training script.
        """
        # Get all training data
        training_data = TrainingData.objects.all()
        
        if not training_data.exists():
            return Response(
                {"error": "No training data available in database."},
                status=status.HTTP_400_BAD_REQUEST
            )
        
        try:
            # Export training data to CSV for the training script
            import csv
            import tempfile
            
            # Create temporary CSV file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False, newline='') as temp_file:
                writer = csv.writer(temp_file)
                
                # Write header
                writer.writerow(['Title', 'Topic Labels', 'Type Labels'])
                
                # Write data
                for td in training_data:
                    writer.writerow([td.content, td.topic_label, td.document_type_label])
                
                temp_csv_path = temp_file.name
            
            # Run the training script
            script_path = os.path.join(os.path.dirname(__file__), '..', 'train_model.py')
            
            # Set environment variable for the CSV path
            env = os.environ.copy()
            env['TRAINING_CSV_PATH'] = temp_csv_path
            
            result = subprocess.run(
                [sys.executable, script_path],
                capture_output=True,
                text=True,
                env=env,
                timeout=3600  # 1 hour timeout
            )
            
            # Clean up temporary file
            try:
                os.unlink(temp_csv_path)
            except:
                pass
            
            if result.returncode == 0:
                # Training successful - reload models
                classifier.reload_models()
                return Response({
                    "status": "Training completed successfully.",
                    "output": result.stdout,
                    "training_samples": training_data.count()
                })
            else:
                return Response({
                    "error": "Training failed.",
                    "details": result.stderr,
                    "output": result.stdout
                }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
                
        except subprocess.TimeoutExpired:
            return Response({
                "error": "Training timed out. Please try again or check the training data."
            }, status=status.HTTP_408_REQUEST_TIMEOUT)
        except Exception as e:
            return Response({
                "error": f"Training failed with error: {str(e)}"
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
    
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
        
        try:
            if filename.endswith('.csv'):
                # Process CSV file
                import csv
                import io
                
                # Read CSV
                file_content = file.read().decode('utf-8')
                csv_data = io.StringIO(file_content)
                reader = csv.DictReader(csv_data)
                
                # Check if we have any data
                rows = list(reader)
                if not rows:
                    return Response(
                        {"error": "CSV file is empty or has no data rows."},
                        status=status.HTTP_400_BAD_REQUEST
                    )
                
                # Check required fields (flexible field names)
                first_row = rows[0]
                field_mapping = {}
                
                # Map common field variations
                for key in first_row.keys():
                    key_lower = key.lower().strip()
                    if any(term in key_lower for term in ['content', 'text', 'title']):
                        field_mapping['content'] = key
                    elif any(term in key_lower for term in ['topic', 'subject']):
                        field_mapping['topic_label'] = key
                    elif any(term in key_lower for term in ['type', 'document_type', 'doc_type']):
                        field_mapping['document_type_label'] = key
                
                # Check if we found all required fields
                missing_fields = []
                if 'content' not in field_mapping:
                    missing_fields.append('content/text/title')
                if 'topic_label' not in field_mapping:
                    missing_fields.append('topic/subject')
                if 'document_type_label' not in field_mapping:
                    missing_fields.append('type/document_type')
                
                if missing_fields:
                    return Response(
                        {
                            "error": f"CSV missing required columns: {', '.join(missing_fields)}",
                            "available_columns": list(first_row.keys()),
                            "note": "Column names are case-insensitive and can be variations of the required names."
                        },
                        status=status.HTTP_400_BAD_REQUEST
                    )
                
                # Process rows
                created_count = 0
                errors = []
                
                for i, row in enumerate(rows, 1):
                    try:
                        # Skip empty rows
                        if not any(row.values()):
                            continue
                            
                        content = row[field_mapping['content']].strip()
                        topic_label = row[field_mapping['topic_label']].strip()
                        doc_type_label = row[field_mapping['document_type_label']].strip()
                        
                        # Skip if any required field is empty
                        if not content or not topic_label or not doc_type_label:
                            errors.append(f"Row {i}: Missing required data")
                            continue
                        
                        TrainingData.objects.create(
                            content=content,
                            topic_label=topic_label,
                            document_type_label=doc_type_label
                        )
                        created_count += 1
                        
                    except Exception as e:
                        errors.append(f"Row {i}: {str(e)}")
                
                response_data = {
                    "status": f"Successfully uploaded {created_count} training samples.",
                    "created_count": created_count,
                    "total_rows": len(rows)
                }
                
                if errors:
                    response_data["errors"] = errors[:10]  # Limit to first 10 errors
                    if len(errors) > 10:
                        response_data["note"] = f"Showing first 10 of {len(errors)} errors"
                
                return Response(response_data)
                
            elif filename.endswith('.json'):
                # Process JSON file
                import json
                
                json_data = json.loads(file.read().decode('utf-8'))
                
                if not isinstance(json_data, list):
                    return Response(
                        {"error": "JSON file must contain a list of training data objects."},
                        status=status.HTTP_400_BAD_REQUEST
                    )
                
                # Process JSON objects
                created_count = 0
                errors = []
                
                for i, item in enumerate(json_data, 1):
                    try:
                        required_fields = ['content', 'topic_label', 'document_type_label']
                        if all(field in item and item[field].strip() for field in required_fields):
                            TrainingData.objects.create(
                                content=item['content'].strip(),
                                topic_label=item['topic_label'].strip(),
                                document_type_label=item['document_type_label'].strip()
                            )
                            created_count += 1
                        else:
                            errors.append(f"Item {i}: Missing or empty required fields")
                    except Exception as e:
                        errors.append(f"Item {i}: {str(e)}")
                
                response_data = {
                    "status": f"Successfully uploaded {created_count} training samples.",
                    "created_count": created_count,
                    "total_items": len(json_data)
                }
                
                if errors:
                    response_data["errors"] = errors[:10]
                    if len(errors) > 10:
                        response_data["note"] = f"Showing first 10 of {len(errors)} errors"
                
                return Response(response_data)
                
            else:
                return Response(
                    {"error": "Unsupported file format. Please upload CSV or JSON."},
                    status=status.HTTP_400_BAD_REQUEST
                )
                
        except Exception as e:
            return Response(
                {"error": f"File processing failed: {str(e)}"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )
    
    @action(detail=False, methods=['get'])
    def export_csv(self, request):
        """Export training data as CSV."""
        from django.http import HttpResponse
        import csv
        
        response = HttpResponse(content_type='text/csv')
        response['Content-Disposition'] = 'attachment; filename="training_data.csv"'
        
        writer = csv.writer(response)
        writer.writerow(['Title', 'Topic Labels', 'Type Labels'])
        
        for td in TrainingData.objects.all():
            writer.writerow([td.content, td.topic_label, td.document_type_label])
        
        return response
    
    @action(detail=False, methods=['get'])
    def statistics(self, request):
        """Get statistics about the training data."""
        from django.db.models import Count
        
        total_count = TrainingData.objects.count()
        
        # Topic distribution
        topic_stats = TrainingData.objects.values('topic_label').annotate(
            count=Count('topic_label')
        ).order_by('-count')
        
        # Document type distribution
        doctype_stats = TrainingData.objects.values('document_type_label').annotate(
            count=Count('document_type_label')
        ).order_by('-count')
        
        return Response({
            'total_samples': total_count,
            'topic_distribution': list(topic_stats),
            'document_type_distribution': list(doctype_stats),
            'unique_topics': len(topic_stats),
            'unique_document_types': len(doctype_stats)
        })