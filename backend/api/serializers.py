from rest_framework import serializers
from .models import Document, ClassificationResult, TrainingData

class DocumentSerializer(serializers.ModelSerializer):
    class Meta:
        model = Document
        fields = ['id', 'file', 'file_name', 'content', 'upload_date']
        read_only_fields = ['content', 'upload_date']

class ClassificationResultSerializer(serializers.ModelSerializer):
    class Meta:
        model = ClassificationResult
        fields = ['id', 'document', 'classification_type', 'result', 'confidence', 'created_at']
        read_only_fields = ['created_at']

class TrainingDataSerializer(serializers.ModelSerializer):
    class Meta:
        model = TrainingData
        fields = ['id', 'content', 'topic_label', 'document_type_label', 'added_date']
        read_only_fields = ['added_date'] 