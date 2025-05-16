from django.contrib import admin
from .models import Document, ClassificationResult, TrainingData

@admin.register(Document)
class DocumentAdmin(admin.ModelAdmin):
    list_display = ('file_name', 'upload_date')
    search_fields = ('file_name', 'content')

@admin.register(ClassificationResult)
class ClassificationResultAdmin(admin.ModelAdmin):
    list_display = ('document', 'classification_type', 'result', 'confidence', 'created_at')
    list_filter = ('classification_type',)
    search_fields = ('document__file_name', 'result')

@admin.register(TrainingData)
class TrainingDataAdmin(admin.ModelAdmin):
    list_display = ('topic_label', 'document_type_label', 'added_date')
    list_filter = ('topic_label', 'document_type_label')
    search_fields = ('content', 'topic_label', 'document_type_label')
