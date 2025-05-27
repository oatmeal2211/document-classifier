from django.db import models

# Create your models here.

class Document(models.Model):
    """Model representing a document uploaded for classification."""
    file = models.FileField(upload_to='documents/')
    file_name = models.CharField(max_length=255)
    content = models.TextField(blank=True)
    upload_date = models.DateTimeField(auto_now_add=True)
    
    def __str__(self):
        return self.file_name

class ClassificationResult(models.Model):
    """Model representing the classification results."""
    document = models.ForeignKey(Document, on_delete=models.CASCADE, related_name='classifications')
    classification_type = models.CharField(max_length=50, choices=[('topic', 'Topic'), ('document_type', 'Document Type')])
    result = models.CharField(max_length=255)
    confidence = models.FloatField(default=0.0)
    created_at = models.DateTimeField(auto_now_add=True)
    
    def __str__(self):
        return f"{self.document.file_name} - {self.classification_type}: {self.result}"

class TrainingData(models.Model):
    """Model for storing training data for the classifier."""
    content = models.TextField()
    topic_label = models.CharField(max_length=255)
    document_type_label = models.CharField(max_length=255)
    added_date = models.DateTimeField(auto_now_add=True)
    
    def __str__(self):
        return f"{self.topic_label} - {self.document_type_label}"
