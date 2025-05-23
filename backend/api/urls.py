from django.urls import path, include
from rest_framework.routers import DefaultRouter
from .views import DocumentViewSet, ClassificationResultViewSet, TrainingDataViewSet

router = DefaultRouter()
router.register(r'documents', DocumentViewSet)
router.register(r'classification-results', ClassificationResultViewSet)
router.register(r'training-data', TrainingDataViewSet)

urlpatterns = [
    path('', include(router.urls)),
] 