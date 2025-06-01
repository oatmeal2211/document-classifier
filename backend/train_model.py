#!/usr/bin/env python
"""
Script to train the document classifier using the NLP document bank CSV file.
"""
import os
import django
import sys

# Setup Django environment
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'backend.settings')
try:
    django.setup()
except Exception as e:
    print(f"Django setup failed: {e}")
    print("Make sure you're running this from the backend directory")

# Hugging Face imports
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from datasets import Dataset
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification, Trainer, TrainingArguments
import torch
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
from transformers import DataCollatorWithPadding
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import pickle

# Configuration
CSV_PATH = os.path.join("backend", "NLP documents bank - Sheet1.csv")
if not os.path.exists(CSV_PATH):
    CSV_PATH = "NLP documents bank - Sheet1.csv"  # Try current directory
    
SAVE_DIR_TOPIC = os.path.join("media", "models", "topic_classifier")
SAVE_DIR_TYPE = os.path.join("media", "models", "doctype_classifier")

# Create directories
os.makedirs(SAVE_DIR_TOPIC, exist_ok=True)
os.makedirs(SAVE_DIR_TYPE, exist_ok=True)

def load_and_preprocess_data():
    """Load and preprocess the CSV data"""
    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(f"CSV file not found at {CSV_PATH}")
    
    print(f"Loading data from {CSV_PATH}")
    df = pd.read_csv(CSV_PATH)
    
    # Print initial data info
    print(f"Initial data shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    
    # Check for required columns
    required_columns = ['Title', 'Topic Labels', 'Type Labels']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        print(f"Warning: Missing columns: {missing_columns}")
        print("Available columns:", df.columns.tolist())
    
    # Clean data
    df['Topic Labels'] = df['Topic Labels'].fillna('')
    df['Type Labels'] = df['Type Labels'].fillna('')
    
    print(f"Empty Topic Labels: {(df['Topic Labels'] == '').sum()}")
    print(f"Empty Type Labels: {(df['Type Labels'] == '').sum()}")
    
    # Filter out empty labels
    df = df[df['Topic Labels'] != '']
    df = df[df['Type Labels'] != '']
    
    print(f"After filtering empty labels: {df.shape}")
    
    if df.empty:
        raise ValueError("No valid data remaining after filtering. Check your CSV file.")
    
    # Use Title as content
    df['content'] = df['Title'].fillna('')
    df = df[df['content'] != '']
    
    print(f"After filtering empty titles: {df.shape}")
    
    # Print label distributions
    print(f"\nTopic Labels distribution:")
    topic_counts = df['Topic Labels'].value_counts()
    print(topic_counts)
    single_topic_classes = topic_counts[topic_counts == 1]
    if len(single_topic_classes) > 0:
        print(f"⚠️  Topic classes with only 1 sample: {list(single_topic_classes.index)}")
    
    print(f"\nType Labels distribution:")
    type_counts = df['Type Labels'].value_counts()
    print(type_counts)
    single_type_classes = type_counts[type_counts == 1]
    if len(single_type_classes) > 0:
        print(f"⚠️  Type classes with only 1 sample: {list(single_type_classes.index)}")
    
    return df

def filter_classes_by_min_samples(texts, labels_encoded, label_encoder, min_samples=3):
    """Filter out classes that don't have enough samples"""
    # Count samples per class
    unique_labels, counts = np.unique(labels_encoded, return_counts=True)
    
    print(f"Original class distribution:")
    for label_idx, count in zip(unique_labels, counts):
        class_name = label_encoder.classes_[label_idx]
        print(f"  {class_name}: {count} samples")
    
    # Find classes with enough samples
    valid_classes = unique_labels[counts >= min_samples]
    
    print(f"Classes with >= {min_samples} samples: {len(valid_classes)} out of {len(unique_labels)}")
    
    if len(valid_classes) < 2:
        # Try with min_samples = 2 if 3 is too restrictive
        if min_samples > 2:
            print(f"Trying with min_samples = 2...")
            return filter_classes_by_min_samples(texts, labels_encoded, label_encoder, min_samples=2)
        else:
            raise ValueError(f"After filtering, only {len(valid_classes)} classes remain. Need at least 2 for classification.")
    
    # Filter data to keep only valid classes
    filtered_texts = []
    filtered_labels = []
    for text, label in zip(texts, labels_encoded):
        if label in valid_classes:
            filtered_texts.append(text)
            filtered_labels.append(label)
    
    # Create new label encoder with only valid classes
    valid_class_names = [label_encoder.classes_[i] for i in valid_classes]
    new_label_encoder = LabelEncoder()
    new_label_encoder.fit(valid_class_names)
    
    # Re-encode labels to be continuous (0, 1, 2, ...)
    original_labels = [label_encoder.classes_[i] for i in filtered_labels]
    new_labels_encoded = new_label_encoder.transform(original_labels)
    
    print(f"After filtering:")
    print(f"  Classes: {len(label_encoder.classes_)} → {len(new_label_encoder.classes_)}")
    print(f"  Samples: {len(texts)} → {len(filtered_texts)}")
    
    # Verify the new distribution
    new_unique, new_counts = np.unique(new_labels_encoded, return_counts=True)
    print(f"New class distribution:")
    for i, count in enumerate(new_counts):
        print(f"  {new_label_encoder.classes_[i]}: {count} samples")
    
    return filtered_texts, new_labels_encoded, new_label_encoder

def clean_text(text):
    """Basic text cleaning for better model input"""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = " ".join(text.split())
    return text

def prepare_data(df, label_column, save_dir):
    """Prepare data and train model for given label column"""
    print(f"\n{'='*50}")
    print(f"Training {label_column} classifier")
    print(f"{'='*50}")
    
    # Use extracted_content if available, else fallback to content/title
    if 'extracted_content' in df.columns:
        texts = df['extracted_content'].fillna('').tolist()
    else:
        texts = df['content'].fillna('').tolist()
    texts = [clean_text(t) for t in texts]
    labels = df[label_column].tolist()
    
    # Encode labels
    label_encoder = LabelEncoder()
    labels_encoded = label_encoder.fit_transform(labels)
    
    print(f"Number of classes: {len(label_encoder.classes_)}")
    print(f"Classes: {list(label_encoder.classes_)}")
    
    # Check for data quality issues
    if len(label_encoder.classes_) < 2:
        raise ValueError(f"Need at least 2 different classes for classification. Found: {len(label_encoder.classes_)}")
    
    # Filter out classes with insufficient samples
    texts, labels_encoded, label_encoder = filter_classes_by_min_samples(
        texts, labels_encoded, label_encoder, min_samples=3
    )
    
    # Check if we have enough data after filtering
    total_samples = len(texts)
    num_classes = len(label_encoder.classes_)
    
    print(f"Final data after filtering:")
    print(f"  Total samples: {total_samples}")
    print(f"  Number of classes: {num_classes}")
    
    # Calculate appropriate test size
    if total_samples <= 10:
        # For very small datasets, use all data for training
        print("⚠️  Very small dataset - using all data for training (no test set)")
        X_train, X_test = texts, []
        y_train, y_test = labels_encoded, []
    elif total_samples <= num_classes * 2:
        # Not enough for stratified split - use minimal test set
        print("⚠️  Limited data - using minimal test set without stratification")
        # Use at most 1 sample per class for testing, or 20% of data, whichever is smaller
        max_test_samples = min(num_classes, int(total_samples * 0.2), total_samples - num_classes)
        test_size = max_test_samples / total_samples
        print(f"  Test size: {max_test_samples} samples ({test_size:.2%})")
        
        X_train, X_test, y_train, y_test = train_test_split(
            texts, labels_encoded, test_size=test_size, random_state=42
        )
    else:
        # Enough data for proper split
        print("Using stratified train/test split")
        unique_labels, counts = np.unique(labels_encoded, return_counts=True)
        min_class_samples = min(counts)
        
        print(f"Class distribution after filtering:")
        for i, count in enumerate(counts):
            print(f"  {label_encoder.classes_[i]}: {count} samples")
        
        # Determine if we can use stratified split
        if min_class_samples >= 2:
            # Safe to use stratified split
            test_size = min(0.3, 0.5)  # Use reasonable test size
            print(f"Using stratified split with test_size = {test_size}")
            X_train, X_test, y_train, y_test = train_test_split(
                texts, labels_encoded, test_size=test_size, stratify=labels_encoded, random_state=42
            )
        else:
            # Fall back to random split
            test_size = 0.2
            print(f"Using random split with test_size = {test_size}")
            X_train, X_test, y_train, y_test = train_test_split(
                texts, labels_encoded, test_size=test_size, random_state=42
            )
    
    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test)}")
    
    if len(X_train) == 0:
        raise ValueError("No training data available after preprocessing")
    
    # Initialize tokenizer
    tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")

    def tokenize(examples):
        return tokenizer(examples["text"], truncation=True, padding=True, max_length=512)

    # Create datasets
    print("Creating datasets...")
    train_dataset = Dataset.from_dict({"text": X_train, "label": y_train}).map(tokenize, batched=True)
    
    # Only create test dataset if we have test data
    test_dataset = None
    if len(X_test) > 0:
        test_dataset = Dataset.from_dict({"text": X_test, "label": y_test}).map(tokenize, batched=True)

    # Initialize model
    print("Initializing model...")
    model = DistilBertForSequenceClassification.from_pretrained(
        "distilbert-base-uncased",
        num_labels=len(label_encoder.classes_),
        id2label={i: label for i, label in enumerate(label_encoder.classes_)},
        label2id={label: i for i, label in enumerate(label_encoder.classes_)}
    )

    # Compute class weights for handling imbalanced data
    class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    class_weights = torch.tensor(class_weights, dtype=torch.float)
    print(f"Class weights: {class_weights}")

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted')
        acc = accuracy_score(labels, preds)
        return {"accuracy": acc, "f1": f1, "precision": precision, "recall": recall}

    # Adjust training arguments based on data size
    if total_samples < 50:
        epochs = 5  # More epochs for small datasets
        batch_size = min(4, total_samples)  # Smaller batch size
    else:
        epochs = 3
        batch_size = 8

    # Training arguments
    training_args = TrainingArguments(
        output_dir=os.path.join(save_dir, "training_output"),
        evaluation_strategy="epoch" if test_dataset is not None else "no",
        save_strategy="epoch",
        logging_strategy="epoch",
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=epochs,
        weight_decay=0.01,
        learning_rate=2e-5,
        logging_dir=os.path.join(save_dir, "logs"),
        load_best_model_at_end=test_dataset is not None,
        metric_for_best_model="f1" if test_dataset is not None else None,
        greater_is_better=True,
        save_total_limit=2,
        dataloader_drop_last=False,
    )

    # Initialize trainer
    from transformers import EarlyStoppingCallback

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics if test_dataset is not None else None,
        data_collator=data_collator,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)] if test_dataset is not None else None
    )

    # Train model
    print("Starting training...")
    trainer.train()

    # Evaluate final model if we have test data
    eval_results = {}
    if test_dataset is not None:
        print("Evaluating final model...")
        eval_results = trainer.evaluate()
        print(f"Final evaluation results: {eval_results}")
    else:
        print("No test set available - skipping evaluation")
        # Create dummy results for consistency
        eval_results = {
            'eval_accuracy': 0.0,
            'eval_f1': 0.0,
            'eval_precision': 0.0,
            'eval_recall': 0.0
        }

    # Save model and tokenizer
    print(f"Saving model to {save_dir}")
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    
    # Save label encoder
    with open(os.path.join(save_dir, "label_encoder.pkl"), "wb") as f:
        pickle.dump(label_encoder, f)
    
    print(f"Model saved successfully to {save_dir}")
    
    return model, tokenizer, label_encoder, eval_results

def main():
    """Main training function"""
    try:
        # Load data
        df = load_and_preprocess_data()
        
        # Check if we have enough data to proceed
        if len(df) < 4:
            print("⚠️  Warning: Very little training data available. Consider adding more samples.")
            print("Training will proceed but model quality may be poor.")
        
        # Train topic classifier
        print("\n" + "="*60)
        print("TRAINING TOPIC CLASSIFIER")
        print("="*60)
        model_topic, tokenizer_topic, le_topic, eval_topic = prepare_data(df, 'Topic Labels', SAVE_DIR_TOPIC)
        
        # Train document type classifier
        print("\n" + "="*60)
        print("TRAINING DOCUMENT TYPE CLASSIFIER")
        print("="*60)
        model_doc, tokenizer_doc, le_doc, eval_doc = prepare_data(df, 'Type Labels', SAVE_DIR_TYPE)
        
        # Summary
        print("\n" + "="*60)
        print("TRAINING COMPLETE - SUMMARY")
        print("="*60)
        print(f"Topic Classifier Performance:")
        print(f"  - Accuracy: {eval_topic.get('eval_accuracy', 'N/A')}")
        print(f"  - F1 Score: {eval_topic.get('eval_f1', 'N/A')}")
        print(f"  - Precision: {eval_topic.get('eval_precision', 'N/A')}")
        print(f"  - Recall: {eval_topic.get('eval_recall', 'N/A')}")
        
        print(f"\nDocument Type Classifier Performance:")
        print(f"  - Accuracy: {eval_doc.get('eval_accuracy', 'N/A')}")
        print(f"  - F1 Score: {eval_doc.get('eval_f1', 'N/A')}")
        print(f"  - Precision: {eval_doc.get('eval_precision', 'N/A')}")
        print(f"  - Recall: {eval_doc.get('eval_recall', 'N/A')}")
        
        print(f"\nModels saved to:")
        print(f"  - Topic: {SAVE_DIR_TOPIC}")
        print(f"  - Document Type: {SAVE_DIR_TYPE}")
        
    except Exception as e:
        print(f"Training failed: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()