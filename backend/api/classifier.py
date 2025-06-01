"""
Enhanced document content extraction utility and document classifier.
This module extracts full text content from various document types and
provides a function to classify documents using trained Hugging Face models.
"""
import os
import pandas as pd
import PyPDF2
import docx
from pathlib import Path
import mimetypes
import chardet
import logging
import threading

# For classification
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
import torch
import pickle

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DocumentContentExtractor:
    """Extract full text content from various document formats"""
    
    def __init__(self, max_content_length=8000):
        self.max_content_length = max_content_length
        self.supported_extensions = {'.txt', '.pdf', '.docx', '.doc', '.csv'}
    
    def detect_encoding(self, file_path):
        try:
            with open(file_path, 'rb') as f:
                raw_data = f.read(10000)
                result = chardet.detect(raw_data)
                return result.get('encoding', 'utf-8')
        except Exception:
            return 'utf-8'
    
    def extract_text_file(self, file_path):
        try:
            encoding = self.detect_encoding(file_path)
            with open(file_path, 'r', encoding=encoding, errors='ignore') as f:
                content = f.read(self.max_content_length)
                return content.strip()
        except Exception as e:
            logger.warning(f"Failed to extract text from {file_path}: {e}")
            return ""
    
    def extract_pdf_content(self, file_path):
        try:
            with open(file_path, 'rb') as f:
                pdf_reader = PyPDF2.PdfReader(f)
                text_content = []
                total_chars = 0
                for page in pdf_reader.pages:
                    page_text = page.extract_text()
                    if not page_text:
                        continue
                    if total_chars + len(page_text) > self.max_content_length:
                        remaining_chars = self.max_content_length - total_chars
                        text_content.append(page_text[:remaining_chars])
                        break
                    text_content.append(page_text)
                    total_chars += len(page_text)
                return ' '.join(text_content).strip()
        except Exception as e:
            logger.warning(f"Failed to extract PDF content from {file_path}: {e}")
            return ""
    
    def extract_docx_content(self, file_path):
        try:
            doc = docx.Document(file_path)
            text_content = []
            total_chars = 0
            for paragraph in doc.paragraphs:
                para_text = paragraph.text
                if total_chars + len(para_text) > self.max_content_length:
                    remaining_chars = self.max_content_length - total_chars
                    text_content.append(para_text[:remaining_chars])
                    break
                text_content.append(para_text)
                total_chars += len(para_text)
            return ' '.join(text_content).strip()
        except Exception as e:
            logger.warning(f"Failed to extract DOCX content from {file_path}: {e}")
            return ""
    
    def clean_text(self, text):
        if not isinstance(text, str):
            return ""
        text = text.lower()
        text = " ".join(text.split())
        return text

    def extract_content(self, file_path):
        if not os.path.exists(file_path):
            logger.warning(f"File not found: {file_path}")
            return ""
        file_path = Path(file_path)
        extension = file_path.suffix.lower()
        if extension == '.txt':
            content = self.extract_text_file(file_path)
        elif extension == '.pdf':
            content = self.extract_pdf_content(file_path)
        elif extension in ['.docx', '.doc']:
            content = self.extract_docx_content(file_path)
        else:
            logger.info(f"Unsupported file type: {extension} for {file_path}")
            content = ""
        return self.clean_text(content)
    
    def process_csv_with_file_paths(self, csv_path, file_path_column='File Path', documents_base_dir='documents'):
        try:
            df = pd.read_csv(csv_path)
            logger.info(f"Loaded CSV with {len(df)} rows")
            if file_path_column not in df.columns:
                logger.warning(f"Column '{file_path_column}' not found. Available columns: {df.columns.tolist()}")
                potential_columns = [col for col in df.columns if 'path' in col.lower() or 'file' in col.lower()]
                if potential_columns:
                    file_path_column = potential_columns[0]
                    logger.info(f"Using column '{file_path_column}' instead")
                else:
                    logger.warning("No file path column found. Using title only.")
                    df['extracted_content'] = df.get('Title', '')
                    return df
            extracted_contents = []
            successful_extractions = 0
            for idx, row in df.iterrows():
                file_path = row.get(file_path_column, '')
                title = row.get('Title', '')
                if pd.isna(file_path) or file_path == '':
                    content = title
                    logger.info(f"Row {idx}: No file path, using title")
                else:
                    if not os.path.isabs(file_path):
                        full_file_path = os.path.join(documents_base_dir, file_path)
                    else:
                        full_file_path = file_path
                    extracted_content = self.extract_content(full_file_path)
                    if extracted_content:
                        content = f"{title}\n\n{extracted_content}" if title else extracted_content
                        successful_extractions += 1
                        logger.info(f"Row {idx}: Successfully extracted {len(extracted_content)} characters from {file_path}")
                    else:
                        content = title
                        logger.warning(f"Row {idx}: Failed to extract content from {file_path}, using title")
                extracted_contents.append(content)
            df['extracted_content'] = extracted_contents
            logger.info(f"Content extraction complete:")
            logger.info(f"  - Total documents: {len(df)}")
            logger.info(f"  - Successful file extractions: {successful_extractions}")
            logger.info(f"  - Using title only: {len(df) - successful_extractions}")
            return df
        except Exception as e:
            logger.error(f"Error processing CSV: {e}")
            raise
    
    def enhance_csv_with_content(self, input_csv_path, output_csv_path=None, documents_base_dir='documents'):
        if output_csv_path is None:
            backup_path = input_csv_path.replace('.csv', '_backup.csv')
            if not os.path.exists(backup_path):
                import shutil
                shutil.copy2(input_csv_path, backup_path)
                logger.info(f"Created backup at {backup_path}")
            output_csv_path = input_csv_path
        enhanced_df = self.process_csv_with_file_paths(
            input_csv_path, 
            documents_base_dir=documents_base_dir
        )
        enhanced_df.to_csv(output_csv_path, index=False)
        logger.info(f"Enhanced CSV saved to {output_csv_path}")
        print("\n" + "="*60)
        print("CONTENT EXTRACTION SUMMARY")
        print("="*60)
        print(f"Total documents processed: {len(enhanced_df)}")
        content_lengths = enhanced_df['extracted_content'].str.len()
        print(f"Content length statistics:")
        print(f"  - Average: {content_lengths.mean():.0f} characters")
        print(f"  - Median: {content_lengths.median():.0f} characters")
        print(f"  - Min: {content_lengths.min():.0f} characters")
        print(f"  - Max: {content_lengths.max():.0f} characters")
        print(f"\nSample extracted content:")
        for i in range(min(3, len(enhanced_df))):
            content = enhanced_df.iloc[i]['extracted_content']
            print(f"\nDocument {i+1}: {content[:200]}...")
        return enhanced_df

_model_cache = {}
_model_cache_lock = threading.Lock()

def load_model_and_tokenizer(classification_type):
    """
    Loads and caches the model, tokenizer, and label encoder for the given classification type.
    """
    with _model_cache_lock:
        if classification_type in _model_cache:
            return _model_cache[classification_type]

        if classification_type == 'topic':
            model_dir = os.path.join("media", "models", "topic_classifier")
        else:
            model_dir = os.path.join("media", "models", "doctype_classifier")

        if not (os.path.exists(model_dir) and os.path.exists(os.path.join(model_dir, "config.json"))):
            raise FileNotFoundError(f"Model not found at {model_dir}. Please train the model first.")

        tokenizer = DistilBertTokenizerFast.from_pretrained(model_dir)
        model = DistilBertForSequenceClassification.from_pretrained(model_dir)
        label_map_path = os.path.join(model_dir, "label_encoder.pkl")
        if os.path.exists(label_map_path):
            with open(label_map_path, "rb") as f:
                label_encoder = pickle.load(f)
        else:
            label_encoder = None

        _model_cache[classification_type] = (tokenizer, model, label_encoder)
        return tokenizer, model, label_encoder

def classify_document(text, classification_type='topic'):
    """
    Classify a document's text as either topic or document type.
    Args:
        text (str): The document text to classify.
        classification_type (str): 'topic' or 'document_type'
    Returns:
        dict: { 'result': predicted_label, 'confidence': probability }
    """
    try:
        tokenizer, model, label_encoder = load_model_and_tokenizer(classification_type)
    except Exception as e:
        return {"error": str(e)}

    inputs = tokenizer(text, truncation=True, padding=True, max_length=512, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=1)
        confidence, pred = torch.max(probs, dim=1)
        label_id = pred.item()
        confidence = confidence.item()

    if label_encoder:
        try:
            label = label_encoder.inverse_transform([label_id])[0]
        except Exception:
            label = str(label_id)
    else:
        label = str(label_id)

    return {"result": label, "confidence": confidence}

def main():
    """Main function for standalone usage"""
    import argparse
    parser = argparse.ArgumentParser(description='Extract content from documents for classification')
    parser.add_argument('csv_path', help='Path to CSV file with document metadata')
    parser.add_argument('--documents-dir', default='documents', help='Base directory containing document files')
    parser.add_argument('--output', help='Output CSV path (default: overwrite input)')
    parser.add_argument('--max-length', type=int, default=8000, help='Maximum content length per document')
    args = parser.parse_args()
    extractor = DocumentContentExtractor(max_content_length=args.max_length)
    extractor.enhance_csv_with_content(
        args.csv_path,
        args.output,
        args.documents_dir
    )

if __name__ == "__main__":
    main()