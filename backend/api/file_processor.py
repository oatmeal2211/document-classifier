import os
import re
import PyPDF2
import docx
import chardet

def detect_encoding(file_path):
    """Detect encoding of a file."""
    with open(file_path, 'rb') as file:
        raw_data = file.read()
        result = chardet.detect(raw_data)
        return result['encoding']

def extract_text_from_file(file_path):
    """
    Extract text from various file types.
    Supported formats: .txt, .pdf, .docx
    """
    file_extension = os.path.splitext(file_path)[1].lower()
    
    try:
        # Text files
        if file_extension == '.txt':
            encoding = detect_encoding(file_path)
            with open(file_path, 'r', encoding=encoding) as f:
                return f.read()
        
        # PDF files
        elif file_extension == '.pdf':
            text = ""
            with open(file_path, 'rb') as f:
                pdf_reader = PyPDF2.PdfReader(f)
                for page_num in range(len(pdf_reader.pages)):
                    text += pdf_reader.pages[page_num].extract_text() + "\n"
            return text
        
        # Word documents
        elif file_extension == '.docx':
            doc = docx.Document(file_path)
            return "\n".join([paragraph.text for paragraph in doc.paragraphs])
        
        # Other file types - just return file name for now
        else:
            return f"Unsupported file format: {os.path.basename(file_path)}"
            
    except Exception as e:
        return f"Error extracting text: {str(e)}"

def clean_text(text):
    """Basic text cleaning."""
    if not text:
        return ""
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Remove special characters that are not useful for classification
    text = re.sub(r'[^\w\s.,;:!?\'"-]', '', text)
    
    return text.strip() 