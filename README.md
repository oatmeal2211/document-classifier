# Document Classifier

A web application for classifying documents by topic and document type. Upload your documents and the application will classify them using machine learning models.

## Features

- Document upload and classification by topic or document type
- Support for various document formats (PDF, DOCX, TXT, etc.)
- Interactive UI for viewing classification results
- Ability to train the model with your own labeled data

## Setup Instructions

### Backend Setup

1. Create and activate a virtual environment:
   ```
   cd backend
   python -m venv env
   source env/bin/activate  # On Windows: env\Scripts\activate
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Apply migrations to set up the database:
   ```
   python manage.py migrate
   ```

4. Create a superuser for admin access:
   ```
   python manage.py createsuperuser
   ```

5. Start the Django development server:
   ```
   python manage.py runserver
   ```

### Frontend Setup

1. Navigate to the frontend directory:
   ```
   cd frontend
   ```

2. Install npm dependencies:
   ```
   npm install
   ```

3. Start the development server:
   ```
   npm start
   ```

## Using the Application

1. Open http://localhost:3000 in your browser to access the frontend.
2. Use the "Upload Documents" page to upload files for classification.
3. Choose whether to classify by topic or document type.
4. View the classification results on the "Classification Results" page.

## Training the Model with Your Own Data

To train the classifier with your own data, you have two options:

### Option 1: Using the Admin Interface

1. Access the Django admin at http://localhost:8000/admin
2. Log in with the superuser credentials you created
3. Navigate to "Training Data" and add entries with:
   - Content: the text content of the document
   - Topic Label: the topic category label
   - Document Type Label: the document type label

4. After adding training data, make a POST request to:
   ```
   http://localhost:8000/api/training-data/train/
   ```

### Option 2: Bulk Upload

You can upload a CSV or JSON file with training examples:

1. Prepare a CSV file with the following columns:
   - `content`: The document text content
   - `topic_label`: The topic category
   - `document_type_label`: The document type

2. Upload the file via the API endpoint:
   ```
   POST http://localhost:8000/api/training-data/bulk_upload/
   ```

3. Train the model using:
   ```
   POST http://localhost:8000/api/training-data/train/
   ```

## API Endpoints

- `/api/documents/` - CRUD operations for documents
- `/api/documents/{id}/classify/` - Classify a document
- `/api/classification-results/` - Get classification results
- `/api/training-data/` - CRUD operations for training data
- `/api/training-data/bulk_upload/` - Bulk upload training data
- `/api/training-data/train/` - Train the classifier models

## Technologies Used

- **Backend**: Django, Django REST Framework, NLTK, scikit-learn
- **Frontend**: React, Axios, React Router
- **Document Processing**: PyPDF2, python-docx, textract