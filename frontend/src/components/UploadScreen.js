import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import axios from 'axios';

const API_URL = 'http://localhost:8000/api';

const UploadScreen = ({ setResults, setDocuments }) => {
  const [files, setFiles] = useState([]);
  const [uploading, setUploading] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [error, setError] = useState('');
  const [success, setSuccess] = useState('');
  const navigate = useNavigate();

  const handleFileChange = (e) => {
    if (e.target.files) {
      const fileArray = Array.from(e.target.files);
      setFiles(fileArray);
    }
  };

  const handleDragOver = (e) => {
    e.preventDefault();
  };

  const handleDrop = (e) => {
    e.preventDefault();
    if (e.dataTransfer.files) {
      const fileArray = Array.from(e.dataTransfer.files);
      setFiles(fileArray);
    }
  };

  const clearMessages = () => {
    setError('');
    setSuccess('');
  };

  const handleUpload = async (classificationType) => {
    if (files.length === 0) {
      setError('Please select at least one file to upload.');
      return;
    }

    clearMessages();
    setUploading(true);
    setUploadProgress(0);

    try {
      const uploadedDocs = [];
      
      // Upload each file
      for (let i = 0; i < files.length; i++) {
        const file = files[i];
        const formData = new FormData();
        formData.append('file', file);
        formData.append('file_name', file.name);

        // Upload the document
        const uploadResponse = await axios.post(`${API_URL}/documents/`, formData, {
          headers: {
            'Content-Type': 'multipart/form-data'
          },
          onUploadProgress: (progressEvent) => {
            const percentCompleted = Math.round((progressEvent.loaded * 100) / progressEvent.total);
            setUploadProgress(percentCompleted);
          }
        });

        // Store the uploaded document
        uploadedDocs.push(uploadResponse.data);

        // Classify the document
        const classifyResponse = await axios.post(
          `${API_URL}/documents/${uploadResponse.data.id}/classify/`,
          { classification_type: classificationType },
          {
            headers: {
              'Content-Type': 'application/json'
            }
          }
        );

        // Add to results
        if (classifyResponse.data) {
          setResults(prevResults => [...prevResults, classifyResponse.data]);
        }
      }

      // Update the documents list
      setDocuments(uploadedDocs);

      // Show success message
      setSuccess(`Successfully uploaded and classified ${files.length} document(s) by ${classificationType}.`);
      
      // Reset files
      setFiles([]);
      
      // Navigate to results
      setTimeout(() => {
        navigate('/results');
      }, 2000);
    } catch (err) {
      console.error('Upload error:', err);
      setError(err.response?.data?.error || 'Error uploading documents. Please try again.');
    } finally {
      setUploading(false);
    }
  };

  return (
    <div className="upload-screen">
      <div className="card">
        <h2>Upload Documents for Classification</h2>
        <p>Upload documents (.txt, .pdf, .docx) to classify them by topic or document type.</p>

        {error && <div className="alert alert-error">{error}</div>}
        {success && <div className="alert alert-success">{success}</div>}

        <div
          className="upload-area"
          onDragOver={handleDragOver}
          onDrop={handleDrop}
          onClick={() => document.getElementById('file-input').click()}
        >
          <div className="upload-icon">📁</div>
          <p>Drag and drop files here, or click to select files</p>
          <input
            id="file-input"
            type="file"
            multiple
            onChange={handleFileChange}
            style={{ display: 'none' }}
            accept=".txt,.pdf,.docx,.doc"
          />
        </div>

        {files.length > 0 && (
          <div className="selected-files mt-3">
            <h3>Selected Files ({files.length})</h3>
            <ul>
              {files.map((file, index) => (
                <li key={index}>
                  {file.name} ({(file.size / 1024).toFixed(2)} KB)
                </li>
              ))}
            </ul>
          </div>
        )}

        {uploading ? (
          <div className="uploading mt-3">
            <div className="progress">
              <div className="progress-bar" style={{ width: `${uploadProgress}%` }}></div>
            </div>
            <p>Uploading... {uploadProgress}%</p>
          </div>
        ) : (
          <div className="classification-options mt-3">
            <h3>Choose Classification Type</h3>
            <div className="button-group">
              <button
                className="btn btn-primary"
                onClick={() => handleUpload('topic')}
                disabled={files.length === 0}
              >
                Classify by Topic
              </button>
              <button
                className="btn btn-secondary"
                onClick={() => handleUpload('document_type')}
                disabled={files.length === 0}
                style={{ marginLeft: '10px' }}
              >
                Classify by Document Type
              </button>
            </div>
          </div>
        )}
      </div>

      <div className="card">
        <h2>Train Your Model</h2>
        <p>
          To train the model with your own data, you can upload a CSV or JSON file with training examples.
          The file should have columns: <strong>content</strong>, <strong>topic_label</strong>, and <strong>document_type_label</strong>.
        </p>
        <p>
          Upload this file through the Django admin interface at <a href="http://localhost:8000/admin/api/trainingdata/" target="_blank" rel="noopener noreferrer">http://localhost:8000/admin/api/trainingdata/</a>
        </p>
        <p>
          Alternatively, you can use the bulk upload API endpoint:
          <code>POST http://localhost:8000/api/training-data/bulk_upload/</code>
        </p>
        <p>
          After uploading, train the model using:
          <code>POST http://localhost:8000/api/training-data/train/</code>
        </p>
      </div>
    </div>
  );
};

export default UploadScreen; 