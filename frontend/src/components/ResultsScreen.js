import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import axios from 'axios';

const API_URL = 'http://localhost:8000/api';

const ResultsScreen = ({ results, setResults, documents }) => {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [activeTab, setActiveTab] = useState('topic');
  const [allDocuments, setAllDocuments] = useState([]);
  const [allResults, setAllResults] = useState([]);
  const navigate = useNavigate();

  useEffect(() => {
    if (!results.length || !documents.length) {
      fetchDocumentsAndResults();
    } else {
      setAllResults(results);
      setAllDocuments(documents);
    }
  }, [results, documents]);

  const fetchDocumentsAndResults = async () => {
    setLoading(true);
    setError('');
    
    try {
      // Fetch all documents
      const docsResponse = await axios.get(`${API_URL}/documents/`);
      setAllDocuments(docsResponse.data);
      
      // Fetch all classification results
      const resultsResponse = await axios.get(`${API_URL}/classification-results/`);
      setAllResults(resultsResponse.data);
      setResults(resultsResponse.data);
    } catch (err) {
      console.error('Error fetching data:', err);
      setError('Failed to load results. Please try again later.');
    } finally {
      setLoading(false);
    }
  };

  const handleClassify = async (documentId, classificationType) => {
    setLoading(true);
    setError('');
    
    try {
      // Classify the document
      const response = await axios.post(
        `${API_URL}/documents/${documentId}/classify/`,
        { classification_type: classificationType },
        {
          headers: {
            'Content-Type': 'application/json'
          }
        }
      );
      
      // Add new result
      if (response.data) {
        setAllResults(prevResults => [...prevResults, response.data]);
        setResults(prevResults => [...prevResults, response.data]);
      }
      
      // Switch to appropriate tab
      setActiveTab(classificationType);
    } catch (err) {
      console.error('Classification error:', err);
      setError(err.response?.data?.error || 'Error classifying document. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  // Filter results based on active tab
  const filteredResults = allResults.filter(result => 
    result.classification_type === activeTab
  );

  // Function to find document name from its ID
  const getDocumentName = (documentId) => {
    const document = allDocuments.find(doc => doc.id === documentId);
    return document ? document.file_name : 'Unknown Document';
  };

  // Function to create a confidence bar
  const ConfidenceBar = ({ confidence }) => {
    const percentage = (confidence * 100).toFixed(0);
    return (
      <div className="confidence-bar">
        <div 
          className="confidence-bar-fill" 
          style={{ width: `${percentage}%` }}
        ></div>
      </div>
    );
  };

  return (
    <div className="results-screen">
      <div className="card">
        <h2>Classification Results</h2>
        
        {error && <div className="alert alert-error">{error}</div>}
        
        <div className="tab-buttons">
          <button 
            className={`tab-button ${activeTab === 'topic' ? 'active' : ''}`}
            onClick={() => setActiveTab('topic')}
          >
            Topic Classification
          </button>
          <button 
            className={`tab-button ${activeTab === 'document_type' ? 'active' : ''}`}
            onClick={() => setActiveTab('document_type')}
          >
            Document Type Classification
          </button>
        </div>
        
        {loading ? (
          <div className="loading">
            <div className="spinner"></div>
          </div>
        ) : filteredResults.length > 0 ? (
          <div className="results-container">
            <table className="results-table">
              <thead>
                <tr>
                  <th>Document</th>
                  <th>{activeTab === 'topic' ? 'Topic' : 'Document Type'}</th>
                  <th>Confidence</th>
                  <th>Actions</th>
                </tr>
              </thead>
              <tbody>
                {filteredResults.map(result => (
                  <tr key={result.id}>
                    <td>{getDocumentName(result.document)}</td>
                    <td>{result.result}</td>
                    <td>
                      {(result.confidence * 100).toFixed(1)}%
                      <ConfidenceBar confidence={result.confidence} />
                    </td>
                    <td>
                      {activeTab === 'topic' ? (
                        <button 
                          className="btn btn-secondary btn-sm"
                          onClick={() => handleClassify(result.document, 'document_type')}
                        >
                          Classify Document Type
                        </button>
                      ) : (
                        <button 
                          className="btn btn-primary btn-sm"
                          onClick={() => handleClassify(result.document, 'topic')}
                        >
                          Classify Topic
                        </button>
                      )}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : (
          <div className="no-results">
            <p>No classification results available for {activeTab === 'topic' ? 'topics' : 'document types'}.</p>
          </div>
        )}
        
        <div className="actions mt-3">
          <button 
            className="btn btn-primary"
            onClick={() => navigate('/')}
          >
            Upload More Documents
          </button>
          <button 
            className="btn"
            onClick={fetchDocumentsAndResults}
            style={{ marginLeft: '10px' }}
          >
            Refresh Results
          </button>
        </div>
      </div>
    </div>
  );
};

export default ResultsScreen; 