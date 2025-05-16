import React, { useState } from 'react';
import { BrowserRouter as Router, Routes, Route, Link } from 'react-router-dom';
import './App.css';
import './styles.css';
import UploadScreen from './components/UploadScreen';
import ResultsScreen from './components/ResultsScreen';

function App() {
  const [results, setResults] = useState([]);
  const [documents, setDocuments] = useState([]);

  return (
    <Router>
      <div className="App">
        <header className="App-header">
          <h1>Document Classifier</h1>
          <nav>
            <ul>
              <li>
                <Link to="/">Upload Documents</Link>
              </li>
              <li>
                <Link to="/results">Classification Results</Link>
              </li>
            </ul>
          </nav>
        </header>
        <main>
          <Routes>
            <Route 
              path="/" 
              element={
                <UploadScreen 
                  setResults={setResults} 
                  setDocuments={setDocuments} 
                />
              } 
            />
            <Route 
              path="/results" 
              element={
                <ResultsScreen 
                  results={results} 
                  setResults={setResults} 
                  documents={documents}
                />
              } 
            />
          </Routes>
        </main>
      </div>
    </Router>
  );
}

export default App;
