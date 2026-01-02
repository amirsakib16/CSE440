import { useState } from 'react';
import './App.css';

function App() {
  const [inputText, setInputText] = useState('');
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  // Smart API URL detection - works in both dev and production
  const getApiUrl = () => {
    // In production (same domain), use relative URLs
    if (window.location.hostname === 'cse440-nlpqna.onrender.com') {
      return '';
    }
    // In development, use full URL
    return import.meta.env.VITE_API_URL || 'http://localhost:5000';
  };

  const API_URL = getApiUrl();

  const handlePredict = async () => {
    if (!inputText.trim()) {
      setError('Please enter some text');
      return;
    }

    setLoading(true);
    setError('');
    setResult(null);

    try {
      const response = await fetch(`${API_URL}/predict`, {
        method: 'POST',
        headers: { 
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ text: inputText }),
      });

      if (!response.ok) {
        const errData = await response.json().catch(() => ({}));
        throw new Error(errData.error || `Server error: ${response.status}`);
      }

      const data = await response.json();
      setResult(data);
    } catch (err) {
      let errorMessage = 'Error connecting to server.';
      
      if (err.message.includes('Failed to fetch')) {
        errorMessage = 'Cannot connect to server. Please check if the backend is running.';
      } else if (err.message) {
        errorMessage = err.message;
      }
      
      setError(errorMessage);
      console.error('Prediction error:', err);
    } finally {
      setLoading(false);
    }
  };

  const handleKeyPress = (e) => {
    // Submit on Ctrl+Enter or Cmd+Enter
    if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) {
      handlePredict();
    }
  };

  return (
    <div className="app-container">
      {/* Animated background elements */}
      <div className="bg-elements">
        <div className="blob blob-1"></div>
        <div className="blob blob-2"></div>
        <div className="blob blob-3"></div>
      </div>

      <div className="content-wrapper">
        <div className="glass-card">
          <div className="header">
            <h1 className="title-gradient">Q&A Topic Classifier</h1>
            <p className="subtitle">Powered by advanced AI • Real-time analysis</p>
          </div>

          <div className="input-group">
            <textarea
              value={inputText}
              onChange={(e) => setInputText(e.target.value)}
              onKeyDown={handleKeyPress}
              placeholder="Enter your question here... (Ctrl+Enter to submit)"
              rows="6"
              className="custom-textarea"
              disabled={loading}
            />
            <div className="input-glow"></div>
          </div>

          <div className="char-count">
            {inputText.length} characters
            {inputText.length > 0 && (
              <span className="hint"> • Press Ctrl+Enter to analyze</span>
            )}
          </div>

          <button 
            onClick={handlePredict} 
            disabled={!inputText.trim() || loading}
            className="predict-btn"
          >
            <div className="btn-glow"></div>
            <div className="btn-content">
              {loading ? (
                <>
                  <div className="spinner"></div>
                  <span>Analyzing...</span>
                </>
              ) : (
                <>
                  <svg className="icon" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M13 10V3L4 14h7v7l9-11h-7z" />
                  </svg>
                  <span>Predict Class</span>
                </>
              )}
            </div>
          </button>

          {result && (
            <div className="result-container animate-fadeIn">
              <div className="result-card">
                <div className="result-card-inner">
                  <div className="result-header">
                    <div className="status-indicator">
                      <div className="dot"></div>
                      <h2>Prediction Result</h2>
                    </div>
                    <svg className="check-icon" fill="currentColor" viewBox="0 0 20 20">
                      <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clipRule="evenodd" />
                    </svg>
                  </div>
                  
                  <div className="result-body">
                    <div>
                      <p className="label">Classification</p>
                      <p className="class-text">{result.class}</p>
                    </div>
                    
                    <div>
                      <p className="label">Confidence Score</p>
                      <div className="progress-wrapper">
                        <div className="progress-bar-bg">
                          <div 
                            className="progress-bar-fill"
                            style={{ 
                              width: result.confidence_raw 
                                ? `${result.confidence_raw * 100}%` 
                                : result.confidence 
                            }}
                          ></div>
                        </div>
                        <span className="confidence-value">
                          {result.confidence}
                        </span>
                      </div>
                    </div>

                    {result.processed_tokens && (
                      <div className="stats">
                        <span className="stat-item">
                          📝 {result.text_length} chars
                        </span>
                        <span className="stat-item">
                          🔤 {result.processed_tokens} tokens
                        </span>
                      </div>
                    )}
                  </div>
                </div>
              </div>
            </div>
          )}

          {error && (
            <div className="error-container animate-fadeIn">
              <div className="error-card">
                <svg className="error-icon" fill="currentColor" viewBox="0 0 20 20">
                  <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z" clipRule="evenodd" />
                </svg>
                <div>
                  <p className="error-title">Error</p>
                  <p className="error-message">{error}</p>
                </div>
              </div>
            </div>
          )}
        </div>

        <div className="floating-particles">
          <div className="particle p1"></div>
          <div className="particle p2"></div>
          <div className="particle p3"></div>
        </div>
      </div>
    </div>
  );
}

export default App;