import { useState } from 'react';
import './App.css';

function App() {
  const [inputText, setInputText] = useState('');
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  // Added 'e' parameter to prevent default form/button behavior
  const handlePredict = async (e) => {
    if (e) e.preventDefault(); 
    
    setLoading(true);
    setError('');
    setResult(null);
    
    try {
      // 1. MUST use https for Render
      // 2. MUST include the trailing slash /
      const response = await fetch('https://v0p22299222nlp.onrender.com/', {
        method: 'POST',
        headers: { 
          'Content-Type': 'application/json',
          'Accept': 'application/json' 
        },
        body: JSON.stringify({ text: inputText }),
      });
      
      // If the server returns 405, it means it's still seeing a GET
      if (response.status === 405) {
        throw new Error('Server received GET instead of POST. Check trailing slash.');
      }

      if (!response.ok) throw new Error('Prediction failed');
      
      const data = await response.json();
      setResult(data);
    } catch (err) {
      setError('Error: ' + err.message);
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="app-container">
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
              placeholder="Enter your question here..."
              rows="6"
              className="custom-textarea"
            />
            <div className="input-glow"></div>
          </div>

          <div className="char-count">{inputText.length} characters</div>

          <button 
            onClick={handlePredict} 
            disabled={!inputText || loading}
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
                            style={{ width: result.confidence }}
                          ></div>
                        </div>
                        <span className="confidence-value">{result.confidence}</span>
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          )}

          {error && (
            <div className="error-container animate-fadeIn">
              <div className="error-card">
                <p>{error}</p>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

export default App;