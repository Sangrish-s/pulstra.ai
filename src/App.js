import React, { useState } from 'react';
import axios from 'axios';
import './App.css';

function App() {
  const [query, setQuery] = useState('');
  const [language, setLanguage] = useState('');
  const [abstracts, setAbstracts] = useState([]);
  const [error, setError] = useState(null);

  const searchPapers = async () => {
    if (!query || !language) {
      alert('Please enter both query and language');
      return;
    }

    try {
      const response = await axios.post('http://184.72.131.74:8088/search', {
        query,
        language,
      });

      setAbstracts(response.data.abstracts);
      setError(null);
    } catch (error) {
      console.error('Error fetching papers:', error.response ? error.response.data : error.message);
      setError('Error fetching papers. Please try again.');
    }
  };

  return (
    <div className="App">
      <header className="App-header">
        <h1>ArXiv Paper Search</h1>
        <div>
          <label htmlFor="query">Query:</label>
          <input
            type="text"
            id="query"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Enter your query"
          />
        </div>
        <div>
          <label htmlFor="language">Language:</label>
          <input
            type="text"
            id="language"
            value={language}
            onChange={(e) => setLanguage(e.target.value)}
            placeholder="Enter language"
          />
        </div>
        <button onClick={searchPapers}>Search</button>
        {error && <p className="error">{error}</p>}
        <div id="results">
          {abstracts.length > 0 ? (
            abstracts.map((abstract, index) => (
              <div key={index} className="paper">
                {abstract}
              </div>
            ))
          ) : (
            <p>No papers found.</p>
          )}
        </div>
      </header>
    </div>
  );
}

export default App;
