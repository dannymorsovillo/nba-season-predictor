
import React, { useState } from 'react';
import axios from 'axios';
import './App.css';

function App() {
  const nbaImages = [
    require('./nbabackground1.jpg'),
    require('./nbabackground2.webp'),
    require('./nbabackground3.webp'),
    require('./nbabackground4.webp'),
  ];

  const [selectedConference, setSelectedConference] = useState(null);
  const[conferenceStandings, setConferenceStandings] = useState(null);
  const[loading, setLoading] = useState(false);

  const handleConferenceSelect = (conference) => {
    setSelectedConference(conference);
    fetchConferenceStandings(conference);
  };

  const fetchConferenceStandings = (conference) => {
    setLoading(true);
    axios.get(`http://localhost:5000/predict_season`)
      .then(response => {
        // Check if data exists and has the expected structure
        if (!response.data || 
            !response.data.Western_Conference || 
            !response.data.Eastern_Conference) {
          throw new Error('Invalid data structure from API');
        }
        
        const standingsData = conference === 'Western' 
          ? response.data.Western_Conference 
          : response.data.Eastern_Conference;
        
        // Additional check for the array content
        if (!Array.isArray(standingsData) || standingsData.length === 0) {
          throw new Error('No standings data available');
        }
        
        setConferenceStandings(standingsData);
        setLoading(false);
      })
      .catch(error => {
        console.error('Error fetching conference standings:', error);
        setConferenceStandings([]); // Reset to empty array on error
        setLoading(false);
      });
  };

  // If a conference is selected, you would render a new component
  if (selectedConference) {
    return (
      <div className="app-container">
        <button
          onClick={() => setSelectedConference(null)}
          className="back-button"
        >
          Back to Home
        </button>
        <h2>{selectedConference} Conference Standings</h2>
        {loading ? (
          <p>Loading...</p>
          ) : conferenceStandings ?  (
          <div className="standings-list">
            {conferenceStandings.map((team, index) => (
              <div key={index} className="standings-item">
                <p>
                  {team.Seeding}. {team.Team} WINS: {team.Wins} LOSSES: {team.Losses}
                </p>
              </div>
            ))}
          </div>
           ) : (
            <p>No standings data available</p>
        )}
      </div>
    );
  }

  return (
    <div className="app-container">
      <h1 className="welcome-header">Welcome to NBA Season Predictor</h1>
      
      <div className="image-scroll-container">
        {nbaImages.map((image, index) => (
          <img 
            key={index} 
            src={image} 
            alt={`NBA Scene ${index + 1}`} 
            className="nba-image"
          />
        ))}
      </div>

      <div className="conference-buttons">
        <button 
          className="conference-btn western-btn"
          onClick={() => handleConferenceSelect('Western')}
        >
          Western Conference Standings
        </button>
        <button 
          className="conference-btn eastern-btn"
          onClick={() => handleConferenceSelect('Eastern')}
        >
          Eastern Conference Standings
        </button>
      </div>
    </div>
  );
}

export default App;
