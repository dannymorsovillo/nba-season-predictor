import React, { useState } from 'react';
import axios from 'axios';

const SeasonResults = () => {
  const [seasonResults, setSeasonResults] = useState(null);
  const [loading, setLoading] = useState(false);

  const fetchSeasonResults = async () => {
    setLoading(true);
    try {
      const response = await axios.get('http://127.0.0.1:5000/predict_season');
      setSeasonResults(response.data);
    } catch (error) {
      console.error('Error fetching season results:', error);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="season-results">
      <h2>Season Results</h2>
      <button onClick={fetchSeasonResults} disabled={loading}>
        {loading ? 'Loading...' : 'View Season Results'}
      </button>

      {seasonResults && (
        <div className="results-table">
          <h3>Predicted Standings</h3>
          <table>
            <thead>
              <tr>
                <th>Team</th>
                <th>Wins</th>
                <th>Losses</th>
                <th>Seeding</th>
              </tr>
            </thead>
            <tbody>
              {seasonResults.map((team, index) => (
                <tr key={index}>
                  <td>{team.Team}</td>
                  <td>{team.Wins}</td>
                  <td>{team.Losses}</td>
                  <td>{team.Points}</td>
                  <td>{team.Rebounds}</td>
                  <td>{team.Assists}</td>
                  <td>{team.Seeding}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
};

export default SeasonResults;