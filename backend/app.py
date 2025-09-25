# app.py
from flask import Flask, jsonify, request
from flask_cors import CORS
import pandas as pd
import numpy as np
import joblib
import os
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Get the directory of the current script
base_dir = os.path.dirname(os.path.abspath(__file__))

# Define file paths
csv_path = os.path.join(base_dir, 'nba-gamelogs-2014-2023.csv')
model_path = os.path.join(base_dir, 'model.pkl')
feature_names_path = os.path.join(base_dir, 'feature_names.joblib')

# Conference mapping
teams_to_conference = {
    "BOS": "East", "BKN": "East", "NYK": "East", "PHI": "East", "TOR": "East",  # Atlantic
    "CHI": "East", "CLE": "East", "DET": "East", "IND": "East", "MIL": "East",  # Central
    "ATL": "East", "CHA": "East", "MIA": "East", "ORL": "East", "WAS": "East",  # Southeast
    "DEN": "West", "MIN": "West", "OKC": "West", "POR": "West", "UTA": "West",  # Northwest
    "GSW": "West", "LAC": "West", "LAL": "West", "PHX": "West", "SAC": "West",  # Pacific
    "DAL": "West", "HOU": "West", "MEM": "West", "NOP": "West", "SAS": "West"   # Southwest
}

# Load data and model
try:
    logger.info(f"Loading data from {csv_path}")
    nba_df = pd.read_csv(csv_path)
    nba_df['Conference'] = nba_df['Team'].map(teams_to_conference)
    
    logger.info(f"Loading model from {model_path}")
    model = joblib.load(model_path)
    
    # Load feature names if available
    if os.path.exists(feature_names_path):
        feature_names = joblib.load(feature_names_path)
        logger.info(f"Loaded {len(feature_names)} feature names")
    else:
        feature_names = None
        logger.warning("Feature names file not found")
    
except FileNotFoundError as e:
    logger.error(f"Error loading files: {str(e)}")
    logger.error("Make sure to run data collection and model training first")
    # We'll continue and handle errors in the endpoints
    nba_df = None
    model = None
    feature_names = None

# Helper function to calculate seeding
def calculate_seeding(conference_teams_data):
    """Calculate team seeding based on predicted wins"""
    sorted_teams = conference_teams_data.sort_values(by='Predicted_Wins', ascending=False)
    sorted_teams['Seeding'] = range(1, len(sorted_teams) + 1)
    return sorted_teams[['Team', 'Seeding']].set_index('Team')['Seeding'].to_dict()

# Root endpoint
@app.route('/')
def home():
    """Root endpoint to check if API is running"""
    return jsonify({"message": "NBA Predictor Backend Running", "status": "ok"})

# Get all teams
@app.route('/teams', methods=['GET'])
def get_all_teams():
    """Return list of all teams in the dataset"""
    if nba_df is None:
        return jsonify({"error": "Dataset not loaded"}), 500
    
    teams = nba_df['Team'].unique().tolist()
    return jsonify({"teams": teams})

# Get team data
@app.route('/team/<team_name>', methods=['GET'])
def get_team_data(team_name):
    """Return data for a specific team"""
    if nba_df is None:
        return jsonify({"error": "Dataset not loaded"}), 500
    
    team_data = nba_df[nba_df['Team'] == team_name.upper()]
    if team_data.empty:
        return jsonify({"error": f"No data found for team {team_name.upper()}"}), 404
    
    return jsonify({
        "team": team_name.upper(),
        "conference": teams_to_conference.get(team_name.upper(), "Unknown"),
        "data": team_data.to_dict(orient="records")
    })

# Predict single game outcome
@app.route('/predict', methods=['POST'])
def predict_game():
    """Predict the outcome of a single game"""
    if model is None:
        return jsonify({"error": "Model not loaded"}), 500
    
    try:
        # Get the input data from the request
        input_data = request.get_json()
        logger.info(f"Received prediction request: {input_data}")
        
        # Convert input data to DataFrame
        input_df = pd.DataFrame([input_data])
        
        # Preprocess the input data
        # Fill missing numeric values
        numeric_cols = input_df.select_dtypes(include=['number']).columns
        if len(numeric_cols) > 0:
            input_df[numeric_cols] = input_df[numeric_cols].fillna(input_df[numeric_cols].mean())
        
        # Convert categorical columns
        input_df['Team'] = input_df['Team'].astype('category').cat.codes
        input_df['Opp'] = input_df['Opp'].astype('category').cat.codes
        
        # Ensure feature order matches training
        if feature_names is not None:
            # Add missing columns with default values
            for col in feature_names:
                if col not in input_df.columns:
                    input_df[col] = 0
            # Select and order columns to match training
            input_df = input_df[feature_names]
        
        # Make prediction
        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0][1]  # Probability of winning
        
        # Return prediction
        result = {
            "prediction": int(prediction),  # 1 for win, 0 for loss
            "win_probability": float(probability),
            "outcome": "Win" if prediction == 1 else "Loss"
        }
        
        return jsonify(result)
    
    except Exception as e:
        logger.error(f"Error in prediction: {str(e)}")
        return jsonify({"error": str(e)}), 400

# Predict season results
@app.route('/predict_season', methods=['GET'])
@app.route('/predict_season', methods=['GET'])
@app.route('/predict_season', methods=['GET'])
@app.route('/predict_season', methods=['GET'])
@app.route('/predict_season', methods=['GET'])
def predict_season():
    try:
        # Get requested season (default: 2023)
        requested_season = request.args.get('season', '2023')
        
        # Convert ALL seasons to integers (handles any CSV format)
        nba_df['Season'] = pd.to_numeric(nba_df['Season'], errors='coerce').dropna().astype(int)
        
        # Convert request to integer
        try:
            season_num = int(requested_season)
        except ValueError:
            return jsonify({
                "error": f"Season must be a number. Received: '{requested_season}'",
                "available_seasons": nba_df['Season'].unique().tolist()
            }), 400
        
        # Find matches
        season_data = nba_df[nba_df['Season'] == season_num]
        
        if season_data.empty:
            return jsonify({
                "error": f"No data for season {season_num}",
                "available_seasons": nba_df['Season'].unique().tolist(),
                "debug": {
                    "your_input": requested_season,
                    "csv_sample": nba_df['Season'].head().tolist()
                }
            }), 404
        
        # ... rest of your function ...
        
        # Rest of your function remains unchanged...
        
        # Rest of your existing function...
        # Get unique teams
        teams = season_data['Team'].unique()
        
        # Create team stats by averaging all games
        team_stats = []
        for team in teams:
            team_games = season_data[season_data['Team'] == team]
            
            # Average stats for this team
            avg_stats = team_games.mean(numeric_only=True).to_dict()
            avg_stats['Team'] = team
            avg_stats['Games'] = len(team_games)
            
            # Store team stats
            team_stats.append(avg_stats)
        
        team_stats_df = pd.DataFrame(team_stats)
        
        # Preprocess for prediction
        team_stats_df['Team'] = team_stats_df['Team'].astype('category').cat.codes
        
        # Add opponent for placeholder (doesn't affect season prediction)
        team_stats_df['Opp'] = 0
        
        # Ensure all required features are present
        if feature_names is not None:
            for col in feature_names:
                if col not in team_stats_df.columns:
                    team_stats_df[col] = 0
            pred_features = team_stats_df[feature_names]
        else:
            # If feature names not known, drop known non-feature columns
            pred_features = team_stats_df.drop(columns=['Games'], errors='ignore')
        
        # Predict win/loss for each team
        predictions = model.predict(pred_features)
        probabilities = model.predict_proba(pred_features)[:, 1]  # Win probabilities
        
        # Create results dataframe
        results_df = pd.DataFrame({
            'Team': teams,
            'Predicted_Wins': predictions * 82,  # Scale to 82 games
            'Win_Probability': probabilities,
            'Conference': [teams_to_conference.get(team, "Unknown") for team in teams]
        })
        
        # Calculate losses
        results_df['Predicted_Losses'] = 82 - results_df['Predicted_Wins']
        
        # Separate conferences
        eastern_teams = results_df[results_df['Conference'] == 'East']
        western_teams = results_df[results_df['Conference'] == 'West']
        
        # Calculate seeding
        eastern_seeding = calculate_seeding(eastern_teams)
        western_seeding = calculate_seeding(western_teams)
        
        # Format results
        eastern_results = []
        for _, row in eastern_teams.iterrows():
            eastern_results.append({
                "Team": row['Team'],
                "Wins": int(row['Predicted_Wins']),
                "Losses": int(row['Predicted_Losses']),
                "WinPct": float(row['Win_Probability']),
                "Seeding": eastern_seeding[row['Team']]
            })
        
        western_results = []
        for _, row in western_teams.iterrows():
            western_results.append({
                "Team": row['Team'],
                "Wins": int(row['Predicted_Wins']),
                "Losses": int(row['Predicted_Losses']),
                "WinPct": float(row['Win_Probability']),
                "Seeding": western_seeding[row['Team']]
            })
        
        # Return results
        return jsonify({
            "Eastern_Conference": eastern_results,
            "Western_Conference": western_results
        })
    
    except Exception as e:
        logger.error(f"Error in season prediction: {str(e)}")
        return jsonify({"error": str(e)}), 400

# === DEBUG ROUTE ===
@app.route('/debug_data')
def debug_data():
    return jsonify({
        "total_records": len(nba_df),
        "seasons_counts": nba_df['Season'].value_counts().to_dict(),
        "sample_data": nba_df.iloc[0].to_dict()
    })

# Main entry point
if __name__ == '__main__':
    # Check if data and model exist
    if not os.path.exists(csv_path):
        logger.warning(f"Data file not found at {csv_path}")
        logger.warning("Run data collection script first")
    
    if not os.path.exists(model_path):
        logger.warning(f"Model file not found at {model_path}")
        logger.warning("Run model training script first")
    
    app.run(debug=True, port=5000)