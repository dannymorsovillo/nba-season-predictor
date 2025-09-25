import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os

def train_model(csv_path, model_save_path):
    # Creating directories if they don't exist
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    
    # Loading data
    data = pd.read_csv(csv_path)
    
    # Preprocessing
    # Drop non-numeric columns that shouldn't be used as features
    columns_to_drop = ['Rk', 'G', 'Date', 'Unnamed: 5', 'Unnamed: 7', 'Tm_points', 'Opp_points', 'W/L']
    data = data.drop(columns=[col for col in columns_to_drop if col in data.columns])
    
    # Convert categorical columns to numeric codes
    categorical_cols = data.select_dtypes(include=['object', 'category']).columns
    for col in categorical_cols:
        data[col] = data[col].astype('category').cat.codes
    
    # Fill missing values - numeric columns with mean, categorical with mode
    numeric_cols = data.select_dtypes(include=['number']).columns
    for col in numeric_cols:
        data[col] = pd.to_numeric(data[col], errors='coerce')  # Convert to numeric, coercing errors
        data[col] = data[col].fillna(data[col].mean())
    
    # Define features and target (assuming 'W/L' is the target column)
    # We need to load it again since we dropped it earlier
    target_data = pd.read_csv(csv_path)
    y = target_data['Rslt'].apply(lambda x: 1 if x == 'W' else 0)
    X = data
    
    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train model 
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluating
    y_pred = model.predict(X_test)
    print(f'Accuracy: {accuracy_score(y_test, y_pred):.2f}')
    print(classification_report(y_test, y_pred))
    
    # Save model and feature names
    print("Saving model...")
    joblib.dump(model, model_save_path)
    joblib.dump(list(X.columns), os.path.join(os.path.dirname(model_save_path), 'feature_names.joblib'))
    print(f"Model saved to {os.path.abspath(model_save_path)}")
    return model

if __name__ == "__main__":
    # Specify full paths to avoid directory issues
    base_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(base_dir, 'nba-gamelogs-2014-2023.csv')
    model_save_path = os.path.join(base_dir, 'model.pkl')  # Save in same directory for now

    # Verify paths
    print(f"CSV path: {csv_path}")
    print(f"Model save path: {model_save_path}")
    
    # Check if CSV exists
    if not os.path.exists(csv_path):
        print(f"Error: CSV file not found at {csv_path}")
    else:
        # Train and save the model
        train_model(csv_path, model_save_path)