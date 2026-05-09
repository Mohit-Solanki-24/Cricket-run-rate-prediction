import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import pickle
import os

# Create directories if they don't exist
os.makedirs('data', exist_ok=True)
os.makedirs('models', exist_ok=True)

print("Starting Data Generation...")
# 1. Generate Synthetic Data
# We use standard IPL teams
teams = ['CSK', 'MI', 'RCB', 'KKR', 'DC', 'PBKS', 'RR', 'SRH', 'LSG', 'GT']

data = []
np.random.seed(42)
for _ in range(10000):
    batting_team = np.random.choice(teams)
    bowling_team = np.random.choice([t for t in teams if t != batting_team])
    
    current_over = np.random.randint(5, 16) # Current over between 5 and 15
    current_score = int(current_over * np.random.uniform(6.0, 10.0))
    wickets_lost = np.random.randint(0, min(10, int(current_over * 0.8) + 1))
    
    target_over = np.random.randint(current_over + 1, 21) # Target over up to 20
    
    # Calculate realistic future runs based on remaining resources
    overs_remaining = target_over - current_over
    wickets_in_hand = 10 - wickets_lost
    run_rate_future = np.random.uniform(6.0, 12.0) * (wickets_in_hand / 10.0 + 0.5)
    future_runs = int(overs_remaining * run_rate_future)
    target_score = current_score + future_runs
    
    data.append([batting_team, bowling_team, current_over, current_score, wickets_lost, target_over, target_score])

df = pd.DataFrame(data, columns=['batting_team', 'bowling_team', 'current_over', 'current_score', 'wickets_lost', 'target_over', 'target_score'])
df.to_csv('data/synthetic_ipl_data.csv', index=False)
print("Saved synthetic dataset to data/synthetic_ipl_data.csv")
print(f"Dataset Shape: {df.shape}")

# 2. Train Model
X = df.drop('target_score', axis=1)
y = df['target_score']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Categorical features for OneHotEncoding
categorical_features = ['batting_team', 'bowling_team']
categorical_transformer = OneHotEncoder(handle_unknown='ignore')

# Preprocessor applies OneHotEncoding to categorical and leaves other numeric features as-is
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', categorical_transformer, categorical_features)
    ],
    remainder='passthrough'
)

# Model Pipeline combines preprocessor and the regressor
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42))
])

print("Training RandomForest Regressor...")
model.fit(X_train, y_train)
score = model.score(X_test, y_test)
print(f"Model R^2 Score on Test Data: {score:.4f}")

# 3. Save Model
with open('models/ipl_model.pkl', 'wb') as f:
    pickle.dump(model, f)
print("Saved trained model to models/ipl_model.pkl")
print("Training Complete!")
