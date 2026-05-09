import streamlit as st
import pandas as pd
import pickle
import os

# Configure the Streamlit page
st.set_page_config(page_title="IPL Runs Predictor", page_icon="🏏", layout="centered")

@st.cache_resource
def load_model():
    """Load the trained machine learning model."""
    model_path = 'models/ipl_model.pkl'
    if os.path.exists(model_path):
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        return model
    else:
        return None

# Attempt to load the model
model = load_model()

# UI Layout
st.title("🏏 IPL Runs Predictor")
st.markdown("Predict the upcoming score of an IPL match based on the current situation!")

if model is None:
    st.error("Model not found! Please run `python train_model.py` first to generate the model.")
else:
    teams = ['CSK', 'MI', 'RCB', 'KKR', 'DC', 'PBKS', 'RR', 'SRH', 'LSG', 'GT']

    st.subheader("Match Status")
    col1, col2 = st.columns(2)
    
    with col1:
        batting_team = st.selectbox("Batting Team", teams, index=0)
    with col2:
        # Prevent selecting the same team
        bowling_team = st.selectbox("Bowling Team", [t for t in teams if t != batting_team], index=0)

    st.subheader("Current Score")
    col3, col4, col5 = st.columns(3)
    
    with col3:
        current_over = st.number_input("Current Over", min_value=1, max_value=19, value=6, step=1)
    with col4:
        current_score = st.number_input("Current Score", min_value=0, max_value=300, value=60, step=1)
    with col5:
        wickets_lost = st.number_input("Wickets Lost", min_value=0, max_value=9, value=1, step=1)

    st.subheader("Prediction Window")
    target_over = st.slider("Predict Score at Over", min_value=current_over + 1, max_value=20, value=min(current_over + 4, 20))

    if st.button("Predict Runs", type="primary", use_container_width=True):
        if wickets_lost >= 10:
            st.warning("Innings is already over (10 wickets down).")
        else:
            # Prepare Input Data exactly as the model expects it
            input_df = pd.DataFrame({
                'batting_team': [batting_team],
                'bowling_team': [bowling_team],
                'current_over': [current_over],
                'current_score': [current_score],
                'wickets_lost': [wickets_lost],
                'target_over': [target_over]
            })

            # Predict
            try:
                prediction = model.predict(input_df)[0]
                
                # The model shouldn't predict fewer runs than the current score in reality
                predicted_runs = max(int(round(prediction)), current_score) 
                
                overs_diff = target_over - current_over
                run_diff = predicted_runs - current_score

                st.success(f"### Predicted Score at Over {target_over}: **{predicted_runs}** runs")
                st.info(f"Projected to score **{run_diff}** runs in the next **{overs_diff}** overs.")
            except Exception as e:
                st.error(f"Prediction Error: {e}")

st.markdown("---")
st.caption("Note: This model is currently trained on a synthetic dataset for demonstration purposes.")
