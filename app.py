import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import xgboost as xgb
import plotly.graph_objects as go
import pickle
import tensorflow as tf
import keras.losses
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras import backend as K
import io
from contextlib import redirect_stdout
import random
import joblib
import requests
import io



# Configure page
st.set_page_config(
    page_title="Robotic Grasp Stability Predictor",
    page_icon="🤖",
    layout="wide"
)

# Custom R² metric
def r2_score(y_true, y_pred):
    ss_res = K.sum(K.square(y_true - y_pred))
    ss_tot = K.sum(K.square(y_true - K.mean(y_true)))
    return 1 - ss_res / (ss_tot + K.epsilon())

@st.cache_resource
def load_models():
    linear_model = joblib.load("linear_reg.pkl")
    poly_scaler = joblib.load("poly_transform.pkl")
    poly_model = joblib.load("poly_reg_model.pkl")

    xgb_model = xgb.XGBRegressor()
    xgb_model.load_model('xgb_model.json')
    return linear_model, poly_model, poly_scaler, xgb_model

@st.cache_resource
def load_nn_model():
    model = tf.keras.models.load_model("neural_net_model.h5", compile=False)
    model.compile(optimizer="adam", loss=MeanSquaredError(), metrics=[r2_score])
    return model

# Create gauge chart
def create_gauge(value):
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = value,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Grasp Stability Score", 'font': {'size': 20}},
        gauge = {
            'axis': {'range': [0, 200], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': "#0072B2"},
            'steps': [
                {'range': [0, 75], 'color': "#D55E00"},
                {'range': [75, 100], 'color': "#E69F00"},
                {'range': [100, 150], 'color': "#CC79A7"},
                {'range': [150, 200], 'color': "#009E73"}],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': value}}))
    
    fig.update_layout(height=300, margin=dict(t=0, b=0, l=0, r=0))
    return fig

# Main application
def main():
    st.title("🤖 Robotic Grasp Stability Predictor")
    st.markdown("Kaggle Dataset: [Grasping Dataset](https://www.kaggle.com/datasets/ugocupcic/grasping-dataset)")
   
    scaler = joblib.load('scaler.pkl')
    initial_columns = joblib.load('columns.pkl')

    linear_model, poly_model, poly_scaler, xgb_model = load_models()
    nn_model = load_nn_model()
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    section = st.sidebar.radio("Go to", 
        ["Introduction", "Problem Statement", "Technical Approach", "Model Summary", "Live Demo"])
    
    # Content sections
    if section == "Introduction":
        st.header("Introduction to Robotic Grasp Stability")
        st.markdown("""
        This application predicts the stability of robotic grasps using sensor data from the Shadow Robot Hand. 
        The goal is to assist robotics researchers and engineers in evaluating the robustness of robotic hand grasps under varying conditions.
        
        ### Key Features:
        - **Real-time predictions**: Evaluate grasp stability dynamically.
        - **Multiple machine learning models**: Compare performance across Linear Regression, Polynomial Regression (Poly), XGBoost (XGB), and Neural Network models.
        - **Interactive UI**: Experiment with joint parameters to observe changes in stability scores.
        
        **Dataset**: The dataset used is sourced from the Shadow Robot Hand's sensor readings.
        """)
        st.image("robot.png", use_container_width=True)

    elif section == "Problem Statement":
        st.header("The Grasp Stability Challenge")
        st.markdown("""
        **Problem Statement**: Robotic grasping requires precise coordination of joint positions and forces to ensure stability. 
        Predicting grasp stability is critical for tasks involving object manipulation in dynamic environments.

        ### Challenges:
        - Real-time assessment of grasp robustness.
        - Integration of complex sensor data (joint positions/velocities/efforts).
        - Variability in object shapes and environmental factors.

        ### Objective:
        Develop a machine learning-based solution to predict grasp stability using sensor data from robotic hands.
        """)

    elif section == "Technical Approach":
        st.header("Technical Approach")
        
        st.subheader("Feature Engineering")
        st.markdown("""
        The dataset contains sensor readings from robotic joints:
        
        - **24 joint measurements**: Position, velocity, and effort for each joint.
        - **Normalization**: StandardScaler was used to normalize feature values for better model performance.
        
        Additional temporal aggregation techniques were applied to capture dynamic behavior over time.
        """)
        
        st.subheader("Model Selection")
        st.markdown("""
        Four machine learning models were evaluated for this task:
        
        | Model                   | R² Score |
        |------------------------|-----------|
        | Linear Regression      | 0.6622    |
        | Polynomial Regression  | 0.6276    |
        | Random Forest          | 0.4531    |
        | XGBoost                | 0.5188    |
        | Neural Network         | 0.8411    |
        
        Neural Network outperformed other models in terms of test  R² Score.
        """)

    elif section == "Model Summary":
        # Capture model.summary() output
        string_io = io.StringIO()
        with redirect_stdout(string_io):
            nn_model.summary()
        summary_string = string_io.getvalue()
        # Display in Streamlit
        st.code(summary_string, language='text')
    
    elif section == "Live Demo":
        st.header("Interactive Prediction Demo")
        st.markdown("Adjust joint parameters to predict grasp stability:")
        
        # Create input sliders for key features
        inputs = {}
        cols = st.columns(3)
        joint_params = ['pos', 'vel', 'eff']
        input_features = ['H1_F1J1', 'H1_F1J2', 'H1_F1J3',
                          'H1_F2J1', 'H1_F2J2', 'H1_F2J3',
                          'H1_F3J1', 'H1_F3J2', 'H1_F3J3']
        feature_ranges = joblib.load("feature_ranges.pkl")

        if "random_defaults" not in st.session_state:
            st.session_state.random_defaults = {
                key: round(random.uniform(bounds["min"], bounds["max"]) / 0.001) * 0.001
                for key, bounds in feature_ranges.items()
            }

        for i, joint in enumerate(input_features):
            with cols[i % 3]:
                st.subheader(joint.replace("_", " "))
                for param in joint_params:
                    key = f"{joint}_{param}"
                    bounds = feature_ranges[key]
                    inputs[key] = st.slider(
                        label=param.upper(),
                        min_value=bounds["min"],
                        max_value=bounds["max"],
                        value=st.session_state.random_defaults[key],
                        step=0.001
                    )
        cols = st.columns(2)
        with cols[0]:
            inputs['measurement_number'] = st.number_input("Enter a number", min_value=0, max_value=30, step=1)
        with cols[1]:
            selected_model = st.selectbox("Select Model", ["Linear", "Polynomial Regression", "XGBoost", "Neural Network"], index=0)

        if st.button("Predict Stability"):
            input_df = pd.DataFrame([inputs])
            input_df = input_df.reindex(columns=initial_columns)
            scaled_input = scaler.transform(input_df)
            if selected_model == "Linear":
                prediction = linear_model.predict(scaled_input)
            elif selected_model == "Polynomial Regression":
                X_test_poly = poly_scaler.transform(scaled_input)
                prediction = poly_model.predict(X_test_poly)
            elif selected_model == "XGBoost":
                prediction = xgb_model.predict(scaled_input)
            elif selected_model == "Neural Network":
                prediction = nn_model.predict(scaled_input, verbose=0)
                st.write(prediction[0])

            st.metric("Predicted Stability Score", f"{prediction[0]:.2f}")
            st.plotly_chart(create_gauge(prediction[0]), use_container_width=True)

            if prediction >=175:
                st.success("Exceptional Grasp: Extremely High stability under disturbance")
            elif prediction >= 100:
                st.success("Excellent Grasp: High stability under disturbance")
            elif prediction >= 75:
                st.warning("Good Grasp: May need minor adjustments")
            else:
                st.error("Unstable Grasp: Recommend reconfiguration")

if __name__ == "__main__":
    main()
