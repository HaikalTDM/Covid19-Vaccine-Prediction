import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Function to load the model and encoder (caching to improve performance)
@st.cache_resource
def load_model():
    model = joblib.load("mortality_model.pkl")
    encoder = joblib.load("label_encoder.pkl")
    return model, encoder

# Function to load the dataset (caching for efficiency)
@st.cache_data
def load_data():
    file_path = "linelist_deaths1.csv"
    data = pd.read_csv(file_path)

    # Preprocess the dataset
    data['vaccine_combo'] = data[['brand1', 'brand2', 'brand3']].apply(
        lambda row: '-'.join(sorted(filter(lambda x: pd.notna(x), row))), axis=1
    )
    data['vaccine_combo'].fillna('No Vaccine', inplace=True)
    data['vaccine_combo_encoded'] = label_encoder.transform(data['vaccine_combo'])
    return data

# Load model, encoder, and dataset
rf_model, label_encoder = load_model()
data = load_data()

# Define vaccine brands for dropdown
vaccine_brands = ['Pfizer', 'Sinovac', 'AstraZeneca', 'Moderna', 'Johnson & Johnson']

# Streamlit app setup
st.title("COVID-19 Vaccine Mortality Prediction and Data Insights")
st.sidebar.title("Navigation")
app_mode = st.sidebar.selectbox(
    "Choose the mode:",
    ["Prediction", "Data Visualization"]
)

# Prediction mode
if app_mode == "Prediction":
    st.header("Predict Mortality Rate")
    st.markdown("""
    Predict the mortality probability based on age and vaccine combination.
    Enter the details below to get started.
    """)

    # User input for prediction
    age = st.number_input("Enter your age:", min_value=0, max_value=120, value=30, step=1)
    dose1 = st.selectbox("Select Dose 1 vaccine:", options=vaccine_brands)
    dose2 = st.selectbox("Select Dose 2 vaccine:", options=vaccine_brands)
    booster = st.selectbox("Select Booster vaccine:", options=vaccine_brands)

    # Submit button for prediction
    if st.button("Predict Mortality Rate"):
        try:
            # Combine doses into a single string
            selected_combo = '-'.join(sorted([dose1, dose2, booster]))

            # Encode the combined vaccine combination
            vaccine_combo_encoded = label_encoder.transform([selected_combo])[0]

            # Prepare input features
            input_features = np.array([[age, vaccine_combo_encoded]])

            # Predict mortality probability
            mortality_probability = rf_model.predict_proba(input_features)[0][1]
            result = round(mortality_probability * 100, 2)

            # Display result
            st.success(f"Predicted Mortality Rate: {result}%")
        except ValueError:
            st.error(f"Invalid vaccine combination: {selected_combo}. Please check your input.")

# Data visualization mode
elif app_mode == "Data Visualization":
    st.header("Data Insights and Visualizations")

    # Display raw data
    if st.checkbox("Show Raw Data"):
        st.write(data.head())

    # Visualization 1: Age Distribution by Mortality Outcome
    st.subheader("Age Distribution by Mortality Outcome")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(data, x='age', hue='bid', multiple='stack', bins=30, kde=False, ax=ax)
    ax.set_title("Age Distribution by Mortality Outcome")
    ax.set_xlabel("Age")
    ax.set_ylabel("Count")
    st.pyplot(fig)

    # Visualization 2: Vaccine Brand Usage by Mortality Outcome
    st.subheader("Vaccine Brand Usage by Mortality Outcome")
    vaccine_counts = data[['brand1', 'brand2', 'brand3', 'bid']].melt(id_vars='bid', value_name='Vaccine Brand').dropna()
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.countplot(data=vaccine_counts, x='Vaccine Brand', hue='bid', ax=ax)
    ax.set_title("Vaccine Brand Usage by Mortality Outcome")
    ax.set_xlabel("Vaccine Brand")
    ax.set_ylabel("Count")
    plt.xticks(rotation=45)
    st.pyplot(fig)

    # Visualization 3: Mortality Outcome Distribution Across States
    st.subheader("Mortality Outcome Distribution Across States")
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.countplot(data=data, x='state', hue='bid', order=data['state'].value_counts().index, ax=ax)
    ax.set_title("Mortality Outcome Distribution Across States")
    ax.set_xlabel("State")
    ax.set_ylabel("Count")
    plt.xticks(rotation=45)
    st.pyplot(fig)

    # Visualization 4: Mortality Probabilities by Vaccine Combination (Interactive)
    st.subheader("Mortality Probabilities by Vaccine Combination")
    data['predicted_proba_mortality'] = rf_model.predict_proba(data[['age', 'vaccine_combo_encoded']])[:, 1]
    vaccine_impact = data.groupby('vaccine_combo').agg(
        avg_predicted_mortality=('predicted_proba_mortality', 'mean'),
        total_cases=('bid', 'count')
    ).reset_index().sort_values(by='avg_predicted_mortality', ascending=False)

    fig = px.bar(
        vaccine_impact,
        x='vaccine_combo',
        y='avg_predicted_mortality',
        color='avg_predicted_mortality',
        labels={'avg_predicted_mortality': 'Avg Predicted Mortality Probability'},
        title='Predicted Mortality Probabilities by Vaccine Combination',
        color_continuous_scale='Viridis'  # Use a valid Plotly color scale
    )
    fig.update_layout(xaxis=dict(tickangle=45))
    st.plotly_chart(fig)

    # Visualization 5: Heatmap of Mortality by Vaccine and Age Group
    # Heatmap: Mortality by Vaccine and Age Group
    st.subheader("Heatmap: Mortality by Vaccine and Age Group")
    data['age_group'] = pd.cut(data['age'], bins=[0, 17, 40, 65, 100], labels=['Child', 'Young Adult', 'Adult', 'Senior'])
    heatmap_data = data.groupby(['vaccine_combo', 'age_group']).agg(
        avg_predicted_mortality=('predicted_proba_mortality', 'mean')
    ).reset_index().pivot(index='vaccine_combo', columns='age_group', values='avg_predicted_mortality').fillna(0)
    
    fig, ax = plt.subplots(figsize=(16, 12))  # Increase figure size
    sns.heatmap(
        heatmap_data, 
        annot=True, 
        cmap='viridis',  # Use a visually distinct colormap
        fmt=".2f", 
        cbar_kws={'label': 'Avg Predicted Mortality'}, 
        linewidths=0.5,  # Add gridlines for clarity
        ax=ax
    )
    ax.set_title("Heatmap: Mortality by Vaccine Combinations and Age Groups", fontsize=18, pad=20)
    ax.set_xlabel("Age Group", fontsize=14)
    ax.set_ylabel("Vaccine Combination", fontsize=14)
    plt.xticks(rotation=45, ha='right', fontsize=12)  # Rotate and align x-axis labels
    plt.yticks(fontsize=10)  # Adjust y-axis label font size
    plt.tight_layout()  # Ensure the plot uses the space efficiently
    st.pyplot(fig)
