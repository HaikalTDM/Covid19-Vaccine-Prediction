import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Function to train the model and save it (for demonstration purposes)
def train_and_save_model():
    # Load the dataset
    file_path = "linelist_deaths1.csv"
    data = pd.read_csv(file_path)

    # Combine vaccine brands into one string
    data['vaccine_combo'] = data[['brand1', 'brand2', 'brand3']].apply(
        lambda row: '-'.join(sorted(filter(lambda x: pd.notna(x), row))), axis=1
    )
    data['vaccine_combo'].fillna('No Vaccine', inplace=True)

    # Encode vaccine combinations
    label_encoder = LabelEncoder()
    data['vaccine_combo_encoded'] = label_encoder.fit_transform(data['vaccine_combo'])

    # Define features and target
    X = data[['age', 'vaccine_combo_encoded']]
    y = data['bid']

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train a Random Forest model
    rf_model = RandomForestClassifier(random_state=42, n_estimators=100, class_weight='balanced')
    rf_model.fit(X_train, y_train)

    # Save the model and encoder
    joblib.dump(rf_model, "mortality_model.pkl")
    joblib.dump(label_encoder, "label_encoder.pkl")

# Uncomment the following line if you need to retrain the model
# train_and_save_model()

# Function to load the model and encoder
@st.cache_resource
def load_model():
    model = joblib.load("mortality_model.pkl")
    encoder = joblib.load("label_encoder.pkl")
    return model, encoder

# Function to load and preprocess the dataset
@st.cache_data
def load_data():
    file_path = "linelist_deaths1.csv"
    data = pd.read_csv(file_path)

    # Combine vaccine brands into one string
    data['vaccine_combo'] = data[['brand1', 'brand2', 'brand3']].apply(
        lambda row: '-'.join(sorted(filter(lambda x: pd.notna(x), row))), axis=1
    )
    data['vaccine_combo'].fillna('No Vaccine', inplace=True)
    return data

# Load model, encoder, and dataset
rf_model, label_encoder = load_model()
data = load_data()

# Add encoded vaccine combinations to the dataset
data['vaccine_combo_encoded'] = label_encoder.transform(data['vaccine_combo'])

# Define vaccine brands for dropdowns
vaccine_brands = ['Pfizer', 'Sinovac', 'AstraZeneca', 'Moderna',  'No Dose']

# Streamlit application setup
st.title("COVID-19 Vaccine Mortality Prediction and Data Insights")
st.sidebar.title("Navigation")
app_mode = st.sidebar.selectbox(
    "Choose the mode:",
    ["Prediction", "Data Visualization", "Dashboard", "Admin Dashboard"]
)

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

    if st.button("Predict Mortality Rate"):
        try:
            # Combine doses into a single string, excluding "No Dose"
            selected_combo = '-'.join(sorted(filter(lambda x: x != "No Dose", [dose1, dose2, booster])))

            # Check if a valid combination was entered
            if not selected_combo:
                st.error("Please select at least one valid vaccine dose.")
            else:
                # Encode vaccine combination
                vaccine_combo_encoded = label_encoder.transform([selected_combo])[0]

                # Prepare input features for prediction
                input_features = np.array([[age, vaccine_combo_encoded]])

                # Predict mortality probability
                mortality_probability = rf_model.predict_proba(input_features)[0][1]
                result = round(mortality_probability * 100, 2)

                # Display the result
                st.success(f"Predicted Mortality Rate: {result}%")

                # Dynamic best combination recommendation
                if dose2 == "No Dose" or booster == "No Dose":
                    lowest_rate = mortality_probability
                    best_combination = None

                    # Iterate through all possible combinations for Dose 2 and Booster
                    for dose2_option in vaccine_brands:
                        if dose2_option != "No Dose" or dose2 != "No Dose":
                            for booster_option in vaccine_brands:
                                if booster_option != "No Dose" or booster != "No Dose":
                                    test_combo = '-'.join(sorted(filter(lambda x: x != "No Dose", [dose1, dose2_option, booster_option])))
                                    test_encoded = label_encoder.transform([test_combo])[0]
                                    test_features = np.array([[age, test_encoded]])
                                    test_rate = rf_model.predict_proba(test_features)[0][1]

                                    # Check if this combination improves the rate
                                    if test_rate < lowest_rate:
                                        lowest_rate = test_rate
                                        best_combination = (dose1, dose2_option, booster_option)

                    # Provide the best recommendation
                    if best_combination:
                        recommended_dose2, recommended_booster = best_combination[1], best_combination[2]
                        st.info(
                            f"Recommendation: Based on your input, consider completing your vaccination with "
                            f"Dose 2: **{recommended_dose2}**, Booster: **{recommended_booster}**. "
                            f"This combination reduces the mortality rate to **{round(lowest_rate * 100, 2)}%**."
                        )
        except ValueError:
            st.error(f"Invalid vaccine combination: {selected_combo}. Please check your input.")



# Dashboard mode
elif app_mode == "Dashboard":
    st.header("Dashboard")

    # Filter options
    selected_state = st.multiselect("Filter by State", options=data['state'].unique(), default=data['state'].unique())
    selected_vaccine = st.multiselect("Filter by Vaccine", options=data['vaccine_combo'].unique(), default=data['vaccine_combo'].unique())
    age_range = st.slider("Select Age Range", int(data['age'].min()), int(data['age'].max()), (int(data['age'].min()), int(data['age'].max())))

    # Apply filters
    filtered_data = data[
        (data['state'].isin(selected_state)) &
        (data['vaccine_combo'].isin(selected_vaccine)) &
        (data['age'].between(age_range[0], age_range[1]))
    ]

    # Display summary statistics
    st.subheader("Summary Statistics")
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Cases", len(filtered_data))
    col2.metric("Average Mortality Rate", f"{filtered_data['bid'].mean() * 100:.2f}%")
    col3.metric("Unique Vaccine Combinations", filtered_data['vaccine_combo'].nunique())

    # Allow data download
    st.download_button(
        label="Download Filtered Data as CSV",
        data=filtered_data.to_csv(index=False),
        file_name="filtered_data.csv",
        mime="text/csv"
    )

    # Show filtered data
    st.subheader("Filtered Data")
    st.write(filtered_data)

    # Visualization 1: Age Distribution by Mortality Outcome
    st.subheader("Age Distribution by Mortality Outcome (Filtered Data)")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(filtered_data, x='age', hue='bid', multiple='stack', bins=30, kde=False, ax=ax)
    ax.set_title("Age Distribution by Mortality Outcome")
    ax.set_xlabel("Age")
    ax.set_ylabel("Count")
    st.pyplot(fig)

    # Visualization 2: Vaccine Brand Usage by Mortality Outcome
    st.subheader("Vaccine Brand Usage by Mortality Outcome (Filtered Data)")
    vaccine_counts = filtered_data[['brand1', 'brand2', 'brand3', 'bid']].melt(id_vars='bid', value_name='Vaccine Brand').dropna()
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.countplot(data=vaccine_counts, x='Vaccine Brand', hue='bid', ax=ax)
    ax.set_title("Vaccine Brand Usage by Mortality Outcome")
    ax.set_xlabel("Vaccine Brand")
    ax.set_ylabel("Count")
    plt.xticks(rotation=45)
    st.pyplot(fig)

    # Visualization 3: State-wise Mortality Distribution
    st.subheader("Mortality Outcome Distribution Across States (Filtered Data)")
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.countplot(data=filtered_data, x='state', hue='bid', order=filtered_data['state'].value_counts().index, ax=ax)
    ax.set_title("Mortality Outcome Distribution Across States")
    ax.set_xlabel("State")
    ax.set_ylabel("Count")
    plt.xticks(rotation=45)
    st.pyplot(fig)

    # Visualization 4: Mortality Probabilities by Vaccine Combination
    st.subheader("Mortality Probabilities by Vaccine Combination (Filtered Data)")
    filtered_data['predicted_proba_mortality'] = rf_model.predict_proba(filtered_data[['age', 'vaccine_combo_encoded']])[:, 1]
    vaccine_impact_filtered = filtered_data.groupby('vaccine_combo').agg(
        avg_predicted_mortality=('predicted_proba_mortality', 'mean'),
        total_cases=('bid', 'count')
    ).reset_index().sort_values(by='avg_predicted_mortality', ascending=False)

    fig, ax = plt.subplots(figsize=(14, 8))  # Increased figure size
    sns.barplot(data=vaccine_impact_filtered, x='vaccine_combo', y='avg_predicted_mortality', ax=ax, palette='coolwarm')
    ax.set_title("Predicted Mortality Probabilities by Vaccine Combination (Filtered Data)", fontsize=16)
    ax.set_xlabel("Vaccine Combination", fontsize=12)
    ax.set_ylabel("Avg Predicted Mortality Probability", fontsize=12)
    plt.xticks(rotation=90, ha='center', fontsize=10)  # Rotate labels 90 degrees
    plt.tight_layout()  # Ensure everything fits nicely
    st.pyplot(fig)

    # Visualization 5: Heatmap
    st.subheader("Heatmap: Mortality by Vaccine and Age Group (Filtered Data)")
    filtered_data['age_group'] = pd.cut(filtered_data['age'], bins=[0, 17, 40, 65, 100], labels=['Child', 'Young Adult', 'Adult', 'Senior'])
    heatmap_data_filtered = filtered_data.groupby(['vaccine_combo', 'age_group']).agg(
        avg_predicted_mortality=('predicted_proba_mortality', 'mean')
    ).reset_index().pivot(index='vaccine_combo', columns='age_group', values='avg_predicted_mortality').fillna(0)

    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(heatmap_data_filtered, annot=True, cmap='coolwarm', fmt=".2f", cbar_kws={'label': 'Avg Predicted Mortality'})
    ax.set_title("Heatmap: Mortality by Vaccine Combinations and Age Groups (Filtered Data)")
    ax.set_xlabel("Age Group")
    ax.set_ylabel("Vaccine Combination")
    st.pyplot(fig)

# Admin Dashboard mode
elif app_mode == "Admin Dashboard":
    st.header("Admin Dashboard")

    # Initialize session state for login
    if "logged_in" not in st.session_state:
        st.session_state.logged_in = False

    # Display login form if not logged in
    if not st.session_state.logged_in:
        admin_username = st.text_input("Admin Username")
        admin_password = st.text_input("Admin Password", type="password")
        login_button = st.button("Login")

        # Dummy credentials for simplicity (these can be replaced with a secure method)
        ADMIN_CREDENTIALS = {"admin": "haikaltdm46"}

        if login_button:
            if admin_username in ADMIN_CREDENTIALS and ADMIN_CREDENTIALS[admin_username] == admin_password:
                st.session_state.logged_in = True
                st.success("Logged in successfully!")
            else:
                st.error("Invalid credentials. Please try again.")
    else:
        # Show the admin dashboard functionalities after login
        st.subheader("Upload New Dataset and Models")

        # Logout button
        if st.button("Logout"):
            st.session_state.logged_in = False

        # File upload section
        dataset_file = st.file_uploader("Upload New Dataset (CSV format)", type=["csv"])
        model_file = st.file_uploader("Upload New Mortality Model (PKL format)", type=["pkl"])
        encoder_file = st.file_uploader("Upload New Label Encoder (PKL format)", type=["pkl"])

        # Handle dataset upload
        if dataset_file:
            try:
                data = pd.read_csv(dataset_file)
                data.to_csv("linelist_deaths1.csv", index=False)  # Save the uploaded dataset
                st.success("Dataset uploaded successfully!")
            except Exception as e:
                st.error(f"Failed to upload dataset: {e}")

        # Handle model upload
        if model_file:
            try:
                with open("mortality_model.pkl", "wb") as f:
                    f.write(model_file.read())  # Save the uploaded model
                st.success("Mortality model uploaded successfully!")
            except Exception as e:
                st.error(f"Failed to upload mortality model: {e}")

        # Handle encoder upload
        if encoder_file:
            try:
                with open("label_encoder.pkl", "wb") as f:
                    f.write(encoder_file.read())  # Save the uploaded encoder
                st.success("Label encoder uploaded successfully!")
            except Exception as e:
                st.error(f"Failed to upload label encoder: {e}")

        # Display Admin Data Overview (Optional)
        st.subheader("Admin Data Overview")
        try:
            st.write(f"Dataset has **{len(data)} records** and the following columns:")
            st.write(data.columns.tolist())
        except NameError:
            st.info("Upload a dataset to view its details.")


