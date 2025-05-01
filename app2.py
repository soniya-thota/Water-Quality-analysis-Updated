import warnings
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import pandas as pd
import joblib
import plotly.express as px  # For interactive plots

# Set page configuration and title
st.set_page_config(page_title="Water Pollution Prediction", layout="wide")

# Navigation Bar (pages)
page = st.sidebar.radio("Navigate", ["Home", "About", "Model Prediction", "Data Analysis"])

# HOME PAGE
if page == "Home":
    st.markdown(
        """
        <style>
        .stApp {
            background-image: url('https://www.slideegg.com/image/webpv2/670/47376-water-ppt-template-670.webp');  /* Replace with your background image URL */
            background-size: cover;
            background-attachment: fixed;
            background-position: center;
        }
        .block-container {
            background-color: rgba(255, 255, 255, 0.8);  /* Add transparency for better readability */
            padding: 2rem;
            border-radius: 10px;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    st.title("Welcome to Water Pollution Level Prediction App")
    st.markdown("""
    This application predicts the water pollution level based on various features such as waste composition, sampling period, etc.
    - **Upload a dataset** to make predictions using the trained model.
    - **View data analysis** for insights into correlations and distributions.
    - **Model predictions** will help you understand the water pollution levels in different water bodies.
    """)

# ABOUT PAGE
elif page == "About":
    st.markdown(
        """
        <style>
        .stApp {
            background-image: url('https://th.bing.com/th/id/OIP.fswnIPiz-pKQnqhhXJBCJwHaEf?rs=1&pid=ImgDetMain');  /* Replace with your background image URL */
            background-size: cover;
            background-attachment: fixed;
            background-position: center;
        }
        .block-container {
            background-color: rgba(255, 255, 255, 0.8);  /* Add transparency for better readability */
            padding: 2rem;
            border-radius: 10px;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    st.title("About This Project")
    st.markdown("""
    This app uses machine learning to predict the **pollution level** of water based on various environmental factors like:
    - **Organic waste**
    - **Metal content**
    - **Plastic and glass waste**
    - **Yard and garden waste**

    The app uses a **Random Forest classifier** trained on environmental data to predict water pollution levels as:
    - **Poor**
    - **Moderate**
    - **High**

    **How to Use**:
    - Upload your **CSV file** with the necessary features.
    - View the predicted pollution levels for the data.
    - Visualize distributions and relationships between features in the dataset.

    The app is designed to help **environmental agencies** and **researchers** to assess water quality and pollution levels in different areas.
    """)

# MODEL PREDICTION PAGE
elif page == "Model Prediction":
    st.markdown(
        """
        <style>
        .stApp {
            background-image: url('https://th.bing.com/th/id/OIP.XADF7amzQRt33rYl51Y3XwHaEK?rs=1&pid=ImgDetMain');  /* Replace with your background image URL */
            background-size: cover;
            background-attachment: fixed;
            background-position: center;
        }
        .block-container {
            background-color: rgba(255, 255, 255, 0.8);  /* Add transparency for better readability */
            padding: 2rem;
            border-radius: 10px;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    st.title("Water Pollution Prediction")
    st.markdown("Upload a CSV file to predict the water pollution level.")

    uploaded_file = st.file_uploader("Upload your CSV file", type="csv")

    if uploaded_file:
        data = pd.read_csv(uploaded_file)
        st.subheader("Uploaded Data Preview:")
        st.dataframe(data.head())

        # Load model
        model = joblib.load("model.pkl")

        # Make predictions
        predictions = model.predict(data)
        label_map = {0: "Poor", 1: "Moderate", 2: "High"}
        pred_labels = pd.Series(predictions).map(label_map)

        result = data.copy()
        result["Predicted_Pollution_Level"] = pred_labels

        st.subheader("Predictions:")
        st.dataframe(result)

        # Download CSV of predictions
        csv = result.to_csv(index=False).encode('utf-8')
        st.download_button("⬇Download Results as CSV", csv, "predictions.csv", "text/csv")
        
        # Visualization Section
        st.subheader("Pollution Level Distribution")

        # Bar Chart for Pollution Levels
        counts = pred_labels.value_counts().sort_index()
        st.bar_chart(counts)

        # Pie Chart for Pollution Levels
        st.subheader("Percentage Distribution")
        fig, ax = plt.subplots()
        ax.pie(counts, labels=counts.index, autopct="%1.1f%%", startangle=90)
        ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
        st.pyplot(fig)

# DATA ANALYSIS PAGE
elif page == "Data Analysis":
    st.markdown(
        """
        <style>
        .stApp {
            background-image: url('https://th.bing.com/th/id/OIP.fAgNlLR4ewYjbVKOwB8KagHaEc?rs=1&pid=ImgDetMain');  /* Replace with your background image URL */
            background-size: cover;
            background-attachment: fixed;
            background-position: center;
        }
        .block-container {
            background-color: rgba(255, 255, 255, 0.8);  /* Add transparency for better readability */
            padding: 2rem;
            border-radius: 10px;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    st.title("Data Analysis")
    st.markdown("""
    Explore the dataset by visualizing the feature relationships and distributions.

    - **Basic Statistics**: View the summary statistics of the dataset.
    - **Correlation Matrix**: See the relationships between different features in the dataset.
    - **Visualizations**: Understand the distribution of features and their impact on pollution levels.
    """)

    uploaded_file = st.file_uploader("Upload your CSV file", type="csv")

    if uploaded_file:
        data = pd.read_csv(uploaded_file)

        # Display basic statistics
        st.subheader("Basic Statistics")
        st.write(data.describe())

        # Correlation matrix
        st.subheader("Correlation Matrix")
        corr = data.corr()
        st.write(corr)

        # Visualize the Correlation Matrix (Heatmap)
        st.subheader("Correlation Matrix Heatmap")
        fig, ax = plt.subplots(figsize=(10, 8))
        cax = ax.matshow(corr, cmap='coolwarm')
        fig.colorbar(cax)
        plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
        plt.yticks(range(len(corr.columns)), corr.columns)
        st.pyplot(fig)

        # Display feature distributions with histograms
        st.subheader("Feature Distributions")
        numerical_columns = data.select_dtypes(include=['float64', 'int64']).columns
        for col in numerical_columns:
            st.subheader(f"Distribution of {col}")
            fig, ax = plt.subplots()
            sns.histplot(data[col], kde=True, ax=ax)
            st.pyplot(fig)
