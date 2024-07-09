import streamlit as st
import pandas as pd
import pickle
from xgboost import XGBRegressor
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import numpy as np

# Define the categories
categories = {
    "Essentials Hub": ["school", "international school", "high school", "academy", "vidya", "Vidyalayam", "college", "university", "clg", "Apartment",
                       "hospital", "clinic", "motherhood", "church", "mosque", "Training",
                       "railway station", "metro station", "junction", "station", "metro", "bus", "airport", "Bus Stop",
                       "bank", "Financial", "Bank", "temple"],
    "Urban Framework": ["property", "building", "block", "street", "main road", "road", "bridge", "layout", "hotel", "flyover", "residence", "park", "Lake", "Garden", "road", "pg", "circle", "Logistics", "petrol", "fire", "nagar", "toll", "Centre", "view point"],
    "Commercial Sphere": ["market", "mall", "shopping", "complex", "plaza", "store", "bazaar", "showroom",
                          "Tech Park", "tech park", "infotech", "Software", "IT", "Resort", "restaurant"]
}

# Categorize landmark function
def categorize_landmark(landmark):
    if not isinstance(landmark, str):
        return "Other"

    landmark_lower = landmark.lower()
    for category, keywords in categories.items():
        for keyword in keywords:
            if keyword in landmark_lower:
                return category
    return "Other"

# Categorize direction function
def categorize_direction(direction):
    if direction in ['East', 'North', 'North - East']:
        return 'Favorable Direction'
    elif direction in ['South', 'South - West', 'South - East']:
        return 'Less Favorable Direction'
    elif direction in ['West', 'North - West']:
        return 'Neutral/Mixed Direction'
    else:
        return 'None'

# Inject CSS for custom styling
def local_css(file_name):
    with open(file_name, 'r') as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# Path to the CSS file (ensure correct path)
css_file_path = 'styles.css'
local_css(css_file_path)

# Centered title
st.markdown('<h1 style="text-align: center;">House Price Prediction</h1>', unsafe_allow_html=True)

# Form background
form_bg_css = """
<style>
div[data-testid="stForm"] {
    background-color: #212121; /* Darker background color */
    padding: 30px; /* Increased padding */
    border-radius: 10px;
}
</style>
"""
st.markdown(form_bg_css, unsafe_allow_html=True)

# Initialize data dictionary
data = {
    'Carpet Area': [],
    'balcony': [],
    'bed': [],
    'bath': [],
    'Lifts': [],
    'Furnished Status': [],
    'Overlooking': [],
    'Construction Age': [],
    'Landmark Category': [],
    'direction_category': [],
    'Standardized_Address': [],
    'Category flooring': []
}

# Function to update data dictionary with form values
def update_data():
    data['Carpet Area'].append(carpet_area)
    data['balcony'].append(balcony)
    data['bed'].append(bed)
    data['bath'].append(bath)
    data['Lifts'].append(lifts)
    data['Furnished Status'].append(furnished_status)
    data['Overlooking'].append(overlooking)
    data['Construction Age'].append(construction_age)
    data['Landmark Category'].append(landmark_category)
    data['direction_category'].append(direction_category)
    data['Standardized_Address'].append(standardized_address)
    data['Category flooring'].append(category_flooring)

# Create form
with st.form(key='property_form'):
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        balcony = st.slider('Balcony', 0, 10, 3)
    with col2:
        bed = st.slider('Bed', 0, 10, 3)
    with col3:
        bath = st.slider('Bath', 0, 10, 3)
    with col4:
        lifts = st.slider('Lifts', 0, 10, 1)

    st.write("")  # Add some space between sections

    col1, col2 = st.columns(2)

    with col1:
        furnished_status_options = ['Semi-Furnished', 'Furnished', 'Unfurnished']
        furnished_status = st.selectbox('Furnished Status', furnished_status_options)
    with col2:
        overlooking_options = ['Single Aspect', 'Not Disclosed', 'Multiple Aspects']
        overlooking = st.selectbox('Overlooking', overlooking_options)

    st.write("")  # Add some space between sections

    col1, col2 = st.columns(2)

    with col1:
        construction_age_options = ['Old Construction', 'New Construction', 'Under Construction']
        construction_age = st.selectbox('Construction Age', construction_age_options)
    with col2:
        landmark_category = st.text_input('Landmark Category')

    st.write("")  # Add some space between sections

    col1, col2 = st.columns(2)

    with col1:
        direction_category_options = ['East', 'West', 'North', 'South', 'North - East', 'North - West', 'South - East', 'South - West']
        direction_category = st.selectbox('Direction Category', direction_category_options)
    with col2:
        standardized_address_options = [
            'Bangalore - South', 'Bangalore - East', 'Bangalore - Central',
            'Bangalore - North', 'Bangalore', 'Bangalore - West', 'Bangalore - Rural'
        ]
        standardized_address = st.selectbox('Standardized Address', standardized_address_options)

    st.write("")  # Add some space between sections

    col1, col2 = st.columns(2)

    with col1:
        category_flooring_options = ['Ceramic-Based', 'Natural Stone', 'other', 'Wood-Based']
        category_flooring = st.selectbox('Category Flooring', category_flooring_options)
    with col2:
        carpet_area = st.number_input('Carpet Area (in sq ft)', min_value=0.0, format="%.2f")

    submit_button = st.form_submit_button(label='Get Price')

# Process form submission
if submit_button:
    update_data()  # Update data dictionary with form value

    # Create DataFrame from updated data dictionary
    df = pd.DataFrame(data)
    if 'Landmark Category' in df.columns:
        df['Landmark Category'] = df['Landmark Category'].apply(categorize_landmark)
    if 'direction_category' in df.columns:
        df['direction_category'] = df['direction_category'].apply(categorize_direction)

    df[["Lifts", "bath", "bed", "balcony"]] = df[["Lifts", "bath", "bed", "balcony"]].astype(int)
    df["Carpet Area"] = df["Carpet Area"].astype(float)
    with open('fitted_preprocessor.pkl', 'rb') as f:
        preprocessor = pickle.load(f)
    
    # Transform the data
    transfered_data = preprocessor.transform(df)
    with open("best_model.pkl",'rb') as f1:
        model = pickle.load(f1)
    y_pred = model.predict(transfered_data)
    st.subheader("Predicted selling price")
    
    # Custom styled prediction text
    st.markdown(f'<p class="prediction-text">{np.exp(y_pred)[0]:,.2f} Lakhs</p>', unsafe_allow_html=True)
if __name__ == '__main__':
    st.run(port=8000, address='0.0.0.0')
