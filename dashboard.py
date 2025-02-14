# Importing required libraries
import streamlit as st 
import numpy as np 
import pandas as pd 
import plotly.graph_objects as go 
import plotly.express as px 
from datetime import datetime, timedelta 
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose 
import matplotlib.pyplot as plt
from twilio.rest import Client  
import pydeck as pdk  
import ee
import geemap.foliumap as geemap

# Initialize Google Earth Engine
try:
    ee.Initialize(project='ee-gangahow')
except Exception as e:
    ee.Authenticate()
    ee.Initialize()

# Function to get satellite-based water quality data
def get_satellite_data():
    roi = ee.Geometry.Rectangle([78.0, 24.0, 88.0, 31.0])  # Ganga Basin Region
    dataset = ee.ImageCollection("COPERNICUS/S2") \
        .filterBounds(roi) \
        .filterDate("2024-01-01", "2024-12-31") \
        .median()

    # Compute NDWI (Water Index)
    ndwi = dataset.normalizedDifference(["B3", "B8"]).rename("NDWI")

    # Compute Chlorophyll-a (Water Quality Indicator)
    chlorophyll = dataset.expression(
        "(B5 - B4) / (B5 + B4)", {
            "B5": dataset.select("B5"),
            "B4": dataset.select("B4")
        }
    ).rename("Chlorophyll-a")

    # Compute Turbidity (Approximate)
    turbidity = dataset.expression(
        "(B2 / B4)", {
            "B2": dataset.select("B2"),
            "B4": dataset.select("B4")
        }
    ).rename("Turbidity")

    return dataset.addBands([ndwi, chlorophyll, turbidity])

# Page Configuration
# Page Configuration with Light Theme
# Page Configuration
st.set_page_config(
    layout="wide",
    page_title="Water Quality Dashboard",
    page_icon="💧",
    initial_sidebar_state="expanded"
)

# Left Column - Select Location and Metrics Dashboard
left_col, right_col = st.columns([1, 2])
with left_col:
    st.markdown("### 📍 Select Location") 
    selected_location = st.selectbox('Choose Location', ['Delhi', 'Varanasi', 'Kolkata'])
    
    # Generate sample water quality data
    date_range = pd.date_range(start="2024-01-01", periods=365, freq='D')
    np.random.seed(42)
    temp = np.random.normal(25, 3, len(date_range))  
    do = np.random.normal(7, 1, len(date_range))  
    ph = np.random.normal(7, 0.5, len(date_range))  
    turbidity = np.random.normal(3, 1, len(date_range))  
    filtered_data = pd.DataFrame({
        'Date': date_range,
        'Temperature (ºC)': temp,
        'D.O. (mg/l)': do,
        'pH': ph,
        'Turbidity (NTU)': turbidity,
    }).set_index('Date')

    # Extract Metrics
    metrics = {
        'pH': filtered_data['pH'], 
        'Turbidity (NTU)': filtered_data['Turbidity (NTU)'], 
        'D.O. (mg/l)': filtered_data['D.O. (mg/l)'], 
        'Temperature (ºC)': filtered_data['Temperature (ºC)'], 
    }

    st.markdown("### 📊 Water Quality Metrics")
    for metric, values in metrics.items(): 
        value = values.iloc[-1]  
        max_value = 14 if 'pH' in metric else values.max()  
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=value,
            gauge={
                'axis': {'range': [0, max_value]},
                'steps': [
                    {'range': [0, max_value*0.5], 'color': '#99d98c'},
                    {'range': [max_value*0.5, max_value*0.75], 'color': '#fcbf49'},
                    {'range': [max_value*0.75, max_value], 'color': '#d62828' }
                ],
            },
            title={'text': f"{metric}", 'font': {'size': 12}},  
            domain={'x': [0, 1], 'y': [0, 0.4]}  
        )) 
        st.plotly_chart(fig, use_container_width=True)

# Right Column - Interactive Map with Satellite Data
with right_col:
    st.markdown("### 🗺 Satellite-Based Water Quality Map")

    # Initialize Map
    m = geemap.Map(center=[27.0, 82.0], zoom=6)

    # Add Satellite Data Layers
    m.addLayer(get_satellite_data(), {"bands": ["NDWI"], "min": -1, "max": 1, "palette": ["blue", "white"]}, "NDWI")
    m.addLayer(get_satellite_data(), {"bands": ["Chlorophyll-a"], "min": 0, "max": 0.1, "palette": ["green", "yellow", "red"]}, "Chlorophyll-a")
    m.addLayer(get_satellite_data(), {"bands": ["Turbidity"], "min": 0, "max": 0.5, "palette": ["blue", "brown"]}, "Turbidity")

    # Render Map in Streamlit
    m.to_streamlit(height=500)

    st.write("### Water Quality Parameters from Satellite Data")
    st.write("- *NDWI*: Higher values indicate clearer water.")
    st.write("- *Chlorophyll-a*: Higher values suggest potential algal blooms.")
    st.write("- *Turbidity*: Higher values indicate more suspended particles in water.")

# Create a new row for the line chart, ETS decomposition, and alerts
with right_col:
    st.markdown("## 📈 Trend Over Time")
    selected_metric = st.selectbox('Choose Metric for Line Chart', list(metrics.keys()), key="line_chart")
    line_fig = px.line(filtered_data, x=filtered_data.index, y=selected_metric, title=f"{selected_metric} Over Time", height=300)
    st.plotly_chart(line_fig)

    st.markdown("## 🔍 ETS Decomposition")
    decomposition = seasonal_decompose(metrics[selected_metric], model='additive', period=30)
    fig, ax = plt.subplots(4, 1, figsize=(8, 10))
    decomposition.observed.plot(ax=ax[0], title='Observed')
    decomposition.trend.plot(ax=ax[1], title='Trend')
    decomposition.seasonal.plot(ax=ax[2], title='Seasonal')
    decomposition.resid.plot(ax=ax[3], title='Residual')
    plt.tight_layout()
    st.pyplot(fig)

# ARIMA Predictions for the Next 7 Days
def arima_predictions(data, metric, days=7):
    model = ARIMA(data[metric], order=(1, 1, 1))
    model_fit = model.fit()
    forecast = model_fit.forecast(steps=days)
    return forecast

predictions = {metric: arima_predictions(filtered_data, metric) for metric in metrics.keys()}
prediction_dates = [filtered_data.index[-1] + timedelta(days=i) for i in range(1, 8)]
predictions_df = pd.DataFrame(predictions, index=prediction_dates)

st.markdown("## 📅 7-Day Predictions")
st.table(predictions_df)

# Chatbot Placeholder
st.markdown("### 🤖 Chatbot")
st.text_input("Ask me anything about water quality metrics!")

# Twilio Alerts UI Element
st.markdown("### 🚨 Twilio Alerts")
st.markdown(
        "<div style='background-color: #ffffff; border: 1px solid #e0e0e0; box-shadow: 2px 2px 5px rgba(0, 0, 0, 0.1); padding: 10px; border-radius: 8px;'>"
        "<strong>Stay Informed!</strong> Receive real-time water quality alerts on your phone.",
        unsafe_allow_html=True
    )