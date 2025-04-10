import streamlit as st
import pandas as pd
import altair as alt
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
import joblib
import numpy as np

# Caching data loading
@st.cache_data
def get_data():
    try:
        data = pd.read_csv('Preprocessed_ForeignX.csv', parse_dates=['Time Serie'], index_col='Time Serie')
        return data
    except FileNotFoundError:
        st.error("Data file not found. Please ensure 'Preprocessed_ForeignX.csv' exists.")
        return pd.DataFrame()

# Caching model loading
@st.cache_resource
def load_currency_model():
    try:
        model_path = 'SINGAPORE - SINGAPORE DOLLAR_US__ARIMA_best_model.joblib'
        return joblib.load(model_path)
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# Forecasting function
def forecast_currency(model, days):
    forecast_result = model.forecast(steps=days)
    forecast_dates = pd.date_range(start=fx_data.index[-1] + pd.Timedelta(days=1), periods=days)
    forecast_df = pd.DataFrame({
        'Date': forecast_dates,
        'Forecasted Value': forecast_result
    })
    return forecast_df

# Load data
fx_data = get_data()

if not fx_data.empty:
    st.sidebar.header("Forecast Settings")
    
    # Currency selection dropdown
    selected_currency = st.sidebar.selectbox("Select Currency to Forecast:", fx_data.columns)

    # Forecast period slider
    forecast_days = st.sidebar.slider("Number of Days to Forecast:", min_value=1, max_value=30, value=7)

    # Main header
    st.title("💹 Forex Exchange Rate Predictor")
    
    # Display historical data
    st.header("Historical Exchange Rate Data")
    st.write(f"Showing historical data for {selected_currency}")
    
    # Plot historical data using Altair
    chart = alt.Chart(
        fx_data.reset_index()
    ).mark_line().encode(
        x='Time Serie:T',
        y=f'{selected_currency}:Q',
        tooltip=['Time Serie', selected_currency]
    ).properties(
        title=f"{selected_currency} - Historical Exchange Rate",
        width=800,
        height=400,
    )
    
    st.altair_chart(chart, use_container_width=True)
    
    # Generate forecast on button click
    if st.button("Generate Forecast"):
        with st.spinner("Generating forecast..."):
            model = load_currency_model()
            if model is not None:
                forecast_data = forecast_currency(model, forecast_days)
                
                # Display results and plot forecast
                st.subheader("Forecast Results")
                st.dataframe(forecast_data)
                
                fig, ax = plt.subplots(figsize=(10, 6))
                historical_period = min(30, len(fx_data))
                historical_data = fx_data[selected_currency].iloc[-historical_period:]
                
                ax.plot(historical_data.index, historical_data.values, label='Historical', color='blue')
                ax.plot(forecast_data['Date'], forecast_data['Forecasted Value'], label='Forecast', color='red', linestyle='--')
                
                ax.set_title(f"{selected_currency} - Forecast for Next {forecast_days} Days")
                ax.set_xlabel("Date")
                ax.set_ylabel("Exchange Rate")
                ax.legend()
                ax.grid(True)
                
                st.pyplot(fig)
