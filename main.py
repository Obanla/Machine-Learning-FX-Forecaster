import streamlit as st
import pandas as pd
import altair as alt
from sklearn.metrics import mean_squared_error
import joblib
import numpy as np
from statsmodels.tsa.arima.model import ARIMA



# Containers for different sections
header = st.container()
dataset = st.container()
features = st.container()
model_training = st.container()


@st.cache_data
def get_data():
    taxi_data = pd.read_csv('Preprocessed_ForeignX.csv')

    return taxi_data


# Prediction Section
st.header("Currency Forecast")
st.write(f"Showing {forecast_days} day forecast for {selected_currency}")

if st.button("Generate Forecast"):
    with st.spinner("Generating forecast..."):
        # Load model for selected currency
        model = load_currency_model(selected_currency)
        
        # Generate forecast
        forecast_data = forecast_currency(model, forecast_days)
        
        # Display forecast results
        st.subheader("Forecast Results")
        st.dataframe(forecast_data)
        
        # Plot forecast
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Get historical data for comparison
        historical_data = fx_data[selected_currency].iloc[-30:]  # Last 30 days
        
        # Plot historical data
        ax.plot(historical_data.index, historical_data.values, 
                label='Historical', color='blue')
        
        # Plot forecast
        ax.plot(forecast_data['Date'], forecast_data['Forecasted Value'], 
                label='Forecast', color='red', linestyle='--')
        
        ax.set_title(f"{selected_currency} - Forecast for Next {forecast_days} Days")
        ax.set_xlabel("Date")
        ax.set_ylabel("Exchange Rate")
        ax.legend()
        ax.grid(True)
        
        # Display the plot
        st.pyplot(fig)
        
        # Display confidence metrics
        st.subheader("Forecast Summary")
        st.write(f"Latest value: {historical_data.iloc[-1]:.4f}")
        st.write(f"Forecasted end value: {forecast_data['Forecasted Value'].iloc[-1]:.4f}")
        change = ((forecast_data['Forecasted Value'].iloc[-1] - historical_data.iloc[-1]) / 
                 historical_data.iloc[-1] * 100)
        st.write(f"Projected change: {change:.2f}%")


# Header Section
with header: 
    st.title('Welcome to my Machine Learning Forex Predictor')
    st.text('In this project, I worked on preprocessing, exploratory data analysis, '
            'and trained state-of-the-art machine learning models to predict foreign exchange rates '
            'for 22 countries.')





# Dataset Section
with dataset:
    st.header('Foreign Exchange Rate Dataset')
    st.text('The foreign exchange dataset includes rates for 22 different countries '
            'from 03-01-1999 to 31-12-2019.')
    
    # Link to dataset
    url = "https://github.com/furkhan67/forex-currency-predictor/tree/main/data"
    st.write("Click here to access the data: [Dataset Link](%s)" % url)
    
    # Load and display data
    try:
        fx_data = get_data()
        st.write(fx_data.head(5))  # Display first 5 rows of the dataset
    except FileNotFoundError:
        st.error("The file 'Preprocessed_ForeignX.csv' was not found. Please ensure it is located in the 'data' folder.")


    currency_columns = [col for col in fx_data.columns 
                    if col not in ['Time Serie']]

    selected_currencies = st.multiselect(
    "Select Currencies:",
    options=currency_columns,
    default=[currency_columns[0]]  # Default to first currency
)



    # Melt the DataFrame to long format
    melted_data = pd.melt(
        fx_data,
        id_vars=['Time Serie'],
        value_vars=currency_columns,
        var_name='Currency',
        value_name='Exchange Rate'
    )


# Filter selected currencies
    filtered_data = melted_data[melted_data['Currency'].isin(selected_currencies)]

# Plot using Altair
    chart = alt.Chart(filtered_data).mark_line().encode(
        x='Time Serie:T',
        y='Exchange Rate:Q',
        color='Currency:N',
        tooltip=['Currency', 'Time Serie', 'Exchange Rate']
        ).properties(
        title="Currency Exchange Rate Trends",
        width=800,
        height=400
    )


    st.altair_chart(chart, use_container_width=True)







# Features Section
with features:
    st.header('The Features I Used')
    st.text('Here are some of the key features used in this project:')

    st.markdown('**Price of a each currency and the date as it is a univariate data analysis**')


# Add UI components
st.title("Model Deployment with Streamlit")
input_value = st.number_input("Enter input value:")
if st.button("Predict"):
    prediction = model.predict([[input_value]])
    st.write(f"Prediction: {prediction}")




# Model Training Section (Placeholder for future content)
with model_training:
    st.header('Model Training')
    st.text('')

    # Load the mode
@st.cache_resource  # Cache the model to prevent reloading on every interactio
def load_model():
    return joblib.load('SINGAPORE - SINGAPORE DOLLAR_US__ARIMA_best_model.joblib')
model = load_model()

# Add currency selection dropdown
st.sidebar.header("Forecast Settings")
currency_options = [col for col in fx_data.columns if col != 'Time Serie']
selected_currency = st.sidebar.selectbox(
    "Select Currency to Forecast:",
    options=currency_options
)

# Add forecast period slider
forecast_days = st.sidebar.slider(
    "Number of Days to Forecast:",
    min_value=1,
    max_value=30,
    value=7
)


@st.cache_resource
def load_currency_model(currency_name):
    try:
        # Create a safe filename by replacing problematic characters
        safe_name = currency_name.replace(" ", "_").replace("/", "_").replace("-", "_")
        model_path = f'{safe_name}_ARIMA_model.joblib'
        return joblib.load(model_path)
    except FileNotFoundError:
        st.error(f"Model for {currency_name} not found. Using default model instead.")
        # If specific model not found, load your best model
        return joblib.load('SINGAPORE_-_SINGAPORE_DOLLAR_US__ARIMA_best_model.joblib')


def forecast_currency(model, days):
    # Get forecast for specified number of days
    forecast_result = model.forecast(steps=days)
    forecast_dates = pd.date_range(start=fx_data.index[-1] + pd.Timedelta(days=1), periods=days)
    
    # Create a DataFrame with the forecasted values
    forecast_df = pd.DataFrame({
        'Date': forecast_dates,
        'Forecasted Value': forecast_result
    })
    
    return forecast_df
