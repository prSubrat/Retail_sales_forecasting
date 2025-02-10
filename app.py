import streamlit as st
import pandas as pd
import joblib
from prophet.plot import plot_plotly
import datetime

# Load the trained Prophet model
model = joblib.load("prophet_model.pkl")

# Streamlit App UI
st.title("Retail Sales Forecasting Application")
st.write("Use this application to forecast future retail sales based on historical data.")

# Sidebar for user inputs
st.sidebar.header("User Input Parameters")

# User input for start date
start_date = st.sidebar.date_input(
    "Select Start Date",
    value=datetime.date.today(),
    min_value=datetime.date(2000, 1, 1),
    max_value=datetime.date.today()
)

# User input for forecast period
n_days = st.sidebar.slider("Forecast Period (days)", min_value=7, max_value=365, value=30)

# "Generate Forecast" button
if st.sidebar.button("Generate Forecast"):
    # Load and prepare data
    df = pd.read_csv("retail_sales_dataset.csv")  # Ensure dataset has a 'Date' column
    df['ds'] = pd.to_datetime(df['Date'])

    # Filter the dataframe based on the selected start date
    df_filtered = df[df['ds'] >= pd.to_datetime(start_date)]

    # Create future dataframe starting from the selected start date
    future = model.make_future_dataframe(periods=n_days)
    future = future[future['ds'] >= pd.to_datetime(start_date)]

    # Predict future sales
    forecast = model.predict(future)

    # Display forecasted values
    st.subheader("Forecasted Sales")
    st.write(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail())

    # Plot the forecast
    fig = plot_plotly(model, forecast)
    st.plotly_chart(fig)

    # Download predictions
    csv = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].to_csv(index=False)
    st.download_button("Download Forecast Data", csv, "forecast.csv", "text/csv")

    st.write("Built with ❤️ using Streamlit and Prophet!")
else:
    st.info("Please set the parameters in the sidebar and click 'Generate Forecast' to view the results.")

import streamlit as st
import pandas as pd

# Create a file uploader widget
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

# Check if the file has been uploaded
if uploaded_file is not None:
    # Read the file as a CSV
    df = pd.read_csv("retail_sales_dataset.csv")

    # Display the DataFrame
    st.write(df)


