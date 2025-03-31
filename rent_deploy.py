import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from prophet import Prophet
from prophet.plot import plot_plotly
import matplotlib.pyplot as plt
from PIL import Image
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import statsmodels.api as sm
#from tensorflow.keras.models import load_model


# Set up Streamlit page layout
st.set_page_config(layout="wide")

# Custom CSS for Helvetica font
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Helvetica:wght@300;400;700&display=swap');
    
    html, body, [class*="css"]  {
        font-family: 'Helvetica', sans-serif;
    }

    .centered-title {
        font-size: 36px;
        font-weight: bold;
        text-align: center;
    }
    </style>
    """, unsafe_allow_html=True)

# Create two columns to place the title and image side by side
col1, col2 = st.columns([2, 1])

with col1:
    # Display the title in the first column
    st.markdown('<p class="centered-title">Comprehensive Time Series Analysis of Monthly Rent Trends Across US States</p>', unsafe_allow_html=True)

with col2:
    # Display the image in the second column
    image = Image.open("/Users/taichowdhury/Documents/Interview_Prep/RentData/rent.jpeg")
    st.image(image, use_column_width=True)

# Sidebar for file upload, date selection, forecast slider, and clear button
st.sidebar.header('Upload Data, Select State, Date Range & Forecast Period')

uploaded_file = st.sidebar.file_uploader("Upload CSV file", type=["csv"])

if st.sidebar.button("Clear"):
    uploaded_file = None
    st.experimental_rerun()



# Create tabs for all sections
tabs = st.tabs(["About", "Stationary", "ARIMA", "SARIMAX", "PROPHET"]) # took off "LSTM", code under construction 

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)

    # Ensure 'Date' column is in datetime format
    data['Date'] = pd.to_datetime(data['Date'], errors='coerce')
    data = data.dropna(subset=['StateName', 'Date', 'Value'])

    # Select state from dropdown
    states = data['StateName'].unique()
    selected_state = st.sidebar.selectbox('Select State:', states)

    # Add start date and end date filter to sidebar
    start_date = st.sidebar.date_input("Start Date", value=data['Date'].min())
    end_date = st.sidebar.date_input("End Date", value=data['Date'].max())

    if start_date > end_date:
        st.sidebar.error("Error: End date must be after start date.")
    else:
        # Filter data based on state and date range
        filtered_data = data[(data['StateName'] == selected_state) & 
                             (data['Date'] >= pd.to_datetime(start_date)) & 
                             (data['Date'] <= pd.to_datetime(end_date))]

        # Ensure the data is sorted by date
        filtered_data = filtered_data.sort_values(by='Date')

        # Add differencing and log transformation for stationarization
        filtered_data['Value_diff'] = filtered_data['Value'].diff()
        filtered_data['Value_log'] = np.log(filtered_data['Value'])

        # Display stationarized data
        #st.write("Stationarized Data (First 10 Rows):")
        #st.write(filtered_data[['Date', 'Value', 'Value_diff', 'Value_log']].head(10))

        # Forecast months slider
        forecast_months = st.sidebar.slider('Select Forecast Period (Months):', 0, 36, 12)


        # Create tabs
        # Initialize RMSE variables for each model

        rmse_values = {"ARIMA": None, "SARIMAX": None, "PROPHET": None}
        

        # About Tab: Data Exploration and Top 5 Regions
        with tabs[0]:
            st.subheader("Data Exploration")
            st.write(f"This dataset is collected from Zillow and covers monthly rent trends from {start_date} to {end_date} for the selected state: {selected_state}. The goal of this analysis is to forecast future rent trends and identify locations (cities/regions) that are driving rent prices in {selected_state}. By analyzing the time series data and its stationarity, we can build ARIMA, SARIMAX, and PROPHET models for rent forecasting.")

            st.write(f"Displaying filtered data for the selected state: {selected_state} from {start_date} to {end_date}")
            st.dataframe(filtered_data)

            st.subheader(f"Historical Rent Values for {selected_state}")
            fig = px.line(filtered_data, x='Date', y='Value')
            st.plotly_chart(fig)


            # Top 5 and Bottom 5 Regions
            st.subheader(f"Top 5 and Lowest 5 Regions Driving Rent Prices for {selected_state}")
            last_10_years = data[data['Date'].dt.year >= data['Date'].dt.year.max() - 10]
            state_data = last_10_years[last_10_years['StateName'] == selected_state]

            top_regions = state_data.groupby('RegionName')['Value'].mean().nlargest(5).reset_index()
            fig_top = px.bar(top_regions, x='RegionName', y='Value', title="Top 5 Regions", labels={'Value': 'Average Rent Value', 'RegionName': 'Region'})
            st.plotly_chart(fig_top)

            bottom_regions = state_data.groupby('RegionName')['Value'].mean().nsmallest(5).reset_index()
            fig_bottom = px.bar(bottom_regions, x='RegionName', y='Value', title="Lowest 5 Regions", labels={'Value': 'Average Rent Value', 'RegionName': 'Region'})
            st.plotly_chart(fig_bottom)
            
                # Adding the two new charts
            st.subheader("Yearly Average Rent Trends")
            filtered_data['Year'] = filtered_data['Date'].dt.year  # Extract year from Date
            yearly_avg_rent = filtered_data.groupby('Year', as_index=False)['Value'].mean()
            yearly_avg_rent.rename(columns={'Value': 'Avg_Rent'}, inplace=True)

            fig_yearly = px.line(yearly_avg_rent, x='Year', y='Avg_Rent', title=f"Yearly Rent Trends in {selected_state}")
            st.plotly_chart(fig_yearly)

            st.subheader("Rent Distribution Across Cities")
            fig_distribution = px.box(filtered_data, x='RegionName', y='Value', title=f"Rent Distribution in Different Cities of {selected_state}")
            st.plotly_chart(fig_distribution)

        # Stationary Tab: ADF Test, PACF and ACF Plots
        with tabs[1]:
            st.subheader(f"ADF Test, ACF, and PACF for {selected_state}")

            if st.checkbox('Perform ADF Test'):
                values = filtered_data['Value'].dropna()
                adf_result = adfuller(values)
                adf_statistic = round(adf_result[0], 2)
                p_value = round(adf_result[1], 2)
                critical_values = {key: round(val, 2) for key, val in adf_result[4].items()}

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("ADF Statistic", adf_statistic)
                with col2:
                    st.metric("p-value", p_value)
                with col3:
                    st.metric("1% Critical Value", critical_values['1%'])

                col4, col5, col6 = st.columns(3)
                with col4:
                    st.metric("5% Critical Value", critical_values['5%'])
                with col5:
                    st.metric("10% Critical Value", critical_values['10%'])

                if adf_statistic > critical_values['1%'] and adf_statistic > critical_values['5%'] and adf_statistic > critical_values['10%']:
                    st.markdown(f"<p style='color:green; font-size:20px;'>The state's rent values are <b>non-stationary</b> as the ADF Statistic is greater than all three critical values.</p>", unsafe_allow_html=True)
                else:
                    st.markdown(f"<p style='color:red; font-size:20px;'>The state's rent values are <b>stationary</b> as the ADF Statistic is less than one or more critical values.</p>", unsafe_allow_html=True)

            if st.checkbox('Show ACF and PACF Plots'):
                values = filtered_data['Value'].dropna()

                fig, axes = plt.subplots(1, 2, figsize=(14, 6))
                plot_acf(values, lags=40, ax=axes[0], title=f"ACF for {selected_state}")
                plot_pacf(values, lags=40, ax=axes[1], title=f"PACF for {selected_state}")
                st.pyplot(fig)

        with tabs[2]:
            st.subheader(f"ARIMA Forecast for {selected_state}")

            if st.checkbox('ARIMA Forecast', key="arima_forecast"):
                # Ensure the 'Date' column is in datetime format
                filtered_data['Date'] = pd.to_datetime(filtered_data['Date'])

                # Split the data into train and test sets
                train_data = filtered_data[filtered_data['Date'] < '2023-01-01']
                test_data = filtered_data[filtered_data['Date'] >= '2023-01-01']

                # Extract the 'Value' column for train and test sets
                train_values = train_data['Value']
                test_values = test_data['Value']

                # Build ARIMA model (order (p, d, q))
                arima_model = ARIMA(train_values, order=(2, 0, 0))  # Example order, adjust as needed
                arima_results = arima_model.fit()

                # Display ARIMA model fit summary
                st.subheader("ARIMA Model Fit Summary")
                st.text(arima_results.summary().as_text())

                # Forecasting on the test set
                forecast_steps = len(test_values)
                arima_forecast_test = arima_results.forecast(steps=forecast_steps)

                # Calculate RMSE on test data
                from sklearn.metrics import mean_squared_error
                import numpy as np

                rmse_test = np.sqrt(mean_squared_error(test_values, arima_forecast_test))
                st.write(f"RMSE on test data: {rmse_test}")

                # Forecasting future values
                forecast_months = st.sidebar.slider('Select Forecast Period (Months):', 1, 36, 12, key="arima_forecast_months")
                arima_forecast_future = arima_results.get_forecast(steps=forecast_months)
                forecast_values_future = arima_forecast_future.predicted_mean

                # Confidence intervals for the future forecast
                confidence_intervals_future = arima_forecast_future.conf_int()
                lower_bounds_future = confidence_intervals_future.iloc[:, 0]
                upper_bounds_future = confidence_intervals_future.iloc[:, 1]

                # Define the forecast index for future values
                last_date = filtered_data['Date'].max()
                forecast_index_future = pd.date_range(
                    start=last_date + pd.DateOffset(months=1),
                    periods=forecast_months,
                    freq='MS'
                )

                # Create the forecast dataframe for future values
                forecast_df_future = pd.DataFrame({
                    'Date': forecast_index_future,
                    'Forecasted Rent Value': forecast_values_future.values,
                    'Lower Bound': lower_bounds_future.values,
                    'Upper Bound': upper_bounds_future.values
                })

                # Display historical and forecast data
                st.subheader(f"ARIMA Forecast with Confidence Interval for {selected_state}")

                fig_arima = go.Figure()

                # Add historical data to the plot
                fig_arima.add_trace(
                    go.Scatter(
                        x=train_data['Date'],
                        y=train_data['Value'],
                        mode='lines',
                        name='Training Data'
                    )
                )

                # Add test data to the plot
                fig_arima.add_trace(
                    go.Scatter(
                        x=test_data['Date'],
                        y=test_data['Value'],
                        mode='lines',
                        name='Test Data'
                    )
                )

                # Add ARIMA forecast on test data
                fig_arima.add_trace(
                    go.Scatter(
                        x=test_data['Date'],
                        y=arima_forecast_test,
                        mode='lines',
                        name='Forecast on Test Data'
                    )
                )

                # Add ARIMA forecast for future values
                fig_arima.add_trace(
                    go.Scatter(
                        x=forecast_df_future['Date'],
                        y=forecast_df_future['Forecasted Rent Value'],
                        mode='lines',
                        name='Future Forecast'
                    )
                )

                # Add shaded confidence intervals for future forecasts
                fig_arima.add_trace(
                    go.Scatter(
                        x=pd.concat([forecast_df_future['Date'], forecast_df_future['Date'][::-1]]),
                        y=pd.concat([forecast_df_future['Upper Bound'], forecast_df_future['Lower Bound'][::-1]]),
                        fill='toself',
                        fillcolor='rgba(0, 100, 200, 0.2)',
                        line=dict(color='rgba(255,255,255,0)'),
                        name='Confidence Interval',
                        hoverinfo="skip",
                        showlegend=True
                    )
                )

                # Customize layout
                fig_arima.update_layout(
                    title="ARIMA Forecasted Rent Values with Confidence Interval",
                    xaxis_title="Date",
                    yaxis_title="Rent Value",
                    legend_title="Legend"
                )

                # Display the chart
                st.plotly_chart(fig_arima)

                # Display the ticker
                st.subheader("Forecasted Rent Values:")
                ticker_data = forecast_df_future[['Date', 'Forecasted Rent Value']].reset_index(drop=True)
                st.table(ticker_data)


        # SARIMAX Tab
        with tabs[3]:
            st.subheader(f"SARIMAX Forecast for {selected_state}")

            if st.checkbox('SARIMAX Forecast', key="sarimax_forecast"):
                # Ensure the 'Date' column is in datetime format
                filtered_data['Date'] = pd.to_datetime(filtered_data['Date'])

                # Compute differenced values for stationarity
                filtered_data['Value_diff'] = filtered_data['Value'].diff()
                filtered_data = filtered_data.dropna(subset=['Value_diff'])  # Drop NaN rows after differencing

                # Split the data into train and test sets
                train_data = filtered_data[filtered_data['Date'] < '2023-01-01']
                test_data = filtered_data[filtered_data['Date'] >= '2023-01-01']

                # Extract the differenced values
                train_values = train_data['Value']
                test_values = test_data['Value']

                # SARIMAX model training
                sarimax_model = SARIMAX(train_values, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
                sarimax_results = sarimax_model.fit()

                # Display SARIMAX model fit summary
                st.subheader("SARIMAX Model Fit Summary")
                st.text(sarimax_results.summary().as_text())

                # Forecasting on the test set
                forecast_steps = len(test_values)
                sarimax_forecast_test = sarimax_results.forecast(steps=forecast_steps)

                # Calculate RMSE on test data
                from sklearn.metrics import mean_squared_error
                import numpy as np

                rmse_test = np.sqrt(mean_squared_error(test_values, sarimax_forecast_test))
                st.write(f"RMSE on test data: {rmse_test}")

                # Plot historical, test, and future forecast data
                st.subheader(f"SARIMAX Forecast with Confidence Interval for {selected_state}")

                fig_sarimax = go.Figure()

                # Add training data
                fig_sarimax.add_trace(
                    go.Scatter(
                        x=train_data['Date'],
                        y=train_data['Value'],
                        mode='lines',
                        name='Training Data'
                    )
                )

                # Add test data
                fig_sarimax.add_trace(
                    go.Scatter(
                        x=test_data['Date'],
                        y=test_data['Value'],
                        mode='lines',
                        name='Test Data'
                    )
                )

                # Add SARIMAX forecast on test data
                fig_sarimax.add_trace(
                    go.Scatter(
                        x=test_data['Date'],
                        y=sarimax_forecast_test,
                        mode='lines',
                        name='Forecast on Test Data'
                    )
                )

                # Customize layout
                fig_sarimax.update_layout(
                    title="SARIMAX Forecast with Confidence Interval",
                    xaxis_title="Date",
                    yaxis_title="Differenced Rent Value",
                    legend_title="Legend"
                )

                # Display the plot
                st.plotly_chart(fig_sarimax)

                # Forecasting future values for table display
                forecast_months = st.sidebar.slider('Forecast Period (Months):', 1, 36, 12, key="sarimax_forecast_months")
                sarimax_forecast_future = sarimax_results.get_forecast(steps=forecast_months)
                forecast_values_future = sarimax_forecast_future.predicted_mean

                # Define the forecast index for future values
                last_date = filtered_data['Date'].max()
                forecast_index_future = pd.date_range(
                    start=last_date + pd.DateOffset(months=1),
                    periods=forecast_months,
                    freq='MS'
                )

                # Create the forecast dataframe for the table
                forecast_df = pd.DataFrame({
                    'Date': forecast_index_future,
                    'Forecasted Rent Value': forecast_values_future
                })

                # Display the ticker table for forecasted values
                st.subheader("Forecasted Rent Values:")
                ticker_data = forecast_df[['Date', 'Forecasted Rent Value']].reset_index(drop=True)
                st.table(ticker_data)





        # PROPHET Tab
        with tabs[4]:
            st.subheader(f"PROPHET Forecast for {selected_state}")

            if st.checkbox('PROPHET Forecast', key="prophet_forecast"):
                # Ensure the 'Date' column is in datetime format
                filtered_data['Date'] = pd.to_datetime(filtered_data['Date'])

                # Rename columns for Prophet
                prophet_data = filtered_data[['Date', 'Value']].dropna()
                prophet_data.columns = ['ds', 'y']

                # Split the data into train and test sets
                train_data = prophet_data[prophet_data['ds'] < '2023-01-01']
                test_data = prophet_data[prophet_data['ds'] >= '2023-01-01']

                # Build Prophet model
                prophet_model = Prophet()
                prophet_model.fit(train_data)

                # Make a dataframe for test data prediction
                test_forecast = prophet_model.predict(test_data[['ds']])

                # Calculate RMSE on test data
                from sklearn.metrics import mean_squared_error
                import numpy as np

                rmse_test = np.sqrt(mean_squared_error(test_data['y'], test_forecast['yhat']))
                st.write(f"RMSE on test data: {rmse_test}")

                # Make future dataframe for forecast
                forecast_months = st.sidebar.slider('Select Forecast Period (Months):', 1, 36, 12, key="prophet_forecast_months")
                future = prophet_model.make_future_dataframe(periods=forecast_months, freq='MS')
                future_forecast = prophet_model.predict(future)

                # Plot Prophet forecast
                st.subheader(f"Prophet Forecast with Confidence Interval for {selected_state}")
                fig_prophet = plot_plotly(prophet_model, future_forecast)
                st.plotly_chart(fig_prophet)

                # Create a dataframe for forecasted future values
                forecast_df_prophet = future_forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(forecast_months)
                forecast_df_prophet.columns = ['Date', 'Forecasted Rent Value', 'Lower Bound', 'Upper Bound']

                # Plot the forecasted future values
                st.subheader(f"Forecasted Rent Values for the Next {forecast_months} Months")
                fig_forecast = px.line(
                    forecast_df_prophet,
                    x='Date',
                    y='Forecasted Rent Value',
                    title=f"Forecasted Rent Values for {selected_state}",
                    labels={'Forecasted Rent Value': 'Rent Value'}
                )
                fig_forecast.add_scatter(
                    x=forecast_df_prophet['Date'],
                    y=forecast_df_prophet['Lower Bound'],
                    mode='lines',
                    name='Lower Bound',
                    line=dict(dash='dot')
                )
                fig_forecast.add_scatter(
                    x=forecast_df_prophet['Date'],
                    y=forecast_df_prophet['Upper Bound'],
                    mode='lines',
                    name='Upper Bound',
                    line=dict(dash='dot')
                )
                st.plotly_chart(fig_forecast)

                # Display forecasted rent values in a table
                st.subheader("Forecasted Rent Values:")
                st.table(forecast_df_prophet[['Date', 'Forecasted Rent Value']].reset_index(drop=True))
        
        # LSTM Tab
        # with tabs[5]:
        #     st.subheader(f"LSTM Forecast for {selected_state}")
            
        #     if uploaded_file is None:
        #         st.write("Upload a dataset to access LSTM forecasting.")
        #     else:
        #         try:
        #             # Load the LSTM model
        #             import pickle
        #             with open('lstm_model_all_states.pkl', 'rb') as file:
        #                 lstm_model = pickle.load(file)

        #             # Example of using the LSTM model
        #             forecast_dates, forecast_values = forecast_with_lstm(
        #                 lstm_model, filtered_data, selected_state, n_forecast=forecast_months
        #             )

        #             forecast_df = pd.DataFrame({'Date': forecast_dates, 'Forecasted Values': forecast_values})
        #             st.write(forecast_df)

        #             fig_lstm = go.Figure()
        #             fig_lstm.add_trace(go.Scatter(x=filtered_data['Date'], y=filtered_data['Value'], mode='lines', name='Historical Data'))
        #             fig_lstm.add_trace(go.Scatter(x=forecast_df['Date'], y=forecast_df['Forecasted Values'], mode='lines', name='LSTM Forecast'))
        #             fig_lstm.update_layout(title=f"LSTM Forecast for {selected_state}", xaxis_title='Date', yaxis_title='Rent Value')
        #             st.plotly_chart(fig_lstm)
        #         except FileNotFoundError:
        #             st.error("LSTM model file not found. Please ensure the model file is in the correct directory.")
        #         except Exception as e:
        #             st.error(f"An error occurred while loading the LSTM model: {e}")

else:
    st.write("Upload a dataset to begin the analysis.")
    
    
    
    




