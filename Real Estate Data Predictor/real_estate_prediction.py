import pandas as pd
from matplotlib import pyplot as plt
from prophet import Prophet

# Step 1: Load and Reshape the Data
file_path = 'data/Metro_Data.csv'
df = pd.read_csv(file_path)

# Identify the date columns (they start with '2000-01-31' to '2024-09-30')
date_columns = df.columns[5:]  # Assuming the first 5 columns are metadata like 'RegionID', 'SizeRank', etc.

# Reshape the data (melt) using only the date columns
df_melted = df.melt(id_vars=['RegionName'], 
                    value_vars=date_columns, 
                    var_name='Date', 
                    value_name='Price')

# Convert 'Date' to datetime format
df_melted['Date'] = pd.to_datetime(df_melted['Date'], format='%Y-%m-%d')

# Filter for a specific region (e.g., 'New York, NY')
region_name = 'New York, NY'
region_data = df_melted[df_melted['RegionName'] == region_name]

# Keep only the 'Date' and 'Price' columns for the time series model
region_data = region_data[['Date', 'Price']]

# Display the first few rows to check the structure
print(f"Data for {region_name}:")
print(region_data.head())

# Step 2: Prepare the Data for Prophet and Train the Model
# Rename columns to match Prophet's expectations ('ds' for dates, 'y' for the target variable)
region_data = region_data.rename(columns={'Date': 'ds', 'Price': 'y'})

# Initialize the Prophet model
model = Prophet()

# Fit the model on historical data
model.fit(region_data)

# Step 3: Forecast Future Prices
# Create a dataframe for future dates (e.g., predict for the next 12 months)
future = model.make_future_dataframe(periods=12, freq='M')  # You can increase 'periods' for more future months

# Predict future values
forecast = model.predict(future)

# Display the forecasted results (first few rows of the future predictions)
print("Forecasted prices:")
print(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail())

# Step 4: Plot the Forecast
# Plot the forecast with historical data
model.plot(forecast)
plt.title(f'Real Estate Price Forecast for {region_name}')
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()

# Optional: If you want to save the forecast data to a CSV file
forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].to_csv(f'forecast_{region_name}.csv', index=False)
