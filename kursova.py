import streamlit as st
import requests
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import statsmodels.api as sm

# Функція для отримання історичних даних про погоду
def get_historical_weather(api_key, city, days):
    try:
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')

        url = f'https://api.weatherbit.io/v2.0/history/daily?city={city}&start_date={start_date}&end_date={end_date}&key={api_key}'
        response = requests.get(url)
        response.raise_for_status()  # Перевірка на помилки HTTP
        weather_data = response.json()

        data_list = [{
            'Date': day['datetime'],
            'Temperature': day['temp'],
            'Humidity': day['rh'],
            'WindSpeed': day['wind_spd']  # Додавання швидкості вітру
        } for day in weather_data['data']]
        return pd.DataFrame(data_list)
    except requests.RequestException as e:
        st.error(f"Помилка отримання даних про погоду: {e}")
        return None

# Функція для прогнозування температури, вологості та швидкості вітру
def forecast_weather(historical_df, forecast_days):
    historical_df['Date'] = pd.to_datetime(historical_df['Date'])
    historical_df.set_index('Date', inplace=True)

    # Параметри моделі
    temp_order = (1, 1, 1)
    temp_seasonal_order = (1, 1, 1, 12)
    humidity_order = (1, 1, 1)
    humidity_seasonal_order = (1, 1, 1, 12)
    wind_speed_order = (1, 1, 1)
    wind_speed_seasonal_order = (1, 1, 1, 12)

    # Прогнозування температури
    model_temperature = sm.tsa.statespace.SARIMAX(
        historical_df['Temperature'],
        order=temp_order,
        seasonal_order=temp_seasonal_order,
        enforce_stationarity=False,
        enforce_invertibility=False
    )
    result_temperature = model_temperature.fit(disp=False)
    forecast_temperature = result_temperature.get_forecast(steps=forecast_days)

    # Прогнозування вологості
    model_humidity = sm.tsa.statespace.SARIMAX(
        historical_df['Humidity'],
        order=humidity_order,
        seasonal_order=humidity_seasonal_order,
        enforce_stationarity=False,
        enforce_invertibility=False
    )
    result_humidity = model_humidity.fit(disp=False)
    forecast_humidity = result_humidity.get_forecast(steps=forecast_days)

    # Прогнозування швидкості вітру
    model_wind_speed = sm.tsa.statespace.SARIMAX(
        historical_df['WindSpeed'],
        order=wind_speed_order,
        seasonal_order=wind_speed_seasonal_order,
        enforce_stationarity=False,
        enforce_invertibility=False
    )
    result_wind_speed = model_wind_speed.fit(disp=False)
    forecast_wind_speed = result_wind_speed.get_forecast(steps=forecast_days)

    # Підготовка даних для прогнозу
    forecast_index = pd.date_range(start=historical_df.index[-1] + timedelta(days=1), periods=forecast_days, freq='D')
    forecast_df = pd.DataFrame({
        'Date': forecast_index,
        'Temperature': forecast_temperature.predicted_mean,
        'Humidity': forecast_humidity.predicted_mean,
        'WindSpeed': forecast_wind_speed.predicted_mean
    })

    return forecast_df

# Функція для візуалізації даних
def process_and_plot_data(historical_df, forecast_df):
    # Візуалізація для температури
    fig, ax = plt.subplots(figsize=(14, 10))
    sns.lineplot(x=historical_df.index, y='Temperature', data=historical_df, marker='o', label='Історичні дані', ax=ax)
    sns.lineplot(x=forecast_df['Date'], y='Temperature', data=forecast_df, marker='o', color='orange', label='Прогноз', ax=ax)
    ax.set(title='Температура (Історичні дані і прогноз)', xlabel='Date', ylabel='Temperature (°C)')
    plt.xticks(rotation=45)
    ax.legend()
    st.pyplot(fig)

    # Візуалізація для вологості
    fig, ax = plt.subplots(figsize=(14, 10))
    sns.lineplot(x=historical_df.index, y='Humidity', data=historical_df, marker='o', color='green', label='Історичні дані', ax=ax)
    sns.lineplot(x=forecast_df['Date'], y='Humidity', data=forecast_df, marker='o', color='purple', label='Прогноз', ax=ax)
    ax.set(title='Вологість (Історичні дані і прогноз)', xlabel='Date', ylabel='Humidity (%)')
    plt.xticks(rotation=45)
    ax.legend()
    st.pyplot(fig)

    # Візуалізація для швидкості вітру
    fig, ax = plt.subplots(figsize=(14, 10))
    sns.lineplot(x=historical_df.index, y='WindSpeed', data=historical_df, marker='o', color='blue', label='Історичні дані', ax=ax)
    sns.lineplot(x=forecast_df['Date'], y='WindSpeed', data=forecast_df, marker='o', color='red', label='Прогноз', ax=ax)
    ax.set(title='Швидкість вітру (Історичні дані і прогноз)', xlabel='Date', ylabel='Швидкість вітру (m/s)')
    plt.xticks(rotation=45)
    ax.legend()
    st.pyplot(fig)

# Основна функція Streamlit
def main():
    st.title("Прогноз погоди")

    api_key = st.sidebar.text_input("API Key Weatherbit", "Ведіть API ключ від Weatherbit")
    city = st.sidebar.text_input("Місто", "Тернопіль")
    historical_days = st.sidebar.number_input("Історичні дні", min_value=1, max_value=365, value=60)
    forecast_days = st.sidebar.number_input("Дні прогнозу", min_value=1, max_value=60, value=10)

    if st.sidebar.button("Отримати дані та зробити прогноз"):
        with st.spinner('Отримання даних...'):
            historical_df = get_historical_weather(api_key, city, historical_days)

        if historical_df is not None:
            forecast_df = forecast_weather(historical_df, forecast_days)

            st.subheader("Історичні дані")
            st.write(historical_df)

            st.subheader("Дані прогнозу")
            st.write(forecast_df)

            st.subheader("Візуалізація даних")
            process_and_plot_data(historical_df, forecast_df)
        else:
            st.error("Історичні дані не отримано.")

# Запуск основної програми
if __name__ == "__main__":
    main()
