import pandas as pd
import numpy as np
import sqlite3
import plotly.express as px
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.preprocessing import OneHotEncoder
import pickle
from prophet import Prophet
import streamlit as st
from datetime import timedelta

demand_file_path = 'data/filtered_demand.csv'
inventory_file_path = 'data/filtered_inventory.csv'
stock_file_path = 'data/filtered_stock.csv'
lead_time_file_path = 'data/average_lead_time.csv'
mean_stock_file_path = 'data/mean_stock.csv'
db_path = 'results.db'
model_dir = 'models/'


def remove_punctuation(column):
    return column.str.replace(',', '', regex=True).replace('.00', '', regex=True).str.strip()

def add_data_to_csv(file_path, data):
    df = pd.read_csv(file_path)
    new_data = pd.DataFrame(data)
    updated_df = pd.concat([df, new_data], ignore_index=True)
    updated_df.to_csv(file_path, index=False)
    return updated_df

def update_stock_csv():
    demand_df = pd.read_csv(demand_file_path)
    inventory_df = pd.read_csv(inventory_file_path)
    stock_df = make_stock_frame(inventory_df, demand_df)
    stock_df.to_csv(stock_file_path, index=False)
    return stock_df

def make_stock_frame(inventory, demand):
    inventory = inventory.copy()
    demand = demand.copy()
    inventory['y'] = inventory['y'].fillna(0)
    demand['y'] = demand['y'].fillna(0)
    combined = pd.concat([inventory, demand])
    combined['y'] = combined.apply(lambda row: -row['y'] if row.name in demand.index else row['y'], axis=1)
    stock = combined.groupby(['i', 't']).sum().groupby('i').cumsum().reset_index()
    stock0 = stock.groupby('i')['y'].transform('min')
    stock['y'] = stock['y'] - stock0
    return stock

def calc_safety_stock(demand, avg_lead_time, service_level):
    L = avg_lead_time / np.timedelta64(1, 'D')
    Z = stats.norm.ppf(service_level)
    sigma_d = demand['y'].std()
    ss = Z * sigma_d * np.sqrt(L)
    return ss

def calc_reorder_point(demand, avg_lead_time, safety_stock):
    L = avg_lead_time / np.timedelta64(1, 'D')
    mean_d = demand['y'].mean()
    return np.ceil(mean_d * L + safety_stock)

def load_forecast_from_db(product_id, model_name):
    conn = sqlite3.connect(db_path)
    query = f"SELECT * FROM {model_name}_forecast WHERE product_id = '{product_id}'"
    df = pd.read_sql(query, conn)
    conn.close()
    return df

def predict_and_plot_prophet(product_id):
    prophet_forecast = load_forecast_from_db(product_id, "prophet")
    if not prophet_forecast.empty:
        prophet_forecast['ds'] = pd.to_datetime(prophet_forecast['ds'])
        st.write(f"상품 ID {product_id}에 대한 Prophet 예측 결과 (4주)")
        st.write(prophet_forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']])
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(prophet_forecast['ds'], prophet_forecast['yhat'], label='예측값', color='blue')
        ax.fill_between(prophet_forecast['ds'], prophet_forecast['yhat_lower'], prophet_forecast['yhat_upper'], color='lightblue', alpha=0.5)
        ax.set_title(f'상품 ID: {product_id}의 Prophet 예측 (4주)')
        ax.set_xlabel('날짜')
        ax.set_ylabel('예측 값')
        ax.legend()
        st.pyplot(fig)
    else:
        st.write(f'상품 ID {product_id}에 대한 Prophet 예측 결과가 없습니다.')

def load_lightgbm_model(product_id):
    model_path = f'{model_dir}lightgbm_model_{product_id}.pkl'
    with open(model_path, 'rb') as file:
        model = pickle.load(file)
    return model

def predict_and_plot_lightgbm(product_id, demand_df):
    model = load_lightgbm_model(product_id)
    if model:
        product_demand = demand_df[demand_df['i'] == product_id].copy()
        product_demand = product_demand.sort_values('t')
        
        last_date = product_demand['t'].max()
        future_dates = pd.date_range(start=last_date + timedelta(weeks=1), periods=4, freq='W-MON')
        
        future_demand = pd.DataFrame({'t': future_dates})
        future_demand['day'] = future_demand['t'].dt.day
        future_demand['month'] = future_demand['t'].dt.month
        future_demand['year'] = future_demand['t'].dt.year
        future_demand['dayofweek'] = future_demand['t'].dt.dayofweek

        for lag in range(1, 7):
            future_demand[f'lag_{lag}'] = product_demand['y'].iloc[-lag] if len(product_demand) >= lag else np.nan
        
        encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
        encoded_features = encoder.fit_transform(product_demand[['i']])
        encoded_df = pd.DataFrame(encoded_features, columns=encoder.get_feature_names_out(['i']))
        future_encoded_df = pd.DataFrame([encoded_features[0]] * len(future_demand), columns=encoder.get_feature_names_out(['i']))
        
        data_encoded = pd.concat([future_demand.reset_index(drop=True), future_encoded_df.reset_index(drop=True)], axis=1)

        data_encoded = data_encoded.dropna()
        
        X = data_encoded.drop(columns=['t'])
        y_pred = model.predict(X)
        
        result_df = future_demand[['t']].copy()
        result_df['Predicted Demand'] = y_pred
        
        st.write(f"상품 ID {product_id}의 향후 4주 LightGBM 예측 결과")
        st.table(result_df)

        fig, ax = plt.subplots(figsize=(14, 7))
        ax.plot(result_df['t'], y_pred, label='예측값', color='red', alpha=0.7)
        ax.set_xlabel('날짜')
        ax.set_ylabel('수요')
        ax.set_title(f'상품 ID: {product_id}의 향후 4주 LightGBM 예측 수요')
        ax.legend()
        st.pyplot(fig)
    else:
        st.write(f'상품 ID {product_id}에 대한 LightGBM 모델이 없습니다.')

def plot_product_demand(product_id, test, y_test, y_pred):
    idx = f'i_{product_id}'
    test_i = test[test[idx] == 1]
    plt.figure(figsize=(14, 7))
    plt.plot(test_i['t'], y_test.loc[test_i.index], label='실제값')
    plt.plot(test_i['t'], y_pred[test_i.index.values], label='예측값', alpha=0.7)
    plt.xlabel('날짜')
    plt.ylabel('수요')
    plt.title(f'상품 ID: {product_id}의 실제 수요와 LightGBM 예측 수요')
    plt.legend()
    st.pyplot()

def simulate_stock(product_id):
    product_id = str(product_id)
    prophet_forecast = load_forecast_from_db(product_id, "prophet")
    prophet_forecast['ds'] = pd.to_datetime(prophet_forecast['ds'])
    lead_time_df = pd.read_csv(lead_time_file_path)
    mean_stock_df = pd.read_csv(mean_stock_file_path)
    avg_lead_time = lead_time_df[['상품코드_WMS_', 'average_lead_time']].drop_duplicates()
    avg_lead_time.set_index('상품코드_WMS_',inplace=True)
    avg_order_quantity = mean_stock_df.set_index('상품코드_WMS_')['mean_stock']
    demand_df = pd.read_csv(demand_file_path)
    stock_df = pd.read_csv(stock_file_path)
    stock_df['t'] = pd.to_datetime(stock_df['t'])
    s = demand_df[demand_df['i'] == product_id]['y'].std()
    lead_time_str = avg_lead_time.loc['650902', 'average_lead_time']
    lead_time_td = pd.to_timedelta(lead_time_str)
    lt = lead_time_td.days
    ss = stats.norm.ppf(0.95) * s * np.sqrt(lt)

    mean_d = demand_df[demand_df['i'] == product_id]['y'].mean()
    ROP = mean_d * lt + ss
    q = avg_order_quantity[product_id]

    init_stock = stock_df[(stock_df['i'] == product_id)]['y'].values[-1]

    current_stock = init_stock
    inventory_levels = []
    order_dates = []

    rows_as_tuples = [tuple(row) for row in prophet_forecast[['ds', 'yhat']].itertuples(index=False, name=None)]
    for date, d in rows_as_tuples:
        current_stock -= d
        inventory_levels.append(current_stock)

        if current_stock <= ROP:
            current_stock += q
            order_dates.append(date)

    fig, ax = plt.subplots(figsize=(14, 7))
    ax.plot([date for date, _ in rows_as_tuples], inventory_levels, label='최적화된 재고 수준', color='blue')
    product_stock = stock_df[stock_df['i'] == product_id]
    min_forecast_date = prophet_forecast['ds'].min()
    filtered_stock_data = product_stock[product_stock['t'] >= min_forecast_date]
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.plot(product_stock[product_stock['t'] < min_forecast_date]['t'],
            product_stock[product_stock['t'] < min_forecast_date]['y'],
            label='현재 재고 수준', color='orange')
    ax.plot(filtered_stock_data['t'],
            filtered_stock_data['y'],
            label='예측 이후 재고 수준', color='blue')
    for order_date in order_dates:
        ax.axvline(x=order_date, color='red', linestyle='--', label='주문 발생' if order_date == order_dates[0] else "")
    ax.set_xlabel('날짜')
    ax.set_ylabel('재고 수준')
    ax.set_title(f'상품 ID: {product_id}의 재고 수준 시뮬레이션')
    ax.legend()
    st.pyplot(fig)

    st.write({'재주문점': ROP, '안전 재고': ss, '평균 주문량': q, '평균 리드 타임': lt})