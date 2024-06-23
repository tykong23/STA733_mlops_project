import pandas as pd
from prophet import Prophet
import lightgbm as lgb
from sklearn.preprocessing import OneHotEncoder
import sqlite3
from datetime import datetime
import os
import pickle
from apscheduler.schedulers.background import BackgroundScheduler

demand_file_path = 'data/filtered_demand.csv'
db_path = 'results.db'
model_dir = 'models/'

os.makedirs(model_dir, exist_ok=True)

demand_df = pd.read_csv(demand_file_path)
demand_df['t'] = pd.to_datetime(demand_df['t'])
product_ids = demand_df['i'].unique()

conn = sqlite3.connect(db_path)
cursor = conn.cursor()

def create_tables():
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS prophet_forecast (
        product_id TEXT,
        ds DATE,
        yhat REAL,
        yhat_lower REAL,
        yhat_upper REAL,
        PRIMARY KEY (product_id, ds)
    )
    ''')
    conn.commit()

def train_and_save_prophet(demand_df, product_id, conn):
    product_demand = demand_df[demand_df['i'] == product_id]
    product_demand = product_demand.rename(columns={'t': 'ds', 'y': 'y'})
    
    m = Prophet()
    m.fit(product_demand[['ds', 'y']])
    
    future = m.make_future_dataframe(periods=4, freq='W')
    forecast = m.predict(future)
    
    forecast = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(4)
    forecast['product_id'] = product_id

    cursor.execute(f"DELETE FROM prophet_forecast WHERE product_id = ?", (product_id,))
    conn.commit()

    forecast.to_sql('prophet_forecast', conn, if_exists='append', index=False)
    print(f"Prophet 모델 예측 결과가 SQL DB에 저장되었습니다: {product_id}")

def train_and_save_lightgbm(demand_df, product_id, conn):
    product_demand = demand_df[demand_df['i'] == product_id]
    product_demand['day'] = product_demand['t'].dt.day
    product_demand['month'] = product_demand['t'].dt.month
    product_demand['year'] = product_demand['t'].dt.year
    product_demand['dayofweek'] = product_demand['t'].dt.dayofweek

    for lag in range(1, 7):
        product_demand[f'lag_{lag}'] = product_demand['y'].shift(lag)
    product_demand = product_demand.dropna()

    encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
    encoded_features = encoder.fit_transform(product_demand[['i']])
    encoded_df = pd.DataFrame(encoded_features, columns=encoder.get_feature_names_out(['i']))
    data_encoded = pd.concat([product_demand.drop(columns=['i']).reset_index(drop=True), encoded_df], axis=1)

    train = data_encoded.loc[data_encoded['year'] <= datetime.now().year - 1].reset_index(drop=True)
    X_train = train.drop(columns=['y', 't'])
    y_train = train['y']

    lgb_train = lgb.Dataset(X_train, y_train)
    params = {
        'objective': 'regression',
        'metric': 'l1',
        'verbosity': -1
    }
    model = lgb.train(params, lgb_train)

    model_path = os.path.join(model_dir, f'lightgbm_model_{product_id}.pkl')
    with open(model_path, 'wb') as model_file:
        pickle.dump(model, model_file)
    print(f"LightGBM 모델이 {model_path}에 저장되었습니다: {product_id}")

    print(f"LightGBM 모델이 pickle 파일로 저장되었습니다: {product_id}")

def retrain_and_save_models():
    create_tables()
    for product_id in product_ids:
        train_and_save_prophet(demand_df, product_id, conn)
        train_and_save_lightgbm(demand_df, product_id, conn)    
    print("모든 모델이 재학습되었습니다.")

if __name__ == "__main__":
    retrain_and_save_models()

    scheduler = BackgroundScheduler()
    scheduler.add_job(retrain_and_save_models, 'interval', weeks=1)
    scheduler.start()

    try:
        print("스케줄러가 실행 중입니다. 모델은 매주 자동으로 재학습됩니다.")
        while True:
            pass  
    except (KeyboardInterrupt, SystemExit):
        scheduler.shutdown()
        print("스케줄러가 종료되었습니다.")

    conn.close()
