from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.providers.common.sql.operators.sql import SQLExecuteQueryOperator
from airflow.providers.postgres.hooks.postgres import PostgresHook

import datetime
import cianparser
import pandas as pd
import re
import os
import logging
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import datetime
from joblib import dump

TRAIN_SIZE = 0.8
MODELS_DIR = "/opt/airflow/models"

def extract_id(url):
    match = re.search(r"/(\d+)/?$", url)
    return int(match.group(1)) if match else 0

moscow_parser = cianparser.CianParser(location="Москва")

def parse(n_rooms=1, end_page=2):
    t = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
    data = moscow_parser.get_flats(
        deal_type="sale",
        rooms=(n_rooms,),
        with_saving_csv=False,
        additional_settings={
            "start_page": 1,
            "end_page": end_page,
            "object_type": "secondary"
        })
    df = pd.DataFrame(data)
    
    # Проверка и преобразование столбцов
    required_columns = [
        "Author", "Author_type", "Url", "Location", "Deal_type", "Accommodation_type",
        "Floor", "Floors_count", "Rooms_count", "Total_meters", "Price_per_month",
        "Commissions", "Price", "District", "Street", "House_number", "Underground", "Residential_complex"
    ]
    
    # Приводим названия столбцов к требуемому виду
    df = df.rename(columns=lambda x: x.strip().replace(" ", "_").capitalize())
    
    # Добавляем url_id
    df['url_id'] = df['Url'].apply(extract_id)
    
    for col in required_columns:
        if col not in df.columns:
            df[col] = None
    
    # Подключение к БД и вставка данных
    hook = PostgresHook(postgres_conn_id="tutorial_pg_conn")
    conn = hook.get_conn()
    cursor = conn.cursor()
    
    # Формируем SQL-запрос с учетом url_id
    columns = ['url_id'] + [f'"{col}"' for col in required_columns]
    placeholders = ', '.join(['%s'] * len(columns))
    insert_sql = f"""
        INSERT INTO raw_data ({', '.join(columns)})
        VALUES ({placeholders})
        ON CONFLICT (url_id) DO UPDATE SET
            "Author" = EXCLUDED."Author",
            "Price" = EXCLUDED."Price",
            "Total_meters" = EXCLUDED."Total_meters"
    """
    
    # Конвертируем данные в кортежи
    data_tuples = [
        tuple([row['url_id']] + [row[col] for col in required_columns])
        for _, row in df.iterrows()
    ]
    
    # Выполняем массовую вставку
    cursor.executemany(insert_sql, data_tuples)
    conn.commit()
    
    # Закрываем соединение
    cursor.close()
    conn.close()

def parse_room_count():
    parse(n_rooms=1, end_page=5)
    parse(n_rooms=2, end_page=5)
    parse(n_rooms=3, end_page=5)

def preprocess_data():
    """Фильтрация и очистка данных прямо в БД"""
    hook = PostgresHook(postgres_conn_id="tutorial_pg_conn")
    conn = hook.get_conn()
    cursor = conn.cursor()
    
    # Создаем временную таблицу для обработки
    cursor.execute("""
        CREATE TEMP TABLE temp_processed AS
        WITH ranked_data AS (
            SELECT 
                url_id,
                "Rooms_count" AS rooms_count,
                "Floor" AS floor,
                "Floors_count" AS floors_count,
                "Total_meters" AS total_meters,
                "Price" AS price,
                ROW_NUMBER() OVER (
                    PARTITION BY url_id 
                    ORDER BY load_date DESC
                ) AS rn
            FROM raw_data
            WHERE 
                "Total_meters" <= 90 AND
                "Price" <= 55000000 AND
                "Rooms_count" != -1
        )
        SELECT 
            url_id,
            rooms_count,
            floor,
            floors_count,
            total_meters,
            price
        FROM ranked_data
        WHERE rn = 1;
    """)
    
    # Обновляем основную таблицу
    cursor.execute("""
        INSERT INTO processed_data (
            url_id, rooms_count, floor, floors_count, total_meters, price
        )
        SELECT 
            url_id,
            rooms_count,
            floor,
            floors_count,
            total_meters,
            price
        FROM temp_processed
        ON CONFLICT (url_id) DO UPDATE SET
            rooms_count = EXCLUDED.rooms_count,
            floor = EXCLUDED.floor,
            floors_count = EXCLUDED.floors_count,
            total_meters = EXCLUDED.total_meters,
            price = EXCLUDED.price;
    """)
    
    conn.commit()
    cursor.close()
    conn.close()
    
def train_model():
    """Train model using data from processed_data table"""
    # Подключение к БД
    hook = PostgresHook(postgres_conn_id="tutorial_pg_conn")
    conn = hook.get_conn()
    
    # Загрузка данных из processed_data
    query = "SELECT rooms_count, floor, floors_count, total_meters, price FROM processed_data;"
    df = pd.read_sql_query(query, conn)
    conn.close()
    
    if df.empty:
        logging.error("No data found in processed_data table!")
        return

    # Подготовка данных
    y = df["price"]
    X = df.drop(columns="price")
    
    # Проверка наличия данных
    if len(X) < 10:
        logging.warning(f"Very small dataset: {len(X)} rows. Model may be unstable")
    
    # Разделение данных
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=1-TRAIN_SIZE, random_state=42
    )
    
    # Обучение модели
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Сохранение модели
    os.makedirs(MODELS_DIR, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = os.path.join(MODELS_DIR, f"model_{timestamp}.joblib")
    dump(model, model_path)
    
    logging.info(f"Model trained and saved to {model_path}")
    logging.info(f"Train R2: {model.score(X_train, y_train):.3f}, Test R2: {model.score(X_test, y_test):.3f}")
    

with DAG(
    "cian_parser_dag",
    schedule="@daily",
    start_date=datetime.datetime(2023, 1, 1),
    catchup=False,
) as dag:
    
    create_raw_data_table = SQLExecuteQueryOperator(
        task_id="create_raw_data_table",
        conn_id="tutorial_pg_conn",
        sql="""
            CREATE TABLE IF NOT EXISTS raw_data (
                url_id BIGINT PRIMARY KEY,
                load_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                "Author" TEXT,
                "Author_type" TEXT,
                "Url" TEXT UNIQUE,
                "Location" TEXT,
                "Deal_type" TEXT,
                "Accommodation_type" TEXT,
                "Floor" INTEGER,
                "Floors_count" INTEGER,
                "Rooms_count" INTEGER,
                "Total_meters" FLOAT,
                "Price_per_month" FLOAT,
                "Commissions" FLOAT,
                "Price" INTEGER,
                "District" TEXT,
                "Street" TEXT,
                "House_number" TEXT,
                "Underground" TEXT,
                "Residential_complex" TEXT            
            );
        """,
    )
    
    create_processed_data_table = SQLExecuteQueryOperator(
        task_id="create_processed_data_table",
        conn_id="tutorial_pg_conn",
        sql="""
            CREATE TABLE IF NOT EXISTS processed_data (
                url_id BIGINT PRIMARY KEY REFERENCES raw_data(url_id),
                rooms_count INTEGER,
                floor INTEGER,
                floors_count INTEGER,
                total_meters FLOAT,
                price INTEGER        
            );
        """,
    )
        
    parse_task = PythonOperator(
        task_id="parse_data",
        python_callable=parse_room_count
    )
    
    process_task = PythonOperator(
        task_id="process_data",
        python_callable=preprocess_data
    )
    
    train_task = PythonOperator(
        task_id="train_model",
        python_callable=train_model
    )
    
    create_raw_data_table >> create_processed_data_table >> parse_task >> process_task >> train_task
