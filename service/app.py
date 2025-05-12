from flask import Flask, render_template, request 
from logging.config import dictConfig
from joblib import dump, load
import pandas as pd
import os
from glob import glob
from datetime import datetime
import argparse

dictConfig(
    {
        "version": 1,
        "formatters": {
            "default": {
                "format": "[%(asctime)s] %(levelname)s in %(module)s: %(message)s",
            }
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "stream": "ext://sys.stdout",
                "formatter": "default",
            },         
            "file": {
                "class": "logging.FileHandler",
                "filename": "flask.log",
                "formatter": "default",
            },
        },
        "root": {"level": "DEBUG", "handlers": ["console", "file"]},
    }
)

app = Flask(__name__)

# Маршрут для отображения формы
@app.route('/')
def index():
    return render_template('index.html')

models_dir = 'models'


model_files = glob(os.path.join(models_dir, '*.joblib'))

def extract_date(filename):
    date_str = os.path.basename(filename).split('_')[2] + '_' + os.path.basename(filename).split('_')[3].split('.')[0]
    return datetime.strptime(date_str, '%Y-%m-%d_%H-%M')

sorted_files = sorted(model_files, key=extract_date)
MODEL_NAME = sorted_files[-1] if sorted_files else None
# model = load(latest_model)
#model = load('../models/best_model.joblib')

# Маршрут для обработки данных формы
@app.route('/api/numbers', methods=['POST'])
def process_numbers():
    data = request.get_json()
    
    app.logger.info(f'Requst data: {data}')
    
    if float(data['total_floors']) < float(data['floor']):
        app.logger.info('status: error, data: ОНекорректное количество этажей')
        return {'result': 'error'}
    elif float(data['area']) >= 0:
        app.logger.info('status: success, data: Числа успешно обработаны')
        
        new_data = pd.DataFrame({
            'rooms_count': [float(data['rooms'])], 
            'floor': [float(data['floor'])], 
            'floors_count': [float(data['total_floors'])],
            'total_meters': [float(data['area'])]})        
        result_sum = model.predict(new_data)[0]
        app.logger.info(f'Стоимость квартиры: {result_sum}')
        return {'result': result_sum}
    else:
        app.logger.info('status: error, data: Отрицательное значение площади')
        return {'result': 'error'}
    

if __name__ == '__main__':
    """Parse arguments and run lifecycle steps"""
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", help="Model name", default=MODEL_NAME)
    args = parser.parse_args()
    print(args.model, MODEL_NAME)

    # app.config["model"] = joblib.load(args.model)
    model = load(args.model)
    app.logger.info(f"Use model: {args.model}")
    app.run(debug=False, port=5050)