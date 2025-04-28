from flask import Flask, render_template, request 
from logging.config import dictConfig

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

# Маршрут для обработки данных формы
@app.route('/api/numbers', methods=['POST'])
def process_numbers():
    # Здесь можно добавить обработку полученных чисел
    # Для примера просто возвращаем их обратно
    data = request.get_json()
    
    app.logger.info(f'Requst data: {data}')
    
    if float(data['area']) >= 0:
        app.logger.info('status: success, data: Числа успешно обработаны')
        result_sum = float(data['area']) * 300000
        app.logger.info(f'Стоимость квартиры: {result_sum}')
        return {'result': result_sum}
    else:
        app.logger.info('status: error, data: Отрицательное значение площади')
        return {'result': 'error'}
    

if __name__ == '__main__':
    app.run(debug=False, port=5050)