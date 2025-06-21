import React, { useState } from 'react';

const App = () => {
  const [numbers, setNumbers] = useState({
    number1: '',
    number2: '',
    number3: '',
    number4: '',
  });

  const [response, setResponse] = useState(null);

  const handleChange = (event) => {
    const { name, value } = event.target;
    setNumbers((prevState) => ({
      ...prevState,
      [name]: value,
    }));
  };

  const handleSubmit = async (event) => {
    event.preventDefault();
    const flat_params = {
      area: numbers.number1,
      rooms: numbers.number2,
      total_floors: numbers.number3,
      floor: numbers.number4
    };

    try {
      
      const response = await fetch('http://89.169.1.154:5051/api/numbers', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(flat_params),
      });

      if (!response.ok) {
        throw new Error('Ошибка при отправке данных');
      }

      const data = await response.json();
      setResponse(data);
    } catch (error) {
      console.error(error);
      setResponse({ error: 'Произошла ошибка при отправке данных' });
    }
  };

  return (
    <div>
      <h1>Это сторонний сервис</h1>

      <form onSubmit={handleSubmit}>
        <div>
          <input
            type="number"
            name="number1"
            value={numbers.number1}
            onChange={handleChange}
            placeholder="Площадь"
          />
        </div>
        <div>
          <input
            type="number"
            name="number2"
            value={numbers.number2}
            onChange={handleChange}
            placeholder="Число комнат"
          />
        </div>
        <div>
          <input
            type="number"
            name="number3"
            value={numbers.number3}
            onChange={handleChange}
            placeholder="Этажей в доме"
          />
        </div>
        <div>
          <input
            type="number"
            name="number4"
            value={numbers.number4}
            onChange={handleChange}
            placeholder="Этаж квартиры"
          />
        </div>

        <button type="submit">Отправить</button>
      </form>

      {response && (
        <div>
          <h2>Ответ сервера:</h2>
          <pre>{JSON.stringify(response, null, 2)}</pre>
        </div>
      )}
    </div>
  );
};

export default App;