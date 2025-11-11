# House Price Prediction

Simple Flask web app and notebook for predicting house prices.

This repository contains a lightweight Flask frontend that sends JSON to a model artifact (stored as a joblib `.pkl`) and returns a predicted price. A Jupyter notebook (`hpp.ipynb`) contains Exploratory Data Analysis (EDA) and model training steps used to build the model artifact.

## Contents

- `app.py` - Flask application and the `HousePricePredictor` class that loads the joblib artifact and serves prediction endpoints.
- `data.csv` - Dataset from [https://www.kaggle.com/datasets/shree1992/housedata](https://www.kaggle.com/datasets/shree1992/housedata) used for analysis and training.
- `hpp.ipynb` - Jupyter notebook used for EDA and model training.
- `static/` - Static files for the web frontend (`style.css`, `script.js`).
- `templates/index.html` - HTML template used by the Flask app.

## Setup

Create a virtual environment:

```powershell
python -m venv myenv
```

Activate the virtual environment:

```powershell
myenv\Scripts\activate
```

Install the packages

```powershell
pip install -r requirements.txt
```

### Run the app

Start the Flask development server:

```powershell
python app.py
```

Open a browser and visit [http://127.0.0.1:5000/](http://127.0.0.1:5000/) to use the simple web UI.

## API

POST /predict

Request JSON body example:

```json
{
  "bedrooms": 3,
  "bathrooms": 2,
  "sqft_living": 2500,
  "city": "Seattle"
}
```

Response example:

```json
{
  "Predicted house price:": 350000.00
}
```

Notes:

- The exact feature names, required fields, and default values depend on the artifact stored in the `.pkl` file. The Flask app will use `default_values` and `default_city` from the artifact for missing fields.
- If the request body isn't valid JSON, the API returns a 400 with an error message.
