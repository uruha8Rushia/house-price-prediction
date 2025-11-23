import pandas as pd
import joblib
import numpy as np

def _make_json_serializable(obj):
    """Recursively convert numpy / pandas types to native Python types so
    Flask/Jinja's tojson can serialize them.

    Handles:
    - numpy scalar types (np.int64, np.float64) -> python int/float via .item()
    - numpy arrays -> list
    - pandas Series/DataFrame -> list/dict
    - dicts/lists recursively
    """
    # numpy scalar (e.g. np.int64)
    if isinstance(obj, np.generic):
        return obj.item()

    # numpy arrays
    if isinstance(obj, np.ndarray):
        return obj.tolist()

    # pandas Series / Index
    if isinstance(obj, pd.Series) or isinstance(obj, pd.Index):
        return obj.tolist()

    # pandas DataFrame
    if isinstance(obj, pd.DataFrame):
        return obj.to_dict(orient="records")

    # dict/list recursion
    if isinstance(obj, dict):
        return {k: _make_json_serializable(v) for k, v in obj.items()}

    if isinstance(obj, (list, tuple)):
        return [_make_json_serializable(v) for v in obj]

    # default: return as-is (json library will error if still unsupported)
    return obj
from flask import Flask, request, jsonify, render_template

class HousePricePredictor:
    def __init__(self, model_path='model/best_model.pkl'):
        self.artifact = joblib.load(model_path)
        self.model = self.artifact['model']
        self.features = self.artifact['features']
        self.discrete_features = self.artifact['discrete_features']
        self.continuous_features = self.artifact['continuous_features']
        self.discrete_scaler = self.artifact['discrete_scaler']
        self.continuous_scaler = self.artifact['continuous_scaler']
        self.default_input = self.artifact['default_input']
        self.default_city = self.artifact['default_city']
        self.city_columns = [c for c in self.features if c.startswith('city_')]
        self.non_city_columns = [c for c in self.features if not c.startswith('city_')]
    
    def prepare_input(self, input_data):
        # create a DataFrame with default values
        input_df = pd.DataFrame([self.default_input])
        
        # update with user input
        for key, value in input_data.items():
            if key in input_df.columns:
                input_df.at[0, key] = value
        
        # handle city one-hot encoding
        city_columns = [col for col in self.features if col.startswith('city_')]
        for col in city_columns:
            input_df[col] = 0
        city_col_name = f"city_{input_data.get('city', self.default_city)}"
        if city_col_name in city_columns:
            input_df.at[0, city_col_name] = 1
        
        return input_df[self.features]
    
    def preprocess_input(self, input_data):
        input_df = self.prepare_input(input_data).copy()

        # scale discrete features
        input_df[self.discrete_features] = self.discrete_scaler.transform(input_df[self.discrete_features])

        # scale continuous features
        input_df[self.continuous_features] = self.continuous_scaler.transform(input_df[self.continuous_features])

        return input_df
    
    def predict(self, input_data):
        processed_input = self.preprocess_input(input_data)
        predicted_price = self.model.predict(processed_input)
        # return a JSON-serializable float
        return float(predicted_price[0])


# ------------------ FLASK APP ------------------

app = Flask(__name__)
model = HousePricePredictor()


@app.route('/')
def index():
    # Ensure defaults are plain Python types (no numpy / pandas scalars)
    defaults_clean = _make_json_serializable(model.default_input)

    return render_template(
        'index.html',
        cities=[c.replace("city_", "") for c in model.city_columns],
        defaults=defaults_clean,
        default_city=model.default_city
    )

@app.route('/predict', methods=['POST'])
def predict():
    # Ensure we have JSON
    input_data = None
    if request.is_json:
        input_data = request.get_json()
    else:
        # Try to be forgiving: attempt to parse body as JSON
        try:
            input_data = request.get_json(force=True)
        except Exception:
            return jsonify({'error': 'Missing or invalid JSON in request body'}), 400

    if input_data is None:
        return jsonify({'error': 'Missing JSON payload'}), 400

    try:
        prediction = model.predict(input_data)
        # return a stable key that frontend expects: `prediction`
        return jsonify({'prediction': prediction})
    except Exception as e:
        # Return error message and 400 so client can show friendly message
        return jsonify({'error': str(e)}), 400


if __name__ == '__main__':
    import os
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=True)
