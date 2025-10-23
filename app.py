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
    def __init__(self, model_path='house_price_prediction_model.pkl'):
        self.artifact = joblib.load(model_path)
        self.model = self.artifact['model']
        self.features = self.artifact['features']
        self.city_columns = self.artifact['city_columns']
        self.non_city_columns = self.artifact['non_city_columns']
        self.default_values = self.artifact['default_values']
        self.default_city = self.artifact['default_city']
    
    def prepare_input(self, input_data):
        input_df = pd.DataFrame(columns=self.features)

        # Fill non-city columns
        for col in self.non_city_columns:
            if col in input_data:
                input_df.at[0, col] = input_data[col]
            else:
                input_df.at[0, col] = self.default_values[col]
        
        # Fill city columns (one-hot)
        for col in self.city_columns:
            input_df.at[0, col] = 0
        
        chosen_city = input_data.get('city', self.default_city)
        city_col_name = f"city_{chosen_city}"
        if city_col_name in self.city_columns:
            input_df.at[0, city_col_name] = 1
        
        return input_df

    def predict(self, input_data):
        input_df = self.prepare_input(input_data)
        prediction = self.model.predict(input_df)
        return float(prediction[0])  # ensure JSON friendly


# ------------------ FLASK APP ------------------

app = Flask(__name__)
model = HousePricePredictor()


@app.route('/')
def index():
    # Ensure defaults are plain Python types (no numpy / pandas scalars)
    defaults_clean = _make_json_serializable(model.default_values)

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
    app.run(debug=True)
