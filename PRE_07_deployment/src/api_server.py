import os
import pickle
import pandas as pd
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)
app.config["SECRET_KEY"] = "you-will-never-guess"

# Características exactas especificadas por el profesor
FEATURES = [
    "bedrooms",
    "bathrooms",
    "sqft_living",
    "sqft_lot",
    "floors",
    "waterfront",
    "condition",
]

# Definición de ruta para cargar el modelo del profesor
MODEL_PATH = "PRE_07_deployment/submission/house_predictor.pkl"

# Si ejecutas el script directamente desde dentro de la carpeta 'src', ajusta la ruta
if not os.path.exists(MODEL_PATH) and os.path.exists("../submission/house_predictor.pkl"):
    MODEL_PATH = "../submission/house_predictor.pkl"

@app.route("/", methods=["GET", "POST"])
def index():
    prediction_val = None
    error_msg = None

    if request.method == "POST":
        try:
            # 1. Obtener datos (soporta tanto JSON como formulario Web)
            if request.is_json:
                args = request.json
            else:
                args = request.form.to_dict()

            # 2. Filtrar y convertir tipos según la lógica del profesor
            filt_args = {key: [float(args[key])] for key in FEATURES}
            df = pd.DataFrame.from_dict(filt_args)

            # 3. Cargar el modelo pickle exactamente como el profesor
            with open(MODEL_PATH, "rb") as file:
                loaded_model = pickle.load(file)

            # 4. Predicción
            pred = loaded_model.predict(df)
            
            # Obtener el valor simple de la matriz de predicción
            raw_pred = pred[0][0] if hasattr(pred[0], '__getitem__') else pred[0]
            prediction_val = round(float(raw_pred), 2)

            # Respuesta para cliente API / cURL
            if request.is_json:
                return str(raw_pred)

        except Exception as e:
            error_msg = f"Error en la predicción: {str(e)}"
            if request.is_json:
                return jsonify({"error": str(e)}), 400

    # Renderiza la vista HTML para solicitudes GET o envíos de formulario Web
    return render_template("index.html", prediction=prediction_val, error=error_msg)

if __name__ == "__main__":
    app.run(debug=True)