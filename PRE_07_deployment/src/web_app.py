import os
import pickle
import pandas as pd  # type: ignore
from flask import Flask, render_template, request, jsonify  # type: ignore

app = Flask(__name__)  # Corregido: __name__ lleva doble guion bajo
app.config["SECRET_KEY"] = "you-will-never-guess"

FOLDER = "PRE_07_deployment"
MODEL_PATH = os.path.join(FOLDER, "submission", "house_predictor.pkl")

# Carga del modelo entrenado
model = None
if os.path.exists(MODEL_PATH):
    with open(MODEL_PATH, "rb") as file:
        model = pickle.load(file)

FEATURES = [
    "bedrooms",
    "bathrooms",
    "sqft_living",
    "sqft_lot",
    "floors",
    "waterfront",
    "condition",
]

@app.route("/", methods=["GET", "POST"])
@app.route("/index", methods=["GET", "POST"])
def index():
    prediction = None

    if request.method == "POST":
        # Permite recibir datos tanto de un formulario HTML como de una petición JSON (curl)
        data = request.get_json() if request.is_json else request.form

        # Extrae los valores convertidos a float
        user_values = {feature: float(data.get(feature, 0)) for feature in FEATURES}

        # Convierte a DataFrame para el modelo
        df_input = pd.DataFrame([user_values])

        # Realiza la predicción si el modelo está cargado
        if model:
            pred_res = model.predict(df_input)[0]
            prediction = round(float(pred_res[0] if hasattr(pred_res, "__len__") else pred_res), 2)

        # Si la petición fue vía API (curl/JSON), retorna respuesta JSON
        if request.is_json:
            return jsonify({"prediccion_precio": prediction})

    return render_template("index.html", prediction=prediction)


if __name__ == "__main__":
    app.run(debug=True)