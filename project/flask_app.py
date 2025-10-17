from flask import Flask, request, jsonify
import joblib
import numpy as np
import os

app = Flask(__name__)


username = 'lutherbanze' 
model_path = f'/home/{username}/mysite/modelo_notas.pkl'

# Carrega o modelo treinado
try:
    model = joblib.load(model_path)
except FileNotFoundError:
    model = None 


@app.route('/')
def home():
    return "A API do modelo de notas está a funcionar!"


@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Modelo não encontrado. Verifique o caminho do ficheiro.'}), 500

    try:
        data = request.get_json(force=True)


        prediction_input = np.array([
            data['idade'],
            data['faltas_percentual'],
            data['nota_teste_1'],
            data['nota_teste_2'],
            data['nota_trabalhos']
        ]).reshape(1, -1)


        prediction = model.predict(prediction_input)


        return jsonify({'nota_prevista': float(prediction[0])})

    except Exception as e:
        return jsonify({'error': str(e)}), 400
