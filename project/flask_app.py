from flask import Flask, request, jsonify
import joblib
import numpy as np
import os

# Inicializa a aplicação Flask
app = Flask(__name__)

# --- IMPORTANTE ---
# O caminho para o seu modelo. No PythonAnywhere, o caminho completo é necessário.
# Substitua 'YourUsername' pelo seu nome de utilizador no PythonAnywhere.
username = 'YourUsername' # TROQUE AQUI
model_path = f'/home/{username}/mysite/modelo_notas.pkl'

# Carrega o modelo treinado
try:
    model = joblib.load(model_path)
except FileNotFoundError:
    model = None # Se o ficheiro não for encontrado, o modelo será None

# Rota principal para verificar se a API está online
@app.route('/')
def home():
    return "A API do modelo de notas está a funcionar!"

# Rota para fazer previsões
@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Modelo não encontrado. Verifique o caminho do ficheiro.'}), 500

    try:
        data = request.get_json(force=True)

        # Organiza os dados na mesma ordem que o modelo foi treinado
        # ['idade', 'faltas_percentual', 'nota_teste_1', 'nota_teste_2', 'nota_trabalhos']
        prediction_input = np.array([
            data['idade'],
            data['faltas_percentual'],
            data['nota_teste_1'],
            data['nota_teste_2'],
            data['nota_trabalhos']
        ]).reshape(1, -1)

        # Usa o modelo para fazer a previsão
        prediction = model.predict(prediction_input)

        # Devolve a previsão em formato JSON
        # A conversão para float é para garantir que o formato é JSON compatível
        return jsonify({'nota_prevista': float(prediction[0])})

    except Exception as e:
        return jsonify({'error': str(e)}), 400