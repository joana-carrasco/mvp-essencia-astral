import joblib
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Carregando o modelo e o encoder
modelo = joblib.load('modelo_essencia_astral.pkl')
encoder = joblib.load('encoder_elementos.pkl')

# Elementos
descricoes = {
    "Fogo": "Você é a centelha da ação! Apaixonado e destemido, sua energia contagiante ilumina qualquer lugar. Impulsivo por natureza, você não apenas lidera, mas vive a vida como uma grande aventura intensa.",
    "Terra": "Você é a âncora da realidade. Prático, leal e construtor, valoriza a segurança e a perseverança para transformar sonhos em resultados palpáveis. Com os pés firmes no chão, você traz estrutura ao mundo.",
    "Ar": "Você é a brisa da renovação. Curioso e sociável, sua mente inquieta busca conectar ideias e pessoas. Adora aprender, inovar e trocar experiências, espalhando conhecimento como o vento. ",
    "Agua": "Você é a profundidade da alma. Sensível e intuitivo, navega pelas emoções com facilidade. Valoriza conexões profundas e genuínas, sentindo o mundo com uma empatia rara e adaptando-se a qualquer corrente. "
}

@app.route('/predict', methods=['POST'])
def predict():
    dados = request.json
    entrada = np.array([[
        dados['extrovertido'],
        dados['emocional'],
        dados['logico'],
        dados['criativo'],
        dados['impulsivo'],
        dados['paciente'],
        dados['comunicativo'],
        dados['intuitivo'],
        dados['mes_nascimento']
    ]])
    predicao = modelo.predict(entrada)
    elemento = encoder.inverse_transform(predicao)[0]
    return jsonify({
        'elemento': elemento,
        'descricao': descricoes[elemento]
    })

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok'})

if __name__ == '__main__':
    app.run(debug=True)
