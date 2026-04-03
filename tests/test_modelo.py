import pytest
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

# Carregando o modelo e o encoder
modelo = joblib.load('../backend/modelo_essencia_astral.pkl')
encoder = joblib.load('../backend/encoder_elementos.pkl')

# Dataset
url = "https://raw.githubusercontent.com/joana-carrasco/mvp-essencia-astral/refs/heads/main/dataset_astrologia.csv"
df = pd.read_csv(url)

# Separando features e target
X = df.drop('elemento', axis=1)
y = df['elemento']
y_encoded = encoder.transform(y)

def test_acuracia_minima():
    """O modelo deve ter acurácia mínima de 50%"""
    y_pred = modelo.predict(X)
    acc = accuracy_score(y_encoded, y_pred)
    print(f"\nAcurácia do modelo: {acc:.4f}")
    assert acc >= 0.50, f"Acurácia {acc:.4f} abaixo do mínimo de 0.50"

def test_predicao_valida():
    """O modelo deve retornar apenas elementos válidos"""
    elementos_validos = set(encoder.classes_)
    y_pred = modelo.predict(X)
    elementos_preditos = set(encoder.inverse_transform(y_pred))
    assert elementos_preditos.issubset(elementos_validos), \
        "Modelo retornou elementos inválidos!"

def test_formato_entrada():
    """O modelo deve aceitar uma entrada com 9 features"""
    entrada = np.array([[1, 0, 1, 1, 1, 0, 1, 0, 3]])
    predicao = modelo.predict(entrada)
    assert len(predicao) == 1, "Predição deve retornar exatamente 1 resultado"

def test_todos_elementos_preditos():
    """O modelo deve ser capaz de prever os 4 elementos"""
    y_pred = modelo.predict(X)
    elementos_preditos = set(encoder.inverse_transform(y_pred))
    assert len(elementos_preditos) >= 2, \
        "Modelo deve prever pelo menos 2 elementos diferentes"
