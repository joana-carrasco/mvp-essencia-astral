# Essência Astral — Preditor de Elemento Astrológico

Projeto desenvolvido como MVP da disciplina - Qualidade de Software, Segurança e Sistemas Inteligentes - pós-graduação em Engenharia de Software — PUC-Rio.

## Sobre o Projeto

O Essência Astral é uma aplicação web que utiliza Machine Learning para prever o elemento astrológico de uma pessoa (Fogo, Terra, Ar ou Água) com base em traços de personalidade e mês de nascimento.

## Estrutura do Repositório

mvp-essencia-astral/
├── notebook/
│   └── essencia_astral.ipynb - Notebook com todo o processo de ML
├── backend/
│   ├── app.py - API Flask com o modelo embarcado
│   ├── modelo_essencia_astral.pkl - Modelo treinado
│   └── encoder_elementos.pkl - Encoder das classes
├── frontend/
│   └── index.html - Interface web
├── tests/
│   └── test_modelo.py - Testes automatizados com PyTest
└── dataset_astrologia.csv - Dataset utilizado no treinamento

## Modelo de Machine Learning

- Algoritmos testados: KNN, Árvore de Decisão, Naive Bayes e SVM
- Melhor modelo: Árvore de Decisão com 60% de acurácia
- Técnicas utilizadas: Pipelines, Cross-Validation, GridSearchCV

## Como executar

**Back-end**
cd backend
py -3.11 app.py

**Front-end**
Abra o arquivo frontend/index.html no navegador.

**Testes**
cd tests
py -3.11 -m pytest test_modelo.py -v

## Tecnologias utilizadas

- Python 3.11
- Scikit-Learn
- Flask
- PyTest
- HTML, CSS e JavaScript
