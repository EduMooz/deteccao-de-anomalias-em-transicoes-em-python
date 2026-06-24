# Detecção de Anomalias em Transações Financeiras com Python

## Descrição do Projeto

Este projeto foi desenvolvido como parte de um desafio da DIO (Digital Innovation One) com o objetivo de aplicar técnicas de Ciência de Dados e Machine Learning para identificar transações financeiras suspeitas, simulando um cenário real de detecção de fraudes em cartões de crédito.

O projeto utiliza algoritmos de classificação supervisionada, técnicas de balanceamento de dados e métodos de interpretabilidade para analisar padrões em transações e identificar possíveis anomalias.

---

## Objetivos

* Realizar análise exploratória dos dados.
* Tratar e preparar os dados para modelagem.
* Aplicar técnicas de engenharia de atributos.
* Lidar com o desbalanceamento das classes utilizando SMOTE.
* Treinar e comparar diferentes modelos de Machine Learning.
* Avaliar o desempenho dos modelos utilizando métricas adequadas para problemas de fraude.
* Interpretar as decisões do modelo utilizando SHAP.

---

## Dataset

Foi utilizado o conjunto de dados público de transações com cartão de crédito disponibilizado pelo TensorFlow.

Características do dataset:

* Transações realizadas por cartões de crédito.
* Classes altamente desbalanceadas.
* Classe 0: Transações normais.
* Classe 1: Transações fraudulentas.
* Variáveis anonimizadas por questões de privacidade.

---

## Tecnologias Utilizadas

* Python
* Pandas
* NumPy
* Matplotlib
* Scikit-Learn
* Imbalanced-Learn (SMOTE)
* XGBoost
* SHAP

---

## Etapas do Projeto

### 1. Carregamento dos Dados

Os dados foram carregados diretamente a partir de uma fonte pública utilizando Pandas.

### 2. Engenharia de Atributos

Foram realizadas transformações na variável de valor da transação:

* Aplicação de transformação logarítmica (`Amount_log`).
* Padronização dos valores (`Amount_scaled`).

### 3. Divisão dos Dados

Os dados foram separados em conjuntos de treino e teste utilizando estratificação para preservar a proporção das classes.

### 4. Balanceamento das Classes

Como o problema apresenta forte desbalanceamento, foi utilizada a técnica SMOTE (Synthetic Minority Over-sampling Technique) para gerar exemplos sintéticos da classe minoritária.

### 5. Treinamento dos Modelos

Foram avaliados diferentes algoritmos:

* Regressão Logística
* Random Forest
* XGBoost

### 6. Ajuste de Threshold

Além da classificação padrão, foram testados diferentes limiares de decisão (thresholds) para melhorar a capacidade de detecção de fraudes.

### 7. Avaliação

Foram utilizadas métricas como:

* Precision
* Recall
* F1-Score
* ROC Curve
* AUC (Area Under Curve)
* Precision-Recall Curve

### 8. Interpretabilidade

Foi utilizada a biblioteca SHAP para identificar quais atributos mais influenciam as previsões do modelo.

---

## Principais Aprendizados

Durante o desenvolvimento deste projeto foi possível praticar conceitos importantes de Ciência de Dados e Machine Learning, como:

* Pré-processamento de dados.
* Engenharia de atributos.
* Tratamento de classes desbalanceadas.
* Avaliação de modelos para detecção de fraudes.
* Otimização de hiperparâmetros.
* Interpretabilidade de modelos com SHAP.

---

## Possíveis Melhorias Futuras

* Implementação de validação cruzada estratificada.
* Teste de novos algoritmos de detecção de anomalias.
* Criação de dashboard para visualização dos resultados.
* Deploy do modelo utilizando APIs.
* Monitoramento contínuo de desempenho.

Projeto desenvolvido por Carlos Eduardo como parte dos estudos em Análise e Desenvolvimento de Sistemas e dos desafios práticos da DIO.
