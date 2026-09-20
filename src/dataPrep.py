# data_prep.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import config as config

def load_and_preprocess_data():
    """Carrega, limpa e prepara os dados para o treinamento."""
   # 1. Carregamento do dataset
    df = pd.read_csv(config.DATASET_PATH)
    
    # Força a conversão da coluna para numérico (textos como '?' viram NaN)
    df['horsepower'] = pd.to_numeric(df['horsepower'], errors='coerce')
        
    # 2. Tratamento de Nulos
    mediana_hp = df['horsepower'].median()
    df['horsepower'] = df['horsepower'].fillna(mediana_hp)
    
    # 3. Engenharia de Atributos: Convertendo para Km/L
    df['kml'] = df['mpg'] * config.MPG_TO_KML
    
    # 4. Separação de Features (X) e Target (y)
    X = df.drop(['mpg', 'kml', 'car name'], axis=1, errors='ignore')
    y = df['kml']
    
    # 5. Codificação Categórica (One-Hot Encoding)
    X = pd.get_dummies(X, columns=['origin'], drop_first=True, dtype=int)
    
    # 6. Divisão em Treino e Teste
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE
    )
    
    # 7. Escalonamento / Normalização
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Retornamos também as colunas de X para usar no Feature Importance depois
    return X_train_scaled, X_test_scaled, y_train, y_test, X.columns