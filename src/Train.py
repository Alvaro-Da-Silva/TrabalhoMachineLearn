# train_and_evaluate.py
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import config as config

def train_models(X_train, y_train):
    """Treina e otimiza os modelos retornando os melhores estimadores."""
    # 1. Regressão Linear Múltipla
    lin_reg = LinearRegression()
    lin_reg.fit(X_train, y_train)
    
    # 2. Regressão Polinomial (Otimizada)
    poly_pipeline = Pipeline([
        ('poly_features', PolynomialFeatures()),
        ('lin_reg', LinearRegression())
    ])
    poly_grid = GridSearchCV(poly_pipeline, config.POLY_PARAMS, cv=5, scoring='neg_mean_absolute_error')
    poly_grid.fit(X_train, y_train)
    
    # 3. KNN Regressor (Otimizado)
    knn = KNeighborsRegressor()
    knn_grid = GridSearchCV(knn, config.KNN_PARAMS, cv=5, scoring='neg_mean_absolute_error')
    knn_grid.fit(X_train, y_train)
    
    return {
        "Regressão Linear": lin_reg,
        f"Regressão Polinomial (Grau {poly_grid.best_params_['poly_features__degree']})": poly_grid.best_estimator_,
        f"KNN Regressor (k={knn_grid.best_params_['n_neighbors']})": knn_grid.best_estimator_
    }

def evaluate_models(models, X_test, y_test):
    """Calcula MAE, RMSE e R² para os modelos treinados."""
    resultados = []
    
    for nome, modelo in models.items():
        y_pred = modelo.predict(X_test)
        
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        
        resultados.append({
            "Modelo": nome,
            "MAE (Km/L)": round(mae, 4),
            "RMSE": round(rmse, 4),
            "R² Score": round(r2, 4)
        })
        
    return pd.DataFrame(resultados)