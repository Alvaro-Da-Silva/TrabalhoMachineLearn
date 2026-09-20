# Caminho do dataset (Kaggle)
DATASET_PATH = 'src/auto-mpg.csv'

# Constantes de Engenharia de Atributos e Separação
MPG_TO_KML = 0.425144
TEST_SIZE = 0.2
RANDOM_STATE = 42

# Hiperparâmetros para Otimização (GridSearchCV)
POLY_PARAMS = {'poly_features__degree': [2, 3]}
KNN_PARAMS = {'n_neighbors': [3, 5, 7, 9, 11, 15]}