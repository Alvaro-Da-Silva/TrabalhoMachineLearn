# main.py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from dataPrep import load_and_preprocess_data
from Train import train_models, evaluate_models

# Configuração visual padrão
sns.set_theme(style="whitegrid")

def plot_metrics(df_resultados):
    """Gera gráfico comparativo de MAE."""
    plt.figure(figsize=(9, 5))
    sns.barplot(
        data=df_resultados, 
        x="Modelo", 
        y="MAE (Km/L)", 
        hue="Modelo", 
        palette="viridis", 
        legend=False
    )
    plt.title("Comparação do Erro Absoluto Médio (MAE) por Modelo")
    plt.ylabel("Erro Médio em Km/L (Menor é Melhor)")
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.show()

def plot_feature_importance(modelo_linear, colunas):
    """Gera gráfico demonstrando o peso de cada atributo no consumo."""
    # Dicionário para traduzir e formatar os rótulos do eixo Y
    nomes_formatados = {
        'model year': 'Ano de Fabricação',
        'displacement': 'Cilindrada',
        'origin_2': 'Origem: Europa',
        'origin_3': 'Origem: Japão',
        'acceleration': 'Aceleração',
        'cylinders': 'Cilindros',
        'horsepower': 'Potência (HP)',
        'weight': 'Peso (Kg)'
    }

    coeficientes = pd.DataFrame({
        'Atributo': colunas,
        'Peso (Coeficiente)': modelo_linear.coef_
    })

    # Aplica a tradução dos nomes e ordena do maior para o menor
    coeficientes['Atributo'] = coeficientes['Atributo'].replace(nomes_formatados)
    coeficientes = coeficientes.sort_values(by='Peso (Coeficiente)', ascending=False)

    plt.figure(figsize=(9, 5))
    sns.barplot(
        data=coeficientes, 
        x='Peso (Coeficiente)', 
        y='Atributo', 
        hue='Atributo', 
        palette="coolwarm", 
        legend=False
    )
    plt.title("Importância dos Atributos - Regressão Linear")
    plt.xlabel("Impacto no Consumo (Km/L)")
    plt.tight_layout()
    plt.show()

def main():
    print("1. Carregando e pré-processando os dados...")
    X_train, X_test, y_train, y_test, feature_names = load_and_preprocess_data()
    print(f"   -> Treino: {X_train.shape[0]} amostras | Teste: {X_test.shape[0]} amostras\n")
    
    print("2. Treinando e otimizando os modelos...")
    modelos = train_models(X_train, y_train)
    
    print("\n3. Avaliando os resultados...")
    df_resultados = evaluate_models(modelos, X_test, y_test)
    
    print("\n--- TABELA DE MÉTRICAS (CONJUNTO DE TESTE) ---")
    print(df_resultados.to_string(index=False))
    
    print("\n4. Gerando visualizações...")
    plot_metrics(df_resultados)
    plot_feature_importance(modelos["Regressão Linear"], feature_names)

if __name__ == "__main__":
    main()