# 1. Entendendo o Problema Prático

## O Cenário:

Montadoras de veículos e consumidores precisam avaliar o impacto do design do motor e do peso do veículo na sua eficiência energética.

## O Objetivo de ML:

Treinar modelos de regressão para estimar com precisão quantos quilômetros por litro (ou Miles Per Gallon — MPG) um carro faz com base em suas características físicas e de engenharia.

## Tipo de Problema:

Regressão (a variável alvo MPG é um valor numérico contínuo).

# 2. Atributos do Dataset (Features)

Você utilizará características numéricas e categóricas diretas e fáceis de interpretar:

**Variável Alvo (Target, $y$):**
mpg: Consumo em milhas por galão (quanto maior, mais econômico é o carro).

**Variáveis de Entrada (Features, $X$):**
    
    cylinders: Quantidade de cilindros do motor (ex: 4, 6, 8).
    
    displacement: Cilindrada / tamanho do motor (volume dos cilindros).
    
    horsepower: Potência do motor em cavalos (CV).
    
    weight: Peso total do veículo (em libras ou kg).
    
    acceleration: Tempo em segundos para ir de 0 a 60 mph.
    
    model_year: Ano de fabricação do veículo.
    
    origin: Origem do fabricante (1: EUA, 2: Europa, 3: Japão).