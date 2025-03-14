import pandas as pd  # Manipular dados
import numpy as np  # Trabalhar com números
import matplotlib.pyplot as plt  # Criar gráficos
from sklearn.model_selection import train_test_split  # Dividir dados
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error  # Avaliar o modelo

df = pd.read_csv('C:\\Users\\destr\\Documents\\Enchentes\\segundo\\dados_novos.csv')

# Remover valores ausentes (se houver)
df = df.dropna()

# Selecionar as colunas mais importantes
colunas_usadas = ["Data", "Taxa Chuva (mm/h)", "Temp (ºC)", "Nível", "Chuva Acum. Dia (mm)"]
df = df[colunas_usadas]

# Separar "x" (dados de entrada) e "y" (o que queremos prever)
x = df.drop("Nível", axis=1)  # Tudo menos o nível do rio
y = df["Nível"]  # Apenas o nível do rio

x = pd.get_dummies(x, drop_first=True)  # Converte texto em números

"""print(x.dtypes)
print(x.head)

print(x.shape, y.shape)
"""

x_treino, x_teste, y_treino, y_teste = train_test_split(x, y, test_size=0.2, random_state=42)

# Criar o modelo de Regressão Linear
modelo = LinearRegression()

# Treinar o modelo com os dados de treino
modelo.fit(x_treino, y_treino)

# Fazer previsões nos dados de teste
y_previsto = modelo.predict(x_teste)

# Calcular erro médio absoluto (MAE)
mae = mean_absolute_error(y_teste, y_previsto)

# Calcular erro quadrático médio (MSE)
mse = mean_squared_error(y_teste, y_previsto)

print(f"Erro Médio Absoluto: {mae:.2f}")
print(f"Erro Quadrático Médio: {mse:.2f}")

print(x_treino.columns)  # Ver as colunas usadas no treinamento
print(x_treino.shape)    # Ver número de colunas

# Exemplo de um novo dia
novo_dado = pd.DataFrame([
    {"Taxa Chuva (mm/h)": 0, "Temp (ºC)": 28, "Chuva Acum. Dia (mm)": 50},
])

# Ajustar colunas para ter o mesmo formato do treino
novo_dado = novo_dado.reindex(columns=x_treino.columns, fill_value=0)

# Fazer previsão
nivel_previsto = modelo.predict(novo_dado)

print(f"Nível do rio previsto: {nivel_previsto[0]} metros")