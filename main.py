import pandas as pd  # Manipular dados
import numpy as np  # Trabalhar com números
import matplotlib.pyplot as plt  # Criar gráficos
from sklearn.model_selection import train_test_split  # Dividir dados
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error  # Avaliar o modelo

df = pd.read_csv('dados_atualizados.csv')

# Remover valores ausentes (se houver)
df = df.dropna()
df = df.set_index('Data')

# Selecionar as colunas mais importantes
colunas_usadas = ['Nível', 'Taxa Chuva (mm/h)', 'Chuva Acum. Dia (mm)','Temp (ºC)', 'umidade_2m', 'velocidade_vento']
df = df[colunas_usadas]

# Separar "x" (dados de entrada) e "y" (o que queremos prever)
x = df.drop("Nível", axis=1)  # Tudo menos o nível do rio
y = df["Nível"]  # Apenas o nível do rio

x = pd.get_dummies(x, drop_first=True)  # Converte texto em números

#print(x.dtypes)
#print(x.head)
#print(x.shape, y.shape)

x_treino, x_teste, y_treino, y_teste = train_test_split(x, y, test_size=0.2, random_state=42)

# Criar e treinar o modelo Random Forest
modelo = RandomForestRegressor(n_estimators=500, random_state=42)
modelo.fit(x_treino, y_treino)

# Fazer previsões nos dados de teste
y_previsto = modelo.predict(x_teste)

# Calcular métricas de erro
mae = mean_absolute_error(y_teste, y_previsto)
mse = mean_squared_error(y_teste, y_previsto)
rmse = np.sqrt(mse)

print(f"Erro Médio Absoluto (MAE): {mae:.2f}")
print(f"Erro Quadrático Médio (MSE): {mse:.2f}")
print(f"Raiz do Erro Quadrático Médio (RMSE): {rmse:.2f}")

# Fazer previsão para um novo dia
novo_dado1 = pd.DataFrame([
    {"Taxa Chuva (mm/h)": 0, "Temp (ºC)": 23.4, "Chuva Acum. Dia (mm)": 0},
    {"Taxa Chuva (mm/h)": 0, "Temp (ºC)": 24, "Chuva Acum. Dia (mm)": 0},
    {"Taxa Chuva (mm/h)": 0, "Temp (ºC)": 20.3, "Chuva Acum. Dia (mm)": 0},
    {"Taxa Chuva (mm/h)": 0, "Temp (ºC)": 21.8, "Chuva Acum. Dia (mm)": 0}
])

# Ajustar colunas do novo dado
novo_dado1 = novo_dado1.reindex(columns=x_treino.columns, fill_value=0)

# Previsão do nível do rio
nivel_previsto = modelo.predict(novo_dado1)
print(f"Nível do rio previsto: {nivel_previsto[0]:.2f} metros")
