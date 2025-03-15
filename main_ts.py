import pandas as pd  # Manipular dados
import numpy as np  # Trabalhar com números
import matplotlib.pyplot as plt  # Criar gráficos
from sklearn.model_selection import TimeSeriesSplit  # Dividir dados
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error  # Avaliar o modelo

df = pd.read_csv('dados_atualizados.csv')

# Remover valores ausentes (se houver)
df = df.dropna()
df = df.set_index('Data')

# Converter o índice para datetime se ainda não estiver nesse formato
df.index = pd.to_datetime(df.index)

# Ordenar o DataFrame por data para garantir a ordem cronológica
df = df.sort_index()

# Selecionar as colunas mais importantes
colunas_usadas = ['Nível', 'Taxa Chuva (mm/h)', 'Chuva Acum. Dia (mm)','Temp (ºC)', 'umidade_2m', 'velocidade_vento']
df = df[colunas_usadas]

# Separar "x" (dados de entrada) e "y" (o que queremos prever)
x = df.drop("Nível", axis=1)  # Tudo menos o nível do rio
y = df["Nível"]  # Apenas o nível do rio

x = pd.get_dummies(x, drop_first=True)  # Converte texto em números

# Configurar o TimeSeriesSplit
tscv = TimeSeriesSplit(n_splits=2, test_size=int(0.2 * len(df)), gap=0)

# Inicializar listas para armazenar métricas de cada fold
mae_scores = []
mse_scores = []
rmse_scores = []

# Criar e treinar o modelo usando validação cruzada de séries temporais
for fold, (train_index, test_index) in enumerate(tscv.split(x)):
    print(f"Treinando fold {fold+1}/5")
    
    # Dividir os dados para este fold
    x_treino, x_teste = x.iloc[train_index], x.iloc[test_index]
    y_treino, y_teste = y.iloc[train_index], y.iloc[test_index]
    
    # Criar e treinar o modelo Random Forest
    modelo = RandomForestRegressor(n_estimators=500, random_state=42)
    modelo.fit(x_treino, y_treino)
    
    # Fazer previsões nos dados de teste
    y_previsto = modelo.predict(x_teste)
    
    # Calcular métricas de erro para este fold
    mae = mean_absolute_error(y_teste, y_previsto)
    mse = mean_squared_error(y_teste, y_previsto)
    rmse = np.sqrt(mse)
    
    # Armazenar as métricas
    mae_scores.append(mae)
    mse_scores.append(mse)
    rmse_scores.append(rmse)
    
    print(f"  Fold {fold+1} - MAE: {mae:.2f}, MSE: {mse:.2f}, RMSE: {rmse:.2f}")

# Calcular e exibir as métricas médias
print("\nMétricas médias de todos os folds:")
print(f"Erro Médio Absoluto (MAE): {np.mean(mae_scores):.2f}")
print(f"Erro Quadrático Médio (MSE): {np.mean(mse_scores):.2f}")
print(f"Raiz do Erro Quadrático Médio (RMSE): {np.mean(rmse_scores):.2f}")

# Treinar o modelo final com todos os dados
modelo_final = RandomForestRegressor(n_estimators=500, random_state=42)
modelo_final.fit(x, y)

# Fazer previsão para um novo dia
novo_dado1 = pd.DataFrame([
    {"Taxa Chuva (mm/h)": 0, "Temp (ºC)": 23.4, "Chuva Acum. Dia (mm)": 0, "umidade_2m": 80, "velocidade_vento": 5},
    {"Taxa Chuva (mm/h)": 0, "Temp (ºC)": 24, "Chuva Acum. Dia (mm)": 0, "umidade_2m": 75, "velocidade_vento": 6},
    {"Taxa Chuva (mm/h)": 0, "Temp (ºC)": 20.3, "Chuva Acum. Dia (mm)": 0, "umidade_2m": 85, "velocidade_vento": 4},
    {"Taxa Chuva (mm/h)": 0, "Temp (ºC)": 21.8, "Chuva Acum. Dia (mm)": 0, "umidade_2m": 82, "velocidade_vento": 3}
])

# Ajustar colunas do novo dado
novo_dado1 = pd.get_dummies(novo_dado1, drop_first=True)
novo_dado1 = novo_dado1.reindex(columns=x.columns, fill_value=0)

# Previsão do nível do rio
nivel_previsto = modelo_final.predict(novo_dado1)
for i, nivel in enumerate(nivel_previsto):
    print(f"Amostra {i+1} - Nível do rio previsto: {nivel:.2f} metros")
