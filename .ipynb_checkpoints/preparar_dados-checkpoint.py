import pandas as pd
import matplotlib.pyplot as plt

#df = pd.read_excel("C:\\Users\\destr\\Documents\\Enchentes\\segundo\\dados_atualizados.xlsx")
#df.to_csv("C:\\Users\\destr\\Documents\\Enchentes\\segundo\\dados_atualizados.csv", index = False)

#Carregar os dados
df = pd.read_csv("C:\\Users\\destr\\Documents\\Enchentes\\segundo\\dados_novos.csv")

df["Data"] = pd.to_datetime(df["Data"])  # Converte para formato de data
df["Data"] = df["Data"].astype(int) // 10**9  # Converte para timestamp (segundos)

#print(df.duplicated())
#df.drop_duplicates(inplace = True)