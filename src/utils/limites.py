import pandas as pd

# Cargamos el dataset
df = pd.read_csv('data/smart_manufacturing_data.csv')

# Imprimimos el mínimo, máximo y promedio de cada sensor
print(df[['temperature', 'vibration', 'humidity', 'pressure', 'energy_consumption']].describe().T[['min', 'mean', 'max']])