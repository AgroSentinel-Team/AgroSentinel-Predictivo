import pandas as pd

df = pd.read_csv('data/smart_manufacturing_data.csv')

# Tu código original (sigue funcionando perfecto)
print("--- ESTADÍSTICAS GENERALES ---")
print(df[['temperature', 'vibration', 'humidity', 'pressure', 'energy_consumption']].describe().T[['min', 'mean', 'max']])


print("\n--- PROMEDIOS POR TIPO DE FALLA ---")
print(df.groupby('failure_type')[['temperature', 'vibration', 'humidity', 'pressure', 'energy_consumption']].mean())