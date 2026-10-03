import pickle

# Se agregó la extensión .pkl al final de la ruta
with open('data/modelo_rf_final.pkl', 'rb') as archivo:
    datos = pickle.load(archivo)

# Imprime el contenido para verlo en la terminal
print(datos)