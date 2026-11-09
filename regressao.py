import pandas as pd
import numpy as np

def calcular_regressao(arquivo):

    df = pd.read_csv(arquivo)

    area = df["area_util_m2"].to_list()
    quartos = df["quantidade_quartos"].to_list()
    preco = df["preco_venda"].to_list()

    X = np.array([
        np.ones(len(area)),
        area,
        quartos
    ]).T

    Y = np.array(preco)

    coeficientes = np.linalg.lstsq(X, Y, rcond=None)[0]

    return coeficientes.tolist()


resultado = calcular_regressao("imoveis_lar_perfeito.csv")

print("Coeficientes:")
print(resultado)

print("\nEquação final:")
print(f"Preco = {resultado[0]:.2f} + "
      f"{resultado[1]:.2f} * Area + "
      f"{resultado[2]:.2f} * Quartos")