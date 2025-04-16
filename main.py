# Importación de librerías necesarias
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
import numpy as np
import os

# 1. Crear carpeta 'output' para guardar los gráficos
os.makedirs("output", exist_ok=True)

# 2. Leer el archivo CSV
df = pd.read_csv("data/ganancias.csv", skiprows=1)

# 3. Transformar datos al formato largo
df_largo = df.melt(id_vars="mes", var_name="año", value_name="ganancias")
orden_meses = ["enero", "febrero", "marzo", "abril", "mayo", "junio"]
df_largo["mes"] = pd.Categorical(df_largo["mes"], categories=orden_meses, ordered=True)

# 4. Gráfico de Barras – Ganancias Totales por Año
totales = df_largo.groupby("año")["ganancias"].sum().reset_index()
plt.figure(figsize=(8, 5))
sns.barplot(data=totales, x="año", y="ganancias", palette="Blues_d")
plt.title("Ganancias Totales por Año")
plt.tight_layout()
plt.savefig("output/ganancias_barras.png", dpi=300)
plt.show()

# 5. Heatmap – Ganancias por Mes y Año
heatmap = df_largo.pivot_table(index="año", columns="mes", values="ganancias")
plt.figure(figsize=(8, 5))
sns.heatmap(heatmap, annot=True, fmt=".0f", cmap="YlGnBu")
plt.title("Heatmap de Ganancias por Mes y Año")
plt.tight_layout()
plt.savefig("output/heatmap_ganancias.png", dpi=300)
plt.show()

# 6. Boxplot – Distribución por Año
plt.figure(figsize=(8, 5))
sns.boxplot(data=df_largo, x="año", y="ganancias", palette="Pastel1")
plt.title("Distribución de Ganancias por Año")
plt.tight_layout()
plt.savefig("output/boxplot_ganancias.png", dpi=300)
plt.show()

# 7. Regresión Lineal – Tendencia por Año
reg_data = df_largo.groupby("año")["ganancias"].sum().reset_index()
reg_data["año"] = reg_data["año"].astype(int)
X = reg_data[["año"]]
y = reg_data["ganancias"]
modelo = LinearRegression().fit(X, y)
y_pred = modelo.predict(X)

plt.figure(figsize=(8, 5))
plt.plot(X, y, marker='o', label="Ganancias reales")
plt.plot(X, y_pred, linestyle='--', color='red', label="Tendencia")
plt.title("Tendencia de Ganancias por Año (Regresión Lineal)")
plt.legend()
plt.tight_layout()
plt.savefig("output/regresion_lineal.png", dpi=300)
plt.show()