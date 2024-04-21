# Importar las bibliotecas necesarias
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Cargar los datos
data = pd.read_csv('customer_data.csv')

# 1. Análisis exploratorio de datos (EDA)
print(data.head())
print(data.describe())
sns.pairplot(data)
plt.show()

# 2. Predicción de la cantidad gastada por el cliente
X = data.drop(['purchase_amount'], axis=1)
y = data['purchase_amount']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
y_pred = lr_model.predict(X_test)
# Evaluación del modelo
mse = mean_squared_error(y_test, y_pred)
print("Error cuadrático medio:", mse)

# 3. Segmentación de clientes basada en su comportamiento de compra
X_clustering = data[['purchase_frequency', 'purchase_amount', 'satisfaction_score']]
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X_clustering)
data['cluster'] = kmeans.labels_
sns.scatterplot(data=data, x='purchase_frequency', y='purchase_amount', hue='cluster')
plt.title('Segmentación de clientes')
plt.show()

# 4. Análisis de la satisfacción del cliente
sns.boxplot(data=data, x='satisfaction_score', y='education')
plt.title('Satisfacción del cliente por nivel educativo')
plt.show()

# 5. Predicción de la utilización de promociones
X_promotion = data.drop(['promotion_usage'], axis=1)
y_promotion = data['promotion_usage']
X_train_promotion, X_test_promotion, y_train_promotion, y_test_promotion = train_test_split(X_promotion, y_promotion, test_size=0.2, random_state=42)
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train_promotion, y_train_promotion)
y_pred_promotion = rf_model.predict(X_test_promotion)
# Evaluación del modelo
accuracy = accuracy_score(y_test_promotion, y_pred_promotion)
print("Precisión del modelo de promoción:", accuracy)
