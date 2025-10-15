# Program Data Mining: Klasifikasi Dataset Iris

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Load dataset
iris = sns.load_dataset("iris")
print("=== Data Awal ===")
print(iris.head())

# Visualisasi data
sns.pairplot(iris, hue="species")
plt.suptitle("Visualisasi Hubungan Antar Fitur Iris Dataset", y=1.02)
plt.show()

# Pisahkan fitur dan label
X = iris.drop("species", axis=1)
y = iris["species"]

# Bagi data menjadi data latih dan uji
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Normalisasi data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Latih model KNN
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_scaled, y_train)

# Prediksi data uji
y_pred = knn.predict(X_test_scaled)

# Evaluasi hasil
print("\n=== Evaluasi Model KNN ===")
print(confusion_matrix(y_test, y_pred))
print("\nLaporan Klasifikasi:")
print(classification_report(y_test, y_pred))
print("Akurasi:", accuracy_score(y_test, y_pred))

# Contoh prediksi baru
contoh = [[5.1, 3.5, 1.4, 0.2]]  # Data baru (sepal & petal)
contoh_scaled = scaler.transform(contoh)
prediksi = knn.predict(contoh_scaled)
print("\nPrediksi untuk data", contoh, "adalah:", prediksi[0])
