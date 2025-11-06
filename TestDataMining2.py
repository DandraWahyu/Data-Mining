# Program Data Mining: Perbandingan Model Klasifikasi Iris

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import numpy as np

# === 1. Load dataset ===
iris = sns.load_dataset("iris")
print("=== Data Awal ===")
print(iris.head())

# === 2. Pisahkan fitur dan label ===
X = iris.drop("species", axis=1)
y = iris["species"]

# === 3. Split data ===
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# === 4. Normalisasi ===
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# === 5. Inisialisasi model ===
models = {
    "KNN (k=5)": KNeighborsClassifier(n_neighbors=5),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "SVM (RBF Kernel)": SVC(kernel='rbf', gamma='scale', random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42)
}

# === 6. Latih dan evaluasi ===
akurasi = {}
for nama, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    skor = accuracy_score(y_test, y_pred)
    akurasi[nama] = skor
    print(f"\n=== {nama} ===")
    print(classification_report(y_test, y_pred))
    print("Akurasi:", round(skor, 4))

# === 7. Visualisasi akurasi model ===
plt.figure(figsize=(8, 5))
sns.barplot(x=list(akurasi.keys()), y=list(akurasi.values()), palette="cool")
plt.title("Perbandingan Akurasi Model Klasifikasi Iris", fontsize=13)
plt.ylabel("Akurasi")
plt.ylim(0, 1)
plt.xticks(rotation=15)
plt.show()

# === 8. Visualisasi Decision Boundary (2 fitur saja) ===
# Gunakan hanya dua fitur agar bisa divisualisasikan (petal_length dan petal_width)
X2 = iris[["petal_length", "petal_width"]]
y2 = iris["species"]

X_train2, X_test2, y_train2, y_test2 = train_test_split(X2, y2, test_size=0.3, random_state=42)
scaler2 = StandardScaler()
X_train2_scaled = scaler2.fit_transform(X_train2)
X_test2_scaled = scaler2.transform(X_test2)

model = SVC(kernel='rbf', gamma='auto', random_state=42)
model.fit(X_train2_scaled, y_train2)

# Membuat grid untuk plot boundary
x_min, x_max = X_train2_scaled[:, 0].min() - 1, X_train2_scaled[:, 0].max() + 1
y_min, y_max = X_train2_scaled[:, 1].min() - 1, X_train2_scaled[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                     np.arange(y_min, y_max, 0.02))
Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Visualisasi decision boundary
plt.figure(figsize=(8, 6))
plt.contourf(xx, yy, Z, alpha=0.3, cmap="coolwarm")
sns.scatterplot(x=X_train2_scaled[:, 0], y=X_train2_scaled[:, 1], hue=y_train2, palette="Set1", edgecolor="k")
plt.title("Decision Boundary (SVM) pada Fitur Petal Length vs Petal Width")
plt.xlabel("Petal Length (scaled)")
plt.ylabel("Petal Width (scaled)")
plt.show()

# === 9. Prediksi contoh baru ===
contoh = [[5.1, 3.5, 1.4, 0.2]]
contoh_scaled = scaler.transform(contoh)
hasil = models["Random Forest"].predict(contoh_scaled)
print("\nPrediksi untuk data", contoh, "adalah:", hasil[0])
