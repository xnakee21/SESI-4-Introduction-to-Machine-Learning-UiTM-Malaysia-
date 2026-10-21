import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# 1. Pembuatan Dataset Transaksi Keuangan
np.random.seed(100)
n_tx = 500

data_transaksi = {
    'Nominal_Transaksi': np.random.exponential(scale=1000000, size=n_tx),
    'Jarak_Lokasi_Km': np.random.uniform(0, 500, n_tx),
    'Frekuensi_24Jam': np.random.randint(1, 20, n_tx),
    'Jam_Transaksi': np.random.randint(0, 24, n_tx),
    'Is_Fraud': np.random.choice([0, 1], size=n_tx, p=[0.9, 0.1])
}

df_s4 = pd.DataFrame(data_transaksi)

X_s4 = df_s4[['Nominal_Transaksi', 'Jarak_Lokasi_Km', 'Frekuensi_24Jam', 'Jam_Transaksi']]
y_s4 = df_s4['Is_Fraud']

# 2. Splitting Train (80%) & Test (20%)
X_train_s4, X_test_s4, y_train_s4, y_test_s4 = train_test_split(X_s4, y_s4, test_size=0.2, random_state=42)

# 3. Model 1: Decision Tree
dt_model = DecisionTreeClassifier(max_depth=5, random_state=42)
dt_model.fit(X_train_s4, y_train_s4)
y_pred_dt = dt_model.predict(X_test_s4)

# 4. Model 2: K-Nearest Neighbors (KNN)
knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train_s4, y_train_s4)
y_pred_knn = knn_model.predict(X_test_s4)

# 5. Evaluasi Kinerja Kedua Algoritma
def evaluasi_model(nama_model, y_true, y_pred):
    print(f"=== EVALUASI ALGORITMA: {nama_model} ===")
    print(f"Akurasi  : {accuracy_score(y_true, y_pred)*100:.2f}%")
    print(f"Presisi  : {precision_score(y_true, y_pred, zero_division=0)*100:.2f}%")
    print(f"Recall   : {recall_score(y_true, y_pred, zero_division=0)*100:.2f}%")
    print(f"F1-Score : {f1_score(y_true, y_pred, zero_division=0)*100:.2f}%\n")

evaluasi_model("Decision Tree", y_test_s4, y_pred_dt)
evaluasi_model("K-Nearest Neighbors", y_test_s4, y_pred_knn)
