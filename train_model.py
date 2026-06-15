# ============================================================
#  train_model.py  —  À LANCER UNE SEULE FOIS
#  Entraîne le modèle XGBoost et l'exporte avec joblib.
#  Reprend la logique de ton notebook :
#  exclusion de duration, feature engineering, one-hot, scaler.
# ============================================================

import pandas as pd
import numpy as np
import re
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

# 1) Chargement
df = pd.read_csv("bank.csv")

# 2) Cible en 0/1
df["y"] = (df["deposit"] == "yes").astype(int)

# 3) DATA LEAKAGE : on retire duration (connue seulement après l'appel)
#    et deposit (qui est la cible elle-même)
df = df.drop(columns=["duration", "deposit"])

# 4) FEATURE ENGINEERING (tes 5 variables)
df["jamais_contacte"] = (df["pdays"] == -1).astype(int)
df["nb_prets"] = (df["housing"] == "yes").astype(int) + (df["loan"] == "yes").astype(int)
df["balance_par_age"] = df["balance"] / df["age"]
df["tranche_jour_mois"] = pd.cut(df["day"], bins=[0, 10, 20, 31],
                                 labels=["debut", "milieu", "fin"])
df["tranche_age"] = pd.cut(df["age"], bins=[0, 25, 35, 50, 60, 100],
                           labels=["<25", "25-35", "35-50", "50-60", "60+"])

# 5) Séparation X / y
y = df["y"]
X = df.drop(columns=["y"])

# 6) One-hot encoding des variables catégorielles
X = pd.get_dummies(X, drop_first=True)

# 7) Nettoyage des noms de colonnes (crochets [] interdits par XGBoost)
X.columns = [re.sub(r"[\[\]<>(),\s]", "_", str(c)) for c in X.columns]

# 8) On mémorise l'ordre exact des colonnes (CRUCIAL pour le scoring)
colonnes_modele = list(X.columns)

# 9) Split stratifié 80/20
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

# 10) Standardisation (fit sur le TRAIN uniquement)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 11) Entraînement XGBoost
model = XGBClassifier(eval_metric="logloss", random_state=42)
model.fit(X_train_scaled, y_train)

# 12) Petit contrôle
acc = model.score(X_test_scaled, y_test)
print(f"Accuracy test : {acc:.4f}")

# 13) EXPORT : modèle + scaler + liste des colonnes
joblib.dump(model, "model_xgb.joblib")
joblib.dump(scaler, "scaler.joblib")
joblib.dump(colonnes_modele, "colonnes_modele.joblib")
print("Export OK : model_xgb.joblib, scaler.joblib, colonnes_modele.joblib")