import streamlit as st
import pandas as pd
import plotly.express as px

import joblib
import re
# 1. Charger les données
df = pd.read_csv("bank.csv")

# 2. Titre
st.title("Dashboard Marketing Bancaire")

# 3. Filtres dans la barre latérale
st.sidebar.header("🔎 Filtres")

# Filtre 1 : tranche d'âge (plage min-max)
age_min, age_max = st.sidebar.slider("Tranche d'âge", 18, 95, (18, 95))

# Filtre 2 : métier (sélection multiple)
jobs = st.sidebar.multiselect(
    "Métier",
    options=sorted(df["job"].unique()),
    default=sorted(df["job"].unique())
)

# Filtre 3 : éducation (sélection multiple)
educations = st.sidebar.multiselect(
    "Éducation",
    options=sorted(df["education"].unique()),
    default=sorted(df["education"].unique())
)

# Filtre 4 : solde / balance (plage min-max)
bal_min, bal_max = st.sidebar.slider(
    "Solde (€)",
    int(df["balance"].min()),
    int(df["balance"].max()),
    (int(df["balance"].min()), int(df["balance"].max()))
)

# Application de tous les filtres en même temps
df_filtre = df[
    (df["age"] >= age_min) & (df["age"] <= age_max) &
    (df["job"].isin(jobs)) &
    (df["education"].isin(educations)) &
    (df["balance"] >= bal_min) & (df["balance"] <= bal_max)
]

# Sécurité : si aucun client ne correspond aux filtres
if len(df_filtre) == 0:
    st.warning("⚠️ Aucun client ne correspond à ces filtres. Élargis ta sélection.")
    st.stop()

# 4. Un chiffre clé
taux = (df_filtre["deposit"] == "yes").mean() * 100
st.metric("Taux de souscription", f"{taux:.1f} %")

# On crée les 3 onglets
tab1, tab2, tab3, tab4 = st.tabs(["Profil client", "Argent", "Historique campagne", "Scoring"])


# ===================== ONGLET 1 : PROFIL CLIENT =====================
with tab1:

    # --- Histogramme des âges ---
    st.subheader("Répartition des âges")
    fig = px.histogram(df_filtre, x="age", color="deposit")
    st.plotly_chart(fig)

    # --- Souscription par tranche d'âge (reprise du graphique d'origine) ---
    st.subheader("Répartition des dépôts par tranche d'âge")

    # Découpage automatique en 5 tranches égales (comme pd.cut(df['age'], 5))
    df_filtre["tranche_age"] = pd.cut(df_filtre["age"], 5)

    rassemblement_par_age = (
        df_filtre.groupby("tranche_age", observed=True)["deposit"]
        .value_counts(normalize=True)
        .rename("pourcentage")
        .reset_index()
    )
    rassemblement_par_age["tranche_age"] = rassemblement_par_age["tranche_age"].astype(str)

    fig_age = px.bar(
        rassemblement_par_age,
        x="tranche_age",
        y="pourcentage",
        color="deposit",
        barmode="group",
        labels={"tranche_age": "tranche age",
                "pourcentage": "Pourcentage",
                "deposit": "Dépôt"},
        title="Répartition des dépôts Oui/Non",
        color_discrete_map={"no": "#3B6B8F", "yes": "#4CA777"}
    )
    st.plotly_chart(fig_age)

    # --- Souscription selon le nombre de prêts ---
    st.subheader("Taux de souscription selon le nombre de prêts")

    df_filtre["nb_prets"] = (
        (df_filtre["housing"] == "yes").astype(int)
        + (df_filtre["loan"] == "yes").astype(int)
    )

    taux_prets = (
        df_filtre.groupby("nb_prets")["deposit"]
        .apply(lambda s: (s == "yes").mean() * 100)
        .reset_index()
    )

    fig_prets = px.bar(
        taux_prets,
        x="nb_prets",
        y="deposit",
        labels={"nb_prets": "Nombre de prêts (0, 1 ou 2)", "deposit": "Taux (%)"},
        title="Souscription selon le nombre de prêts",
        color="deposit",
        color_continuous_scale="Teal"
    )
    fig_prets.update_xaxes(tickmode="linear")
    st.plotly_chart(fig_prets)


# ===================== ONGLET 2 : ARGENT =====================
with tab2:

    st.subheader("Nombre de souscriptions par tranche de solde")

    # qcut peut échouer si trop peu de valeurs distinctes après filtrage -> sécurité
    try:
        df_filtre["tranche_balance"] = pd.qcut(df_filtre["balance"], q=5, duplicates="drop")

        df_succes = df_filtre[df_filtre["deposit"] == "yes"]

        compte_balance = df_succes["tranche_balance"].value_counts().sort_index().reset_index()
        compte_balance.columns = ["tranche_balance", "nombre"]
        compte_balance["tranche_balance"] = compte_balance["tranche_balance"].astype(str)

        fig_bal = px.bar(
            compte_balance, x="tranche_balance", y="nombre",
            labels={"tranche_balance": "Tranche de solde (€)", "nombre": "Nb de souscriptions"},
            title="Nombre de souscriptions par tranche de solde"
        )
        st.plotly_chart(fig_bal)
    except ValueError:
        st.info("Pas assez de diversité de soldes pour créer des tranches avec ces filtres.")


# ===================== ONGLET 3 : HISTORIQUE CAMPAGNE =====================
with tab3:

    # --- Taux de souscription selon le résultat passé ---
    st.subheader("Taux de souscription selon le résultat de la campagne précédente")

    taux_poutcome = (
        df_filtre.groupby("poutcome", observed=True)["deposit"]
        .apply(lambda s: (s == "yes").mean() * 100)
        .reset_index()
        .sort_values("deposit", ascending=False)
    )

    fig_pout = px.bar(
        taux_poutcome,
        x="poutcome",
        y="deposit",
        labels={"poutcome": "Résultat campagne précédente", "deposit": "Taux (%)"},
        title="Souscription selon le résultat passé",
        color="deposit",
        color_continuous_scale="Teal"
    )
    st.plotly_chart(fig_pout)

    # --- Résultat de campagne par métier ---
    st.subheader("Résultat des campagnes par métier")

    df_job = df_filtre[df_filtre["poutcome"] != "unknown"]

    compte_job = (
        df_job.groupby(["job", "poutcome"], observed=True)
        .size()
        .reset_index(name="nombre")
    )

    fig_job = px.bar(
        compte_job,
        x="job",
        y="nombre",
        color="poutcome",
        barmode="group",
        labels={"job": "Métier", "nombre": "Nombre de clients", "poutcome": "Résultat"},
        title="Résultat des campagnes précédentes par métier"
    )
    fig_job.update_xaxes(tickangle=-45)
    st.plotly_chart(fig_job)

# ===================== ONGLET 4 : SCORING =====================
with tab4:
    st.subheader("🎯 Estimer la probabilité de souscription d'un client")

    # ---------- ARCHÉTYPE DU CLIENT À CONTACTER ----------
    st.markdown("### 👤 Portrait-robot du client à cibler")
    st.caption("Profil type des clients ayant réellement souscrit (deposit = yes)")

    # On isole les souscripteurs
    souscripteurs = df[df["deposit"] == "yes"]

    # Caractéristiques numériques : on prend la médiane
    archetype = {
        "Âge médian": f"{int(souscripteurs['age'].median())} ans",
        "Solde médian": f"{int(souscripteurs['balance'].median())} €",
        "Nb de contacts médian": f"{int(souscripteurs['campaign'].median())}",
    }
    # Caractéristiques catégorielles : on prend la valeur la plus fréquente (mode)
    for col, label in [("job", "Métier le plus fréquent"),
                       ("marital", "Statut marital"),
                       ("education", "Éducation"),
                       ("housing", "Prêt immobilier"),
                       ("loan", "Prêt personnel"),
                       ("contact", "Type de contact")]:
        valeur = souscripteurs[col].mode()[0]
        pct = (souscripteurs[col] == valeur).mean() * 100
        archetype[label] = f"{valeur} ({pct:.0f} %)"

    # Affichage en tableau
    tableau = pd.DataFrame(
        list(archetype.items()),
        columns=["Caractéristique", "Valeur type"]
    )
    st.table(tableau)

    st.info("💡 En résumé : viser des clients **sans prêt**, **contactés par téléphone**, "
            "avec un **solde positif** — ce sont les traits dominants des souscripteurs.")

    st.divider()


    # Charger le modèle exporté (et gérer le cas où il n'existe pas)
    try:
        model = joblib.load("model_xgb.joblib")
        scaler = joblib.load("scaler.joblib")
        colonnes = joblib.load("colonnes_modele.joblib")
    except FileNotFoundError:
        st.error("Modèle introuvable. Lance d'abord : python train_model.py")
        st.stop()

    # Champs de saisie, organisés en 3 colonnes
    c1, c2, c3 = st.columns(3)
    with c1:
        in_age = st.number_input("Âge", 18, 95, 40)
        in_balance = st.number_input("Solde (€)", -8000, 100000, 1000)
        in_job = st.selectbox("Métier", sorted(df["job"].unique()))
        in_marital = st.selectbox("Statut marital", sorted(df["marital"].unique()))
        in_education = st.selectbox("Éducation", sorted(df["education"].unique()))
    with c2:
        in_housing = st.selectbox("Prêt immobilier", ["no", "yes"])
        in_loan = st.selectbox("Prêt personnel", ["no", "yes"])
        in_default = st.selectbox("Défaut de crédit", ["no", "yes"])
        in_contact = st.selectbox("Type de contact", sorted(df["contact"].unique()))
        in_poutcome = st.selectbox("Résultat campagne précédente", sorted(df["poutcome"].unique()))
    with c3:
        in_campaign = st.number_input("Nb contacts (campagne)", 1, 60, 2)
        in_pdays = st.number_input("Jours depuis dernier contact (-1 = jamais)", -1, 900, -1)
        in_previous = st.number_input("Nb contacts précédents", 0, 100, 0)
        in_day = st.number_input("Jour du mois", 1, 31, 15)
        in_month = st.selectbox("Mois", sorted(df["month"].unique()))

    # Bouton de calcul
    if st.button("Calculer la probabilité", type="primary"):
        # 1) Construire une ligne au même format que les données d'origine
        ligne = pd.DataFrame([{
            "age": in_age, "job": in_job, "marital": in_marital,
            "education": in_education, "default": in_default, "balance": in_balance,
            "housing": in_housing, "loan": in_loan, "contact": in_contact,
            "day": in_day, "month": in_month, "campaign": in_campaign,
            "pdays": in_pdays, "previous": in_previous, "poutcome": in_poutcome
        }])

        # 2) MÊME feature engineering qu'à l'entraînement
        ligne["jamais_contacte"] = (ligne["pdays"] == -1).astype(int)
        ligne["nb_prets"] = (ligne["housing"] == "yes").astype(int) + (ligne["loan"] == "yes").astype(int)
        ligne["balance_par_age"] = ligne["balance"] / ligne["age"]
        ligne["tranche_jour_mois"] = pd.cut(ligne["day"], [0, 10, 20, 31],
                                            labels=["debut", "milieu", "fin"])
        ligne["tranche_age"] = pd.cut(ligne["age"], [0, 25, 35, 50, 60, 100],
                                      labels=["<25", "25-35", "35-50", "50-60", "60+"])

        # 3) One-hot + nettoyage des noms (comme à l'entraînement)
        ligne = pd.get_dummies(ligne, drop_first=True)
        ligne.columns = [re.sub(r"[\[\]<>(),\s]", "_", str(c)) for c in ligne.columns]

        # 4) ÉTAPE CRUCIALE : aligner sur les colonnes exactes du modèle
        ligne = ligne.reindex(columns=colonnes, fill_value=0)

        # 5) Standardiser puis prédire
        ligne_scaled = scaler.transform(ligne)
        proba = model.predict_proba(ligne_scaled)[0][1]

        # 6) Affichage du résultat
        st.metric("Probabilité de souscription", f"{proba*100:.1f} %")
        if proba >= 0.5:
            st.success("✅ Client prioritaire à contacter")
        else:
            st.warning("⚠️ Client peu susceptible de souscrire")