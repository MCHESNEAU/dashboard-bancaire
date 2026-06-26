import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

import joblib
import re

# Doit être la TOUTE PREMIÈRE commande Streamlit : passe l'app en pleine largeur.
st.set_page_config(
    page_title="Dashboard Marketing Bancaire",
    page_icon="🏦",
    layout="wide",
)

# ===================== OUTILS MODÈLE (chargement + scoring) =====================
# Colonnes brutes minimales attendues dans un fichier de clients à scorer.
COLONNES_REQUISES = [
    "age", "job", "marital", "education", "default", "balance", "housing",
    "loan", "contact", "day", "month", "campaign", "pdays", "previous", "poutcome"
]


@st.cache_resource
def charger_modele():
    """Charge le modèle, le scaler et la liste des colonnes (mis en cache)."""
    model = joblib.load("model_xgb.joblib")
    scaler = joblib.load("scaler.joblib")
    colonnes = joblib.load("colonnes_modele.joblib")
    return model, scaler, colonnes


def _feature_engineering(df_in):
    """Reproduit EXACTEMENT les transformations de train_model.py."""
    d = df_in.copy()
    # Colonnes jamais utilisées par le modèle (cible / fuite de données)
    d = d.drop(columns=[c for c in ["duration", "deposit", "y"] if c in d.columns])

    d["jamais_contacte"] = (d["pdays"] == -1).astype(int)
    d["nb_prets"] = (d["housing"] == "yes").astype(int) + (d["loan"] == "yes").astype(int)
    d["balance_par_age"] = d["balance"] / d["age"]
    d["tranche_jour_mois"] = pd.cut(d["day"], [0, 10, 20, 31],
                                    labels=["debut", "milieu", "fin"])
    d["tranche_age"] = pd.cut(d["age"], [0, 25, 35, 50, 60, 100],
                              labels=["<25", "25-35", "35-50", "50-60", "60+"])

    d = pd.get_dummies(d, drop_first=True)
    d.columns = [re.sub(r"[\[\]<>(),\s]", "_", str(c)) for c in d.columns]
    return d


def preparer_features(df_in, colonnes):
    """Feature engineering + alignement exact sur les colonnes du modèle."""
    X = _feature_engineering(df_in)
    return X.reindex(columns=colonnes, fill_value=0)


def scorer(df_in, model, scaler, colonnes):
    """Renvoie un tableau de probabilités de souscription (0-1) pour chaque ligne."""
    X = preparer_features(df_in, colonnes)
    X_scaled = scaler.transform(X)
    return model.predict_proba(X_scaled)[:, 1]


@st.cache_resource
def charger_explainer():
    """Explainer SHAP (TreeExplainer) pour le modèle XGBoost (mis en cache)."""
    import shap
    model, _, _ = charger_modele()
    return shap.TreeExplainer(model)


# Noms de variables plus lisibles pour l'affichage SHAP.
def joli_nom(col):
    remplacements = {
        "poutcome": "Résultat précédent", "contact": "Contact", "month": "Mois",
        "job": "Métier", "education": "Éducation", "marital": "Statut marital",
        "housing": "Prêt immo", "loan": "Prêt perso", "default": "Défaut",
        "balance_par_age": "Solde/âge", "nb_prets": "Nb de prêts",
        "jamais_contacte": "Jamais contacté", "tranche_age": "Tranche d'âge",
        "tranche_jour_mois": "Période du mois", "campaign": "Nb contacts",
        "pdays": "Jours dernier contact", "previous": "Contacts précédents",
        "balance": "Solde", "age": "Âge", "day": "Jour",
    }
    for cle, val in remplacements.items():
        if col.startswith(cle):
            suffixe = col[len(cle):].lstrip("_")
            return f"{val} = {suffixe}" if suffixe else val
    return col


@st.cache_data
def importance_globale_shap():
    """Importance moyenne (|SHAP|) de chaque variable, sur un échantillon."""
    explainer = charger_explainer()
    _, scaler, colonnes = charger_modele()
    echantillon = df.sample(min(500, len(df)), random_state=42)
    X = preparer_features(echantillon, colonnes)
    sv = explainer.shap_values(scaler.transform(X))
    return pd.DataFrame({
        "variable": [joli_nom(c) for c in colonnes],
        "importance": np.abs(sv).mean(axis=0),
    })


def graphe_taux(data, colonne, titre, label_x, tri=False, ordre=None):
    """Barres du taux de souscription (%) par catégorie, avec ligne de moyenne.

    Permet de lire d'un coup d'œil ce qui FAVORISE le dépôt : toute barre
    au-dessus de la ligne moyenne sur-performe.
    """
    t = (
        data.groupby(colonne, observed=True)["deposit"]
        .apply(lambda s: (s == "yes").mean() * 100)
        .reset_index(name="taux")
    )
    t[colonne] = t[colonne].astype(str)
    if tri:
        t = t.sort_values("taux", ascending=False)

    fig = px.bar(
        t, x=colonne, y="taux",
        labels={colonne: label_x, "taux": "Taux de souscription (%)"},
        title=titre,
        color="taux",
        color_continuous_scale="Teal",
        category_orders={colonne: ordre} if ordre else None,
    )
    moyenne = (data["deposit"] == "yes").mean() * 100
    fig.add_hline(
        y=moyenne, line_dash="dash", line_color="grey",
        annotation_text=f"Moyenne {moyenne:.1f} %", annotation_position="top left",
    )
    return fig


def extremes_taux(data, colonne, min_effectif=30):
    """Renvoie (meilleure_cat, taux_max, pire_cat, taux_min) selon le taux de
    souscription. Ignore les catégories trop peu représentées (< min_effectif)
    pour éviter qu'un faible échantillon ne fausse la synthèse."""
    taux = (
        data.groupby(colonne, observed=True)["deposit"]
        .apply(lambda s: (s == "yes").mean() * 100)
    )
    eff = data.groupby(colonne, observed=True).size()
    valides = taux[eff >= min_effectif].dropna()
    if valides.empty:
        valides = taux.dropna()
    if valides.empty:
        return None
    return valides.idxmax(), valides.max(), valides.idxmin(), valides.min()


def afficher_synthese(data, dimensions):
    """Affiche une synthèse dynamique « ce qui favorise le dépôt ».

    dimensions = liste de (colonne, label_indicateur, label_texte).
    Recalculée selon les filtres ; ignore les catégories peu représentées.
    """
    moyenne_g = (data["deposit"] == "yes").mean() * 100
    resultats = [
        (lbl_m, lbl_t, extremes_taux(data, col)) for col, lbl_m, lbl_t in dimensions
    ]

    st.markdown("#### 🧭 Synthèse — ce qui favorise le dépôt")
    cols = st.columns(1 + len(resultats))
    cols[0].metric("Taux moyen (sélection)", f"{moyenne_g:.1f} %")
    for i, (lbl_m, _, ex) in enumerate(resultats, start=1):
        if ex:
            cols[i].metric(lbl_m, str(ex[0]), f"{ex[1]:.0f} %")

    points = []
    for _, lbl_t, ex in resultats:
        if ex:
            points.append(
                f"**{lbl_t}** : `{ex[0]}` performe le mieux ({ex[1]:.0f} %), "
                f"`{ex[2]}` le moins ({ex[3]:.0f} %)."
            )
    if points:
        st.info("\n\n".join("• " + p for p in points))
    st.divider()


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

# 4. Rangée d'indicateurs clés (KPI)
taux = (df_filtre["deposit"] == "yes").mean() * 100
nb_clients = len(df_filtre)
nb_souscripteurs = int((df_filtre["deposit"] == "yes").sum())
solde_median = int(df_filtre["balance"].median())

k1, k2, k3, k4 = st.columns(4)
k1.metric("Clients (sélection)", f"{nb_clients:,}".replace(",", " "))
k2.metric("Taux de souscription", f"{taux:.1f} %")
k3.metric("Souscripteurs", f"{nb_souscripteurs:,}".replace(",", " "))
k4.metric("Solde médian", f"{solde_median:,} €".replace(",", " "))

st.divider()

# On crée les onglets — narration : on explique, on score, on cible
tab_decryptage, tab_scoring, tab_cible = st.tabs(
    ["Décryptage campagne", "Scoring", "Campagne ciblée"]
)


# ===================== ONGLET : SCORING =====================
with tab_scoring:
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
        model, scaler, colonnes = charger_modele()
    except FileNotFoundError:
        st.error("Modèle introuvable. Lance d'abord : python train_model.py")
        st.stop()

    # ---------- IMPORTANCE GLOBALE (SHAP) ----------
    with st.expander("📊 Importance globale des variables (SHAP)"):
        st.caption(
            "Quelles variables pèsent le plus dans les décisions du modèle "
            "(impact moyen sur le score, calculé sur un échantillon de clients)."
        )
        try:
            imp = importance_globale_shap().sort_values("importance").tail(12)
            fig_imp = px.bar(
                imp, x="importance", y="variable", orientation="h",
                labels={"importance": "Impact moyen sur le score", "variable": ""},
                title="Variables les plus influentes",
                color="importance", color_continuous_scale="Teal",
            )
            st.plotly_chart(fig_imp, use_container_width=True)
        except Exception as e:
            st.info(f"Importance SHAP indisponible : {e}")

    st.divider()

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
        # Construire une ligne au même format que les données d'origine
        ligne = pd.DataFrame([{
            "age": in_age, "job": in_job, "marital": in_marital,
            "education": in_education, "default": in_default, "balance": in_balance,
            "housing": in_housing, "loan": in_loan, "contact": in_contact,
            "day": in_day, "month": in_month, "campaign": in_campaign,
            "pdays": in_pdays, "previous": in_previous, "poutcome": in_poutcome
        }])

        # Scoring via la fonction commune (même feature engineering qu'à l'entraînement)
        proba = scorer(ligne, model, scaler, colonnes)[0]

        st.metric("Probabilité de souscription", f"{proba*100:.1f} %")
        if proba >= 0.5:
            st.success("✅ Client prioritaire à contacter")
        else:
            st.warning("⚠️ Client peu susceptible de souscrire")

        # --- Explication SHAP de cette prédiction ---
        st.markdown("#### 🔬 Pourquoi ce score ?")
        try:
            explainer = charger_explainer()
            X_ligne = preparer_features(ligne, colonnes)
            shap_vals = explainer.shap_values(scaler.transform(X_ligne))[0]

            contrib = pd.DataFrame({
                "variable": [joli_nom(c) for c in colonnes],
                "impact": shap_vals,
            })
            contrib = contrib[contrib["impact"].abs() > 1e-6]
            top = contrib.reindex(
                contrib["impact"].abs().sort_values(ascending=False).index
            ).head(8).sort_values("impact")
            top["sens"] = np.where(top["impact"] >= 0, "Favorise", "Défavorise")

            fig_shap = px.bar(
                top, x="impact", y="variable", orientation="h", color="sens",
                color_discrete_map={"Favorise": "#4CA777", "Défavorise": "#3B6B8F"},
                labels={"impact": "Impact sur le score", "variable": "", "sens": ""},
                title="Facteurs qui ont le plus influencé ce client",
            )
            st.plotly_chart(fig_shap, use_container_width=True)
            st.caption(
                "🟢 Vert = pousse vers la souscription · 🔵 Bleu = pousse contre. "
                "Plus la barre est longue, plus l'effet est fort."
            )
        except Exception as e:
            st.info(f"Explication SHAP indisponible : {e}")


# ===================== ONGLET : CAMPAGNE CIBLÉE =====================
with tab_cible:
    st.subheader("📋 Sélectionner les clients à contacter")
    st.caption(
        "Importe un fichier CSV au format de bank.csv. Le modèle score chaque client, "
        "puis sélectionne ceux à contacter selon le critère que tu choisis."
    )

    fichier = st.file_uploader("Fichier CSV de clients", type=["csv"])

    if fichier is None:
        st.info("💡 Dépose un CSV pour lancer le scoring. Tu peux réutiliser bank.csv comme exemple.")
    else:
        try:
            clients = pd.read_csv(fichier)
        except Exception as e:
            st.error(f"Impossible de lire le fichier : {e}")
            clients = None

        if clients is not None:
            manquantes = [c for c in COLONNES_REQUISES if c not in clients.columns]
            if manquantes:
                st.error(
                    "Le fichier ne contient pas toutes les colonnes nécessaires. "
                    f"Colonnes manquantes : {', '.join(manquantes)}"
                )
            else:
                try:
                    model, scaler, colonnes = charger_modele()
                except FileNotFoundError:
                    st.error("Modèle introuvable. Lance d'abord : python train_model.py")
                    st.stop()

                # Scoring de tous les clients du fichier
                resultat = clients.copy()
                resultat["proba_souscription"] = (
                    scorer(clients, model, scaler, colonnes) * 100
                ).round(1)
                resultat = resultat.sort_values(
                    "proba_souscription", ascending=False
                ).reset_index(drop=True)

                st.success(f"✅ {len(resultat)} clients scorés.")

                # --- Choix du critère de sélection ---
                mode = st.radio(
                    "Critère de sélection",
                    ["Par seuil de probabilité", "Par taux de réussite moyen visé"],
                    horizontal=True,
                )

                if mode == "Par seuil de probabilité":
                    seuil = st.slider(
                        "Probabilité minimale pour contacter un client (%)",
                        0, 100, 50, 5,
                    )
                    selection = resultat[resultat["proba_souscription"] >= seuil]
                else:
                    cible = st.slider(
                        "Taux de réussite moyen visé pour la liste (%)",
                        0, 100, 60, 5,
                    )
                    # Les clients sont triés par proba décroissante : la moyenne
                    # cumulée est donc décroissante. On garde le plus grand groupe
                    # (le mieux scoré) dont la moyenne reste ≥ à la cible.
                    moyenne_cumulee = resultat["proba_souscription"].expanding().mean()
                    k = int((moyenne_cumulee >= cible).sum())
                    selection = resultat.head(k)

                # --- Indicateurs ---
                col1, col2, col3 = st.columns(3)
                col1.metric("Clients dans le fichier", len(resultat))
                col2.metric("Clients à contacter", len(selection))
                taux_attendu = (
                    selection["proba_souscription"].mean() if len(selection) else 0
                )
                col3.metric("Taux de réussite attendu", f"{taux_attendu:.1f} %")

                if len(selection) == 0:
                    st.warning("⚠️ Aucun client ne respecte ce critère. Abaisse le seuil / la cible.")
                else:
                    # --- Si le vrai résultat est connu : performance réelle ---
                    if "deposit" in resultat.columns:
                        perf_selection = (selection["deposit"] == "yes").mean() * 100
                        perf_globale = (resultat["deposit"] == "yes").mean() * 100
                        gain = perf_selection - perf_globale
                        st.success(
                            f"🎯 Performance réelle : **{perf_selection:.1f} %** des clients ciblés "
                            f"ont réellement souscrit, contre **{perf_globale:.1f} %** sur l'ensemble "
                            f"du fichier (gain de **{gain:+.1f} points**)."
                        )

                    # --- Tableau de la sélection ---
                    colonnes_affichees = ["proba_souscription"] + [
                        c for c in ["age", "job", "marital", "education", "balance",
                                    "housing", "loan", "contact"]
                        if c in selection.columns
                    ]
                    if "deposit" in selection.columns:
                        colonnes_affichees.append("deposit")

                    st.dataframe(
                        selection[colonnes_affichees],
                        use_container_width=True,
                        hide_index=True,
                    )

                    # --- Téléchargement de la liste d'appels ---
                    csv_export = selection.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        "⬇️ Télécharger la liste d'appels (CSV)",
                        data=csv_export,
                        file_name="clients_a_contacter.csv",
                        mime="text/csv",
                    )

                    # --- Courbe de gain / lift (si le vrai résultat est connu) ---
                    if "deposit" in resultat.columns:
                        y_true = (resultat["deposit"] == "yes").astype(int).values
                        total_sousc = int(y_true.sum())
                        if total_sousc > 0:
                            st.divider()
                            st.markdown("#### 📈 Courbe de gain — efficacité du ciblage")

                            n = len(resultat)
                            courbe = pd.DataFrame({
                                "pct_clients": np.arange(1, n + 1) / n * 100,
                                "pct_captes": np.cumsum(y_true) / total_sousc * 100,
                            })
                            fig_gain = px.line(
                                courbe, x="pct_clients", y="pct_captes",
                                labels={"pct_clients": "% de clients contactés (mieux notés d'abord)",
                                        "pct_captes": "% de souscripteurs captés"},
                                title="Souscripteurs captés selon l'effort d'appel",
                            )
                            fig_gain.update_traces(line_color="#4CA777")
                            fig_gain.add_shape(
                                type="line", x0=0, y0=0, x1=100, y1=100,
                                line=dict(dash="dash", color="grey"),
                            )
                            pct_sel = len(selection) / n * 100
                            captes_sel = (selection["deposit"] == "yes").sum() / total_sousc * 100
                            fig_gain.add_scatter(
                                x=[pct_sel], y=[captes_sel], mode="markers",
                                marker=dict(size=12, color="#1A2733"),
                                name="Ta sélection",
                            )
                            st.plotly_chart(fig_gain, use_container_width=True)

                            top10 = resultat.head(max(1, n // 10))
                            lift10 = ((top10["deposit"] == "yes").mean() * 100) / (
                                (resultat["deposit"] == "yes").mean() * 100
                            )
                            st.caption(
                                f"💡 Lecture : en appelant **{pct_sel:.0f} %** des clients (ta sélection), "
                                f"tu captes **{captes_sel:.0f} %** des souscripteurs. "
                                f"La ligne grise = ciblage au hasard. "
                                f"Lift du top 10 % : **×{lift10:.2f}**."
                            )


# ===================== ONGLET : DÉCRYPTAGE CAMPAGNE =====================
with tab_decryptage:
    st.subheader("🔍 Qu'est-ce qui favorise le dépôt ?")
    st.caption(
        "Pour chaque caractéristique, on lit le **taux de souscription** par catégorie. "
        "La ligne grise = moyenne globale : toute barre au-dessus sur-performe. "
        "Les filtres de la barre latérale s'appliquent ici aussi."
    )

    ax1, ax2, ax3 = st.tabs(
        ["💰 Profil bancaire", "📣 Profil de campagne", "👥 Profil socio-démographique"]
    )

    # -------- AXE 1 : PROFIL BANCAIRE --------
    with ax1:
        st.markdown("**Solde, prêts et défaut de paiement**")
        data_b = df_filtre.copy()

        # ----- Synthèse dynamique -----
        data_syn_b = df_filtre.copy()
        data_syn_b["nb_prets"] = (
            (data_syn_b["housing"] == "yes").astype(int)
            + (data_syn_b["loan"] == "yes").astype(int)
        )
        dims_b = [
            ("housing", "Prêt immobilier", "Prêt immobilier"),
            ("loan", "Prêt personnel", "Prêt personnel"),
            ("nb_prets", "Nb de prêts optimal", "Nombre de prêts"),
        ]
        try:
            data_syn_b["tranche_balance"] = pd.qcut(
                data_syn_b["balance"], 5, duplicates="drop"
            ).astype(str)
            dims_b = [("tranche_balance", "Meilleur solde", "Solde")] + dims_b
        except ValueError:
            pass
        afficher_synthese(data_syn_b, dims_b)

        # Taux par tranche de solde
        try:
            data_b["tranche_balance"] = pd.qcut(data_b["balance"], 5, duplicates="drop")
            st.plotly_chart(
                graphe_taux(data_b, "tranche_balance",
                            "Souscription selon le solde", "Tranche de solde (€)"),
                use_container_width=True,
            )
        except ValueError:
            st.info("Pas assez de diversité de soldes pour créer des tranches avec ces filtres.")

        col_a, col_b = st.columns(2)
        with col_a:
            st.plotly_chart(
                graphe_taux(df_filtre, "housing",
                            "Selon le prêt immobilier", "Prêt immobilier"),
                use_container_width=True,
            )
        with col_b:
            st.plotly_chart(
                graphe_taux(df_filtre, "loan",
                            "Selon le prêt personnel", "Prêt personnel"),
                use_container_width=True,
            )

        # Taux selon le nombre de prêts cumulés
        data_b["nb_prets"] = (
            (df_filtre["housing"] == "yes").astype(int)
            + (df_filtre["loan"] == "yes").astype(int)
        )
        st.plotly_chart(
            graphe_taux(data_b, "nb_prets",
                        "Selon le nombre de prêts (0, 1 ou 2)", "Nombre de prêts"),
            use_container_width=True,
        )

    # -------- AXE 2 : PROFIL DE CAMPAGNE --------
    with ax2:
        st.markdown("**Canal de contact, calendrier et pression commerciale**")
        data_c = df_filtre.copy()

        # ----- Synthèse automatique (recalculée selon les filtres) -----
        data_c["tranche_jour"] = pd.cut(
            data_c["day"], [0, 10, 20, 31], labels=["debut", "milieu", "fin"]
        )
        data_c["nb_contacts"] = pd.cut(
            data_c["campaign"], [0, 1, 2, 3, 5, 1000],
            labels=["1", "2", "3", "4-5", "6+"]
        )
        moyenne_g = (data_c["deposit"] == "yes").mean() * 100

        ex_contact = extremes_taux(data_c, "contact")
        ex_mois = extremes_taux(data_c, "month")
        ex_nb = extremes_taux(data_c, "nb_contacts")
        ex_pout = extremes_taux(data_c, "poutcome")

        st.markdown("#### 🧭 Synthèse — ce qui favorise le dépôt")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Taux moyen (sélection)", f"{moyenne_g:.1f} %")
        if ex_contact:
            m2.metric("Meilleur canal", str(ex_contact[0]), f"{ex_contact[1]:.0f} %")
        if ex_mois:
            m3.metric("Meilleur mois", str(ex_mois[0]), f"{ex_mois[1]:.0f} %")
        if ex_nb:
            m4.metric("Nb de contacts optimal", str(ex_nb[0]), f"{ex_nb[1]:.0f} %")

        points = []
        if ex_contact:
            points.append(
                f"**Canal** : `{ex_contact[0]}` est le plus efficace "
                f"({ex_contact[1]:.0f} %), `{ex_contact[2]}` le moins ({ex_contact[3]:.0f} %)."
            )
        if ex_mois:
            points.append(
                f"**Calendrier** : le mois `{ex_mois[0]}` performe le mieux "
                f"({ex_mois[1]:.0f} %), `{ex_mois[2]}` le moins ({ex_mois[3]:.0f} %)."
            )
        if ex_nb:
            points.append(
                f"**Pression commerciale** : `{ex_nb[0]}` contact(s) donne le meilleur "
                f"résultat ({ex_nb[1]:.0f} %) — multiplier les appels fait souvent chuter le taux."
            )
        if ex_pout:
            points.append(
                f"**Historique** : un résultat passé `{ex_pout[0]}` mène à {ex_pout[1]:.0f} % "
                "de souscription — réactiver d'anciens clients convertis est très rentable."
            )
        if points:
            st.info("\n\n".join("• " + p for p in points))

        st.divider()

        col_c, col_d = st.columns(2)
        with col_c:
            st.plotly_chart(
                graphe_taux(df_filtre, "contact",
                            "Selon le type de contact", "Type de contact", tri=True),
                use_container_width=True,
            )
        with col_d:
            ordre_jour = ["debut", "milieu", "fin"]
            data_c["tranche_jour"] = pd.cut(
                data_c["day"], [0, 10, 20, 31], labels=ordre_jour
            )
            st.plotly_chart(
                graphe_taux(data_c, "tranche_jour",
                            "Selon la période du mois", "Période du mois",
                            ordre=ordre_jour),
                use_container_width=True,
            )

        # Taux par mois (ordonné chronologiquement)
        ordre_mois = ["jan", "feb", "mar", "apr", "may", "jun",
                      "jul", "aug", "sep", "oct", "nov", "dec"]
        st.plotly_chart(
            graphe_taux(df_filtre, "month",
                        "Selon le mois de contact", "Mois", ordre=ordre_mois),
            use_container_width=True,
        )

        # Taux selon le nombre de contacts durant la campagne
        data_c["nb_contacts"] = pd.cut(
            data_c["campaign"], [0, 1, 2, 3, 5, 1000],
            labels=["1", "2", "3", "4-5", "6+"]
        )
        st.plotly_chart(
            graphe_taux(data_c, "nb_contacts",
                        "Selon le nombre de contacts durant la campagne",
                        "Nombre de contacts", ordre=["1", "2", "3", "4-5", "6+"]),
            use_container_width=True,
        )
        st.caption("➡️ Au-delà de quelques contacts, l'acharnement commercial devient souvent contre-productif.")

    # -------- AXE 3 : PROFIL SOCIO-DÉMOGRAPHIQUE --------
    with ax3:
        st.markdown("**Âge, métier, éducation et situation maritale**")
        data_s = df_filtre.copy()

        ordre_age = ["<25", "25-35", "35-50", "50-60", "60+"]
        data_s["tranche_age"] = pd.cut(
            data_s["age"], [0, 25, 35, 50, 60, 100], labels=ordre_age
        )

        # ----- Synthèse dynamique -----
        afficher_synthese(data_s, [
            ("tranche_age", "Meilleure tranche d'âge", "Tranche d'âge"),
            ("job", "Meilleur métier", "Métier"),
            ("education", "Éducation", "Niveau d'éducation"),
            ("marital", "Situation maritale", "Situation maritale"),
        ])
        st.plotly_chart(
            graphe_taux(data_s, "tranche_age",
                        "Selon la tranche d'âge", "Tranche d'âge", ordre=ordre_age),
            use_container_width=True,
        )

        st.plotly_chart(
            graphe_taux(df_filtre, "job",
                        "Selon le métier", "Métier", tri=True),
            use_container_width=True,
        )

        col_e, col_f = st.columns(2)
        with col_e:
            st.plotly_chart(
                graphe_taux(df_filtre, "education",
                            "Selon le niveau d'éducation", "Éducation", tri=True),
                use_container_width=True,
            )
        with col_f:
            st.plotly_chart(
                graphe_taux(df_filtre, "marital",
                            "Selon la situation maritale", "Situation maritale", tri=True),
                use_container_width=True,
            )