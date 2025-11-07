
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, auc
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

st.set_page_config(page_title="Universal Bank: Personal Loan Propensity", layout="wide")

# -----------------------------
# Helpers
# -----------------------------
TARGET_CANON = "Personal_Loan"

def canonize_columns(df):
    df = df.copy()
    df.columns = [c.strip().replace(" ", "_").replace(".", "_") for c in df.columns]
    # Normalize a few known columns
    ren = {"ZIP_Code":"Zip_code", "ZIP":"Zip_code", "ZIPCODE":"Zip_code"}
    df = df.rename(columns=ren)
    # Fix Experience negatives if any
    if "Experience" in df.columns:
        nonneg = df.loc[df["Experience"] >= 0, "Experience"]
        if not nonneg.empty:
            med = nonneg.median()
            df.loc[df["Experience"] < 0, "Experience"] = med
    return df

def find_target(df):
    candidates = ["Personal_Loan", "PersonalLoan", "Personal_Loan_", "PersonalLoan_", "Personal", "Loan", "personal_loan"]
    for c in candidates:
        if c in df.columns:
            return c
    return None

def ensure_binary(series):
    try:
        s = pd.to_numeric(series, errors="coerce").fillna(0).astype(int)
        s = s.clip(0,1)
        return s
    except Exception:
        return series

def split_features_target(df, target_col):
    X = df.drop(columns=[target_col]).copy()
    y = ensure_binary(df[target_col])
    return X, y

def compute_metrics(y_true, y_pred, y_proba):
    return dict(
        Accuracy = accuracy_score(y_true, y_pred),
        Precision = precision_score(y_true, y_pred, zero_division=0),
        Recall = recall_score(y_true, y_pred, zero_division=0),
        F1_Score = f1_score(y_true, y_pred, zero_division=0),
        AUC = roc_auc_score(y_true, y_proba)
    )

def model_dict(random_state=42):
    return {
        "Decision Tree": DecisionTreeClassifier(random_state=random_state),
        "Random Forest": RandomForestClassifier(n_estimators=300, random_state=random_state, n_jobs=-1),
        "Gradient Boosting": GradientBoostingClassifier(random_state=random_state)
    }

def get_proba(clf, X):
    if hasattr(clf, "predict_proba"):
        return clf.predict_proba(X)[:,1]
    else:
        # should not happen for chosen models
        return clf.decision_function(X)

def confusion_df(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    return pd.DataFrame(cm, index=["Actual 0","Actual 1"], columns=["Pred 0","Pred 1"])

@st.cache_data(show_spinner=False)
def deciles(s, n=10):
    return pd.qcut(s.rank(method="first"), q=n, labels=[f"D{d}" for d in range(1, n+1)])

# -----------------------------
# Data Section
# -----------------------------
st.title("📈 Universal Bank — Personal Loan Propensity & Insights")

with st.expander("Upload data (CSV) and basic options", expanded=True):
    uploaded = st.file_uploader("Upload UniversalBank-like CSV", type=["csv"])
    if uploaded is not None:
        df = pd.read_csv(uploaded)
    else:
        st.info("Please upload your dataset CSV (the classic UniversalBank.csv works).")
        st.stop()
    df = canonize_columns(df)
    tgt = find_target(df)
    if tgt is None:
        st.warning("No target column found. Expected something like 'Personal_Loan'. "
                   "You can still explore Insights, but Modeling requires the target.")
    else:
        st.success(f"Detected target column: **{tgt}**")
    st.write("Preview:", df.head())

# -----------------------------
# Tabs
# -----------------------------
tab1, tab2, tab3 = st.tabs(["🔎 Customer Insights (5 charts)", "🧠 Train & Evaluate Models", "🚀 Predict & Download"])

# -----------------------------
# Tab 1: Customer Insights
# -----------------------------
with tab1:
    st.subheader("Marketing Insights")
    if find_target(df) is None:
        st.warning("For acceptance-rate charts, a target column is needed.")
    else:
        target = find_target(df)
        # 1) Heatmap: Income decile vs CCAvg decile -> acceptance rate
        cols_needed = ["Income","CCAvg",target]
        can1 = all(c in df.columns for c in cols_needed)
        if can1:
            dtmp = df[[ "Income","CCAvg",target ]].copy()
            dtmp["Income_decile"] = deciles(dtmp["Income"])
            dtmp["CCAvg_decile"] = deciles(dtmp["CCAvg"])
            heat = (dtmp
                    .groupby(["Income_decile","CCAvg_decile"], observed=True)[target]
                    .mean()
                    .reset_index(name="AcceptanceRate"))
            fig1 = px.density_heatmap(
                heat, x="Income_decile", y="CCAvg_decile", z="AcceptanceRate",
                color_continuous_scale="Blues", title="Acceptance Rate by Income & CCAvg Deciles"
            )
            st.plotly_chart(fig1, use_container_width=True)
            st.caption("Insight: Prioritize segments in the upper-right deciles where both income and card spend are high.")

        # 2) Treemap: Education -> Family sized by count, colored by acceptance rate
        if all(c in df.columns for c in [target,"Education","Family"]):
            tmp = df.groupby(["Education","Family"], observed=True).agg(
                cnt=(target,"size"),
                rate=(target,"mean")
            ).reset_index()
            fig2 = px.treemap(tmp, path=["Education","Family"], values="cnt",
                              color="rate", color_continuous_scale="Teal",
                              title="Segment Size & Acceptance by Education → Family")
            st.plotly_chart(fig2, use_container_width=True)
            st.caption("Insight: Large green tiles ≈ big segments with high conversion potential — ideal for tailored offers.")

        # 3) Stacked bar: Product holdings vs acceptance rate (CDAccount, CreditCard, Online, Securities)
        prods = [c for c in ["CD_Account","CreditCard","Online","Securities_Account"] if c in df.columns]
        if len(prods) >= 2:
            long = []
            for p in prods:
                t = df.groupby(p, observed=True)[target].mean().reset_index()
                t["Product"] = p
                t[p] = t[p].map({0:"No",1:"Yes"})
                long.append(t.rename(columns={p:"Has"}))
            long = pd.concat(long, ignore_index=True)
            fig3 = px.bar(long, x="Product", y=target, color="Has", barmode="group",
                          title="Acceptance Rate by Product Ownership")
            st.plotly_chart(fig3, use_container_width=True)
            st.caption("Insight: Products with strong uplift when 'Yes' can be used for cross-sell targeting.")

        # 4) Boxplot: Income by label
        if all(c in df.columns for c in [target,"Income"]):
            fig4 = px.box(df, x=target, y="Income", points=False,
                          title="Income Distribution by Personal Loan Label")
            st.plotly_chart(fig4, use_container_width=True)
            st.caption("Insight: Higher-income customers skew toward acceptance; consider premium loan variants and tailored messaging.")

        # 5) Grouped bars: Acceptance by Education & Online usage
        if all(c in df.columns for c in [target,"Education","Online"]):
            grp = df.groupby(["Education","Online"], observed=True)[target].mean().reset_index()
            grp["Online"] = grp["Online"].map({0:"No",1:"Yes"})
            fig5 = px.bar(grp, x="Education", y=target, color="Online", barmode="group",
                          title="Acceptance Rate by Education & Online Banking")
            st.plotly_chart(fig5, use_container_width=True)
            st.caption("Insight: Digital-first education segments respond better — push in-app journeys and personalized nudges.")

# -----------------------------
# Tab 2: Modeling
# -----------------------------
with tab2:
    st.subheader("Apply Decision Tree, Random Forest, Gradient Boosting")
    if find_target(df) is None:
        st.warning("Target column required for training (e.g., 'Personal_Loan').")
    else:
        target = find_target(df)
        X, y = split_features_target(df, target)
        test_size = st.slider("Test size (fraction)", 0.1, 0.4, 0.3, 0.05)
        seed = st.number_input("Random seed", 0, 9999, 42)

        if st.button("Train All Models"):
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, stratify=y, random_state=seed)

            models = model_dict(random_state=seed)
            metrics_rows = []
            roc_traces = []

            for name, clf in models.items():
                clf.fit(X_train, y_train)
                ytr_pred = clf.predict(X_train)
                yte_pred = clf.predict(X_test)
                ytr_p = get_proba(clf, X_train)
                yte_p = get_proba(clf, X_test)

                train_m = compute_metrics(y_train, ytr_pred, ytr_p)
                test_m  = compute_metrics(y_test,  yte_pred, yte_p)

                metrics_rows.append({
                    "Algorithm": name,
                    "Training Accuracy": train_m["Accuracy"],
                    "Testing Accuracy":  test_m["Accuracy"],
                    "Precision":          test_m["Precision"],
                    "Recall":             test_m["Recall"],
                    "F1-Score":           test_m["F1_Score"],
                    "AUC":                test_m["AUC"]
                })

                fpr, tpr, _ = roc_curve(y_test, yte_p)
                roc_traces.append(go.Scatter(x=fpr, y=tpr, name=f"{name} (AUC={test_m['AUC']:.3f})", mode="lines"))

                # Confusion matrices
                cm_tr = confusion_df(y_train, ytr_pred)
                cm_te = confusion_df(y_test,  yte_pred)

                c1, c2 = st.columns(2)
                with c1:
                    st.markdown(f"**{name} — Train Confusion Matrix**")
                    st.dataframe(cm_tr)
                with c2:
                    st.markdown(f"**{name} — Test Confusion Matrix**")
                    st.dataframe(cm_te)

                # Feature importance (if available)
                if hasattr(clf, "feature_importances_"):
                    imp = pd.Series(clf.feature_importances_, index=X.columns).sort_values(ascending=False)
                    fig_imp = px.bar(imp.head(20), title=f"Top Features — {name}")
                    st.plotly_chart(fig_imp, use_container_width=True)

            # Metrics table
            met_df = pd.DataFrame(metrics_rows).set_index("Algorithm").sort_values("AUC", ascending=False)
            st.markdown("### 📊 Performance Summary")
            st.dataframe(met_df.style.format("{:.4f}"))

            # ROC overlay
            fig_roc = go.Figure(roc_traces)
            fig_roc.add_trace(go.Scatter(x=[0,1], y=[0,1], mode="lines", name="baseline", line=dict(dash="dash")))
            fig_roc.update_layout(title="ROC Curves (Test Set)", xaxis_title="FPR", yaxis_title="TPR")
            st.plotly_chart(fig_roc, use_container_width=True)

            st.session_state["best_model_name"] = met_df.index[0]
            st.session_state["best_model_metrics"] = met_df.loc[met_df.index[0]].to_dict()

            # Persist the best model for prediction tab
            best_name = met_df.index[0]
            best_model = model_dict(random_state=seed)[best_name].fit(X_train, y_train)
            st.session_state["fitted_model"] = best_model
            st.session_state["feature_columns"] = list(X.columns)

# -----------------------------
# Tab 3: Predict & Download
# -----------------------------
with tab3:
    st.subheader("Upload new customer file and score propensity")
    st.write("Upload a CSV with the same feature columns used for training (exclude the target).")

    new_file = st.file_uploader("Upload new dataset (for scoring)", type=["csv"], key="score_uploader")
    thresh = st.slider("Decision threshold (for label 1)", 0.05, 0.95, 0.5, 0.05)

    if new_file is not None:
        new_df = pd.read_csv(new_file)
        new_df = canonize_columns(new_df)

        # If a trained model exists in session, use it. Otherwise, attempt quick fit using available data with target.
        model = st.session_state.get("fitted_model", None)
        feat_cols = st.session_state.get("feature_columns", None)

        if model is None:
            st.info("No trained model in session. Attempting to train a Gradient Boosting model quickly from the uploaded dataset (requires target).")
            tgt = find_target(df)
            if tgt is None:
                st.error("Cannot auto-train because the main dataset has no target. Please train models in the previous tab first.")
                st.stop()
            X, y = split_features_target(df, tgt)
            model = GradientBoostingClassifier(random_state=42).fit(X, y)
            feat_cols = list(X.columns)

        # Align columns
        missing = [c for c in feat_cols if c not in new_df.columns]
        extra   = [c for c in new_df.columns if c not in feat_cols]
        if missing:
            st.error(f"Missing required columns: {missing}")
            st.stop()
        Xnew = new_df[feat_cols].copy()

        proba = get_proba(model, Xnew)
        label = (proba >= thresh).astype(int)

        out = new_df.copy()
        out["Loan_Probability"] = proba
        out["Predicted_Label"] = label

        st.markdown("### Preview of Scored Output")
        st.dataframe(out.head())

        # Download
        csv = out.to_csv(index=False).encode("utf-8")
        st.download_button("📥 Download predictions CSV", data=csv, file_name="predictions.csv", mime="text/csv")

    else:
        st.info("Upload a file above to generate predictions.")
