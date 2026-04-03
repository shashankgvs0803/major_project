import os
import pickle

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix,
)

st.set_page_config(
    page_title="IIoT IDS — Multi-Class",
    layout="wide",
)

ARTIFACT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results_multi_class")


# -- Model loading --------------------------------------------
@st.cache_resource(show_spinner="Loading models …")
def load_artifacts():
    from tensorflow.keras.models import load_model
    paths = {
        "centralized": os.path.join(ARTIFACT_DIR, "model_mc.keras"),
        "federated":   os.path.join(ARTIFACT_DIR, "model_mc_fl.keras"),
        "autoencoder": os.path.join(ARTIFACT_DIR, "autoencoder_ids_mc.keras"),
        "scaler":      os.path.join(ARTIFACT_DIR, "scaler_mc.pkl"),
        "threshold":   os.path.join(ARTIFACT_DIR, "ae_threshold_mc.pkl"),
        "label_map":   os.path.join(ARTIFACT_DIR, "label_map_mc.pkl"),
    }
    missing = [k for k, p in paths.items() if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError(f"Missing in {ARTIFACT_DIR}: {missing}")
    models = {
        "centralized": load_model(paths["centralized"]),
        "federated":   load_model(paths["federated"]),
        "autoencoder": load_model(paths["autoencoder"]),
    }
    with open(paths["scaler"],    "rb") as f: scaler    = pickle.load(f)
    with open(paths["threshold"], "rb") as f: threshold = pickle.load(f)
    with open(paths["label_map"], "rb") as f: label_map = pickle.load(f)
    return models, scaler, float(threshold), label_map


@st.cache_data(show_spinner="Reading CSV…")
def load_csv(uploaded_file):
    df     = pd.read_csv(uploaded_file)
    y_true = df["label"].values if "label" in df.columns else None
    X_raw  = (df.drop(columns=["label"], errors="ignore")
                .select_dtypes(include=[np.number])
                .fillna(0).values.astype(np.float32))
    return X_raw, y_true


# -- Inference functions ---------------------------------------
def run_autoencoder(X_scaled, models, threshold):
    recon    = models["autoencoder"].predict(X_scaled, verbose=0)
    errors   = np.mean(np.square(X_scaled - recon), axis=1)
    ae_flags = errors > threshold
    n_total  = len(X_scaled)
    return {
        "errors":    errors,
        "ae_flags":  ae_flags,
        "threshold": threshold,
        "n_total":   n_total,
        "n_flagged": int(ae_flags.sum()),
        "n_normal":  n_total - int(ae_flags.sum()),
    }


def run_classifier(X_scaled, ae_result, models, label_map, model_key):
    X_3d         = X_scaled.reshape(X_scaled.shape[0], X_scaled.shape[1], 1)
    probs        = models[model_key].predict(X_3d, verbose=0)
    preds        = np.argmax(probs, axis=1)
    final_labels = np.array([label_map[p] for p in preds])
    is_attack    = final_labels != "benign"
    if ae_result is not None:
        ae_flags   = ae_result["ae_flags"]
        confidence = np.where(
            is_attack  & ae_flags,  "High - AE confirmed",
            np.where(
            is_attack  & ~ae_flags, "Medium - AE not flagged",
            np.where(
            ~is_attack & ae_flags,  "Review - AE flagged benign",
                                    "Normal")))
    else:
        confidence = np.where(is_attack, "Attack (AE not run)", "Benign (AE not run)")
    n_total   = len(X_scaled)
    n_anomaly = int(is_attack.sum())
    return {
        "final_labels": final_labels,
        "confidence":   confidence,
        "n_total":      n_total,
        "n_benign":     n_total - n_anomaly,
        "n_anomaly":    n_anomaly,
    }


# -- Render functions ------------------------------------------
def _build_table(result, ae_result, y_true):
    r     = result
    table = pd.DataFrame({
        "Row":        np.arange(1, r["n_total"] + 1),
        "Prediction": r["final_labels"],
        "Confidence": r["confidence"],
    })
    if ae_result is not None:
        table["AE Flag"]     = np.where(ae_result["ae_flags"], "Anomaly", "Normal")
        table["Recon Error"] = ae_result["errors"].round(6)
    if y_true is not None:
        table["Ground Truth"] = y_true
        table["Correct"]      = (result["final_labels"] == y_true)
    return table


def render_ae_panel(ae_result, y_true):
    r      = ae_result
    has_gt = y_true is not None
    c1, c2 = st.columns(2)
    c1.metric("Flagged Normal",  f"{r['n_normal']:,}",  f"{r['n_normal']/r['n_total']:.1%}")
    c2.metric("Flagged Anomaly", f"{r['n_flagged']:,}", f"{r['n_flagged']/r['n_total']:.1%}")
    if has_gt:
        y_ae_true = (y_true != "benign").astype(int)
        ae_acc  = accuracy_score( y_ae_true, r["ae_flags"].astype(int))
        ae_prec = precision_score(y_ae_true, r["ae_flags"].astype(int), zero_division=0)
        ae_rec  = recall_score(   y_ae_true, r["ae_flags"].astype(int), zero_division=0)
        ae_f1   = f1_score(       y_ae_true, r["ae_flags"].astype(int), zero_division=0)
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("AE Accuracy",  f"{ae_acc:.3f}")
        m2.metric("AE Precision", f"{ae_prec:.3f}")
        m3.metric("AE Recall",    f"{ae_rec:.3f}")
        m4.metric("AE F1",        f"{ae_f1:.3f}")
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=r["errors"][~r["ae_flags"]], name="Normal",
                               marker_color="#1a9e60", opacity=0.7, nbinsx=40))
    fig.add_trace(go.Histogram(x=r["errors"][r["ae_flags"]],  name="Anomaly",
                               marker_color="#d62828", opacity=0.7, nbinsx=40))
    fig.add_vline(x=r["threshold"], line_dash="dash", line_color="white",
                  annotation_text=f"Threshold {r['threshold']:.4f}")
    fig.update_layout(title=dict(text="Reconstruction Error Distribution", x=0.5, xanchor="center"),
                      height=260, barmode="overlay",
                      margin=dict(t=70, b=10, l=5, r=5),
                      legend=dict(orientation="h", yanchor="bottom", y=1.08, x=0),
                      xaxis_title="Reconstruction Error (MSE)")
    st.plotly_chart(fig, use_container_width=True, key="ae_hist")


def render_panel(result, ae_result, y_true, title, table):
    r      = result
    has_gt = y_true is not None
    c1, c2 = st.columns(2)
    c1.metric("Benign", f"{r['n_benign']:,}",  f"{r['n_benign']/r['n_total']:.1%}")
    c2.metric("Attack", f"{r['n_anomaly']:,}", f"{r['n_anomaly']/r['n_total']:.1%}")
    if has_gt:
        acc  = accuracy_score( y_true, r["final_labels"])
        prec = precision_score(y_true, r["final_labels"], average="macro", zero_division=0)
        rec  = recall_score(   y_true, r["final_labels"], average="macro", zero_division=0)
        f1   = f1_score(       y_true, r["final_labels"], average="macro", zero_division=0)
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Accuracy",  f"{acc:.3f}")
        m2.metric("Precision", f"{prec:.3f}")
        m3.metric("Recall",    f"{rec:.3f}")
        m4.metric("F1",        f"{f1:.3f}")
    unique, counts = np.unique(r["final_labels"], return_counts=True)
    attack_types = {k: v for k, v in zip(unique, counts) if k != "benign"}
    if attack_types:
        fig2 = px.pie(names=list(attack_types.keys()), values=list(attack_types.values()),
                      title="Attack Type Distribution", hole=0.4)
        fig2.update_layout(height=280, margin=dict(t=40, b=10, l=5, r=5))
        st.plotly_chart(fig2, use_container_width=True, key=f"pie_{title}")
        for label, count in sorted(attack_types.items(), key=lambda x: -x[1]):
            st.markdown(f"- **{label}**: {count:,} samples")
    else:
        st.success("No attacks detected — all traffic classified as benign.")
    if has_gt:
        all_classes = sorted(set(y_true) | set(r["final_labels"]))
        cm = confusion_matrix(y_true, r["final_labels"], labels=all_classes)
        fig_cm = px.imshow(cm, x=all_classes, y=all_classes,
                           color_continuous_scale="Blues",
                           labels=dict(x="Predicted", y="True", color="Count"),
                           title="Confusion Matrix", text_auto=True)
        fig_cm.update_layout(height=500, margin=dict(t=50, b=10, l=5, r=5),
                             xaxis=dict(tickangle=45))
        with st.expander("Confusion Matrix"):
            st.plotly_chart(fig_cm, use_container_width=True, key=f"cm_{title}")
    with st.expander(f"Full results ({r['n_total']:,} rows)"):
        st.dataframe(table, use_container_width=True, hide_index=True)
        st.download_button("Download CSV", table.to_csv(index=False).encode(),
                           f"{title.lower()}_results.csv", "text/csv",
                           key=f"dl_{title}")


def render_comparison(res_central, res_fed, y_true):
    if y_true is None:
        st.info("Upload a CSV with a label column to see metric comparison.")
        return
    rows = []
    for label, r in [("Centralized", res_central), ("Federated", res_fed)]:
        rows.append({
            "Model":     label,
            "Accuracy":  accuracy_score( y_true, r["final_labels"]),
            "Precision": precision_score(y_true, r["final_labels"], average="macro", zero_division=0),
            "Recall":    recall_score(   y_true, r["final_labels"], average="macro", zero_division=0),
            "F1":        f1_score(       y_true, r["final_labels"], average="macro", zero_division=0),
        })
    comp      = pd.DataFrame(rows)
    comp_melt = comp.melt(id_vars="Model", var_name="Metric", value_name="Score")
    fig = px.bar(comp_melt, x="Metric", y="Score", color="Model", barmode="group",
                 color_discrete_map={"Centralized": "#3b82f6", "Federated": "#8b5cf6"},
                 title="Centralized vs Federated - CNN-GRU Metrics", text_auto=".3f")
    fig.update_layout(height=400, yaxis_range=[0, 1.05],
                      margin=dict(t=80, b=10, l=5, r=5),
                      title=dict(text="Centralized vs Federated - CNN-GRU Metrics", y=0.97, x=0.5, xanchor="center"),
                      legend=dict(orientation="h", yanchor="bottom", y=1.08, x=0))
    st.plotly_chart(fig, use_container_width=True, key="comparison_bar")
    st.dataframe(comp.set_index("Model").style.format("{:.4f}"), use_container_width=True)


def render_dataset_summary(y_true):
    counts  = pd.Series(y_true).value_counts().sort_values(ascending=False)
    n_total = len(y_true)
    n_benign = int(counts.get("benign", 0))
    n_attack = n_total - n_benign

    c1, c2, c3 = st.columns(3)
    c1.metric("Total Samples", f"{n_total:,}")
    c2.metric("Benign",        f"{n_benign:,}  ({n_benign/n_total:.1%})")
    c3.metric("Attacks",       f"{n_attack:,}  ({n_attack/n_total:.1%})")

    attack_counts = {k: v for k, v in counts.items() if k != "benign"}
    if attack_counts:
        rows = [
            {
                "Attack Type":   k,
                "Count":         f"{v:,}",
                "% of Total":    f"{v/n_total:.2%}",
                "% of Attacks":  f"{v/n_attack:.2%}",
            }
            for k, v in sorted(attack_counts.items(), key=lambda x: -x[1])
        ]
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    else:
        st.success("Dataset contains benign samples only.")


# -- Sidebar --------------------------------------------------
with st.sidebar:
    st.title("IIoT IDS")
    st.caption("Multi-Class · Parallel Pipeline")
    st.divider()
    uploaded = st.file_uploader("Upload CSV", type="csv", key="uploader")

# -- Main -----------------------------------------------------
st.title("IIoT Intrusion Detection System")
st.caption("Parallel Pipeline: Autoencoder (anomaly detector) + CNN-GRU (classifier) — Multi-Class")

if uploaded is None:
    st.info("Upload a CSV file in the sidebar to get started.")
    st.stop()

X_raw, y_true = load_csv(uploaded)

file_id = f"{uploaded.name}_{uploaded.size}"
if st.session_state.get("_file_id") != file_id:
    st.session_state["_file_id"] = file_id
    for k in ("_res_ae", "_res_central", "_res_fed", "_X_scaled", "_table_central", "_table_fed"):
        st.session_state.pop(k, None)

with st.sidebar:
    st.success(f"**{len(X_raw):,}** samples · {X_raw.shape[1]} features"
               + (" · labels detected" if y_true is not None else ""))

# -- Stage 1: Dataset -----------------------------------------
st.markdown('<div style="display:flex;align-items:center;gap:12px;margin:8px 0 2px"><span style="background:#3b82f6;color:white;border-radius:50%;width:34px;height:34px;display:inline-flex;align-items:center;justify-content:center;font-weight:700;font-size:16px;flex-shrink:0">1</span><span style="font-size:1.4rem;font-weight:600">Dataset Overview</span></div>', unsafe_allow_html=True)
st.caption("Upload your CSV in the sidebar — ground truth class distribution is shown below")

if y_true is not None:
    with st.expander("Ground Truth Dataset Summary", expanded=True):
        render_dataset_summary(y_true)
else:
    st.info("No label column detected — ground truth summary unavailable.")

st.divider()

# -- Stage 2: Autoencoder -------------------------------------
st.markdown('<div style="display:flex;align-items:center;gap:12px;margin:8px 0 2px"><span style="background:#3b82f6;color:white;border-radius:50%;width:34px;height:34px;display:inline-flex;align-items:center;justify-content:center;font-weight:700;font-size:16px;flex-shrink:0">2</span><span style="font-size:1.4rem;font-weight:600">Autoencoder Anomaly Detection</span></div>', unsafe_allow_html=True)
st.caption("Shared across both pipelines — reconstruction error flags anomalous traffic")

if st.button("Run Autoencoder", use_container_width=True, key="btn_ae"):
    try:
        models, scaler, threshold, label_map = load_artifacts()
        with st.spinner("Running autoencoder..."):
            X_scaled = scaler.transform(X_raw).astype(np.float32)
            st.session_state["_X_scaled"] = X_scaled
            st.session_state["_res_ae"] = run_autoencoder(X_scaled, models, threshold)
            for k in ("_res_central", "_res_fed", "_table_central", "_table_fed"):
                st.session_state.pop(k, None)
    except Exception as e:
        st.error(f"Failed: {e}")

if "_res_ae" in st.session_state:
    render_ae_panel(st.session_state["_res_ae"], y_true)

st.divider()

# -- Stage 3: CNN-GRU -----------------------------------------
st.markdown('<div style="display:flex;align-items:center;gap:12px;margin:8px 0 2px"><span style="background:#3b82f6;color:white;border-radius:50%;width:34px;height:34px;display:inline-flex;align-items:center;justify-content:center;font-weight:700;font-size:16px;flex-shrink:0">3</span><span style="font-size:1.4rem;font-weight:600">CNN-GRU Classification</span></div>', unsafe_allow_html=True)
st.caption("Centralized (full dataset) vs Federated (FedAvg) — both receive original scaled features")

ae_result = st.session_state.get("_res_ae", None)
if ae_result is None:
    st.info("Run the Autoencoder above first to enable confidence tags in predictions.")

col_left, col_right = st.columns(2)

with col_left:
    with st.container(border=True):
        st.markdown("## Centralized")
        if st.button("Predict", use_container_width=True, key="btn_central"):
            try:
                models, scaler, threshold, label_map = load_artifacts()
                with st.spinner("Running centralized CNN-GRU..."):
                    if "_X_scaled" not in st.session_state:
                        st.session_state["_X_scaled"] = scaler.transform(X_raw).astype(np.float32)
                    X_scaled = st.session_state["_X_scaled"]
                    res = run_classifier(X_scaled, ae_result, models, label_map, "centralized")
                    st.session_state["_res_central"]   = res
                    st.session_state["_table_central"] = _build_table(res, ae_result, y_true)
            except Exception as e:
                st.error(f"Failed: {e}")
        if "_res_central" in st.session_state:
            render_panel(st.session_state["_res_central"], ae_result, y_true, "Centralized",
                         st.session_state["_table_central"])

with col_right:
    with st.container(border=True):
        st.markdown("## Federated")
        if st.button("Predict", use_container_width=True, key="btn_fed"):
            try:
                models, scaler, threshold, label_map = load_artifacts()
                with st.spinner("Running federated CNN-GRU..."):
                    if "_X_scaled" not in st.session_state:
                        st.session_state["_X_scaled"] = scaler.transform(X_raw).astype(np.float32)
                    X_scaled = st.session_state["_X_scaled"]
                    res = run_classifier(X_scaled, ae_result, models, label_map, "federated")
                    st.session_state["_res_fed"]   = res
                    st.session_state["_table_fed"] = _build_table(res, ae_result, y_true)
            except Exception as e:
                st.error(f"Failed: {e}")
        if "_res_fed" in st.session_state:
            render_panel(st.session_state["_res_fed"], ae_result, y_true, "Federated",
                         st.session_state["_table_fed"])

# -- Comparison ------------------------------------------------
if "_res_central" in st.session_state and "_res_fed" in st.session_state:
    st.divider()
    st.markdown('<div style="display:flex;align-items:center;gap:12px;margin:8px 0 2px"><span style="background:#3b82f6;color:white;border-radius:50%;width:34px;height:34px;display:inline-flex;align-items:center;justify-content:center;font-weight:700;font-size:16px;flex-shrink:0">4</span><span style="font-size:1.4rem;font-weight:600">Centralized vs Federated Comparison</span></div>', unsafe_allow_html=True)
    render_comparison(
        st.session_state["_res_central"],
        st.session_state["_res_fed"],
        y_true,
    )

