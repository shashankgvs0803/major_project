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
    page_title="IIoT IDS — Binary",
    layout="wide",
)

ARTIFACT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

STAGE_HTML = (
    '<div style="display:flex;align-items:center;gap:12px;margin:8px 0 2px">'
    '<span style="background:#3b82f6;color:white;border-radius:50%;width:34px;height:34px;'
    'display:inline-flex;align-items:center;justify-content:center;font-weight:700;'
    'font-size:16px;flex-shrink:0">{n}</span>'
    '<span style="font-size:1.4rem;font-weight:600">{title}</span></div>'
)


# -- Model loading --------------------------------------------
@st.cache_resource(show_spinner="Loading models …")
def load_artifacts():
    from tensorflow.keras.models import load_model
    paths = {
        "centralized": os.path.join(ARTIFACT_DIR, "cnn_gru_ids.keras"),
        "federated":   os.path.join(ARTIFACT_DIR, "federated_cnn_gru_ids.keras"),
        "autoencoder": os.path.join(ARTIFACT_DIR, "autoencoder_ids.keras"),
        "scaler":      os.path.join(ARTIFACT_DIR, "scaler.pkl"),
        "threshold":   os.path.join(ARTIFACT_DIR, "ae_threshold.pkl"),
        "label_map":   os.path.join(ARTIFACT_DIR, "label_map.pkl"),
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
    y_true = df["label"].values.astype(int) if "label" in df.columns else None
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
    X_3d   = X_scaled.reshape(X_scaled.shape[0], X_scaled.shape[1], 1)
    probs  = models[model_key].predict(X_3d, verbose=0).ravel()
    preds  = (probs > 0.5).astype(int)
    labels = np.array([label_map[p] for p in preds])
    conf   = np.where(preds == 1, probs, 1 - probs)
    is_attack = preds == 1
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
        "labels":     labels,
        "preds":      preds,
        "probs":      probs,
        "conf":       conf,
        "confidence": confidence,
        "n_total":    n_total,
        "n_benign":   n_total - n_anomaly,
        "n_anomaly":  n_anomaly,
    }


# -- Cache builders -------------------------------------------
def _build_ae_stats(ae_result, y_true):
    if y_true is None:
        return None
    flags = ae_result["ae_flags"].astype(int)
    return {
        "acc":  accuracy_score( y_true, flags),
        "prec": precision_score(y_true, flags, zero_division=0),
        "rec":  recall_score(   y_true, flags, zero_division=0),
        "f1":   f1_score(       y_true, flags, zero_division=0),
    }


def _build_stats(result, y_true):
    out = {"metrics": None, "cm": None, "all_classes": None}
    if y_true is not None:
        out["metrics"] = {
            "acc":  accuracy_score( y_true, result["preds"]),
            "prec": precision_score(y_true, result["preds"], zero_division=0),
            "rec":  recall_score(   y_true, result["preds"], zero_division=0),
            "f1":   f1_score(       y_true, result["preds"], zero_division=0),
        }
        all_classes        = [0, 1]
        out["all_classes"] = ["Benign", "Attack"]
        out["cm"]          = confusion_matrix(y_true, result["preds"], labels=all_classes)
    return out


def _build_table(result, ae_result, y_true, label_map):
    table = pd.DataFrame({
        "Row":        np.arange(1, result["n_total"] + 1),
        "Prediction": result["labels"],
        "Conf %":     (result["conf"] * 100).round(1),
        "Confidence": result["confidence"],
    })
    if ae_result is not None:
        table["AE Flag"]     = np.where(ae_result["ae_flags"], "Anomaly", "Normal")
        table["Recon Error"] = ae_result["errors"].round(6)
    if y_true is not None:
        table["Ground Truth"] = [label_map[v] for v in y_true]
        table["Correct"]      = (result["preds"] == y_true)
    return table


def _build_comparison_data(res_central, res_fed, y_true):
    if y_true is None:
        return None
    rows = []
    for label, r in [("Centralized", res_central), ("Federated", res_fed)]:
        rows.append({
            "Model":     label,
            "Accuracy":  accuracy_score( y_true, r["preds"]),
            "Precision": precision_score(y_true, r["preds"], zero_division=0),
            "Recall":    recall_score(   y_true, r["preds"], zero_division=0),
            "F1":        f1_score(       y_true, r["preds"], zero_division=0),
        })
    return pd.DataFrame(rows)


# -- Render functions ------------------------------------------
def render_dataset_summary(y_true, label_map):
    n_total  = len(y_true)
    n_benign = int((y_true == 0).sum())
    n_attack = int((y_true == 1).sum())
    c1, c2, c3 = st.columns(3)
    c1.metric("Total Samples", f"{n_total:,}")
    c2.metric(label_map[0],    f"{n_benign:,}  ({n_benign/n_total:.1%})")
    c3.metric(label_map[1],    f"{n_attack:,}  ({n_attack/n_total:.1%})")


def render_ae_panel(ae_result, ae_stats):
    r      = ae_result
    c1, c2 = st.columns(2)
    c1.metric("Flagged Normal",  f"{r['n_normal']:,}",  f"{r['n_normal']/r['n_total']:.1%}")
    c2.metric("Flagged Anomaly", f"{r['n_flagged']:,}", f"{r['n_flagged']/r['n_total']:.1%}")
    if ae_stats is not None:
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("AE Accuracy",  f"{ae_stats['acc']:.3f}")
        m2.metric("AE Precision", f"{ae_stats['prec']:.3f}")
        m3.metric("AE Recall",    f"{ae_stats['rec']:.3f}")
        m4.metric("AE F1",        f"{ae_stats['f1']:.3f}")
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=r["errors"][~r["ae_flags"]], name="Normal",
                               marker_color="#1a9e60", opacity=0.7, nbinsx=40))
    fig.add_trace(go.Histogram(x=r["errors"][r["ae_flags"]],  name="Anomaly",
                               marker_color="#d62828", opacity=0.7, nbinsx=40))
    fig.add_vline(x=r["threshold"], line_dash="dash", line_color="white",
                  annotation_text=f"Threshold {r['threshold']:.4f}")
    fig.update_layout(
        title=dict(text="Reconstruction Error Distribution", x=0.5, xanchor="center"),
        height=260, barmode="overlay",
        margin=dict(t=70, b=10, l=5, r=5),
        legend=dict(orientation="h", yanchor="bottom", y=1.08, x=0),
        xaxis_title="Reconstruction Error (MSE)",
    )
    st.plotly_chart(fig, use_container_width=True, key="ae_hist")


def render_panel(result, title, table, stats):
    r      = result
    c1, c2 = st.columns(2)
    c1.metric("Benign", f"{r['n_benign']:,}", f"{r['n_benign']/r['n_total']:.1%}")
    c2.metric("Attack", f"{r['n_anomaly']:,}", f"{r['n_anomaly']/r['n_total']:.1%}")
    if stats["metrics"] is not None:
        m = stats["metrics"]
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Accuracy",  f"{m['acc']:.3f}")
        m2.metric("Precision", f"{m['prec']:.3f}")
        m3.metric("Recall",    f"{m['rec']:.3f}")
        m4.metric("F1",        f"{m['f1']:.3f}")
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=r["probs"][r["preds"] == 0], name="Benign",
        marker_color="#1a9e60", opacity=0.7, nbinsx=30,
    ))
    fig.add_trace(go.Histogram(
        x=r["probs"][r["preds"] == 1], name="Attack",
        marker_color="#d62828", opacity=0.7, nbinsx=30,
    ))
    fig.add_vline(x=0.5, line_dash="dash", line_color="white",
                  annotation_text="Threshold 0.5")
    fig.update_layout(
        title=dict(text="Confidence Distribution", x=0.5, xanchor="center"),
        height=260, barmode="overlay",
        margin=dict(t=70, b=10, l=5, r=5),
        legend=dict(orientation="h", yanchor="bottom", y=1.08, x=0),
        xaxis_title="Attack Probability",
    )
    st.plotly_chart(fig, use_container_width=True, key=f"hist_{title}")
    if stats["cm"] is not None:
        fig_cm = px.imshow(
            stats["cm"],
            x=stats["all_classes"], y=stats["all_classes"],
            color_continuous_scale="Blues",
            labels=dict(x="Predicted", y="True", color="Count"),
            title="Confusion Matrix", text_auto=True,
        )
        fig_cm.update_layout(height=350, margin=dict(t=50, b=10, l=5, r=5))
        with st.expander("Confusion Matrix"):
            st.plotly_chart(fig_cm, use_container_width=True, key=f"cm_{title}")
    with st.expander(f"Full results ({r['n_total']:,} rows)"):
        st.dataframe(table, use_container_width=True, hide_index=True)
        st.download_button(
            "Download CSV", table.to_csv(index=False).encode(),
            f"{title.lower()}_results.csv", "text/csv",
            key=f"dl_{title}",
        )


def render_comparison(comp):
    if comp is None:
        st.info("Upload a CSV with a label column to see metric comparison.")
        return
    comp_melt = comp.melt(id_vars="Model", var_name="Metric", value_name="Score")
    fig = px.bar(
        comp_melt, x="Metric", y="Score", color="Model", barmode="group",
        color_discrete_map={"Centralized": "#3b82f6", "Federated": "#8b5cf6"},
        text_auto=".3f",
    )
    fig.update_layout(
        height=400, yaxis_range=[0, 1.05],
        margin=dict(t=80, b=10, l=5, r=5),
        title=dict(text="Centralized vs Federated — CNN-GRU Metrics",
                   y=0.97, x=0.5, xanchor="center"),
        legend=dict(orientation="h", yanchor="bottom", y=1.08, x=0),
    )
    st.plotly_chart(fig, use_container_width=True, key="comparison_bar")
    st.dataframe(comp.set_index("Model").style.format("{:.4f}"), use_container_width=True)


# -- Sidebar --------------------------------------------------
with st.sidebar:
    st.title("IIoT IDS")
    st.caption("Binary · Parallel Pipeline")
    st.divider()
    uploaded = st.file_uploader("Upload CSV", type="csv", key="uploader")

# -- Main -----------------------------------------------------
st.title("IIoT Intrusion Detection System")
st.caption("Parallel Pipeline: Autoencoder (anomaly detector) + CNN-GRU (classifier) — Binary")

if uploaded is None:
    st.info("Upload a CSV file in the sidebar to get started.")
    st.stop()

X_raw, y_true = load_csv(uploaded)

file_id = f"{uploaded.name}_{uploaded.size}"
if st.session_state.get("_file_id") != file_id:
    st.session_state["_file_id"] = file_id
    for k in ("_res_ae", "_res_central", "_res_fed", "_X_scaled",
              "_table_central", "_table_fed",
              "_stats_ae", "_stats_central", "_stats_fed", "_comp_data",
              "_label_map"):
        st.session_state.pop(k, None)

with st.sidebar:
    st.success(f"**{len(X_raw):,}** samples · {X_raw.shape[1]} features"
               + (" · labels detected" if y_true is not None else ""))

# -- Stage 1: Dataset -----------------------------------------
st.markdown(STAGE_HTML.format(n=1, title="Dataset Overview"), unsafe_allow_html=True)
st.caption("Upload your CSV in the sidebar — ground truth class distribution is shown below")

if y_true is not None:
    # Need label_map for display — load artifacts if not yet loaded
    if "_label_map" not in st.session_state:
        try:
            _, _, _, lm = load_artifacts()
            st.session_state["_label_map"] = lm
        except Exception:
            st.session_state["_label_map"] = {0: "Benign", 1: "Attack"}
    with st.expander("Ground Truth Dataset Summary", expanded=True):
        render_dataset_summary(y_true, st.session_state["_label_map"])
else:
    st.info("No label column detected — ground truth summary unavailable.")

st.divider()

# -- Stage 2: Autoencoder -------------------------------------
st.markdown(STAGE_HTML.format(n=2, title="Autoencoder Anomaly Detection"), unsafe_allow_html=True)
st.caption("Shared across both pipelines — reconstruction error flags anomalous traffic")

if st.button("Run Autoencoder", use_container_width=True, key="btn_ae"):
    try:
        models, scaler, threshold, label_map = load_artifacts()
        st.session_state["_label_map"] = label_map
        with st.spinner("Running autoencoder..."):
            X_scaled = scaler.transform(X_raw).astype(np.float32)
            st.session_state["_X_scaled"] = X_scaled
            ae_res = run_autoencoder(X_scaled, models, threshold)
            st.session_state["_res_ae"]   = ae_res
            st.session_state["_stats_ae"] = _build_ae_stats(ae_res, y_true)
            for k in ("_res_central", "_res_fed", "_table_central", "_table_fed",
                      "_stats_central", "_stats_fed", "_comp_data"):
                st.session_state.pop(k, None)
    except Exception as e:
        st.error(f"Failed: {e}")

if "_res_ae" in st.session_state:
    render_ae_panel(st.session_state["_res_ae"], st.session_state.get("_stats_ae"))

st.divider()

# -- Stage 3: CNN-GRU -----------------------------------------
st.markdown(STAGE_HTML.format(n=3, title="CNN-GRU Classification"), unsafe_allow_html=True)
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
                st.session_state["_label_map"] = label_map
                with st.spinner("Running centralized CNN-GRU..."):
                    if "_X_scaled" not in st.session_state:
                        st.session_state["_X_scaled"] = scaler.transform(X_raw).astype(np.float32)
                    X_scaled = st.session_state["_X_scaled"]
                    res = run_classifier(X_scaled, ae_result, models, label_map, "centralized")
                    st.session_state["_res_central"]   = res
                    st.session_state["_table_central"] = _build_table(res, ae_result, y_true, label_map)
                    st.session_state["_stats_central"] = _build_stats(res, y_true)
                    if "_res_fed" in st.session_state:
                        st.session_state["_comp_data"] = _build_comparison_data(
                            res, st.session_state["_res_fed"], y_true)
                    else:
                        st.session_state.pop("_comp_data", None)
            except Exception as e:
                st.error(f"Failed: {e}")
        if "_res_central" in st.session_state:
            render_panel(st.session_state["_res_central"], "Centralized",
                         st.session_state["_table_central"],
                         st.session_state["_stats_central"])

with col_right:
    with st.container(border=True):
        st.markdown("## Federated")
        if st.button("Predict", use_container_width=True, key="btn_fed"):
            try:
                models, scaler, threshold, label_map = load_artifacts()
                st.session_state["_label_map"] = label_map
                with st.spinner("Running federated CNN-GRU..."):
                    if "_X_scaled" not in st.session_state:
                        st.session_state["_X_scaled"] = scaler.transform(X_raw).astype(np.float32)
                    X_scaled = st.session_state["_X_scaled"]
                    res = run_classifier(X_scaled, ae_result, models, label_map, "federated")
                    st.session_state["_res_fed"]   = res
                    st.session_state["_table_fed"] = _build_table(res, ae_result, y_true, label_map)
                    st.session_state["_stats_fed"] = _build_stats(res, y_true)
                    if "_res_central" in st.session_state:
                        st.session_state["_comp_data"] = _build_comparison_data(
                            st.session_state["_res_central"], res, y_true)
                    else:
                        st.session_state.pop("_comp_data", None)
            except Exception as e:
                st.error(f"Failed: {e}")
        if "_res_fed" in st.session_state:
            render_panel(st.session_state["_res_fed"], "Federated",
                         st.session_state["_table_fed"],
                         st.session_state["_stats_fed"])

# -- Stage 4: Comparison --------------------------------------
if "_res_central" in st.session_state and "_res_fed" in st.session_state:
    st.divider()
    st.markdown(STAGE_HTML.format(n=4, title="Centralized vs Federated Comparison"),
                unsafe_allow_html=True)
    render_comparison(st.session_state.get("_comp_data"))
