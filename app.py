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

ARTIFACT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results (1)")

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
        "centralized": os.path.join(ARTIFACT_DIR, "cnn_gru_centralized.keras"),
        "federated":   os.path.join(ARTIFACT_DIR, "cnn_gru_federated.keras"),
        "autoencoder": os.path.join(ARTIFACT_DIR, "autoencoder.keras"),
        "scaler":      os.path.join(ARTIFACT_DIR, "scaler.pkl"),
        "threshold":   os.path.join(ARTIFACT_DIR, "ae_threshold.pkl"),
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
    label_map_path = os.path.join(ARTIFACT_DIR, "label_map.pkl")
    if os.path.exists(label_map_path):
        with open(label_map_path, "rb") as f: label_map = pickle.load(f)
    else:
        label_map = {0: "Benign", 1: "Attack"}
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


def run_classifier(X_scaled, ae_result, models, label_map, model_key, mode="parallel"):
    X_3d   = X_scaled.reshape(X_scaled.shape[0], X_scaled.shape[1], 1)
    probs  = models[model_key].predict(X_3d, verbose=0).ravel()
    cnn_preds = (probs > 0.5).astype(int)   # raw CNN-GRU before AND
    preds     = cnn_preds.copy()

    # Cascading AND: final Attack only when both AE and CNN-GRU agree
    if mode == "cascading_and" and ae_result is not None:
        preds = (preds & ae_result["ae_flags"].astype(int))

    labels = np.array([label_map[p] for p in preds])
    conf   = np.where(preds == 1, probs, 1 - probs)
    is_attack = preds == 1
    if ae_result is not None:
        ae_flags   = ae_result["ae_flags"]
        if mode == "cascading_and":
            confidence = np.where(
                is_attack,                              "High - AE + CNN-GRU agreed",
                np.where(
                (probs > 0.5) & ~ae_flags,              "Suppressed - AE cleared",
                np.where(
                ae_flags & ~(probs > 0.5).astype(bool), "Suppressed - CNN-GRU cleared",
                                                         "Normal")))
        else:
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
        "cnn_preds":  cnn_preds,
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
    cm = confusion_matrix(y_true, flags, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    n_benign_true = int((y_true == 0).sum())
    return {
        "tp":  int(tp),
        "fp":  int(fp),
        "fn":  int(fn),
        "fpr": fp / n_benign_true if n_benign_true > 0 else 0.0,
    }


def _build_stats(result, y_true):
    out = {"metrics": None, "cm": None, "all_classes": None}
    if y_true is not None:
        cm = confusion_matrix(y_true, result["preds"], labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        n_benign_true = int((y_true == 0).sum())
        out["metrics"] = {
            "acc":  accuracy_score( y_true, result["preds"]),
            "prec": precision_score(y_true, result["preds"], zero_division=0),
            "rec":  recall_score(   y_true, result["preds"], zero_division=0),
            "f1":   f1_score(       y_true, result["preds"], zero_division=0),
            "tp":   int(tp),
            "fp":   int(fp),
            "fn":   int(fn),
            "fpr":  fp / n_benign_true if n_benign_true > 0 else 0.0,
        }
        all_classes        = [0, 1]
        out["all_classes"] = ["Benign", "Attack"]
        out["cm"]          = cm
    return out


def _build_table(result, ae_result, y_true, label_map, mode="Parallel"):
    if mode == "Cascading (AND)" and ae_result is not None:
        # Per-stage transparent view
        cnn_labels = np.array([label_map[p] for p in result["cnn_preds"]])
        table = pd.DataFrame({
            "Row":                np.arange(1, result["n_total"] + 1),
            "CNN-GRU Prediction": cnn_labels,
            "AE Flag":            np.where(ae_result["ae_flags"], "Anomaly", "Normal"),
            "Final Decision":     result["labels"],
            "Recon Error":        ae_result["errors"].round(6),
        })
        if y_true is not None:
            table["Ground Truth"] = [label_map[v] for v in y_true]
            table["Correct"]      = (result["preds"] == y_true)
        # Reorder to put Ground Truth first if present
        if y_true is not None:
            table = table[["Row", "Ground Truth", "CNN-GRU Prediction",
                           "AE Flag", "Final Decision", "Recon Error", "Correct"]]
    else:
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
        m1.metric("AE Attacks Caught",   f"{ae_stats['tp']:,}",    help="True Positives — attacks correctly flagged by AE")
        m2.metric("AE False Alarms",     f"{ae_stats['fp']:,}",    help="False Positives — benign samples wrongly flagged by AE")
        m3.metric("AE Missed Attacks",   f"{ae_stats['fn']:,}",    help="False Negatives — attacks that passed through the AE gate")
        m4.metric("AE False Alarm Rate", f"{ae_stats['fpr']:.3f}", help="FP / total benign — fraction of benign traffic wrongly flagged")
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


def render_panel(result, title, table, stats, mode="Parallel"):
    r      = result
    c1, c2 = st.columns(2)
    c1.metric("Benign", f"{r['n_benign']:,}", f"{r['n_benign']/r['n_total']:.1%}")
    c2.metric("Attack", f"{r['n_anomaly']:,}", f"{r['n_anomaly']/r['n_total']:.1%}")
    if stats["metrics"] is not None:
        m = stats["metrics"]
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Attacks Caught",   f"{m['tp']:,}",    help="True Positives — attacks correctly flagged")
        m2.metric("False Alarms",     f"{m['fp']:,}",    help="False Positives — benign samples wrongly flagged")
        m3.metric("Missed Attacks",   f"{m['fn']:,}",    help="False Negatives — attacks that slipped through")
        m4.metric("False Alarm Rate", f"{m['fpr']:.3f}", help="FP / total benign — fraction of benign traffic wrongly blocked")
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
    st.divider()
    mode = st.radio(
        "Pipeline Mode",
        options=["Parallel", "Cascading (AND)"],
        captions=["AE and CNN-GRU run independently",
                  "Attack only if both AE and CNN-GRU agree"],
        key="pipeline_mode",
    )
    st.divider()
    uploaded = st.file_uploader("Upload CSV", type="csv", key="uploader")

# -- Main -----------------------------------------------------
st.title("IIoT Intrusion Detection System")
_mode_caption = ("Cascading AND Pipeline: AE flags → CNN-GRU classifies → Attack only if both agree — Binary"
                 if mode == "Cascading (AND)" else
                 "Parallel Pipeline: Autoencoder (anomaly detector) + CNN-GRU (classifier) — Binary")
st.caption(_mode_caption)

if uploaded is None:
    st.info("Upload a CSV file in the sidebar to get started.")
    st.stop()

X_raw, y_true = load_csv(uploaded)

_PRED_KEYS = ("_res_central", "_res_fed", "_table_central", "_table_fed",
              "_stats_central", "_stats_fed", "_comp_data")

file_id = f"{uploaded.name}_{uploaded.size}"
if st.session_state.get("_file_id") != file_id:
    st.session_state["_file_id"] = file_id
    for k in ("_res_ae", "_X_scaled", "_stats_ae", "_label_map") + _PRED_KEYS:
        st.session_state.pop(k, None)

# Invalidate predictions when mode changes (AE result is still valid)
_mode_key = mode
if st.session_state.get("_active_mode") != _mode_key:
    st.session_state["_active_mode"] = _mode_key
    for k in _PRED_KEYS:
        st.session_state.pop(k, None)

# Invalidate stale stats that are missing new keys (tp/fp/fn/fpr)
for _sk in ("_stats_central", "_stats_fed"):
    _s = st.session_state.get(_sk)
    if _s is not None and _s.get("metrics") is not None and "tp" not in _s["metrics"]:
        for k in _PRED_KEYS:
            st.session_state.pop(k, None)
        break

# Invalidate stale AE stats missing new keys
_ae_s = st.session_state.get("_stats_ae")
if _ae_s is not None and "tp" not in _ae_s:
    st.session_state.pop("_stats_ae", None)

with st.sidebar:
    st.success(f"**{len(X_raw):,}** samples · {X_raw.shape[1]} features"
               + (" · labels detected" if y_true is not None else ""))
    st.caption(f"Mode: **{mode}**")

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

# -- Stage 2: Autoencoder (Parallel mode only) ---------------
if mode == "Parallel":
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
if mode == "Cascading (AND)":
    st.caption("Cascading AND — final Attack label requires both AE anomaly flag AND CNN-GRU Attack prediction")
else:
    st.caption("Centralized (full dataset) vs Federated (FedAvg) — both receive original scaled features")

ae_result = st.session_state.get("_res_ae", None)
if ae_result is None and mode == "Parallel":
    st.info("Run the Autoencoder above first to enable confidence tags in predictions.")

col_left, col_right = st.columns(2)

with col_left:
    with st.container(border=True):
        st.markdown("## Centralized")
        _btn_label_c = "Predict (Centralized)" if mode == "Cascading (AND)" else "Predict"
        if st.button(_btn_label_c, use_container_width=True, key="btn_central"):
            try:
                models, scaler, threshold, label_map = load_artifacts()
                st.session_state["_label_map"] = label_map
                with st.spinner("Running centralized CNN-GRU..."):
                    if "_X_scaled" not in st.session_state:
                        st.session_state["_X_scaled"] = scaler.transform(X_raw).astype(np.float32)
                    X_scaled = st.session_state["_X_scaled"]
                    # In cascading mode, auto-run AE if not already done
                    if mode == "Cascading (AND)" and "_res_ae" not in st.session_state:
                        ae_res = run_autoencoder(X_scaled, models, threshold)
                        st.session_state["_res_ae"]   = ae_res
                        st.session_state["_stats_ae"] = _build_ae_stats(ae_res, y_true)
                    ae_result = st.session_state.get("_res_ae", None)
                    _mode_arg = "cascading_and" if mode == "Cascading (AND)" else "parallel"
                    res = run_classifier(X_scaled, ae_result, models, label_map, "centralized", _mode_arg)
                    st.session_state["_res_central"]   = res
                    st.session_state["_table_central"] = _build_table(res, ae_result, y_true, label_map, mode)
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
                         st.session_state["_stats_central"], mode)

with col_right:
    with st.container(border=True):
        st.markdown("## Federated")
        _btn_label_f = "Predict (Federated)" if mode == "Cascading (AND)" else "Predict"
        if st.button(_btn_label_f, use_container_width=True, key="btn_fed"):
            try:
                models, scaler, threshold, label_map = load_artifacts()
                st.session_state["_label_map"] = label_map
                with st.spinner("Running federated CNN-GRU..."):
                    if "_X_scaled" not in st.session_state:
                        st.session_state["_X_scaled"] = scaler.transform(X_raw).astype(np.float32)
                    X_scaled = st.session_state["_X_scaled"]
                    # In cascading mode, auto-run AE if not already done
                    if mode == "Cascading (AND)" and "_res_ae" not in st.session_state:
                        ae_res = run_autoencoder(X_scaled, models, threshold)
                        st.session_state["_res_ae"]   = ae_res
                        st.session_state["_stats_ae"] = _build_ae_stats(ae_res, y_true)
                    ae_result = st.session_state.get("_res_ae", None)
                    _mode_arg = "cascading_and" if mode == "Cascading (AND)" else "parallel"
                    res = run_classifier(X_scaled, ae_result, models, label_map, "federated", _mode_arg)
                    st.session_state["_res_fed"]   = res
                    st.session_state["_table_fed"] = _build_table(res, ae_result, y_true, label_map, mode)
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
                         st.session_state["_stats_fed"], mode)

# -- Stage 4: Comparison --------------------------------------
if "_res_central" in st.session_state and "_res_fed" in st.session_state:
    st.divider()
    st.markdown(STAGE_HTML.format(n=4, title="Centralized vs Federated Comparison"),
                unsafe_allow_html=True)
    render_comparison(st.session_state.get("_comp_data"))
