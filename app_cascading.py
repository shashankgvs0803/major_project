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
    page_title="IIoT IDS — Cascading",
    layout="wide",
)

# ── Paths ─────────────────────────────────────────────────────
# Change ARTIFACT_DIR to "results" once new notebook outputs are ready.
# Currently uses the old multi-class results folder.
ARTIFACT_DIR     = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results_multi_class")
TEST_SAMPLES_CSV = os.path.join(ARTIFACT_DIR, "test_samples_mc.csv")


# ── Artifact loading ──────────────────────────────────────────
# Using old multi-class artifact filenames.
# When new notebook results are ready, update paths to:
#   model_centralized.keras, model_federated.keras, autoencoder.keras,
#   scaler.pkl, ae_threshold.pkl, le.pkl, benign_idx.pkl
@st.cache_resource(show_spinner="Loading models…")
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
        raise FileNotFoundError(f"Missing artifacts in {ARTIFACT_DIR}: {missing}")

    models = {
        "centralized": load_model(paths["centralized"]),
        "federated":   load_model(paths["federated"]),
        "autoencoder": load_model(paths["autoencoder"]),
    }
    with open(paths["scaler"],    "rb") as f: scaler    = pickle.load(f)
    with open(paths["threshold"], "rb") as f: threshold = pickle.load(f)
    with open(paths["label_map"], "rb") as f: label_map = pickle.load(f)

    # Derive benign index from label_map {int: class_name}
    benign_idx = next(k for k, v in label_map.items() if v == "benign")
    return models, scaler, float(threshold), label_map, int(benign_idx)


@st.cache_data(show_spinner="Reading CSV…")
def load_csv(source):
    """Accepts a Streamlit UploadedFile or a plain file path string."""
    df     = pd.read_csv(source)
    y_true = df["label"].values if "label" in df.columns else None
    X_raw  = (df.drop(columns=["label"], errors="ignore")
                .select_dtypes(include=[np.number])
                .fillna(0).values.astype(np.float32))
    return X_raw, y_true


# ── Cascading inference ───────────────────────────────────────
def run_cascading(X_scaled, models, threshold, label_map, benign_idx, model_key):
    """
    Stage 1 — Autoencoder (anomaly gate):
        reconstruction error ≤ threshold  →  Benign  (stop here)
        reconstruction error  > threshold  →  forward to Stage 2

    Stage 2 — CNN-GRU (attack classifier):
        applied only to samples that passed the AE gate.
    """
    n = len(X_scaled)

    # Stage 1 — AE gate
    recon      = models["autoencoder"].predict(X_scaled, verbose=0)
    ae_errors  = np.mean(np.square(X_scaled - recon), axis=1)
    ae_flagged = ae_errors > threshold                          # True = anomaly

    # Initialise all samples as benign (AE said normal)
    final_labels = np.full(n, "benign", dtype=object)
    stage_used   = np.full(n, "ae_gate", dtype=object)         # which stage gave the label

    # Stage 2 — CNN-GRU on flagged samples only
    n_flagged = int(ae_flagged.sum())
    if n_flagged > 0:
        flagged_idx = np.where(ae_flagged)[0]
        X_flagged   = X_scaled[flagged_idx].reshape(n_flagged, X_scaled.shape[1], 1)
        probs       = models[model_key].predict(X_flagged, verbose=0)
        preds       = np.argmax(probs, axis=1)
        for arr_i, orig_i in enumerate(flagged_idx):
            final_labels[orig_i] = label_map[preds[arr_i]]
            stage_used[orig_i]   = "cnn_gru"

    is_attack = final_labels != "benign"
    return {
        "final_labels": final_labels,
        "stage_used":   stage_used,
        "ae_errors":    ae_errors,
        "ae_flagged":   ae_flagged,
        "n_total":      n,
        "n_ae_benign":  n - n_flagged,          # labelled benign by AE alone
        "n_cnn_gru":    n_flagged,              # forwarded to CNN-GRU
        "n_benign":     int((~is_attack).sum()),
        "n_attack":     int(is_attack.sum()),
        "threshold":    threshold,
    }


# ── Render helpers ────────────────────────────────────────────
def render_verdict_banner(r):
    """Single file-level verdict — majority attack type or BENIGN."""
    attack_labels = r["final_labels"][r["final_labels"] != "benign"]
    attack_pct    = r["n_attack"] / r["n_total"] * 100

    if len(attack_labels) == 0:
        st.markdown(
            '<div style="background:#1b5e20;border-radius:12px;padding:24px 32px;text-align:center;">'
            '<span style="font-size:2rem;font-weight:700;color:#a5d6a7;">✔  BENIGN TRAFFIC</span><br>'
            f'<span style="color:#c8e6c9;font-size:1rem;">All {r["n_total"]:,} samples cleared by AE gate</span>'
            '</div>',
            unsafe_allow_html=True,
        )
    else:
        unique, counts = np.unique(attack_labels, return_counts=True)
        dominant       = unique[np.argmax(counts)]
        st.markdown(
            '<div style="background:#b71c1c;border-radius:12px;padding:24px 32px;text-align:center;">'
            f'<span style="font-size:2rem;font-weight:700;color:#ffcdd2;">⚠  ATTACK DETECTED</span><br>'
            f'<span style="color:#ef9a9a;font-size:1.3rem;font-weight:600;">{dominant.upper()}</span><br>'
            f'<span style="color:#ffcdd2;font-size:0.95rem;">'
            f'{r["n_attack"]:,} of {r["n_total"]:,} samples flagged ({attack_pct:.1f}%)'
            f'</span></div>',
            unsafe_allow_html=True,
        )
        st.markdown("")
        # Attack type breakdown
        rows = [{"Attack Type": u, "Samples": f"{c:,}", "% of Flagged": f"{c/len(attack_labels):.1%}"}
                for u, c in sorted(zip(unique, counts), key=lambda x: -x[1])]
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


def render_dataset_summary(y_true):
    """Class distribution breakdown for a ground-truth label array."""
    counts   = pd.Series(y_true).value_counts().sort_values(ascending=False)
    n_total  = len(y_true)
    n_benign = int(counts.get("benign", 0))
    n_attack = n_total - n_benign

    c1, c2, c3 = st.columns(3)
    c1.metric("Total Samples", f"{n_total:,}")
    c2.metric("Benign",  f"{n_benign:,}",  f"{n_benign/n_total:.1%}")
    c3.metric("Attacks", f"{n_attack:,}", f"{n_attack/n_total:.1%}")

    attack_counts = {k: v for k, v in counts.items() if k != "benign"}
    if attack_counts:
        rows = [{"Attack Type": k, "Count": f"{v:,}",
                 "% of Dataset": f"{v/n_total:.2%}",
                 "% of Attacks": f"{v/n_attack:.2%}"}
                for k, v in sorted(attack_counts.items(), key=lambda x: -x[1])]
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


def render_gate_stats(r, key_suffix=""):
    """AE gate summary — how many were stopped vs forwarded."""
    c1, c2, c3 = st.columns(3)
    c1.metric("Total Samples",     f"{r['n_total']:,}")
    c2.metric("Stopped at Gate",   f"{r['n_ae_benign']:,}",
              f"{r['n_ae_benign']/r['n_total']:.1%}",
              help="AE reconstruction error ≤ threshold → labelled Benign")
    c3.metric("Forwarded to CNN-GRU", f"{r['n_cnn_gru']:,}",
              f"{r['n_cnn_gru']/r['n_total']:.1%}",
              help="AE reconstruction error > threshold → classified by CNN-GRU")

    # Reconstruction error histogram
    errors, flagged = r["ae_errors"], r["ae_flagged"]
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=errors[~flagged], name="Stopped (Benign)",
                               marker_color="#1a9e60", opacity=0.7, nbinsx=40))
    fig.add_trace(go.Histogram(x=errors[flagged],  name="Forwarded (Anomaly)",
                               marker_color="#d62828", opacity=0.7, nbinsx=40))
    fig.add_vline(x=r["threshold"], line_dash="dash", line_color="white",
                  annotation_text=f"Threshold {r['threshold']:.5f}")
    fig.update_layout(
        title=dict(text="AE Reconstruction Error — Gate Decision", x=0.5, xanchor="center"),
        height=280, barmode="overlay",
        xaxis_title="Reconstruction Error (MSE)",
        margin=dict(t=70, b=10, l=5, r=5),
        legend=dict(orientation="h", yanchor="bottom", y=1.08, x=0),
    )
    st.plotly_chart(fig, use_container_width=True, key=f"gate_hist_{key_suffix}")


def render_final_results(r, y_true, model_label, key_suffix):
    c1, c2 = st.columns(2)
    c1.metric("Final Benign",  f"{r['n_benign']:,}",  f"{r['n_benign']/r['n_total']:.1%}")
    c2.metric("Final Attacks", f"{r['n_attack']:,}", f"{r['n_attack']/r['n_total']:.1%}")

    # Metrics vs ground truth
    if y_true is not None:
        acc  = accuracy_score( y_true, r["final_labels"])
        prec = precision_score(y_true, r["final_labels"], average="macro", zero_division=0)
        rec  = recall_score(   y_true, r["final_labels"], average="macro", zero_division=0)
        f1   = f1_score(       y_true, r["final_labels"], average="macro", zero_division=0)
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Accuracy",  f"{acc:.4f}")
        m2.metric("Precision", f"{prec:.4f}")
        m3.metric("Recall",    f"{rec:.4f}")
        m4.metric("F1-score",  f"{f1:.4f}")

    # Attack type pie
    unique, counts = np.unique(r["final_labels"], return_counts=True)
    attack_types = {k: v for k, v in zip(unique, counts) if k != "benign"}
    if attack_types:
        fig_pie = px.pie(names=list(attack_types.keys()), values=list(attack_types.values()),
                         title="Detected Attack Types", hole=0.4)
        fig_pie.update_layout(height=280, margin=dict(t=40, b=10, l=5, r=5))
        st.plotly_chart(fig_pie, use_container_width=True, key=f"pie_{key_suffix}")
        for label, count in sorted(attack_types.items(), key=lambda x: -x[1]):
            st.markdown(f"- **{label}**: {count:,}")
    else:
        st.success("No attacks detected — all traffic classified as benign.")

    # Stage breakdown
    with st.expander("Stage breakdown"):
        stage_counts = pd.Series(r["stage_used"]).value_counts().reset_index()
        stage_counts.columns = ["Stage", "Count"]
        stage_counts["Stage"] = stage_counts["Stage"].map(
            {"ae_gate": "AE Gate (Benign)", "cnn_gru": "CNN-GRU (Attack/Benign)"})
        st.dataframe(stage_counts, use_container_width=True, hide_index=True)

    # Confusion matrix
    if y_true is not None:
        all_classes = sorted(set(y_true) | set(r["final_labels"]))
        cm = confusion_matrix(y_true, r["final_labels"], labels=all_classes)
        fig_cm = px.imshow(cm, x=all_classes, y=all_classes,
                           color_continuous_scale="Blues", text_auto=True,
                           labels=dict(x="Predicted", y="True", color="Count"),
                           title="Confusion Matrix")
        fig_cm.update_layout(height=500, margin=dict(t=50, b=10, l=5, r=5),
                             xaxis=dict(tickangle=45))
        with st.expander("Confusion Matrix"):
            st.plotly_chart(fig_cm, use_container_width=True, key=f"cm_{key_suffix}")

    # Full results table
    table = pd.DataFrame({
        "Row":        np.arange(1, r["n_total"] + 1),
        "Prediction": r["final_labels"],
        "Stage":      r["stage_used"],
        "Recon Error": r["ae_errors"].round(6),
        "AE Flag":    np.where(r["ae_flagged"], "Forwarded", "Stopped"),
    })
    if y_true is not None:
        table["Ground Truth"] = y_true
        table["Correct"]      = r["final_labels"] == y_true
    with st.expander(f"Full results ({r['n_total']:,} rows)"):
        st.dataframe(table, use_container_width=True, hide_index=True)
        st.download_button("Download CSV", table.to_csv(index=False).encode(),
                           f"cascading_{model_label.lower()}_results.csv", "text/csv",
                           key=f"dl_{key_suffix}")


def render_comparison(res_central, res_fed, y_true):
    if y_true is None:
        st.info("Upload a CSV with a 'label' column to see metric comparison.")
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
                 title="Cascading: Centralized vs Federated CNN-GRU", text_auto=".4f")
    fig.update_layout(height=400, yaxis_range=[0, 1.05],
                      title=dict(y=0.97, x=0.5, xanchor="center"),
                      margin=dict(t=80, b=10, l=5, r=5),
                      legend=dict(orientation="h", yanchor="bottom", y=1.08, x=0))
    st.plotly_chart(fig, use_container_width=True, key="comparison_bar")
    st.dataframe(comp.set_index("Model").style.format("{:.4f}"), use_container_width=True)


# ── Sidebar ───────────────────────────────────────────────────
def _section(n, title, subtitle=""):
    st.markdown(
        f'<div style="display:flex;align-items:center;gap:12px;margin:8px 0 2px">'
        f'<span style="background:#3b82f6;color:white;border-radius:50%;width:34px;height:34px;'
        f'display:inline-flex;align-items:center;justify-content:center;font-weight:700;font-size:16px;flex-shrink:0">'
        f'{n}</span><span style="font-size:1.4rem;font-weight:600">{title}</span></div>',
        unsafe_allow_html=True,
    )
    if subtitle:
        st.caption(subtitle)


with st.sidebar:
    st.title("IIoT IDS")
    st.caption("Cascading Pipeline")
    st.divider()
    st.markdown(
        "**How it works**\n\n"
        "1. **AE Gate** — all samples pass through the autoencoder.\n"
        "   - Error ≤ threshold → **Benign** ✓ *(stops here)*\n"
        "   - Error > threshold → forwarded to Stage 2\n\n"
        "2. **CNN-GRU** — only anomalous samples get classified."
    )
    st.divider()
    mode = st.radio(
        "Mode",
        ["Live Detection", "Evaluate on Test Set"],
        help="Live Detection: upload any CSV and get a verdict.\n"
             "Evaluate: run on the saved test set with full metrics.",
    )
    st.divider()
    if mode == "Live Detection":
        uploaded = st.file_uploader("Upload CSV (115 features)", type="csv", key="uploader_live")
    else:
        uploaded = st.file_uploader(
            "Upload test CSV with 'label' column",
            type="csv",
            key="uploader_eval",
            help="Upload your test_samples_mc.csv or any labelled N-BaIoT CSV for evaluation.",
        )

# ── Main ──────────────────────────────────────────────────────
st.title("IIoT Intrusion Detection System")
st.caption("Cascading Pipeline: AE Anomaly Gate → CNN-GRU Attack Classifier (anomalies only)")

# ══════════════════════════════════════════════════════════════
# MODE 1 — LIVE DETECTION
# ══════════════════════════════════════════════════════════════
if mode == "Live Detection":
    if uploaded is None:
        st.info("Upload a CSV file in the sidebar to get started.")
        st.stop()

    X_raw, y_true = load_csv(uploaded)

    file_id = f"{uploaded.name}_{uploaded.size}"
    if st.session_state.get("_file_id") != file_id:
        st.session_state["_file_id"] = file_id
        for k in ("_X_scaled", "_live_central", "_live_fed"):
            st.session_state.pop(k, None)

    with st.sidebar:
        st.success(f"**{len(X_raw):,}** samples · {X_raw.shape[1]} features")

    # Dataset overview
    _section(1, "Uploaded File Overview")
    if y_true is not None:
        with st.expander("Ground Truth Distribution", expanded=True):
            render_dataset_summary(y_true)
    else:
        st.info("No 'label' column — running inference only.")
    st.divider()

    # Detection
    _section(2, "Live Detection", "AE gate + CNN-GRU verdict on your file")
    col_left, col_right = st.columns(2)

    with col_left:
        with st.container(border=True):
            st.markdown("### Centralized CNN-GRU")
            if st.button("Detect (Centralized)", use_container_width=True, key="btn_live_c"):
                try:
                    models, scaler, threshold, label_map, benign_idx = load_artifacts()
                    if "_X_scaled" not in st.session_state:
                        st.session_state["_X_scaled"] = scaler.transform(X_raw).astype(np.float32)
                    with st.spinner("Running cascading pipeline…"):
                        st.session_state["_live_central"] = run_cascading(
                            st.session_state["_X_scaled"], models, threshold,
                            label_map, benign_idx, "centralized")
                except Exception as e:
                    st.error(f"Failed: {e}")
            if "_live_central" in st.session_state:
                render_verdict_banner(st.session_state["_live_central"])
                with st.expander("Gate statistics"):
                    render_gate_stats(st.session_state["_live_central"], "live_c")

    with col_right:
        with st.container(border=True):
            st.markdown("### Federated CNN-GRU")
            if st.button("Detect (Federated)", use_container_width=True, key="btn_live_f"):
                try:
                    models, scaler, threshold, label_map, benign_idx = load_artifacts()
                    if "_X_scaled" not in st.session_state:
                        st.session_state["_X_scaled"] = scaler.transform(X_raw).astype(np.float32)
                    with st.spinner("Running cascading pipeline…"):
                        st.session_state["_live_fed"] = run_cascading(
                            st.session_state["_X_scaled"], models, threshold,
                            label_map, benign_idx, "federated")
                except Exception as e:
                    st.error(f"Failed: {e}")
            if "_live_fed" in st.session_state:
                render_verdict_banner(st.session_state["_live_fed"])
                with st.expander("Gate statistics"):
                    render_gate_stats(st.session_state["_live_fed"], "live_f")

# ══════════════════════════════════════════════════════════════
# MODE 2 — EVALUATE ON TEST SET
# ══════════════════════════════════════════════════════════════
else:
    if uploaded is None:
        st.info("Upload a labelled test CSV in the sidebar to evaluate the cascading pipeline.")
        st.stop()

    X_raw, y_true = load_csv(uploaded)

    file_id = f"{uploaded.name}_{uploaded.size}"
    if st.session_state.get("_eval_file_id") != file_id:
        st.session_state["_eval_file_id"] = file_id
        for k in ("_X_scaled_eval", "_res_central", "_res_fed"):
            st.session_state.pop(k, None)

    # Dataset overview
    _section(1, "Test Set Overview")
    if y_true is not None:
        with st.expander("Ground Truth Distribution", expanded=True):
            render_dataset_summary(y_true)
    st.divider()

    # Evaluation
    _section(2, "Run Cascading Pipeline",
             "AE gate runs first — only anomalies are forwarded to CNN-GRU")
    col_left, col_right = st.columns(2)

    with col_left:
        with st.container(border=True):
            st.markdown("### Centralized CNN-GRU")
            if st.button("Evaluate (Centralized)", use_container_width=True, key="btn_eval_c"):
                try:
                    models, scaler, threshold, label_map, benign_idx = load_artifacts()
                    if "_X_scaled_eval" not in st.session_state:
                        st.session_state["_X_scaled_eval"] = scaler.transform(X_raw).astype(np.float32)
                    with st.spinner("Running…"):
                        st.session_state["_res_central"] = run_cascading(
                            st.session_state["_X_scaled_eval"], models, threshold,
                            label_map, benign_idx, "centralized")
                except Exception as e:
                    st.error(f"Failed: {e}")
            if "_res_central" in st.session_state:
                r = st.session_state["_res_central"]
                st.markdown("#### Gate Statistics")
                render_gate_stats(r, "eval_c")
                st.markdown("#### Results")
                render_final_results(r, y_true, "Centralized", "central")

    with col_right:
        with st.container(border=True):
            st.markdown("### Federated CNN-GRU")
            if st.button("Evaluate (Federated)", use_container_width=True, key="btn_eval_f"):
                try:
                    models, scaler, threshold, label_map, benign_idx = load_artifacts()
                    if "_X_scaled_eval" not in st.session_state:
                        st.session_state["_X_scaled_eval"] = scaler.transform(X_raw).astype(np.float32)
                    with st.spinner("Running…"):
                        st.session_state["_res_fed"] = run_cascading(
                            st.session_state["_X_scaled_eval"], models, threshold,
                            label_map, benign_idx, "federated")
                except Exception as e:
                    st.error(f"Failed: {e}")
            if "_res_fed" in st.session_state:
                r = st.session_state["_res_fed"]
                st.markdown("#### Gate Statistics")
                render_gate_stats(r, "eval_f")
                st.markdown("#### Results")
                render_final_results(r, y_true, "Federated", "fed")

    # Comparison
    if "_res_central" in st.session_state and "_res_fed" in st.session_state:
        st.divider()
        _section(3, "Centralized vs Federated Comparison",
                 "Both use the same AE gate — only the CNN-GRU stage differs")
        render_comparison(
            st.session_state["_res_central"],
            st.session_state["_res_fed"],
            y_true,
        )

