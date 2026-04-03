# IIoT Intrusion Detection System

Binary and multi-class IDS using CNN-GRU + Autoencoder with Federated Learning (FedAvg).

---

## Quickstart

**1. Install [uv](https://docs.astral.sh/uv/getting-started/installation/)** (if not already installed)
```bash
pip install uv
```

**2. Create environment and install dependencies**
```bash
uv sync
```

> **Alternative (without uv):** Use a standard virtual environment instead:
> ```bash
> python -m venv .venv
> .venv\Scripts\activate        # Windows
> # source .venv/bin/activate   # macOS / Linux
> pip install -r requirements.txt
> ```

---

## Run

Activate the virtual environment first (if not already active):
```bash
# uv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS / Linux
```

Then run either app:
```bash
streamlit run app.py            # Binary classifier (Benign / Attack)
streamlit run app_multi_class.py  # Multi-class classifier (Benign + attack types)
```

> **Without activating**, you can also use `uv run streamlit run app.py` directly.

Then upload the test CSV from the sidebar:
- Binary → `test_samples.csv`
- Multi-class → `test_samples_mc.csv`

> **Note:** The `results/` and `results_multi_class/` folders containing trained model files must be present. These are not included in the repo by default — download them separately and place them in the project root.

---

## Structure

| Path | Description |
|------|-------------|
| `app.py` | Binary IDS app — Centralized + Federated CNN-GRU + Autoencoder |
| `app_multi_class.py` | Multi-class IDS app — same pipeline, 10+ attack types |
| `results/` | Binary model artifacts (`.keras`, `.pkl`) |
| `results_multi_class/` | Multi-class model artifacts (`.keras`, `.pkl`) |
| `notebooks/` | Training notebooks for all models |
| `extra notebooks/` | Experimental / exploratory notebooks |
| `requirements.txt` | Python dependencies |

