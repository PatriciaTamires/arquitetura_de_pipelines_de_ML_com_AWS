# tests/unit_tests.py

#PARA EXECUTAR O CÓDIGO, NO TERMINAL RODE:
# python -m venv venv
# venv\Scripts\activate
# pip install -r requirements.txt

#pytest -v
#deactivate


import math
import numpy as np
import pandas as pd
import pytest
import deeproot as dr

MODEL = "telcoChurn"
CLEAN_CSV = "telco_clean"

# =========================
# Fixtures (rodam 1x por sessão)
# =========================
@pytest.fixture(scope="session")
def df_clean():
    """Dataset limpo produzido pelo preprocess."""
    try:
        return dr.load_data(CLEAN_CSV)
    except Exception as e:
        pytest.skip(f"Sem telco_clean disponível (rode o preprocess antes). Motivo: {e}")

@pytest.fixture(scope="session")
def metrics():
    """Métricas salvas no train."""
    try:
        m = dr.load_metrics(MODEL)
        if not isinstance(m, dict):
            raise TypeError("Métricas não são dict.")
        return m
    except Exception as e:
        pytest.skip(f"Sem métricas disponíveis (rode o train antes). Motivo: {e}")

@pytest.fixture(scope="session")
def pipe():
    """Pipeline salvo (modelo final)."""
    try:
        return dr.load_model(MODEL)
    except Exception as e:
        pytest.skip(f"Sem modelo salvo (rode o train antes). Motivo: {e}")


# =========================
# Helper (lógica do register)
# =========================
def _passed_by_register_logic(metrics, acc_t=0.80, f1_t=0.80, auc_t=0.80, per_class_min=0.75):
    def to_float(x):
        try:
            return float(x)
        except (TypeError, ValueError):
            return None

    acc = to_float(metrics.get("accuracy"))
    f1_macro = to_float(metrics.get("f1_macro"))
    auc = to_float(metrics.get("auc"))

    per_class = metrics.get("per_class") or []
    f1_vals = []
    for row in per_class:
        try:
            v = float(row.get("f1"))
        except (TypeError, ValueError):
            v = None
        if v is not None and not math.isnan(v):
            f1_vals.append(v)
    min_f1 = min(f1_vals) if f1_vals else None

    passed = True
    if f1_macro is None or f1_macro < f1_t:
        passed = False
    if acc is not None and acc < acc_t:
        passed = False
    if auc is None or auc < auc_t:
        passed = False
    if (min_f1 is not None) and (min_f1 < per_class_min):
        passed = False
    return passed


# =========================
# PREPROCESS (gates pelo output)
# =========================
def test_preprocess_has_clean_output_preprocess(df_clean: pd.DataFrame):
    df = df_clean

    # alvo existe e é binário 0/1
    assert "Churn" in df.columns, "Coluna 'Churn' deveria existir no dataset limpo."
    assert set(df["Churn"].unique()) <= {0, 1}, "Churn deve ser binário {0,1}."

    # sem ids proibidos
    lowered = [c.lower() for c in df.columns]
    assert "customerid" not in lowered and "id" not in lowered, "IDs deveriam ter sido removidos."

    # sem vazamentos (qualquer coluna com 'churn', exceto o alvo)
    leaky = [c for c in df.columns if "churn" in c.lower() and c != "Churn"]
    assert not leaky, f"Leakage detectado: {leaky}"

    # TotalCharges numérica se existir
    if "TotalCharges" in df.columns:
        assert pd.api.types.is_numeric_dtype(df["TotalCharges"]), "TotalCharges deveria ser numérica."

    # numéricas sem NaN (imputação aplicada)
    num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    assert all(df[c].isna().sum() == 0 for c in num_cols), "Colunas numéricas não devem ter NaN após imputação."

def test_preprocess_strings_are_stripped_preprocess(df_clean: pd.DataFrame):
    df = df_clean
    cat_cols = [c for c in df.columns if df[c].dtype.name in ("object", "string")]
    for c in cat_cols:
        s = df[c].astype("string")
        assert (s == s.str.strip()).all(), f"Coluna '{c}' contém espaços não removidos nas bordas."

def test_preprocess_no_constant_columns_preprocess(df_clean: pd.DataFrame):
    df = df_clean
    nun = df.nunique()
    const = nun[nun <= 1].index.tolist()
    assert not const, f"Colunas sem variação deveriam ter sido removidas: {const}"


# =========================
# TRAIN (artefatos + sanidade)
# =========================
def test_train_saved_model_and_metrics_exist_train(metrics: dict):
    m = metrics
    required = {
        "accuracy",
        "f1_macro",
        "auc",
        "per_class",
        "confusion_matrix",
        "classification_report",
        "n_features",
    }
    assert required.issubset(m.keys()), f"Métricas faltando: {required - set(m.keys())}"

    # sanidade das métricas
    acc = float(m["accuracy"])
    auc = float(m["auc"])
    assert 0.0 <= acc <= 1.0, "Accuracy deve estar em [0,1]."
    assert 0.0 <= auc <= 1.0, "AUC deve estar em [0,1]."

    # per_class coerente e cm 2x2
    per_class = m["per_class"]
    assert isinstance(per_class, list) and len(per_class) >= 2, "per_class deve ter pelo menos duas classes."
    cm = np.array(m["confusion_matrix"])
    assert cm.shape == (2, 2), "Confusion matrix deve ser 2x2 para problema binário."

def test_train_pipeline_can_predict_train(pipe, df_clean: pd.DataFrame):
    X = df_clean.drop(columns=["Churn"])
    sample = X.head(5)
    proba = pipe.predict_proba(sample)[:, 1]
    pred = pipe.predict(sample)
    assert len(proba) == len(sample)
    assert len(pred) == len(sample)
    assert np.all((proba >= 0.0) & (proba <= 1.0)), "Probabilidades devem estar em [0,1]."

def test_train_pipeline_raises_on_wrong_schema_train(pipe):
    """Exemplo com pytest.raises: quebramos o schema de propósito."""
    bad = pd.DataFrame({"totally_unknown_feature": [0, 1, 2]})
    with pytest.raises(Exception):
        _ = pipe.predict(bad)


# =========================
# REGISTER (gates pela decisão)
# =========================
def test_register_decision_under_strict_thresholds_register(metrics):
    ok = _passed_by_register_logic(metrics, acc_t=0.95, f1_t=0.95, auc_t=0.95, per_class_min=0.95)
    assert ok is False, "Deveria reprovar sob thresholds extremamente altos."

def test_register_decision_under_relaxed_thresholds_register(metrics):
    ok = _passed_by_register_logic(metrics, acc_t=0.50, f1_t=0.50, auc_t=0.50, per_class_min=0.40)
    assert ok is True, "Deveria aprovar sob thresholds relaxados."
