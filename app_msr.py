

import io
import itertools
import warnings
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import statsmodels.api as sm
from statsmodels.formula.api import ols
from scipy import stats
from scipy.optimize import minimize

warnings.filterwarnings("ignore")
st.set_page_config(page_title="MSR Pro | Diseño y Optimización", page_icon="📈", layout="wide")

st.markdown("""
<style>
.main .block-container {padding-top: 1.3rem; padding-bottom: 2rem;}
h1, h2, h3 {color:#17365D;}
div[data-testid="stMetric"] {background:#F5F8FC; padding:12px; border-radius:10px;}
</style>
""", unsafe_allow_html=True)

st.title("📈 MSR Pro — Metodología de Superficie de Respuesta")
st.caption("Diseño experimental, modelación cuadrática, diagnóstico y optimización de procesos.")
st.markdown("---")

# ---------------------------------------------------------------------
# FUNCIONES DE GENERACIÓN DE DISEÑOS
# ---------------------------------------------------------------------
def factorial_design(k):
    return np.asarray(list(itertools.product([-1.0, 1.0], repeat=k)), dtype=float)

def center_rows(k, n):
    return np.zeros((n, k), dtype=float)

def box_behnken_design(k, n_center):
    if k < 3:
        raise ValueError("Box-Behnken requiere al menos 3 factores.")
    rows = []
    for i, j in itertools.combinations(range(k), 2):
        for a, b in itertools.product([-1.0, 1.0], repeat=2):
            row = np.zeros(k)
            row[i], row[j] = a, b
            rows.append(row)
    rows.extend(center_rows(k, n_center))
    return np.asarray(rows, dtype=float)

def ccd_design(k, n_center, alpha_mode="Rotatable", placement="Aleatoria",
               replicates=1, randomize=True, blocked=False):
    fact = factorial_design(k)
    n_fact = len(fact)
    if alpha_mode == "Rotatable":
        alpha = float(n_fact ** 0.25)
    elif alpha_mode == "Face centered":
        alpha = 1.0
    elif alpha_mode == "Orthogonal":
        # Elección de alpha que hace ortogonales los contrastes cuadráticos
        # centrados entre factores para el esquema CCD indicado.
        n_total = n_fact + 2*k + n_center
        alpha_sq = (np.sqrt(n_fact*n_total) - n_fact) / 2.0
        alpha = float(np.sqrt(max(alpha_sq, 0.0)))
        if alpha == 0:
            alpha = 1.0
    else:
        alpha = float(alpha_mode)

    axial = []
    for i in range(k):
        for sign in [-1.0, 1.0]:
            row = np.zeros(k)
            row[i] = sign * alpha
            axial.append(row)
    axial = np.asarray(axial, dtype=float)
    centers = center_rows(k, n_center)

    if blocked:
        # Bloque 1: puntos factoriales. Bloque 2: axiales y centrales.
        base = np.vstack([fact, axial, centers])
        blocks = np.array([1]*len(fact) + [2]*(len(axial)+len(centers)))
    else:
        base = np.vstack([fact, axial, centers])
        blocks = np.ones(len(base), dtype=int)

    base = np.tile(base, (replicates, 1))
    blocks = np.tile(blocks, replicates)
    std_order = np.arange(1, len(base)+1)

    if placement in ["Primera", "Última", "Espaciada", "Aleatoria"]:
        n_base = len(fact) + len(axial)
        center_idx = np.where(np.all(np.isclose(base, 0), axis=1))[0]
        noncenter_idx = np.where(~np.all(np.isclose(base, 0), axis=1))[0]
        if placement == "Primera":
            order = np.concatenate([center_idx, noncenter_idx])
        elif placement == "Última":
            order = np.concatenate([noncenter_idx, center_idx])
        elif placement == "Aleatoria":
            order = np.arange(len(base))
            rng = np.random.default_rng()
            rng.shuffle(order)
        else:
            # Ubica los centros aproximadamente a intervalos iguales en la secuencia.
            order = np.empty(len(base), dtype=int)
            if len(center_idx) == 1:
                center_positions = [len(base) // 2]
            else:
                center_positions = np.linspace(0, len(base)-1, len(center_idx)+2)[1:-1].round().astype(int).tolist()
            center_set = set(center_positions)
            other_iter = iter(noncenter_idx.tolist())
            center_iter = iter(center_idx.tolist())
            for pos in range(len(base)):
                order[pos] = next(center_iter) if pos in center_set else next(other_iter)
    else:
        order = np.arange(len(base))

    base = base[order]
    blocks = blocks[order]
    std_order = std_order[order]
    if randomize and placement != "Aleatoria":
        rng = np.random.default_rng()
        center_mask = np.all(np.isclose(base, 0), axis=1)
        if blocked:
            # Aleatorización dentro de cada bloque, conservando las posiciones centrales.
            for b in np.unique(blocks):
                idx_block = np.where(blocks == b)[0]
                noncenter_slots = idx_block[~center_mask[idx_block]]
                shuffled = rng.permutation(noncenter_slots)
                base[noncenter_slots] = base[shuffled].copy()
                blocks[noncenter_slots] = blocks[shuffled].copy()
                std_order[noncenter_slots] = std_order[shuffled].copy()
        else:
            slots = np.where(~center_mask)[0]
            shuffled = rng.permutation(slots)
            base[slots] = base[shuffled].copy()
            blocks[slots] = blocks[shuffled].copy()
            std_order[slots] = std_order[shuffled].copy()

    return base, blocks, std_order, alpha

def make_design(design_type, k, n_center, alpha_mode, center_placement,
                replicates, randomize, blocked):
    if design_type == "Box-Behnken":
        base = box_behnken_design(k, n_center)
        blocks = np.ones(len(base), dtype=int)
        std_order = np.arange(1, len(base)+1)
        base = np.tile(base, (replicates, 1))
        blocks = np.tile(blocks, replicates)
        std_order = np.tile(std_order, replicates)
        center_mask = np.all(np.isclose(base, 0), axis=1)
        if center_placement == "Primera":
            order = np.concatenate([np.where(center_mask)[0], np.where(~center_mask)[0]])
        elif center_placement == "Última":
            order = np.concatenate([np.where(~center_mask)[0], np.where(center_mask)[0]])
        elif center_placement == "Aleatoria":
            order = np.random.default_rng().permutation(len(base))
        elif center_placement == "Espaciada":
            center_idx = np.where(center_mask)[0].tolist()
            other_idx = np.where(~center_mask)[0].tolist()
            ordered = list(other_idx)
            for j, idx in enumerate(center_idx, start=1):
                pos = int(round(j * (len(ordered) + 1) / (len(center_idx) + 1)))
                ordered.insert(min(max(pos, 0), len(ordered)), idx)
            order = np.asarray(ordered, dtype=int)
        else:
            order = np.arange(len(base))
        base, blocks, std_order = base[order], blocks[order], std_order[order]
        if randomize and center_placement != "Aleatoria":
            # Aleatoriza los tratamientos no centrales sin desplazar las posiciones
            # reservadas a los puntos centrales (First/Last/Spaced).
            center_mask_after = np.all(np.isclose(base, 0), axis=1)
            slots = np.where(~center_mask_after)[0]
            shuffled = np.random.default_rng().permutation(slots)
            base[slots] = base[shuffled].copy()
            blocks[slots] = blocks[shuffled].copy()
            std_order[slots] = std_order[shuffled].copy()
        return base, blocks, std_order, None
    return ccd_design(k, n_center, alpha_mode, center_placement,
                      replicates, randomize, blocked)

def coded_to_actual(x, lows, highs):
    centers = (np.asarray(lows) + np.asarray(highs)) / 2
    half_ranges = (np.asarray(highs) - np.asarray(lows)) / 2
    return centers + np.asarray(x) * half_ranges

def actual_to_coded(values, lows, highs):
    centers = (np.asarray(lows) + np.asarray(highs)) / 2
    half_ranges = (np.asarray(highs) - np.asarray(lows)) / 2
    return (np.asarray(values) - centers) / half_ranges

def quadratic_formula(response, coded_names):
    terms = list(coded_names)
    terms += [f"I({x} ** 2)" for x in coded_names]
    terms += [f"{a}:{b}" for a, b in itertools.combinations(coded_names, 2)]
    return f"Q('{response}') ~ " + " + ".join(terms)

def model_prediction(model, points, coded_names):
    frame = pd.DataFrame(points, columns=coded_names)
    return np.asarray(model.predict(frame), dtype=float)

def stationary_point(model, coded_names):
    p = model.params
    k = len(coded_names)
    b = np.array([p.get(x, 0.0) for x in coded_names], dtype=float)
    B = np.zeros((k, k), dtype=float)
    for i, x in enumerate(coded_names):
        B[i, i] = p.get(f"I({x} ** 2)", 0.0)
    for i, j in itertools.combinations(range(k), 2):
        key1, key2 = f"{coded_names[i]}:{coded_names[j]}", f"{coded_names[j]}:{coded_names[i]}"
        bij = p.get(key1, p.get(key2, 0.0))
        B[i, j] = B[j, i] = bij / 2.0
    try:
        point = np.linalg.solve(2*B, -b)
        eig = np.linalg.eigvalsh(2*B)
        if np.all(eig < -1e-9):
            kind = "Máximo local"
        elif np.all(eig > 1e-9):
            kind = "Mínimo local"
        elif np.all(np.abs(eig) <= 1e-9):
            kind = "Curvatura prácticamente nula"
        else:
            kind = "Punto silla"
        return point, kind, eig, B
    except np.linalg.LinAlgError:
        return None, "No existe solución única (Hessiana singular)", None, B

# ---------------------------------------------------------------------
# SIDEBAR: CONFIGURACIÓN GENERAL
# ---------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Configuración del estudio")
    study_name = st.text_input("Nombre del estudio", "Estudio de Superficie de Respuesta")
    st.markdown("### Clase de diseño")
    design_class = st.selectbox("Seleccionar clase", ["Superficie de Respuesta"])
    st.caption("Esta versión desarrolla la clase Response Surface del asistente de Statgraphics.")
    design_type = st.selectbox("Diseño de superficie", ["Diseño Central Compuesto (DCC)", "Box-Behnken"])
    k = st.number_input("Número de factores experimentales", min_value=2, max_value=8,
                        value=3 if design_type == "Box-Behnken" else 3, step=1)
    if design_type == "Box-Behnken" and k < 3:
        st.warning("Box-Behnken requiere k ≥ 3.")
    n_center_default = 3 if design_type == "Box-Behnken" else 2
    n_center = st.number_input("Número de puntos centrales", min_value=1, max_value=100, value=n_center_default)
    replicates = st.number_input("Réplicas del diseño completo", min_value=1, max_value=10, value=1)
    n_responses = st.number_input("Número de variables respuesta", min_value=1, max_value=5, value=1)
    center_placement = st.selectbox("Ubicación de puntos centrales",
                                    ["Aleatoria", "Espaciada", "Primera", "Última"])
    randomize = st.checkbox("Aleatorizar orden de ejecución", value=True)
    blocked = st.checkbox("Separar diseño en bloques (factorial / axial-central)", value=False,
                          disabled=(design_type != "Diseño Central Compuesto (DCC)"))
    alpha_mode = "Rotatable"
    if design_type == "Diseño Central Compuesto (DCC)":
        alpha_mode = st.selectbox("Características del DCC",
                                  ["Rotatable", "Orthogonal", "Face centered"])
    response_names = []
    response_units_list = []
    for ri in range(int(n_responses)):
        response_names.append(st.text_input(f"Nombre de la respuesta {ri+1}", "Y" if ri == 0 else f"Y{ri+1}", key=f"response_name_{ri}"))
        response_units_list.append(st.text_input(f"Unidades de {response_names[-1]}", "", key=f"response_unit_{ri}"))
    response_name = response_names[0] if response_names else "Y"
    response_units = response_units_list[0] if response_units_list else ""
    st.caption("La codificación utiliza −1 y +1 como límites factoriales; los puntos axiales pueden extenderse más allá de ese intervalo.")

# ---------------------------------------------------------------------
# DEFINICIÓN DE FACTORES Y GENERACIÓN
# ---------------------------------------------------------------------
st.header("1. Configuración y creación del diseño experimental")
with st.expander("Definir factores experimentales", expanded=True):
    factor_defs = []
    cols = st.columns(min(k, 4))
    for i in range(k):
        with cols[i % len(cols)]:
            st.markdown(f"**Factor {i+1}**")
            default_name = f"Factor_{chr(65+i)}"
            fname = st.text_input("Nombre", default_name, key=f"factor_name_{i}")
            low = st.number_input("Nivel bajo", value=-1.0, key=f"factor_low_{i}", format="%.6g")
            high = st.number_input("Nivel alto", value=1.0, key=f"factor_high_{i}", format="%.6g")
            unit = st.text_input("Unidad", "", key=f"factor_unit_{i}")
            factor_defs.append({"name": fname.strip() or default_name, "low": float(low),
                                "high": float(high), "unit": unit})
    invalid = [f["name"] for f in factor_defs if f["high"] <= f["low"]]
    if len({f["name"] for f in factor_defs}) != len(factor_defs):
        st.error("Los nombres de los factores deben ser únicos.")
    if invalid:
        st.error("El nivel alto debe ser mayor que el nivel bajo para: " + ", ".join(invalid))

c1, c2, c3 = st.columns(3)
with c1:
    st.info("**DCC:** factorial + puntos axiales + puntos centrales.")
with c2:
    st.info("**Box-Behnken:** combinaciones por pares en niveles −1 y +1, con los demás factores en cero.")
with c3:
    st.info("**Codificación:** xᵢ = (Xᵢ − centro) / semirango.")

if st.button("🧪 Generar diseño experimental", type="primary", disabled=bool(invalid) or len({f["name"] for f in factor_defs}) != len(factor_defs)):
    try:
        coded, blocks, std_order, alpha = make_design(
            design_type, int(k), int(n_center), alpha_mode, center_placement,
            int(replicates), randomize, blocked
        )
        names = [f["name"] for f in factor_defs]
        lows = [f["low"] for f in factor_defs]
        highs = [f["high"] for f in factor_defs]
        actual = coded_to_actual(coded, lows, highs)
        d = pd.DataFrame({"Run": np.arange(1, len(coded)+1),
                          "StdOrder": std_order, "Block": blocks})
        for j, name in enumerate(names):
            d[name] = actual[:, j]
        for j in range(k):
            d[f"x{j+1}"] = coded[:, j]
        if len(set(response_names)) != len(response_names) or any(not x.strip() for x in response_names):
            st.error("Los nombres de las variables respuesta deben ser únicos y no estar vacíos.")
            st.stop()
        for rname in response_names:
            if rname in d.columns:
                st.error(f"El nombre de respuesta '{rname}' coincide con una columna del diseño.")
                st.stop()
            d[rname] = np.nan
        st.session_state["msr_design"] = d
        st.session_state["msr_factor_names"] = names
        st.session_state["msr_response"] = response_name
        st.session_state["msr_response_names"] = response_names
        st.session_state["msr_units"] = response_units
        st.session_state["msr_lows"] = lows
        st.session_state["msr_highs"] = highs
        st.session_state["msr_coded_names"] = [f"x{i+1}" for i in range(k)]
        st.session_state["msr_design_type"] = design_type
        st.session_state["msr_alpha"] = alpha
        st.session_state["msr_study"] = study_name
        if alpha is not None:
            st.info(f"DCC generado con distancia axial α = {alpha:.5f}.")
        st.success(f"Diseño generado: {len(d)} corridas, {k} factores y {n_center} puntos centrales por réplica.")
        st.rerun()
    except Exception as e:
        st.error(f"No fue posible generar el diseño: {e}")

# ---------------------------------------------------------------------
# IMPORTAR DATOS O EDITAR MATRIZ
# ---------------------------------------------------------------------
st.header("2. Matriz experimental y captura de respuestas")
data_method = st.radio("Origen de los datos", ["Diseño generado / edición manual", "Cargar archivo CSV"], horizontal=True)
df_data = None

if data_method == "Diseño generado / edición manual":
    if "msr_design" in st.session_state:
        base_df = st.session_state["msr_design"].copy()
        resp = st.session_state["msr_response"]
        st.caption("Introduce las observaciones en la columna de respuesta. Las columnas x1, x2, … contienen los factores codificados.")
        edited = st.data_editor(base_df, num_rows="fixed", use_container_width=True, key="msr_editor")
        df_data = edited
        st.session_state["msr_design"] = edited
    else:
        st.info("Define los factores y genera un diseño para comenzar, o carga un CSV.")
else:
    uploaded = st.file_uploader("Selecciona un archivo CSV", type=["csv"])
    if uploaded is not None:
        try:
            df_data = pd.read_csv(uploaded)
            st.dataframe(df_data.head(10), use_container_width=True)
        except Exception as e:
            st.error(f"Error al leer CSV: {e}")

# ---------------------------------------------------------------------
# ANÁLISIS DE PRIMER ORDEN / ASCENSO RÁPIDO
# ---------------------------------------------------------------------
with st.expander("🚀 Fase 1 — Modelo de primer orden y ascenso más rápido", expanded=False):
    st.markdown("Ajuste exploratorio de primer orden. Selecciona los factores y una respuesta numérica.")
    if df_data is not None:
        numeric_cols = df_data.select_dtypes(include=np.number).columns.tolist()
        excluded = ["Run", "StdOrder", "Block"]
        candidates = [c for c in numeric_cols if c not in excluded and not c.startswith("x")]
        if "msr_factor_names" in st.session_state:
            f1_names = [n for n in st.session_state["msr_factor_names"] if n in df_data.columns]
        else:
            f1_names = candidates[:-1] if len(candidates) > 1 else candidates
        f1_factors = st.multiselect("Factores del modelo lineal", candidates, default=f1_names, key="f1_select")
        f1_response_options = [c for c in numeric_cols if c not in f1_factors and c not in excluded]
        if f1_response_options and f1_factors:
            f1_y = st.selectbox("Respuesta para primer orden", f1_response_options, key="f1_y")
            if st.button("Ajustar modelo de primer orden", key="fit_first"):
                temp = df_data[f1_factors + [f1_y]].apply(pd.to_numeric, errors="coerce").dropna()
                if len(temp) <= len(f1_factors)+1:
                    st.error("No hay observaciones suficientes para estimar el modelo.")
                else:
                    m1 = ols(f"Q('{f1_y}') ~ " + " + ".join([f"Q('{x}')" for x in f1_factors]), data=temp).fit()
                    st.write(m1.summary())
                    b = m1.params.iloc[1:]
                    norm = np.linalg.norm(b.values)
                    if norm > 0:
                        step = st.number_input("Tamaño del paso en unidades codificadas", min_value=0.01, value=0.5, key="f1_step")
                        direction = b.values / norm
                        path = pd.DataFrame([{"Paso": i, **{f"Δ {f}": float(step*i*direction[j]) for j, f in enumerate(f1_factors)}} for i in range(1, 6)])
                        st.dataframe(path, use_container_width=True)

# ---------------------------------------------------------------------
# ANÁLISIS DE SEGUNDO ORDEN
# ---------------------------------------------------------------------
st.header("3. Análisis de superficie de respuesta — Modelo de segundo orden")
if df_data is not None:
    if "msr_factor_names" in st.session_state and data_method == "Diseño generado / edición manual":
        factor_names = st.session_state["msr_factor_names"]
        coded_names = st.session_state["msr_coded_names"]
        response_default = st.session_state["msr_response"]
        lows = st.session_state["msr_lows"]
        highs = st.session_state["msr_highs"]
        # Recalcular codificación desde los valores reales editados, para mantener consistencia.
        for i, fname in enumerate(factor_names):
            if fname in df_data.columns:
                df_data[coded_names[i]] = actual_to_coded(
                    pd.to_numeric(df_data[fname], errors="coerce").to_numpy(), lows[i], highs[i]
                )
        response_options = [r for r in st.session_state.get("msr_response_names", [response_default]) if r in df_data.columns]
    else:
        numeric = df_data.select_dtypes(include=np.number).columns.tolist()
        factor_names = st.multiselect("Selecciona los factores (escala natural)", numeric, key="csv_factors")
        response_options = [c for c in numeric if c not in factor_names]
        coded_names = [f"x{i+1}" for i in range(len(factor_names))]
        lows, highs = [], []
        if factor_names:
            for name in factor_names:
                lo, hi = float(df_data[name].min()), float(df_data[name].max())
                lows.append(lo); highs.append(hi)
            for i, name in enumerate(factor_names):
                df_data[coded_names[i]] = actual_to_coded(pd.to_numeric(df_data[name], errors="coerce"), lows[i], highs[i])

    if factor_names and response_options:
        y_choice = st.selectbox("Variable respuesta a modelar", response_options, key="msr_y_choice")
        if st.button("📐 Ajustar modelo cuadrático completo", type="primary", key="fit_quad"):
            try:
                required = coded_names + [y_choice]
                work = df_data[required].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
                k_model = len(coded_names)
                n_terms = 1 + 2*k_model + k_model*(k_model-1)//2
                if len(work) <= n_terms:
                    st.error(f"Se requieren más observaciones que parámetros ({n_terms}); hay {len(work)} observaciones válidas.")
                else:
                    formula = quadratic_formula(y_choice, coded_names)
                    model = ols(formula, data=work).fit()
                    st.session_state["msr_model"] = model
                    st.session_state["msr_work"] = work
                    st.session_state["msr_formula"] = formula
                    st.session_state["msr_coded_names_fit"] = coded_names
                    st.session_state["msr_factor_names_fit"] = factor_names
                    st.session_state["msr_response_fit"] = y_choice
                    st.session_state["msr_low_fit"] = lows
                    st.session_state["msr_high_fit"] = highs
                    st.success("Modelo cuadrático ajustado correctamente.")
            except Exception as e:
                st.error(f"Error al ajustar el modelo: {e}")

if "msr_model" in st.session_state:
    model = st.session_state["msr_model"]
    work = st.session_state["msr_work"]
    coded_names = st.session_state["msr_coded_names_fit"]
    factor_names = st.session_state["msr_factor_names_fit"]
    y_choice = st.session_state["msr_response_fit"]
    lows, highs = st.session_state["msr_low_fit"], st.session_state["msr_high_fit"]
    k_model = len(coded_names)

    st.markdown("### 3.1 Resumen del modelo")
    met = st.columns(5)
    met[0].metric("Corridas válidas", len(work))
    met[1].metric("R²", f"{model.rsquared:.4f}")
    met[2].metric("R² ajustado", f"{model.rsquared_adj:.4f}")
    met[3].metric("AIC", f"{model.aic:.3f}")
    met[4].metric("Error estándar", f"{np.sqrt(model.mse_resid):.4g}")

    with st.expander("Ecuación estimada del modelo", expanded=True):
        st.code(model.model.formula, language="text")
        coef = pd.DataFrame({"Coeficiente": model.params, "Error estándar": model.bse,
                             "t": model.tvalues, "Valor-p": model.pvalues})
        st.dataframe(coef.style.format(precision=5), use_container_width=True)

    # ANOVA tipo II y prueba de falta de ajuste
    st.markdown("### 3.2 ANOVA del modelo cuadrático")
    try:
        anova = sm.stats.anova_lm(model, typ=2)
        anova = anova.rename(columns={"sum_sq":"Suma de cuadrados","df":"GL","F":"Razón-F","PR(>F)":"Valor-p"})
        st.dataframe(anova.style.format(precision=5), use_container_width=True)
    except Exception as e:
        st.warning(f"No se pudo construir la tabla ANOVA: {e}")

    st.markdown("### 3.3 Error puro y falta de ajuste")
    # Repeticiones exactas de las mismas combinaciones codificadas.
    grouped = work.groupby(coded_names, dropna=False)[y_choice]
    ss_pure = float(grouped.apply(lambda s: ((s-s.mean())**2).sum()).sum())
    n_unique = work[coded_names].drop_duplicates().shape[0]
    df_pure = len(work) - n_unique
    ss_resid = float(np.sum(model.resid**2))
    df_resid = int(model.df_resid)
    ss_lof = max(0.0, ss_resid - ss_pure)
    df_lof = df_resid - df_pure
    ms_pure = ss_pure/df_pure if df_pure > 0 else np.nan
    ms_lof = ss_lof/df_lof if df_lof > 0 else np.nan
    f_lof = ms_lof/ms_pure if df_pure > 0 and ms_pure > 0 and df_lof > 0 else np.nan
    p_lof = stats.f.sf(f_lof, df_lof, df_pure) if np.isfinite(f_lof) else np.nan
    lof_table = pd.DataFrame([
        ["Error residual", ss_resid, df_resid, ss_resid/df_resid if df_resid>0 else np.nan],
        ["Error puro", ss_pure, df_pure, ms_pure],
        ["Falta de ajuste", ss_lof, df_lof, ms_lof],
    ], columns=["Fuente","Suma de cuadrados","GL","Cuadrado medio"])
    st.dataframe(lof_table.style.format(precision=5), use_container_width=True)
    if np.isfinite(p_lof):
        st.write(f"**Prueba de falta de ajuste:** F = {f_lof:.4f}; p = {p_lof:.5f}.")
        if p_lof < 0.05:
            st.warning("La falta de ajuste es significativa al 5%; el modelo cuadrático puede no describir adecuadamente la respuesta.")
        else:
            st.success("No se detecta falta de ajuste significativa al 5%.")
    else:
        st.info("No hay grados de libertad de error puro suficientes. Se necesitan réplicas en puntos de diseño para probar falta de ajuste.")

    # Diagnósticos
    st.markdown("### 3.4 Diagnóstico del modelo")
    fitted = np.asarray(model.fittedvalues)
    resid = np.asarray(model.resid)
    d1, d2, d3, d4 = st.tabs(["Residuos vs. predichos", "Gráfico normal Q-Q", "Observados vs. predichos", "Distribución de residuos"])
    with d1:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=fitted, y=resid, mode="markers", name="Residuos"))
        fig.add_hline(y=0, line_dash="dash")
        fig.update_layout(xaxis_title="Valores predichos", yaxis_title="Residuos", height=430)
        st.plotly_chart(fig, use_container_width=True)
    with d2:
        (osm, osr), (slope, intercept, r) = stats.probplot(resid, dist="norm")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=osm, y=osr, mode="markers", name="Residuos"))
        fig.add_trace(go.Scatter(x=osm, y=intercept+slope*np.asarray(osm), mode="lines", name="Referencia"))
        fig.update_layout(xaxis_title="Cuantiles teóricos", yaxis_title="Residuos ordenados", height=430)
        st.plotly_chart(fig, use_container_width=True)
    with d3:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=work[y_choice], y=fitted, mode="markers", name="Predicción"))
        lo = min(float(work[y_choice].min()), float(np.min(fitted)))
        hi = max(float(work[y_choice].max()), float(np.max(fitted)))
        fig.add_trace(go.Scatter(x=[lo,hi], y=[lo,hi], mode="lines", name="Identidad"))
        fig.update_layout(xaxis_title="Observados", yaxis_title="Predichos", height=430)
        st.plotly_chart(fig, use_container_width=True)
    with d4:
        fig = px.histogram(x=resid, nbins=max(8, int(np.sqrt(len(resid)))), labels={"x":"Residuo"})
        fig.update_layout(height=430)
        st.plotly_chart(fig, use_container_width=True)
    if len(resid) >= 3:
        shapiro = stats.shapiro(resid[:5000])
        st.caption(f"Shapiro–Wilk para residuos: W = {shapiro.statistic:.4f}; p = {shapiro.pvalue:.5f}.")

    # Punto estacionario y optimización restringida
    st.markdown("### 3.5 Punto estacionario y optimización")
    point, kind, eig, B = stationary_point(model, coded_names)
    if point is not None:
        pred_stationary = float(model_prediction(model, [point], coded_names)[0])
        stationary_df = pd.DataFrame({
            "Factor": factor_names,
            "Coordenada codificada": point,
            "Coordenada real": coded_to_actual(point, lows, highs)
        })
        st.write(f"**Clasificación:** {kind}")
        st.dataframe(stationary_df.style.format(precision=5), use_container_width=True)
        st.write(f"**Respuesta estimada en el punto estacionario:** {pred_stationary:.6g} {st.session_state.get('msr_units','')}")
        st.caption("Autovalores de la Hessiana: " + ", ".join(f"{v:.5g}" for v in eig))
        if np.any(np.abs(point) > 1):
            st.warning("El punto estacionario está fuera de la región factorial codificada [−1, +1] en al menos un factor. Verifica si sigue dentro de la región experimental del diseño.")
    else:
        st.warning(kind)

    objective = st.selectbox("Criterio de optimización", ["Maximizar respuesta", "Minimizar respuesta", "Aproximar a un objetivo"])
    target = None
    if objective == "Aproximar a un objetivo":
        target = st.number_input("Valor objetivo de Y", value=float(work[y_choice].mean()))
    bounds = [(float(work[x].min()), float(work[x].max())) for x in coded_names]
    def objective_fn(x):
        yhat = float(model_prediction(model, [x], coded_names)[0])
        if objective == "Maximizar respuesta":
            return -yhat
        if objective == "Minimizar respuesta":
            return yhat
        return (yhat-float(target))**2
    opt = minimize(objective_fn, np.zeros(k_model), method="L-BFGS-B", bounds=bounds)
    if opt.success:
        opt_y = float(model_prediction(model, [opt.x], coded_names)[0])
        opt_df = pd.DataFrame({"Factor":factor_names, "Codificado":opt.x,
                               "Valor real":coded_to_actual(opt.x,lows,highs)})
        st.write(f"**Resultado de optimización restringida:** Y predicha = {opt_y:.6g}")
        st.dataframe(opt_df.style.format(precision=5), use_container_width=True)
    else:
        st.warning("El optimizador no encontró una solución convergente: " + str(opt.message))

    # Superficie y contorno para dos factores seleccionados, manteniendo los demás fijos.
    st.markdown("### 3.6 Superficie de respuesta y curvas de contorno")
    if k_model >= 2:
        pair_cols = st.columns(2)
        with pair_cols[0]:
            factor_x = st.selectbox("Factor del eje X", factor_names, index=0, key="surface_x")
        with pair_cols[1]:
            factor_y = st.selectbox("Factor del eje Y", [f for f in factor_names if f != factor_x],
                                    index=0, key="surface_y")
        ix, iy = factor_names.index(factor_x), factor_names.index(factor_y)
        fixed = np.zeros(k_model)
        other_indices = [i for i in range(k_model) if i not in [ix, iy]]
        if other_indices:
            st.caption("Los demás factores se mantienen en los siguientes valores codificados:")
            fixed_cols = st.columns(min(len(other_indices), 4))
            for n, idx in enumerate(other_indices):
                with fixed_cols[n % len(fixed_cols)]:
                    fixed[idx] = st.slider(f"{factor_names[idx]} (codificado)", 
                                           float(work[coded_names[idx]].min()),
                                           float(work[coded_names[idx]].max()), 0.0,
                                           key=f"fixed_{idx}")
        xvals = np.linspace(float(work[coded_names[ix]].min()), float(work[coded_names[ix]].max()), 60)
        yvals = np.linspace(float(work[coded_names[iy]].min()), float(work[coded_names[iy]].max()), 60)
        XX, YY = np.meshgrid(xvals, yvals)
        grid = np.tile(fixed, (XX.size, 1))
        grid[:, ix], grid[:, iy] = XX.ravel(), YY.ravel()
        ZZ = model_prediction(model, grid, coded_names).reshape(XX.shape)
        contour_tab, surface_tab = st.tabs(["Mapa de contornos", "Superficie tridimensional"])
        with contour_tab:
            figc = go.Figure(go.Contour(x=xvals, y=yvals, z=ZZ, colorscale="Viridis",
                                        contours={"showlabels":True, "labelfont":{"size":11}},
                                        colorbar={"title":y_choice}))
            figc.add_trace(go.Scatter(x=work[coded_names[ix]], y=work[coded_names[iy]],
                                     mode="markers", marker={"color":"black","size":7,"symbol":"x"},
                                     name="Puntos experimentales"))
            figc.update_layout(xaxis_title=f"{factor_x} (codificado)", yaxis_title=f"{factor_y} (codificado)", height=560)
            st.plotly_chart(figc, use_container_width=True)
        with surface_tab:
            figs = go.Figure(go.Surface(x=xvals, y=yvals, z=ZZ, colorscale="Viridis",
                                        colorbar={"title":y_choice}, opacity=0.92))
            figs.update_layout(scene={"xaxis_title":f"{factor_x} (codificado)",
                                      "yaxis_title":f"{factor_y} (codificado)",
                                      "zaxis_title":y_choice}, height=650)
            st.plotly_chart(figs, use_container_width=True)
    else:
        st.info("Se requieren al menos dos factores para representar una superficie.")

    # Exportación de resultados
    st.markdown("### 3.7 Exportar resultados")
    export_df = work.copy()
    export_df["Predicho"] = model.fittedvalues
    export_df["Residuo"] = model.resid
    e1, e2 = st.columns(2)
    with e1:
        st.download_button("⬇️ Descargar datos y predicciones (CSV)",
                           export_df.to_csv(index=False).encode("utf-8-sig"),
                           file_name="MSR_datos_predicciones.csv", mime="text/csv")
    with e2:
        summary_text = model.summary().as_text()
        st.download_button("⬇️ Descargar resumen del modelo (TXT)",
                           summary_text.encode("utf-8"),
                           file_name="MSR_resumen_modelo.txt", mime="text/plain")


# ---------------------------------------------------------------------
# OPTIMIZACIÓN SIMULTÁNEA DE MÚLTIPLES RESPUESTAS — DESEABILIDAD
# ---------------------------------------------------------------------
    st.markdown("---")
    st.header("4. Optimización simultánea de múltiples respuestas")
    st.markdown(
        "Ajusta un modelo cuadrático para cada respuesta y optimiza la "
        "deseabilidad global de Derringer–Suich. La optimización busca una "
        "combinación común de factores para todas las respuestas seleccionadas."
    )

    if factor_names and response_options:
        multi_responses = st.multiselect(
            "Respuestas que participarán en la optimización",
            response_options,
            default=response_options,
            key="msr_multi_responses"
        )

        if len(multi_responses) >= 2:
            multi_models = {}
            multi_data = {}
            model_rows = []
            n_terms_multi = 1 + 2*len(coded_names) + len(coded_names)*(len(coded_names)-1)//2

            for resp_name in multi_responses:
                needed = coded_names + [resp_name]
                temp_multi = df_data[needed].apply(
                    pd.to_numeric, errors="coerce"
                ).replace([np.inf, -np.inf], np.nan).dropna()

                if len(temp_multi) <= n_terms_multi:
                    st.error(
                        f"{resp_name}: se requieren más de {n_terms_multi} "
                        f"observaciones válidas para estimar el modelo cuadrático."
                    )
                    continue

                try:
                    formula_multi = quadratic_formula(resp_name, coded_names)
                    fitted_multi = ols(formula_multi, data=temp_multi).fit()
                    if np.linalg.matrix_rank(fitted_multi.model.exog) < fitted_multi.model.exog.shape[1]:
                        st.warning(
                            f"{resp_name}: la matriz del modelo no tiene rango completo. "
                            "Revise el diseño o las corridas disponibles."
                        )
                        continue
                    multi_models[resp_name] = fitted_multi
                    multi_data[resp_name] = temp_multi
                    model_rows.append({
                        "Respuesta": resp_name,
                        "Corridas": int(fitted_multi.nobs),
                        "R²": fitted_multi.rsquared,
                        "R² ajustado": fitted_multi.rsquared_adj,
                        "RMSE": float(np.sqrt(fitted_multi.mse_resid)),
                        "AIC": fitted_multi.aic
                    })
                except Exception as err_multi:
                    st.warning(f"No fue posible ajustar {resp_name}: {err_multi}")

            if model_rows:
                st.markdown("#### 4.1 Ajuste de modelos por respuesta")
                st.dataframe(
                    pd.DataFrame(model_rows).style.format({
                        "R²": "{:.5f}", "R² ajustado": "{:.5f}",
                        "RMSE": "{:.5g}", "AIC": "{:.3f}"
                    }),
                    use_container_width=True
                )

            if len(multi_models) >= 2:
                st.markdown("#### 4.2 Especificación de metas y deseabilidades")
                st.caption(
                    "Límite bajo y alto delimitan la región aceptable. "
                    "Peso controla la forma de la deseabilidad individual; "
                    "Importancia controla la contribución de cada respuesta "
                    "a la deseabilidad global."
                )

                desirability_specs = {}
                for ri, resp_name in enumerate(multi_models):
                    y_obs = pd.to_numeric(df_data[resp_name], errors="coerce").dropna()
                    default_low = float(y_obs.min())
                    default_high = float(y_obs.max())
                    target_reference = {"Y1": 170.0, "Y2": 1300.0, "Y3": 450.0, "Y4": 67.5}
                    default_target = float(target_reference.get(resp_name, y_obs.mean()))

                    with st.expander(f"Configuración de {resp_name}", expanded=True):
                        c_goal, c_low, c_high, c_target, c_weight, c_importance = st.columns(6)
                        with c_goal:
                            goal = st.selectbox(
                                "Objetivo",
                                ["Hit (alcanzar meta)", "Maximizar", "Minimizar"],
                                index=0, key=f"msr_goal_{ri}"
                            )
                        with c_low:
                            low_lim = st.number_input(
                                "Límite bajo", value=default_low,
                                key=f"msr_low_{ri}", format="%.6g"
                            )
                        with c_high:
                            high_lim = st.number_input(
                                "Límite alto", value=default_high,
                                key=f"msr_high_{ri}", format="%.6g"
                            )
                        with c_target:
                            target_val = st.number_input(
                                "Objetivo", value=default_target,
                                key=f"msr_target_{ri}", format="%.6g"
                            )
                        with c_weight:
                            shape_weight = st.number_input(
                                "Peso", min_value=0.01, value=1.0,
                                key=f"msr_weight_{ri}", format="%.3f"
                            )
                        with c_importance:
                            importance = st.number_input(
                                "Importancia", min_value=0.01, value=3.0,
                                key=f"msr_importance_{ri}", format="%.3f"
                            )

                    if high_lim <= low_lim:
                        st.error(f"{resp_name}: el límite alto debe ser mayor que el límite bajo.")
                    elif goal.startswith("Hit") and not (low_lim < target_val < high_lim):
                        st.error(
                            f"{resp_name}: para el objetivo Hit, la meta debe estar "
                            "estrictamente entre el límite bajo y el límite alto."
                        )
                    else:
                        desirability_specs[resp_name] = {
                            "goal": goal, "low": float(low_lim), "high": float(high_lim),
                            "target": float(target_val), "weight": float(shape_weight),
                            "importance": float(importance)
                        }

                def individual_desirability(y_value, spec):
                    """Deseabilidad Derringer–Suich, acotada entre 0 y 1."""
                    y_value = np.asarray(y_value, dtype=float)
                    lo, hi = spec["low"], spec["high"]
                    exponent = spec["weight"]
                    goal = spec["goal"]

                    if goal == "Maximizar":
                        raw = (y_value - lo) / (hi - lo)
                        dval = np.clip(raw, 0.0, 1.0) ** exponent
                    elif goal == "Minimizar":
                        raw = (hi - y_value) / (hi - lo)
                        dval = np.clip(raw, 0.0, 1.0) ** exponent
                    else:
                        target_value = spec["target"]
                        dval = np.zeros_like(y_value, dtype=float)
                        left = y_value <= target_value
                        right = ~left
                        dval[left] = np.clip(
                            (y_value[left] - lo) / (target_value - lo), 0.0, 1.0
                        ) ** exponent
                        dval[right] = np.clip(
                            (hi - y_value[right]) / (hi - target_value), 0.0, 1.0
                        ) ** exponent
                    return dval

                if len(desirability_specs) == len(multi_models):
                    st.markdown("#### 4.3 Optimización de la deseabilidad global")
                    valid_rows = df_data[coded_names].apply(
                        pd.to_numeric, errors="coerce"
                    ).replace([np.inf, -np.inf], np.nan).dropna()
                    bounds_multi = [
                        (float(valid_rows[col].min()), float(valid_rows[col].max()))
                        for col in coded_names
                    ]

                    response_order = list(multi_models.keys())
                    importance_total = sum(
                        desirability_specs[r]["importance"] for r in response_order
                    )

                    # Precalcular coeficientes. Evita llamar model.predict() de
                    # statsmodels (que construye DataFrames) miles de veces.
                    quadratic_terms = (
                        list(coded_names)
                        + [f"I({x} ** 2)" for x in coded_names]
                        + [f"{a}:{b}" for a, b in itertools.combinations(coded_names, 2)]
                    )
                    coefficient_vectors = {}
                    for r in response_order:
                        params = multi_models[r].params
                        coefficient_vectors[r] = np.array(
                            [float(params.get("Intercept", 0.0))]
                            + [float(params.get(term, 0.0)) for term in quadratic_terms],
                            dtype=float
                        )

                    def fast_predict_one(point_coded, response):
                        x = np.asarray(point_coded, dtype=float)
                        features = np.concatenate((
                            [1.0],
                            x,
                            x ** 2,
                            np.array([x[i] * x[j] for i, j in itertools.combinations(range(len(x)), 2)])
                        ))
                        return float(features @ coefficient_vectors[response])

                    def evaluate_multi(point_coded):
                        predicted = {
                            r: fast_predict_one(point_coded, r) for r in response_order
                        }
                        d_individual = {
                            r: float(individual_desirability(
                                np.array([predicted[r]]), desirability_specs[r]
                            )[0])
                            for r in response_order
                        }
                        if any(v <= 0 for v in d_individual.values()):
                            global_d = 0.0
                        else:
                            global_d = float(np.exp(sum(
                                desirability_specs[r]["importance"] * np.log(d_individual[r])
                                for r in response_order
                            ) / importance_total))
                        return global_d, predicted, d_individual

                    def objective_multi(point_coded):
                        return -evaluate_multi(point_coded)[0]

                    # Optimización rápida con pocos puntos iniciales informativos:
                    # centro del dominio y las dos mejores corridas observadas.
                    starts = [np.array([(a+b)/2 for a, b in bounds_multi], dtype=float)]
                    observed_candidates = valid_rows[coded_names].to_numpy(dtype=float)
                    if len(observed_candidates):
                        scored = sorted(
                            ((evaluate_multi(row)[0], row) for row in observed_candidates),
                            key=lambda z: z[0], reverse=True
                        )
                        for _, row in scored[:2]:
                            starts.append(row.copy())

                    best_result = None
                    with st.spinner("Calculando la deseabilidad global…"):
                        for start_point in starts:
                            try:
                                candidate = minimize(
                                    objective_multi, start_point, method="L-BFGS-B",
                                    bounds=bounds_multi,
                                    options={"maxiter": 500, "ftol": 1e-9}
                                )
                                candidate_d = evaluate_multi(candidate.x)[0]
                                if best_result is None or candidate_d > best_result[0]:
                                    best_result = (candidate_d, candidate.x.copy(), candidate)
                            except Exception:
                                continue

                    if best_result is not None:
                        global_d, optimum_coded, optimizer_result = best_result
                        _, predicted_opt, individual_opt = evaluate_multi(optimum_coded)
                        optimum_actual = coded_to_actual(optimum_coded, lows, highs)

                        st.metric("Deseabilidad global óptima (D)", f"{global_d:.6f}")
                        factor_result = pd.DataFrame({
                            "Factor": factor_names,
                            "Óptimo codificado": optimum_coded,
                            "Óptimo en unidades reales": optimum_actual,
                            "Límite codificado inferior": [b[0] for b in bounds_multi],
                            "Límite codificado superior": [b[1] for b in bounds_multi]
                        })
                        response_result = pd.DataFrame([
                            {
                                "Respuesta": r,
                                "Objetivo": desirability_specs[r]["goal"],
                                "Predicción óptima": predicted_opt[r],
                                "Deseabilidad individual": individual_opt[r],
                                "Importancia": desirability_specs[r]["importance"],
                                "Peso": desirability_specs[r]["weight"]
                            } for r in response_order
                        ])

                        result_left, result_right = st.columns(2)
                        with result_left:
                            st.markdown("**Factores óptimos comunes**")
                            st.dataframe(factor_result.style.format(precision=6), use_container_width=True)
                        with result_right:
                            st.markdown("**Respuestas predichas y deseabilidades**")
                            st.dataframe(response_result.style.format(precision=6), use_container_width=True)

                        st.caption(
                            "La deseabilidad global es la media geométrica ponderada: "
                            "D = [∏ dᵢ^(rᵢ)]^(1/Σrᵢ), donde rᵢ representa la importancia."
                        )

                        # Tabla de configuración completa, útil para reproducibilidad.
                        config_result = pd.DataFrame([
                            {"Respuesta": r, **desirability_specs[r]}
                            for r in response_order
                        ])
                        with st.expander("Ver configuración completa de deseabilidad"):
                            st.dataframe(config_result, use_container_width=True)

                        # Gráfico de deseabilidad individual en el óptimo.
                        fig_d = go.Figure()
                        fig_d.add_trace(go.Bar(
                            x=response_order,
                            y=[individual_opt[r] for r in response_order],
                            text=[f"{individual_opt[r]:.3f}" for r in response_order],
                            textposition="outside",
                            name="Deseabilidad individual"
                        ))
                        fig_d.update_layout(
                            title="Deseabilidad individual en la solución óptima",
                            yaxis_title="Deseabilidad (0–1)",
                            yaxis=dict(range=[0, 1.12]),
                            height=420
                        )
                        st.plotly_chart(fig_d, use_container_width=True)

                        # Overlay de contornos de deseabilidad para dos factores.
                        if len(coded_names) >= 2:
                            st.markdown("#### 4.4 Gráfico de superposición (Overlay Plot)")
                            ov1, ov2 = st.columns(2)
                            with ov1:
                                ov_x = st.selectbox(
                                    "Factor del eje X (Overlay)",
                                    factor_names, index=0, key="msr_overlay_x"
                                )
                            with ov2:
                                ov_y = st.selectbox(
                                    "Factor del eje Y (Overlay)",
                                    [f for f in factor_names if f != ov_x],
                                    index=0, key="msr_overlay_y"
                                )

                            ox = factor_names.index(ov_x)
                            oy = factor_names.index(ov_y)
                            x_grid = np.linspace(bounds_multi[ox][0], bounds_multi[ox][1], 70)
                            y_grid = np.linspace(bounds_multi[oy][0], bounds_multi[oy][1], 70)
                            OX, OY = np.meshgrid(x_grid, y_grid)
                            overlay_points = np.tile(optimum_coded, (OX.size, 1))
                            overlay_points[:, ox] = OX.ravel()
                            overlay_points[:, oy] = OY.ravel()

                            fig_overlay = go.Figure()
                            for resp_name in response_order:
                                z_pred = model_prediction(
                                    multi_models[resp_name], overlay_points, coded_names
                                )
                                d_surface = individual_desirability(
                                    z_pred, desirability_specs[resp_name]
                                ).reshape(OX.shape)
                                fig_overlay.add_trace(go.Contour(
                                    x=x_grid, y=y_grid, z=d_surface,
                                    name=f"{resp_name}: deseabilidad",
                                    contours={
                                        "start": 0.2, "end": 1.0, "size": 0.2,
                                        "showlabels": True
                                    },
                                    showscale=False,
                                    opacity=0.78,
                                    line={"width": 2}
                                ))
                            fig_overlay.add_trace(go.Scatter(
                                x=[optimum_coded[ox]], y=[optimum_coded[oy]],
                                mode="markers+text",
                                marker={"size": 13, "color": "black", "symbol": "star"},
                                text=["Óptimo global"], textposition="top center",
                                name="Óptimo global"
                            ))
                            fig_overlay.update_layout(
                                title="Overlay Plot — contornos de deseabilidad individual",
                                xaxis_title=f"{ov_x} (codificado)",
                                yaxis_title=f"{ov_y} (codificado)",
                                height=620
                            )
                            st.plotly_chart(fig_overlay, use_container_width=True)
                            st.caption(
                                "Los demás factores se fijan en sus valores codificados óptimos. "
                                "La región donde coinciden deseabilidades altas representa la "
                                "zona favorable para la optimización conjunta."
                            )

                        # Exportación de los resultados de optimización múltiple.
                        export_multi = response_result.copy()
                        export_multi["Deseabilidad_global"] = global_d
                        export_multi["Factores_optimos_codificados"] = [
                            ", ".join(f"{v:.8g}" for v in optimum_coded)
                        ] * len(export_multi)
                        export_multi["Factores_optimos_reales"] = [
                            ", ".join(f"{v:.8g}" for v in optimum_actual)
                        ] * len(export_multi)
                        csv_multi = export_multi.to_csv(index=False).encode("utf-8-sig")
                        st.download_button(
                            "⬇️ Descargar resultados de optimización múltiple (CSV)",
                            csv_multi,
                            file_name="MSR_optimizacion_multiple_deseabilidad.csv",
                            mime="text/csv",
                            key="download_multi_desirability"
                        )
                    else:
                        st.error("No fue posible encontrar una solución para la optimización múltiple.")
        else:
            st.info("Selecciona al menos dos respuestas para activar la optimización múltiple.")
    else:
        st.info("Selecciona los factores y las variables respuesta antes de iniciar la optimización múltiple.")

else:
    st.info("Genera un diseño, carga los datos y completa la columna de respuesta para habilitar el análisis de segundo orden.")

st.markdown("---")
st.caption("MSR Pro — herramienta didáctica y analítica. Verifica supuestos, región experimental, unidades y corridas de confirmación antes de adoptar condiciones óptimas.")

