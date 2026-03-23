"""
dashboard/app.py — light sidebar navigation
"""

import json, os, pickle, sqlite3, sys
import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _ROOT)
from config import DB_PATH, MODEL_PATH, METRICS_PATH, CLF_REPORT_PATH

st.set_page_config(page_title="Dropout Risk Analytics", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Sora:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');
html, body, [class*="css"] { font-family: 'Sora', sans-serif; }
.stApp { background-color: #f4f6f9; }

/* LIGHT BLUE sidebar */
section[data-testid="stSidebar"] { background-color: #dbeafe !important; border-right: 1px solid #bfdbfe; }
section[data-testid="stSidebar"] * { color: #1a1d23 !important; }

/* sidebar buttons — visible dark text on white */
section[data-testid="stSidebar"] .stButton > button {
    background: transparent !important;
    color: #374151 !important;
    border: none !important;
    border-left: 3px solid transparent !important;
    border-radius: 0 !important;
    width: 100% !important;
    text-align: left !important;
    padding: 0.6rem 1rem !important;
    font-size: 0.88rem !important;
    font-weight: 500 !important;
    box-shadow: none !important;
    margin: 1px 0 !important;
}
section[data-testid="stSidebar"] .stButton > button:hover {
    background: #f4f6f9 !important;
    color: #1a1d23 !important;
    border-left: 3px solid #3b82f6 !important;
    transform: none !important;
}

/* KPI cards */
div[data-testid="metric-container"] {
    background: #ffffff; border: 1px solid #e8eaed; border-radius: 10px;
    padding: 1.1rem 1.3rem; box-shadow: 0 1px 3px rgba(0,0,0,0.04);
}
div[data-testid="metric-container"] label {
    color: #374151 !important; font-size: 0.7rem !important;
    font-weight: 600 !important; text-transform: uppercase !important; letter-spacing: 0.08em !important;
}
div[data-testid="metric-container"] [data-testid="stMetricValue"] {
    color: #000000 !important; font-size: 1.7rem !important; font-weight: 700 !important;
}

/* main content buttons */
div[data-testid="stMain"] .stButton > button {
    background: #1a1d23 !important; color: #fff !important;
    border-radius: 8px !important; border: none !important;
    font-weight: 600 !important; padding: 0.6rem 1.4rem !important;
}
div[data-testid="stMain"] .stButton > button:hover { background: #374151 !important; transform: none !important; }

.section-title {
    font-size: 0.68rem; font-weight: 700; letter-spacing: 0.14em;
    text-transform: uppercase; color: #9ca3af;
    margin: 0 0 1rem 0; padding-bottom: 0.5rem; border-bottom: 1px solid #e8eaed;
}
.page-header {
    background: #fff; border-bottom: 1px solid #e8eaed;
    padding: 1.1rem 2rem; margin: -3rem -3rem 2rem -3rem;
    display: flex; align-items: center; gap: 1rem;
}
.badge-live { background:#dcfce7; color:#15803d; font-size:0.68rem; font-weight:700; padding:3px 10px; border-radius:999px; text-transform:uppercase; letter-spacing:0.08em; }
.badge-ml   { background:#dbeafe; color:#1d4ed8; font-size:0.68rem; font-weight:700; padding:3px 10px; border-radius:999px; text-transform:uppercase; letter-spacing:0.08em; }
.result-card { background:#fff; border:1px solid #e8eaed; border-radius:10px; padding:1.5rem; }
.result-card.risk { border-left:4px solid #ef4444; }
.result-card.warn { border-left:4px solid #f59e0b; }
.result-card.safe { border-left:4px solid #22c55e; }
.status-tag { display:inline-block; padding:4px 12px; border-radius:6px; font-size:0.72rem; font-weight:700; letter-spacing:0.08em; text-transform:uppercase; margin-bottom:0.8rem; }
.tag-risk { background:#fee2e2; color:#b91c1c; }
.tag-warn { background:#fef3c7; color:#b45309; }
.tag-safe { background:#dcfce7; color:#15803d; }
.stProgress > div > div > div > div { background: #1a1d23 !important; }
[data-testid="stMetricValue"] { color: #000000 !important; font-weight: 700 !important; }
[data-testid="stMetricLabel"] { color: #374151 !important; }
[data-testid="stMetricLabel"] p { color: #374151 !important; }
div[data-testid="metric-container"] p { color: #000000 !important; }
h1,h2,h3,h4,h5,h6 { color:#1a1d23 !important; }
</style>
""", unsafe_allow_html=True)

# Altair theme
def clean_theme():
    return {"config": {"background": "#ffffff","view": {"stroke": "transparent"},
        "axis": {"grid":True,"gridColor":"#f3f4f6","labelColor":"#9ca3af","titleColor":"#6b7280","labelFont":"Sora","titleFont":"Sora","labelFontSize":11,"titleFontSize":11},
        "legend": {"labelColor":"#6b7280","titleColor":"#9ca3af","labelFont":"Sora","titleFont":"Sora"},
        "title": {"color":"#1a1d23","font":"Sora","fontSize":12,"fontWeight":600,"anchor":"start","offset":10}}}
alt.themes.register("clean", clean_theme)
alt.themes.enable("clean")

@st.cache_data
def load_db():
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql("SELECT * FROM students", conn)
    conn.close()
    return df

@st.cache_resource
def load_model():
    with open(MODEL_PATH, "rb") as f: return pickle.load(f)

def load_metrics():
    return json.load(open(METRICS_PATH)) if os.path.exists(METRICS_PATH) else {}

def load_clf_reports():
    return json.load(open(CLF_REPORT_PATH)) if os.path.exists(CLF_REPORT_PATH) else {}

try:
    model = load_model()
except FileNotFoundError:
    st.error("Model not found. Run: python models/train_model.py"); st.stop()

try:
    df = load_db()
except:
    st.warning("Database not found. Run: python database/etl_pipeline.py")
    df = pd.DataFrame()

SAFE_C = "#4ade80"
RISK_C = "#f87171"

if "page" not in st.session_state:
    st.session_state.page = "Overview"

# ── SIDEBAR ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## Dropout Risk")
    st.caption("Student Analytics Platform")
    st.markdown("---")

    st.markdown("**ANALYTICS**")
    if st.button("  Overview",                key="n1", use_container_width=True): st.session_state.page = "Overview"
    if st.button("  AI Risk Predictor",       key="n2", use_container_width=True): st.session_state.page = "AI Risk Predictor"
    if st.button("  Model Comparison",        key="n3", use_container_width=True): st.session_state.page = "Model Comparison"
    if st.button("  Classification Reports",  key="n4", use_container_width=True): st.session_state.page = "Classification Reports"

    st.markdown("---")
    if not df.empty:
        st.markdown("**LIVE DATA**")
        st.markdown(f"Total Students &nbsp;&nbsp;&nbsp;&nbsp; **{len(df):,}**", unsafe_allow_html=True)
        st.markdown(f"At-Risk Count &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; **{int(df['dropout_risk'].sum()):,}**", unsafe_allow_html=True)
        st.markdown(f"Risk Rate &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; **{df['dropout_risk'].mean()*100:.1f}%**", unsafe_allow_html=True)
        st.markdown(f"Avg Attendance &nbsp;&nbsp; **{df['attendance_percentage'].mean():.1f}%**", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("**PIPELINE**")
    st.caption("1. data/generate_data.py\n2. database/etl_pipeline.py\n3. models/train_model.py\n4. dashboard/app.py")

page = st.session_state.page

# ── HEADER ────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="page-header">
    <div style="flex:1">
        <div style="font-size:1.1rem;font-weight:700;color:#1a1d23;">Student Dropout Risk Analytics</div>
        <div style="font-size:0.78rem;color:#9ca3af;">End-to-end ML pipeline for early identification of at-risk students</div>
    </div>
    <span class="badge-live">Live</span>&nbsp;
    <span class="badge-ml">ML Enabled</span>
</div>
""", unsafe_allow_html=True)

# ── PAGE: OVERVIEW ────────────────────────────────────────────────────────────
if page == "Overview":
    if df.empty:
        st.info("No data. Run ETL pipeline first.")
    else:
        df_plot = df.copy()
        df_plot["Status"] = df_plot["dropout_risk"].map({0:"Safe", 1:"At Risk"})

        st.markdown('<p class="section-title">Cohort Overview</p>', unsafe_allow_html=True)
        k1,k2,k3,k4,k5,k6 = st.columns(6)
        k1.metric("Total Students",   f"{len(df):,}")
        k2.metric("At-Risk Students", f"{int(df['dropout_risk'].sum()):,}", f"{df['dropout_risk'].mean()*100:.1f}% of cohort", delta_color="inverse")
        k3.metric("Safe Students",    f"{len(df)-int(df['dropout_risk'].sum()):,}")
        k4.metric("Avg Attendance",   f"{df['attendance_percentage'].mean():.1f}%")
        k5.metric("Avg Assignment",   f"{df['assignment_score'].mean():.1f}")
        k6.metric("Avg Mid-Sem",      f"{df['midsem_score'].mean():.1f}")
        st.markdown("<br>", unsafe_allow_html=True)

        st.markdown('<p class="section-title">Feature Correlation</p>', unsafe_allow_html=True)
        corr = df.drop(columns=["student_id"]).corr().round(3)
        corr_long = corr.reset_index().melt(id_vars="index", var_name="Variable", value_name="Correlation").rename(columns={"index":"Feature"})
        heatmap = alt.Chart(corr_long).mark_rect(cornerRadius=3).encode(
            x=alt.X("Feature:O", sort=None, axis=alt.Axis(labelAngle=-35)),
            y=alt.Y("Variable:O", sort=None),
            color=alt.Color("Correlation:Q", scale=alt.Scale(scheme="blueorange", domain=[-1,1]), legend=alt.Legend(title="r")),
            tooltip=["Feature","Variable", alt.Tooltip("Correlation:Q", format=".3f")]
        ).properties(height=300, title="Feature Correlation Matrix")
        st.altair_chart(heatmap, use_container_width=True)
        st.markdown("<br>", unsafe_allow_html=True)

        st.markdown('<p class="section-title">Distributions</p>', unsafe_allow_html=True)
        ca, cb = st.columns(2)
        with ca:
            risk_sem = df.groupby("semester")["dropout_risk"].mean().mul(100).reset_index().rename(columns={"dropout_risk":"Risk (%)"})
            st.altair_chart(alt.Chart(risk_sem).mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4, color="#1a1d23").encode(
                x=alt.X("semester:O", title="Semester"),
                y=alt.Y("Risk (%):Q", scale=alt.Scale(domain=[0,100])),
                opacity=alt.condition(alt.datum["Risk (%)"] > risk_sem["Risk (%)"].mean(), alt.value(1.0), alt.value(0.4)),
                tooltip=[alt.Tooltip("semester:O"), alt.Tooltip("Risk (%):Q", format=".1f")]
            ).properties(height=280, title="Avg Dropout Risk % by Semester"), use_container_width=True)
        with cb:
            st.altair_chart(alt.Chart(df_plot).transform_density("attendance_percentage", as_=["attendance_percentage","density"], groupby=["Status"]).mark_area(opacity=0.6, interpolate="monotone").encode(
                x=alt.X("attendance_percentage:Q", title="Attendance %"),
                y=alt.Y("density:Q", title="Density"),
                color=alt.Color("Status:N", scale=alt.Scale(domain=["Safe","At Risk"], range=[SAFE_C, RISK_C]))
            ).properties(height=280, title="Attendance Distribution by Risk Status"), use_container_width=True)
        st.markdown("<br>", unsafe_allow_html=True)

        st.markdown('<p class="section-title">Score Analysis</p>', unsafe_allow_html=True)
        st.altair_chart(alt.Chart(df_plot.sample(min(600,len(df_plot)), random_state=1)).mark_circle(opacity=0.65, size=50).encode(
            x=alt.X("assignment_score:Q", title="Assignment Score"),
            y=alt.Y("midsem_score:Q", title="Mid-Sem Score"),
            color=alt.Color("Status:N", scale=alt.Scale(domain=["Safe","At Risk"], range=[SAFE_C, RISK_C])),
            size=alt.Size("attendance_percentage:Q", scale=alt.Scale(range=[20,160]), legend=None),
            tooltip=[alt.Tooltip("assignment_score:Q",format=".1f"), alt.Tooltip("midsem_score:Q",format=".1f"), alt.Tooltip("attendance_percentage:Q",format=".1f"), "Status"]
        ).properties(height=320, title="Assignment Score vs Mid-Sem Score"), use_container_width=True)
        st.markdown("<br>", unsafe_allow_html=True)

        st.markdown('<p class="section-title">Key Insights</p>', unsafe_allow_html=True)
        ia,ib,ic,id_ = st.columns(4)
        ia.metric("Highest Risk Semester", f"Sem {int(risk_sem.loc[risk_sem['Risk (%)'].idxmax(),'semester'])}")
        ib.metric("Low Attendance Risk",   f"{df[df['attendance_percentage']<65]['dropout_risk'].mean()*100:.1f}%", "Attendance < 65%", delta_color="inverse")
        ic.metric("High Difficulty Risk",  f"{df[df['course_difficulty']>=4]['dropout_risk'].mean()*100:.1f}%",  "Difficulty >= 4",  delta_color="inverse")
        id_.metric("High Attend Safety",   f"{df[df['attendance_percentage']>=80]['dropout_risk'].mean()*100:.1f}%", "Attendance >= 80%")

# ── PAGE: AI RISK PREDICTOR ───────────────────────────────────────────────────
elif page == "AI Risk Predictor":
    st.markdown('<p class="section-title">New Prediction</p>', unsafe_allow_html=True)
    fc1,fc2,fc3 = st.columns(3)
    with fc1:
        attendance = st.slider("Attendance (%)",   0.0,100.0,75.0,1.0)
        assignment = st.slider("Assignment Score", 0.0,100.0,60.0,1.0)
    with fc2:
        midsem   = st.slider("Mid-Sem Score",    0.0,100.0,55.0,1.0)
        semester = st.selectbox("Semester", list(range(1,9)), index=2)
    with fc3:
        difficulty = st.select_slider("Course Difficulty", options=[1,2,3,4,5], value=3,
            format_func=lambda x:{1:"Easy",2:"Moderate",3:"Medium",4:"Hard",5:"Very Hard"}[x])
        st.markdown("<br>", unsafe_allow_html=True)
        run_btn = st.button("Run Assessment", use_container_width=True, key="run")

    if run_btn:
        X = np.array([[attendance, assignment, midsem, semester, difficulty]])
        try:    prob = model.predict_proba(X)[0][1]*100
        except: prob = 95.0 if model.predict(X)[0]==1 else 5.0

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<p class="section-title">Assessment Results</p>', unsafe_allow_html=True)
        r1,r2 = st.columns([1.3,1])
        with r1:
            if   prob>=70: card_cls,tag_cls,tag_txt,headline,body = "risk","tag-risk","Critical Risk","High Dropout Risk Detected","Immediate intervention and counselling recommended."
            elif prob>=45: card_cls,tag_cls,tag_txt,headline,body = "warn","tag-warn","Moderate Risk","Moderate Dropout Risk","Close monitoring and academic support advised."
            else:          card_cls,tag_cls,tag_txt,headline,body = "safe","tag-safe","Low Risk","Healthy Student Profile","Student is within safe academic parameters."
            st.markdown(f"""<div class="result-card {card_cls}">
                <span class="status-tag {tag_cls}">{tag_txt}</span>
                <p style="font-size:1rem;font-weight:600;color:#1a1d23;margin:0 0 0.3rem">{headline}</p>
                <p style="font-size:0.85rem;color:#6b7280;margin:0 0 1rem">{body}</p>
                <table style="width:100%;font-size:0.83rem;border-collapse:collapse">
                    <tr style="border-bottom:1px solid #f3f4f6"><td style="padding:0.4rem 0;color:#9ca3af">Attendance</td><td style="color:#1a1d23;font-weight:600;font-family:'JetBrains Mono',monospace">{attendance:.1f}%</td></tr>
                    <tr style="border-bottom:1px solid #f3f4f6"><td style="padding:0.4rem 0;color:#9ca3af">Assignment</td><td style="color:#1a1d23;font-weight:600;font-family:'JetBrains Mono',monospace">{assignment:.1f}</td></tr>
                    <tr style="border-bottom:1px solid #f3f4f6"><td style="padding:0.4rem 0;color:#9ca3af">Mid-Sem</td><td style="color:#1a1d23;font-weight:600;font-family:'JetBrains Mono',monospace">{midsem:.1f}</td></tr>
                    <tr style="border-bottom:1px solid #f3f4f6"><td style="padding:0.4rem 0;color:#9ca3af">Semester</td><td style="color:#1a1d23;font-weight:600;font-family:'JetBrains Mono',monospace">{semester}</td></tr>
                    <tr><td style="padding:0.4rem 0;color:#9ca3af">Difficulty</td><td style="color:#1a1d23;font-weight:600;font-family:'JetBrains Mono',monospace">{difficulty}/5</td></tr>
                </table></div>""", unsafe_allow_html=True)
        with r2:
            st.metric("Dropout Probability", f"{prob:.1f}%")
            st.progress(int(prob)/100)
            factors = pd.DataFrame({"Factor":["Low Attendance","Low Assignment","Low Mid-Sem","High Difficulty"],
                "Triggered":[1 if attendance<65 else 0,1 if assignment<50 else 0,1 if midsem<45 else 0,1 if difficulty>=4 else 0]})
            st.altair_chart(alt.Chart(factors).mark_bar(cornerRadiusTopLeft=4,cornerRadiusTopRight=4).encode(
                x=alt.X("Factor:N",title="",axis=alt.Axis(labelAngle=-20)),
                y=alt.Y("Triggered:Q",scale=alt.Scale(domain=[0,1.2])),
                color=alt.condition(alt.datum.Triggered==1, alt.value(RISK_C), alt.value(SAFE_C)),
                tooltip=["Factor","Triggered"]
            ).properties(height=200, title="Risk Factor Flags"), use_container_width=True)

# ── PAGE: MODEL COMPARISON ────────────────────────────────────────────────────
elif page == "Model Comparison":
    st.markdown('<p class="section-title">Model Performance</p>', unsafe_allow_html=True)
    metrics = load_metrics()
    if not metrics:
        st.info("Run: python models/train_model.py")
    else:
        pivot = pd.DataFrame(metrics).T.reset_index().rename(columns={"index":"Model"})
        numeric_cols = ["Accuracy","Precision","Recall","F1 Score"]
        st.markdown('<p class="section-title">Summary Table</p>', unsafe_allow_html=True)
        st.dataframe(pivot.style.highlight_max(subset=numeric_cols,color="#dcfce7",axis=0).highlight_min(subset=numeric_cols,color="#fee2e2",axis=0).format({c:"{:.3f}" for c in numeric_cols}), use_container_width=True, hide_index=True)
        st.caption("Green = best   |   Red = lowest")
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<p class="section-title">Visual Comparison</p>', unsafe_allow_html=True)
        records = [{"Model":m,"Metric":k,"Score":v} for m,mets in metrics.items() for k,v in mets.items()]
        st.altair_chart(alt.Chart(pd.DataFrame(records)).mark_bar(cornerRadiusTopLeft=3,cornerRadiusTopRight=3).encode(
            x=alt.X("Model:N",title="",axis=alt.Axis(labelAngle=-30)),
            y=alt.Y("Score:Q",scale=alt.Scale(domain=[0,1.05])),
            color=alt.Color("Model:N",scale=alt.Scale(scheme="tableau10"),legend=None),
            column=alt.Column("Metric:N",header=alt.Header(labelFontSize=12,labelColor="#6b7280",titleFontSize=0)),
            tooltip=["Model","Metric",alt.Tooltip("Score:Q",format=".4f")]
        ).properties(width=160,height=260))
        best = pivot.set_index("Model")["F1 Score"].idxmax()
        st.success(f"Best model by F1-Score: **{best}** — F1 = {pivot.set_index('Model').loc[best,'F1 Score']:.4f}")

# ── PAGE: CLASSIFICATION REPORTS ─────────────────────────────────────────────
elif page == "Classification Reports":
    st.markdown('<p class="section-title">Classification Reports</p>', unsafe_allow_html=True)
    reports = load_clf_reports()
    if not reports:
        st.info("Run: python models/train_model.py")
    else:
        chosen = st.selectbox("Select Model", list(reports.keys()))
        report = reports[chosen]
        rows = [{"Class":c,"Precision":report[c].get("precision",0),"Recall":report[c].get("recall",0),"F1-Score":report[c].get("f1-score",0),"Support":int(report[c].get("support",0))}
                for c in ["Not At Risk","At Risk","macro avg","weighted avg"] if c in report]
        rdf = pd.DataFrame(rows)
        st.dataframe(rdf.style.background_gradient(subset=["Precision","Recall","F1-Score"],cmap="Blues",vmin=0,vmax=1).format({"Precision":"{:.4f}","Recall":"{:.4f}","F1-Score":"{:.4f}"}), use_container_width=True, hide_index=True)
        st.markdown("<br>", unsafe_allow_html=True)
        class_df = rdf[rdf["Class"].isin(["Not At Risk","At Risk"])].melt(id_vars=["Class"],value_vars=["Precision","Recall","F1-Score"],var_name="Metric",value_name="Score")
        st.altair_chart(alt.Chart(class_df).mark_bar(cornerRadiusTopLeft=4,cornerRadiusTopRight=4).encode(
            x=alt.X("Metric:N",title=""),
            y=alt.Y("Score:Q",scale=alt.Scale(domain=[0,1.05])),
            color=alt.Color("Class:N",scale=alt.Scale(domain=["Not At Risk","At Risk"],range=[SAFE_C,RISK_C])),
            xOffset="Class:N",
            tooltip=["Class","Metric",alt.Tooltip("Score:Q",format=".4f")]
        ).properties(height=300, title=f"Per-Class Metrics — {chosen}"), use_container_width=True)
        if report.get("accuracy"): st.metric("Overall Accuracy", f"{report['accuracy']:.4f}")
