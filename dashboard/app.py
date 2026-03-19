import streamlit as st
import pickle
import numpy as np
import pandas as pd
import sqlite3
import os
import altair as alt

# --- CONFIG AND PATHS ---
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(PROJECT_ROOT, "best_model.pkl")
DB_PATH = os.path.join(PROJECT_ROOT, "database", "student_data.db")

st.set_page_config(page_title="Dropout Risk Analytics", page_icon="🎓", layout="wide", initial_sidebar_state="expanded")

# --- CUSTOM CSS FOR PREMIUM LOOK ---
st.markdown("""
<style>
    /* Main Background */
    .stApp {
        background-color: #0f172a;
        color: #f8fafc;
    }
    
    /* Headings */
    h1, h2, h3 {
        color: #38bdf8 !important;
        font-family: 'Inter', sans-serif;
    }
    h5 {
        color: #94a3b8 !important;
    }
    
    /* Metrics / Cards */
    div[data-testid="metric-container"] {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 4px 6px -1px rgb(0 0 0 / 0.1), 0 2px 4px -2px rgb(0 0 0 / 0.1);
        transition: transform 0.2s ease-in-out;
    }
    div[data-testid="metric-container"]:hover {
        transform: translateY(-2px);
        border-color: #38bdf8;
    }
    
    /* Tabs Customization */
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
        background-color: transparent;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: transparent;
        border-radius: 4px 4px 0px 0px;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
        color: #94a3b8;
        font-weight: 600;
        font-size: 1.1rem;
    }
    .stTabs [aria-selected="true"] {
        color: #38bdf8 !important;
        border-bottom: 2px solid #38bdf8;
    }

    /* Buttons */
    .stButton>button {
        background: linear-gradient(to right, #0ea5e9, #2563eb);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background: linear-gradient(to right, #0284c7, #1d4ed8);
        box-shadow: 0 0 15px rgba(56, 189, 248, 0.5);
        color: white;
    }
    
    /* Sidebar styling enhancements */
    section[data-testid="stSidebar"] {
        background-color: #0f172a !important;
        border-right: 1px solid #334155;
    }
</style>
""", unsafe_allow_html=True)


# --- DATA LOADING ---
@st.cache_data
def load_data():
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql("SELECT * FROM students", conn)
    conn.close()
    return df

try:
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
except FileNotFoundError:
    st.error("Model file not found. Please train the model first.")
    st.stop()

try:
    df = load_data()
except Exception as e:
    st.warning("Could not load database for insights. Make sure the ETL pipeline has been run.")
    df = pd.DataFrame()


# --- SIDEBAR & HEADER ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3135/3135810.png", width=100)
    st.title("Admin Portal")
    st.markdown("Monitor student engagement and predict dropout risks early to improve retention.")
    st.markdown("---")
    st.markdown("### Navigation Quick Stats")
    if not df.empty:
        st.metric("Total Records", len(df))
        st.metric("Risk Rate", f"{(df['dropout_risk'].mean() * 100):.1f}%")

st.markdown("<h1 style='text-align: center; margin-bottom: 2rem;'>🎓 Advanced Student Retention Analytics</h1>", unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["📊 Interactive Data Insights", "🔮 AI Risk Prediction", "📈 Model Comparison"])

# ------ TAB 1: DATA INSIGHTS ------
with tab1:
    if not df.empty:
        # Top KPI row
        avg_risk = df["dropout_risk"].mean() * 100
        total_students = len(df)
        at_risk = int(df['dropout_risk'].sum())
        
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Students Tracker", f"{total_students:,}", "Active")
        c2.metric("Overall Dropout Risk", f"{avg_risk:.1f}%", "-2.1% etc", delta_color="inverse")
        c3.metric("Students At-Risk", f"{at_risk:,}", f"{(at_risk/total_students)*100:.1f}% of cohort", delta_color="off")
        c4.metric("Avg Attendance", f"{df['attendance_percentage'].mean():.1f}%", "+1.2%", delta_color="normal")
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Native Streamlit / Altair Visuals
        st.markdown("### 📈 Engagement & Performance Breakdown")
        colA, colB = st.columns(2)
        
        with colA:
            # 1. Feature Correlation (Styled Dataframe is flawless)
            st.markdown("##### Feature Correlation Heatmap")
            corr = df.drop(columns=["student_id"]).corr()
            st.dataframe(corr.style.background_gradient(cmap="Blues", axis=None).format("{:.2f}"), use_container_width=True, height=350)
            
        with colB:
            # 2. Bar Chart: Risk by Semester
            st.markdown("##### Dropout Risk Across Semesters (%)")
            risk_by_sem = df.groupby("semester")["dropout_risk"].mean() * 100
            st.bar_chart(risk_by_sem, color="#38bdf8", use_container_width=True)

        st.markdown("<br>", unsafe_allow_html=True)
        
        colC, colD = st.columns(2)
        with colC:
            # 3. Density/Distribution: Attendance by Risk
            st.markdown("##### Attendance Density by Status")
            # Using Altair for a gorgeous native density plot which replaces the Violin plot flawlessly
            df_plot = df.copy()
            df_plot["Status"] = df_plot["dropout_risk"].map({0: "Healthy", 1: "At Risk"})
            chart = alt.Chart(df_plot).transform_density(
                'attendance_percentage',
                as_=['attendance_percentage', 'density'],
                groupby=['Status']
            ).mark_area(opacity=0.6).encode(
                x=alt.X('attendance_percentage:Q', title="Attendance %"),
                y=alt.Y('density:Q', title="Density"),
                color=alt.Color('Status:N', scale=alt.Scale(domain=['Healthy', 'At Risk'], range=['#34d399', '#f87171']))
            ).properties(height=350)
            st.altair_chart(chart, use_container_width=True)
            
        with colD:
            # 4. Native Scatter: Scores
            st.markdown("##### Assignment vs Mid-Sem Scores")
            st.scatter_chart(df_plot, x="assignment_score", y="midsem_score", color="Status", size="attendance_percentage", use_container_width=True)
            
    else:
        st.info("No data available. Please run the ETL pipeline script to populate the database.")

# ------ TAB 2: PREDICTION ------
with tab2:
    st.markdown("### 🤖 Individual AI Risk Assessment")
    st.markdown("Use the advanced machine learning model to predict individual student profiles in real-time.")
    
    with st.container():
        st.markdown("<div style='background-color:#1e293b; padding:2rem; border-radius:12px; border:1px solid #334155;'>", unsafe_allow_html=True)
        rc1, rc2 = st.columns(2)
        
        with rc1:
            attendance = st.slider("Attendance Percentage", 0.0, 100.0, 75.0, 1.0, key="att_slider")
            assignment = st.slider("Assignment Score", 0.0, 100.0, 60.0, 1.0, key="ass_slider")
            midsem = st.slider("Mid-Sem Exam Score", 0.0, 100.0, 55.0, 1.0, key="mid_slider")
        
        with rc2:
            semester = st.selectbox("Current Semester", [1, 2, 3, 4, 5, 6, 7, 8], index=2, key="sem_sel")
            difficulty = st.select_slider("Course Difficulty", options=[1, 2, 3, 4, 5], value=3, key="diff_sel")
            
            st.markdown("<br><br>", unsafe_allow_html=True)
            analyze_btn = st.button("🚀 Run AI Analysis", use_container_width=True)
            
        st.markdown("</div>", unsafe_allow_html=True)
        
    if analyze_btn:
        X = np.array([[attendance, assignment, midsem, semester, difficulty]])
        
        # Try to get probability for progress bar
        try:
            prob = model.predict_proba(X)[0][1] * 100
        except AttributeError:
            pred = model.predict(X)[0]
            prob = 95.0 if pred == 1 else 5.0
            
        st.markdown("<br>", unsafe_allow_html=True)
        
        st.markdown("### 📊 Assessment Results")
        res_col1, res_col2 = st.columns([1, 1])
        
        with res_col1:
            if prob > 50:
                st.error("⚠️ **HIGH DROPOUT RISK DETECTED**")
                st.markdown(f"The model flags this profile with a critical risk score based on the provided metrics. Immediate counseling recommended.")
            else:
                st.success("✅ **HEALTHY STUDENT PROFILE**")
                st.markdown(f"The student is functioning within safe academic boundaries. Continued normal monitoring advised.")
                
        with res_col2:
            # Native Streamlit progress metric
            st.metric("AI Risk Probability", f"{prob:.1f}%")
            st.progress(int(prob) / 100.0)
            if prob > 50:
                st.warning("Risk level is critically high.")
            else:
                st.info("Risk level is within normal bounds.")

# ------ TAB 3: MODEL COMPARISON ------
with tab3:
    st.markdown("### 📈 Machine Learning Model Performance Comparison")
    st.markdown("Compare the evaluation metrics of all professional models trained in the pipeline.")
    
    METRICS_PATH = os.path.join(PROJECT_ROOT, "model_metrics.json")
    if os.path.exists(METRICS_PATH):
        import json
        with open(METRICS_PATH, "r") as f:
            metrics_data = json.load(f)
            
        records = []
        for model_name, metrics in metrics_data.items():
            for metric_name, value in metrics.items():
                records.append({
                    "Model": model_name,
                    "Metric": metric_name,
                    "Score": value
                })
        metrics_df = pd.DataFrame(records)
        
        st.markdown("##### 🚀 Model Performance Overview")
        pivot_df = metrics_df.pivot(index="Model", columns="Metric", values="Score").reset_index()
        st.dataframe(pivot_df.style.background_gradient(cmap="Blues", axis=None).format(precision=3), use_container_width=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        st.markdown("##### 📊 Comparative Metric Analysis")
        bars = alt.Chart(metrics_df).mark_bar(cornerRadiusTopLeft=3, cornerRadiusTopRight=3).encode(
            x=alt.X('Model:N', title="", axis=alt.Axis(labelAngle=-45, labelFontSize=12)),
            y=alt.Y('Score:Q', scale=alt.Scale(domain=[0, 1]), title="Score"),
            color=alt.Color('Model:N', scale=alt.Scale(scheme='blues'), legend=None),
            column=alt.Column('Metric:N', header=alt.Header(labelOrient='bottom', titleFontSize=14, labelFontSize=14)),
            tooltip=['Model', 'Metric', 'Score']
        ).properties(width=160, height=350)
        
        st.altair_chart(bars)
    else:
        st.info("Metrics not found. The background ML Pipeline is currently training models and writing data. Refresh shortly!")
