import os
import joblib
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from streamlit_option_menu import option_menu


# ============================================================
# PATH CONFIGURATION
# ============================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "sentiment_model.pkl")
DATA_PATH = os.path.join(BASE_DIR, "data", "processed", "cleaned_policies.csv")
RESULTS_PATH = os.path.join(BASE_DIR, "models", "model_comparison.csv")


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Policy Sentiment AI",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# DESIGN SYSTEM
# ============================================================

SENTIMENT_COLORS = {
    "positive": "#10b981",
    "negative": "#f43f5e",
    "neutral":  "#f59e0b",
}

PLOTLY_BASE = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter, sans-serif", color="#94a3b8", size=13),
    title=dict(font=dict(color="#f1f5f9", size=16)),
    xaxis=dict(gridcolor="#1e293b", linecolor="#334155", tickcolor="#64748b"),
    yaxis=dict(gridcolor="#1e293b", linecolor="#334155", tickcolor="#64748b"),
    legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#94a3b8")),
    margin=dict(t=50, b=30, l=20, r=20),
)


def style_fig(fig, height=380):
    fig.update_layout(**PLOTLY_BASE, height=height)
    return fig


# ============================================================
# CSS
# ============================================================

# Build CSS as a single string — no leading newline so Streamlit
# does not treat the block as a markdown paragraph / code block.
_CSS = (
    '<link href="https://fonts.googleapis.com/css2?family=Inter'
    ':wght@300;400;500;600;700;800&display=swap" rel="stylesheet">'
    "<style>"
    "html,body,[class*='css']{font-family:'Inter',sans-serif!important;}"
    ".stApp{background-color:#0f172a;color:#f1f5f9;}"
    ".block-container{padding:2rem 2.5rem 3rem 2.5rem;max-width:1300px;}"
    "[data-testid='stSidebar']{background:#0a1120;border-right:1px solid #1e293b;}"
    "[data-testid='stSidebar'] *{color:#cbd5e1!important;}"
    "[data-testid='stSidebar'] .stRadio label{padding:8px 12px;border-radius:8px;transition:background 0.2s;cursor:pointer;font-size:14px;font-weight:500;}"
    "[data-testid='stSidebar'] .stRadio label:hover{background:#1e293b;}"
    ".hero-banner{background:linear-gradient(135deg,#1e1b4b 0%,#0f172a 50%,#1e293b 100%);border:1px solid #334155;border-radius:20px;padding:40px 44px;margin-bottom:28px;position:relative;overflow:hidden;}"
    ".hero-banner::before{content:'';position:absolute;top:-60px;right:-60px;width:260px;height:260px;background:radial-gradient(circle,rgba(99,102,241,0.18) 0%,transparent 70%);border-radius:50%;}"
    ".hero-badge{display:inline-block;background:rgba(99,102,241,0.15);border:1px solid rgba(99,102,241,0.4);color:#a5b4fc;font-size:12px;font-weight:600;letter-spacing:1.2px;text-transform:uppercase;padding:4px 14px;border-radius:20px;margin-bottom:16px;}"
    ".hero-title{font-size:36px;font-weight:800;color:#f8fafc;line-height:1.2;margin-bottom:10px;}"
    ".hero-title span{background:linear-gradient(135deg,#818cf8,#6366f1);-webkit-background-clip:text;-webkit-text-fill-color:transparent;}"
    ".hero-subtitle{font-size:15px;color:#94a3b8;max-width:560px;line-height:1.6;}"
    ".kpi-card{background:#1e293b;border:1px solid #334155;border-radius:14px;padding:22px 24px;transition:transform 0.2s,border-color 0.2s;}"
    ".kpi-card:hover{transform:translateY(-2px);border-color:#475569;}"
    ".kpi-label{font-size:12px;font-weight:600;letter-spacing:0.8px;text-transform:uppercase;color:#64748b;margin-bottom:8px;}"
    ".kpi-value{font-size:36px;font-weight:800;color:#f1f5f9;line-height:1;}"
    ".kpi-sub{font-size:12px;color:#64748b;margin-top:4px;}"
    ".section-header{font-size:16px;font-weight:700;color:#f1f5f9;margin:8px 0 16px 0;padding-bottom:10px;border-bottom:1px solid #1e293b;}"
    ".result-card{border-radius:16px;padding:28px 24px;text-align:center;margin:18px 0;border:1px solid;}"
    ".result-positive{background:rgba(16,185,129,0.12);border-color:rgba(16,185,129,0.35);color:#6ee7b7;}"
    ".result-negative{background:rgba(244,63,94,0.12);border-color:rgba(244,63,94,0.35);color:#fda4af;}"
    ".result-neutral{background:rgba(245,158,11,0.12);border-color:rgba(245,158,11,0.35);color:#fcd34d;}"
    ".result-icon{font-size:48px;line-height:1;margin-bottom:10px;}"
    ".result-label{font-size:13px;font-weight:600;letter-spacing:1.5px;text-transform:uppercase;opacity:0.7;margin-bottom:4px;}"
    ".result-text{font-size:26px;font-weight:800;}"
    ".insight-card{background:#1e293b;border:1px solid #334155;border-radius:12px;padding:18px 20px;font-size:14px;color:#cbd5e1;line-height:1.6;}"
    ".insight-card b{color:#f1f5f9;}"
    ".tech-pill{display:inline-block;background:#1e293b;border:1px solid #334155;border-radius:20px;padding:5px 14px;font-size:12px;font-weight:500;color:#94a3b8;margin:3px 3px 3px 0;}"
    ".pipeline-step{background:#1e293b;border:1px solid #334155;border-radius:10px;padding:12px 18px;font-size:13px;font-weight:500;color:#cbd5e1;text-align:center;margin:4px 0;}"
    ".pipeline-arrow{text-align:center;color:#6366f1;font-size:18px;margin:0;}"
    "[data-testid='metric-container']{background:#1e293b;border:1px solid #334155;border-radius:12px;padding:16px 20px;}"
    "[data-testid='stMetricValue']{color:#f1f5f9!important;font-weight:700!important;}"
    "[data-testid='stMetricLabel']{color:#64748b!important;font-size:12px!important;font-weight:600!important;text-transform:uppercase;letter-spacing:0.6px;}"
    "div[data-testid='stSelectbox'] label,div[data-testid='stTextArea'] label,div[data-testid='stTextInput'] label{color:#94a3b8!important;font-size:13px!important;font-weight:600!important;letter-spacing:0.4px;text-transform:uppercase;}"
    "div[data-testid='stTextArea'] textarea,div[data-testid='stTextInput'] input{background:#1e293b!important;border:1px solid #334155!important;color:#f1f5f9!important;border-radius:10px!important;font-family:'Inter',sans-serif!important;font-size:14px!important;}"
    "div[data-testid='stSelectbox']>div{background:#1e293b!important;border:1px solid #334155!important;color:#f1f5f9!important;border-radius:10px!important;}"
    ".stButton>button[kind='primary']{background:linear-gradient(135deg,#6366f1,#4f46e5)!important;border:none!important;border-radius:10px!important;font-weight:600!important;font-size:14px!important;padding:12px 24px!important;color:white!important;box-shadow:0 4px 15px rgba(99,102,241,0.3)!important;}"
    ".stButton>button[kind='secondary']{background:#1e293b!important;border:1px solid #334155!important;border-radius:10px!important;color:#cbd5e1!important;font-weight:500!important;}"
    ".stDownloadButton>button{background:#1e293b!important;border:1px solid #334155!important;color:#a5b4fc!important;border-radius:10px!important;font-weight:600!important;}"
    ".stDataFrame,[data-testid='stDataFrame']{border:1px solid #334155!important;border-radius:12px!important;overflow:hidden;}"
    "hr{border-color:#1e293b!important;margin:24px 0!important;}"
    "h1{font-size:26px!important;font-weight:800!important;color:#f1f5f9!important;margin-bottom:6px!important;}"
    "h2{font-size:18px!important;font-weight:700!important;color:#e2e8f0!important;}"
    "h3{font-size:15px!important;font-weight:600!important;color:#cbd5e1!important;}"
    "p,li{color:#94a3b8!important;}"
    "code{background:#1e293b!important;color:#a5b4fc!important;border-radius:4px!important;}"
    "</style>"
)
st.markdown(_CSS, unsafe_allow_html=True)


# ============================================================
# LOAD DATA
# ============================================================


@st.cache_data
def load_dataset():
    if not os.path.exists(DATA_PATH):
        return None
    return pd.read_csv(DATA_PATH)

@st.cache_data
def load_results():
    if not os.path.exists(RESULTS_PATH):
        return None
    return pd.read_csv(RESULTS_PATH)

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        return None
    return joblib.load(MODEL_PATH)


df = load_dataset()
results_df = load_results()
model = load_model()

if df is None:
    st.error("Dataset not found. Run `python src/data_preprocessing.py` first.")
    st.stop()

if model is None:
    st.error("ML model not found. Run `python src/train_model.py` first.")
    st.stop()


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.markdown("""
    <div style="padding: 8px 4px 16px 4px;">
        <div style="font-size:20px; font-weight:800; color:#f1f5f9; margin-bottom:4px;">
            🏛️ PolicySentiment
        </div>
        <div style="font-size:12px; color:#475569; font-weight:500; letter-spacing:0.5px;">
            AI-POWERED ANALYSIS
        </div>
    </div>
    """, unsafe_allow_html=True)

    page = option_menu(
        menu_title=None,
        options=["Dashboard", "Sentiment Analyzer", "Policy Analytics", "Dataset Explorer", "Model Performance", "About"],
        icons=["house-fill", "search", "bar-chart-fill", "database-fill", "cpu-fill", "info-circle-fill"],
        default_index=0,
        styles={
            "container": {"padding": "4px 0", "background-color": "transparent"},
            "icon": {"color": "#6366f1", "font-size": "15px"},
            "nav-link": {
                "font-size": "13px",
                "font-weight": "500",
                "text-align": "left",
                "margin": "2px 0",
                "padding": "9px 14px",
                "border-radius": "8px",
                "color": "#94a3b8",
                "--hover-color": "#1e293b",
            },
            "nav-link-selected": {
                "background-color": "#1e293b",
                "color": "#f1f5f9",
                "font-weight": "600",
                "border": "1px solid #334155",
            },
        }
    )

    st.markdown("""
    <div style="margin-top:16px; padding:12px 14px; background:#0f172a; border-radius:10px;">
        <div style="font-size:11px; color:#334155; line-height:1.9;">
            <div style="color:#475569; font-weight:600; margin-bottom:6px; letter-spacing:0.5px;">TECH STACK</div>
            Python · Scikit-learn<br>
            TF-IDF · Streamlit<br>
            Plotly · Pandas
        </div>
    </div>
    """, unsafe_allow_html=True)


# ============================================================
# PRE-COMPUTE
# ============================================================

total_records  = len(df)
positive_count = int((df["sentiment"] == "positive").sum())
negative_count = int((df["sentiment"] == "negative").sum())
neutral_count  = int((df["sentiment"] == "neutral").sum())


# ============================================================
# DASHBOARD
# ============================================================

if page == "Dashboard":

    st.markdown("""
    <div class="hero-banner">
        <div class="hero-badge">🇮🇳 &nbsp;Government Policy Intelligence</div>
        <div class="hero-title">Policy <span>Sentiment</span> AI</div>
        <div class="hero-subtitle">
            Classify and visualize sentiment in government policy statements
            using NLP and Machine Learning — instantly, accurately, at scale.
        </div>
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""<div class="kpi-card">
            <div class="kpi-label">Total Policies</div>
            <div class="kpi-value">{total_records}</div>
            <div class="kpi-sub">Analysed records</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""<div class="kpi-card" style="border-top:3px solid #10b981;">
            <div class="kpi-label" style="color:#10b981;">Positive</div>
            <div class="kpi-value" style="color:#10b981;">{positive_count}</div>
            <div class="kpi-sub">{positive_count/total_records*100:.0f}% of dataset</div>
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown(f"""<div class="kpi-card" style="border-top:3px solid #f43f5e;">
            <div class="kpi-label" style="color:#f43f5e;">Negative</div>
            <div class="kpi-value" style="color:#f43f5e;">{negative_count}</div>
            <div class="kpi-sub">{negative_count/total_records*100:.0f}% of dataset</div>
        </div>""", unsafe_allow_html=True)
    with c4:
        st.markdown(f"""<div class="kpi-card" style="border-top:3px solid #f59e0b;">
            <div class="kpi-label" style="color:#f59e0b;">Neutral</div>
            <div class="kpi-value" style="color:#f59e0b;">{neutral_count}</div>
            <div class="kpi-sub">{neutral_count/total_records*100:.0f}% of dataset</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<div style='height:24px'></div>", unsafe_allow_html=True)

    col_l, col_r = st.columns(2, gap="large")

    sentiment_counts = df["sentiment"].value_counts().reset_index()
    sentiment_counts.columns = ["sentiment", "count"]

    with col_l:
        st.markdown('<div class="section-header">Sentiment Distribution</div>', unsafe_allow_html=True)
        fig = go.Figure(data=[go.Pie(
            labels=sentiment_counts["sentiment"],
            values=sentiment_counts["count"],
            hole=0.52,
            marker=dict(
                colors=[SENTIMENT_COLORS.get(s, "#6366f1") for s in sentiment_counts["sentiment"]],
                line=dict(color="#0f172a", width=3)
            ),
            textinfo="label+percent",
            textfont=dict(size=13, color="#f1f5f9"),
            hovertemplate="<b>%{label}</b><br>Count: %{value}<br>Share: %{percent}<extra></extra>",
        )])
        fig.add_annotation(
            text=f"<b>{total_records}</b><br>policies",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16, color="#f1f5f9"),
        )
        style_fig(fig, 360)
        st.plotly_chart(fig, use_container_width=True)

    category_counts = df["category"].value_counts().reset_index()
    category_counts.columns = ["category", "count"]

    with col_r:
        st.markdown('<div class="section-header">Policies by Category</div>', unsafe_allow_html=True)
        fig = px.bar(
            category_counts, x="category", y="count",
            color="count", color_continuous_scale=["#312e81", "#6366f1", "#a5b4fc"],
        )
        fig.update_layout(coloraxis_showscale=False, xaxis_tickangle=-30)
        fig.update_traces(hovertemplate="<b>%{x}</b><br>Count: %{y}<extra></extra>", marker_line_width=0)
        style_fig(fig, 360)
        st.plotly_chart(fig, use_container_width=True)

    st.divider()
    st.markdown('<div class="section-header">📌 Key Insights</div>', unsafe_allow_html=True)
    most_common_cat = df["category"].value_counts().index[0]
    i1, i2, i3 = st.columns(3)
    with i1:
        st.markdown(f'<div class="insight-card">📈 <b>{positive_count/total_records*100:.1f}%</b> of all policies carry a <b>positive</b> sentiment — indicating overall optimistic policy coverage.</div>', unsafe_allow_html=True)
    with i2:
        st.markdown(f'<div class="insight-card">🗂️ The most represented category is <b>{most_common_cat}</b>, with {df["category"].value_counts().iloc[0]} entries in the dataset.</div>', unsafe_allow_html=True)
    with i3:
        st.markdown('<div class="insight-card">🤖 The classifier uses <b>TF-IDF</b> vectorization with n-gram (1,2) features and a max of 5,000 vocabulary terms.</div>', unsafe_allow_html=True)


# ============================================================
# SENTIMENT ANALYZER
# ============================================================

elif page == "Sentiment Analyzer":

    st.title("🔍 Policy Sentiment Analyzer")
    st.markdown('<p style="color:#64748b; margin-top:-8px; margin-bottom:24px;">Enter any government policy statement or public opinion to classify its sentiment.</p>', unsafe_allow_html=True)

    col_input, col_result = st.columns([3, 2], gap="large")

    with col_input:
        text = st.text_area(
            "Policy Statement",
            height=200,
            placeholder="e.g. The new education policy provides students with better learning opportunities..."
        )

        col_btn, col_clear = st.columns([3, 1])
        with col_btn:
            analyze = st.button("🔍  Analyze Sentiment", type="primary", use_container_width=True)
        with col_clear:
            clear = st.button("Clear", use_container_width=True)

        if clear:
            st.rerun()

        st.markdown("""
        <div style="margin-top:16px; padding:14px 18px; background:#1e293b;
                    border:1px solid #334155; border-radius:10px; font-size:12px; color:#64748b;">
            <b style="color:#94a3b8;">💡 Try these examples:</b><br>
            • "The healthcare scheme provides affordable treatment to millions."<br>
            • "The policy has failed to provide sufficient support to citizens."<br>
            • "The government announced new measures for agricultural development."
        </div>
        """, unsafe_allow_html=True)

    with col_result:
        if analyze:
            if not text.strip():
                st.warning("Please enter a policy statement to analyze.")
            else:
                prediction = model.predict([text])[0]
                icons  = {"positive": "✅", "negative": "❌", "neutral": "⚖️"}
                labels = {"positive": "POSITIVE SENTIMENT", "negative": "NEGATIVE SENTIMENT", "neutral": "NEUTRAL SENTIMENT"}
                css    = {"positive": "result-positive", "negative": "result-negative", "neutral": "result-neutral"}

                st.markdown(f"""
                <div class="result-card {css[prediction]}">
                    <div class="result-icon">{icons[prediction]}</div>
                    <div class="result-label">Detected Sentiment</div>
                    <div class="result-text">{labels[prediction]}</div>
                </div>
                """, unsafe_allow_html=True)

                if hasattr(model, "decision_function"):
                    scores  = model.decision_function([text])[0]
                    classes = model.classes_
                    exp_s   = np.exp(scores - np.max(scores))
                    probs   = exp_s / exp_s.sum()
                    score_df = pd.DataFrame({
                        "Sentiment": [c.capitalize() for c in classes],
                        "Confidence": probs * 100,
                    })
                    fig = px.bar(
                        score_df, x="Sentiment", y="Confidence",
                        color="Sentiment",
                        color_discrete_map={c.capitalize(): SENTIMENT_COLORS.get(c, "#6366f1") for c in classes},
                        text="Confidence",
                    )
                    fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside", marker_line_width=0)
                    fig.update_layout(showlegend=False, yaxis_range=[0, 115], yaxis_title="Confidence %", xaxis_title=None)
                    style_fig(fig, 260)
                    st.markdown('<div style="font-size:13px; font-weight:600; color:#94a3b8; margin-top:8px;">Confidence Scores</div>', unsafe_allow_html=True)
                    st.plotly_chart(fig, use_container_width=True)
        else:
            st.markdown("""
            <div style="height:340px; display:flex; align-items:center; justify-content:center;
                        background:#1e293b; border:1px dashed #334155; border-radius:16px; text-align:center;">
                <div>
                    <div style="font-size:48px; margin-bottom:12px;">🔍</div>
                    <div style="font-size:14px; font-weight:600; color:#475569;">Enter text and click Analyze</div>
                    <div style="font-size:12px; color:#334155; margin-top:4px;">Results will appear here</div>
                </div>
            </div>
            """, unsafe_allow_html=True)


# ============================================================
# POLICY ANALYTICS
# ============================================================

elif page == "Policy Analytics":

    st.title("📊 Policy Analytics")
    st.markdown('<p style="color:#64748b; margin-top:-8px; margin-bottom:24px;">Explore sentiment patterns across different policy categories.</p>', unsafe_allow_html=True)

    categories = sorted(df["category"].unique())
    selected_category = st.selectbox("Filter by Category", ["All Categories"] + categories)

    filtered_df = df[df["category"] == selected_category].copy() if selected_category != "All Categories" else df.copy()

    m1, m2, m3, m4 = st.columns(4)
    f_pos = int((filtered_df["sentiment"] == "positive").sum())
    f_neg = int((filtered_df["sentiment"] == "negative").sum())
    f_neu = int((filtered_df["sentiment"] == "neutral").sum())
    with m1: st.metric("Policies", len(filtered_df))
    with m2: st.metric("Positive", f_pos)
    with m3: st.metric("Negative", f_neg)
    with m4: st.metric("Neutral",  f_neu)

    st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)
    st.markdown('<div class="section-header">Sentiment by Policy Category</div>', unsafe_allow_html=True)

    grouped = (
        filtered_df.groupby(["category", "sentiment"]).size().reset_index(name="count")
    )
    fig = px.bar(
        grouped, x="category", y="count", color="sentiment", barmode="group",
        color_discrete_map=SENTIMENT_COLORS,
    )
    fig.update_layout(xaxis_tickangle=-25, legend_title_text="Sentiment", bargap=0.2, bargroupgap=0.08)
    fig.update_traces(marker_line_width=0)
    style_fig(fig, 400)
    st.plotly_chart(fig, use_container_width=True)

    col_pct, col_tbl = st.columns([2, 3], gap="large")

    with col_pct:
        st.markdown('<div class="section-header">Sentiment Share</div>', unsafe_allow_html=True)
        pct_df = filtered_df["sentiment"].value_counts(normalize=True).mul(100).reset_index()
        pct_df.columns = ["sentiment", "percentage"]
        fig = go.Figure(go.Bar(
            x=pct_df["sentiment"].str.capitalize(),
            y=pct_df["percentage"],
            marker_color=[SENTIMENT_COLORS.get(s, "#6366f1") for s in pct_df["sentiment"]],
            text=pct_df["percentage"].round(1).astype(str) + "%",
            textposition="outside",
        ))
        fig.update_layout(yaxis_range=[0, 115], yaxis_title="Share %", xaxis_title=None, showlegend=False)
        style_fig(fig, 320)
        st.plotly_chart(fig, use_container_width=True)

    with col_tbl:
        st.markdown('<div class="section-header">Policies in Selection</div>', unsafe_allow_html=True)
        st.dataframe(
            filtered_df[["policy_name", "category", "sentiment"]].rename(columns={
                "policy_name": "Policy Name", "category": "Category", "sentiment": "Sentiment"
            }),
            use_container_width=True, hide_index=True, height=320,
        )


# ============================================================
# DATASET EXPLORER
# ============================================================

elif page == "Dataset Explorer":

    st.title("📁 Dataset Explorer")
    st.markdown('<p style="color:#64748b; margin-top:-8px; margin-bottom:24px;">Browse, search, and download the full policy sentiment dataset.</p>', unsafe_allow_html=True)

    col_search, col_filter = st.columns([3, 1])
    with col_search:
        search = st.text_input("Search Policies", placeholder="Search by policy name or text...")
    with col_filter:
        sentiment_filter = st.selectbox("Sentiment", ["All", "positive", "negative", "neutral"])

    display_df = df.copy()
    if search:
        mask = (
            display_df["policy_name"].str.contains(search, case=False, na=False)
            | display_df["text"].str.contains(search, case=False, na=False)
        )
        display_df = display_df[mask]
    if sentiment_filter != "All":
        display_df = display_df[display_df["sentiment"] == sentiment_filter]

    col_count, col_dl = st.columns([3, 1])
    with col_count:
        st.markdown(f'<p style="color:#64748b; font-size:13px;">Showing <b style="color:#f1f5f9;">{len(display_df)}</b> of {total_records} records</p>', unsafe_allow_html=True)
    with col_dl:
        st.download_button(
            label="⬇️ Export CSV",
            data=display_df.to_csv(index=False),
            file_name="policy_sentiment_data.csv",
            mime="text/csv",
            use_container_width=True,
        )

    st.dataframe(
        display_df.rename(columns={
            "policy_id": "ID", "policy_name": "Policy Name",
            "category": "Category", "text": "Original Text",
            "clean_text": "Cleaned Text", "sentiment": "Sentiment",
        }),
        use_container_width=True, hide_index=True, height=480,
    )


# ============================================================
# MODEL PERFORMANCE
# ============================================================

elif page == "Model Performance":

    st.title("🤖 Model Performance")
    st.markdown('<p style="color:#64748b; margin-top:-8px; margin-bottom:24px;">Benchmarking of all three classification algorithms on the test split.</p>', unsafe_allow_html=True)

    if results_df is not None:

        best_row = results_df.loc[results_df["F1 Score"].idxmax()]
        st.markdown(f"""
        <div style="background:rgba(99,102,241,0.1); border:1px solid rgba(99,102,241,0.3);
                    border-radius:12px; padding:16px 22px; margin-bottom:24px;">
            <span style="font-size:13px; color:#a5b4fc; font-weight:600; letter-spacing:0.5px;">🏆 BEST MODEL</span>
            <div style="font-size:22px; font-weight:800; color:#f1f5f9; margin-top:4px;">
                {best_row['Model']}
                <span style="font-size:15px; color:#6366f1; font-weight:600; margin-left:12px;">
                    F1 Score: {best_row['F1 Score']:.2%}
                </span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="section-header">Full Comparison Table</div>', unsafe_allow_html=True)
        st.dataframe(
            results_df.style.format({
                "Accuracy": "{:.2%}", "Precision": "{:.2%}",
                "Recall": "{:.2%}", "F1 Score": "{:.2%}",
            }).background_gradient(subset=["Accuracy", "F1 Score"], cmap="Blues"),
            use_container_width=True, hide_index=True,
        )

        st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)

        col_acc, col_f1 = st.columns(2, gap="large")

        with col_acc:
            st.markdown('<div class="section-header">Accuracy</div>', unsafe_allow_html=True)
            fig = px.bar(
                results_df, x="Model", y="Accuracy", color="Model",
                color_discrete_sequence=["#6366f1", "#818cf8", "#a5b4fc"], text="Accuracy",
            )
            fig.update_traces(texttemplate="%{text:.2%}", textposition="outside", marker_line_width=0)
            fig.update_layout(showlegend=False, yaxis_range=[0, 1.2], yaxis_title="Accuracy", xaxis_title=None)
            style_fig(fig, 320)
            st.plotly_chart(fig, use_container_width=True)

        with col_f1:
            st.markdown('<div class="section-header">F1 Score</div>', unsafe_allow_html=True)
            fig = px.bar(
                results_df, x="Model", y="F1 Score", color="Model",
                color_discrete_sequence=["#10b981", "#34d399", "#6ee7b7"], text="F1 Score",
            )
            fig.update_traces(texttemplate="%{text:.2%}", textposition="outside", marker_line_width=0)
            fig.update_layout(showlegend=False, yaxis_range=[0, 1.2], yaxis_title="F1 Score", xaxis_title=None)
            style_fig(fig, 320)
            st.plotly_chart(fig, use_container_width=True)

        st.markdown('<div class="section-header">Multi-metric Radar</div>', unsafe_allow_html=True)
        metrics = ["Accuracy", "Precision", "Recall", "F1 Score"]
        r_colors = ["#6366f1", "#10b981", "#f59e0b"]
        fig = go.Figure()
        for i, row in results_df.iterrows():
            vals = [row[m] for m in metrics] + [row[metrics[0]]]
            fig.add_trace(go.Scatterpolar(
                r=vals, theta=metrics + [metrics[0]], fill="toself",
                name=row["Model"],
                line=dict(color=r_colors[i % len(r_colors)], width=2),
                fillcolor=r_colors[i % len(r_colors)] + "22",
            ))
        fig.update_layout(
            polar=dict(
                radialaxis=dict(visible=True, range=[0, 1], gridcolor="#1e293b", linecolor="#1e293b", tickfont=dict(color="#475569")),
                angularaxis=dict(tickfont=dict(color="#94a3b8")),
                bgcolor="#0f172a",
            ),
        )
        style_fig(fig, 380)
        st.plotly_chart(fig, use_container_width=True)

    else:
        st.warning("Model comparison file not found. Run `python src/train_model.py` first.")


# ============================================================
# ABOUT
# ============================================================

elif page == "About":

    st.title("ℹ️ About the Project")
    st.markdown('<p style="color:#64748b; margin-top:-8px; margin-bottom:28px;">Technical overview of the Government Policy Sentiment Analysis system.</p>', unsafe_allow_html=True)

    col_info, col_pipe = st.columns([3, 2], gap="large")

    with col_info:
        st.markdown('<div class="section-header">Project Overview</div>', unsafe_allow_html=True)
        st.markdown("""
        <p>This project applies <b style="color:#a5b4fc;">Natural Language Processing (NLP)</b> and
        <b style="color:#a5b4fc;">Machine Learning</b> to automatically classify government
        policy statements into three sentiment categories.</p>
        <br>
        <b style="color:#f1f5f9;">Sentiment Classes</b>
        <div style="margin-top:10px; display:flex; gap:10px; flex-wrap:wrap;">
            <span style="background:rgba(16,185,129,0.12); border:1px solid rgba(16,185,129,0.3);
                         color:#6ee7b7; padding:6px 16px; border-radius:20px; font-size:13px; font-weight:600;">✅ Positive</span>
            <span style="background:rgba(244,63,94,0.12); border:1px solid rgba(244,63,94,0.3);
                         color:#fda4af; padding:6px 16px; border-radius:20px; font-size:13px; font-weight:600;">❌ Negative</span>
            <span style="background:rgba(245,158,11,0.12); border:1px solid rgba(245,158,11,0.3);
                         color:#fcd34d; padding:6px 16px; border-radius:20px; font-size:13px; font-weight:600;">⚖️ Neutral</span>
        </div>
        <br>
        <b style="color:#f1f5f9;">ML Algorithms Compared</b>
        <ul style="margin-top:10px; color:#94a3b8;">
            <li>Logistic Regression</li>
            <li>Multinomial Naive Bayes</li>
            <li>Linear Support Vector Machine (LinearSVC)</li>
        </ul>
        <br>
        <b style="color:#f1f5f9;">Technology Stack</b><br>
        <div style="margin-top:10px;">
            <span class="tech-pill">Python 3.12</span>
            <span class="tech-pill">Pandas</span>
            <span class="tech-pill">NumPy</span>
            <span class="tech-pill">Scikit-learn</span>
            <span class="tech-pill">NLTK</span>
            <span class="tech-pill">TF-IDF</span>
            <span class="tech-pill">Plotly</span>
            <span class="tech-pill">Streamlit</span>
            <span class="tech-pill">Joblib</span>
        </div>
        <br>
        <div style="background:#1e293b; border:1px solid #334155; border-radius:10px;
                    padding:14px 18px; font-size:12px; color:#64748b; line-height:1.7;">
            <b style="color:#94a3b8;">⚠️ Dataset Note</b><br>
            The current dataset is a curated demo dataset with 30 synthetic records.
            For production or academic evaluation, a larger real-world corpus is recommended.
        </div>
        """, unsafe_allow_html=True)

    with col_pipe:
        st.markdown('<div class="section-header">ML Pipeline</div>', unsafe_allow_html=True)
        steps = [
            ("📄", "Raw Policy Text"),
            ("🧹", "Text Cleaning"),
            ("🔢", "TF-IDF Vectorization"),
            ("🤖", "ML Classifier Training"),
            ("📊", "Evaluation & Selection"),
            ("💾", "Model Serialization"),
            ("🌐", "Streamlit Deployment"),
        ]
        for icon, label in steps:
            st.markdown(f'<div class="pipeline-step">{icon} &nbsp; {label}</div>', unsafe_allow_html=True)
            if label != steps[-1][1]:
                st.markdown('<div class="pipeline-arrow">↓</div>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("""
        <div style="background:#1e293b; border:1px solid #334155; border-radius:10px; padding:16px 18px;">
            <div style="font-size:12px; font-weight:600; color:#64748b; letter-spacing:0.5px; margin-bottom:10px;">FEATURE ENGINEERING</div>
            <div style="font-size:13px; color:#94a3b8; line-height:1.8;">
                • Lowercase normalization<br>
                • URL &amp; punctuation removal<br>
                • English stopword removal<br>
                • Unigram + Bigram n-grams (1,2)<br>
                • Top 5,000 vocabulary terms
            </div>
        </div>
        """, unsafe_allow_html=True)
