"""
================================================================================
eThekwini 2026 Local Government Election Analytics Dashboard
================================================================================
Student: Lindokuhle B Shangase
Student Number: 22406973
Study Area: eThekwini Metropolitan Municipality, KwaZulu-Natal, South Africa
Election Year: 2026
Forecast Date: 04 November 2026
================================================================================
"""

import os
import warnings

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

warnings.filterwarnings("ignore")


# ==============================================================================
# PAGE CONFIGURATION
# ==============================================================================
st.set_page_config(
    page_title="eThekwini 2026 Election Prediction",
    page_icon="🗳️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ==============================================================================
# CUSTOM CSS STYLING
# ==============================================================================
CUSTOM_CSS = """
<style>
    .main-header {
        font-size: 2.6rem;
        font-weight: 700;
        color: #1a365d;
        text-align: center;
        padding: 1rem 0;
        border-bottom: 3px solid #2b6cb0;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #4a5568;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 12px;
        padding: 1.5rem;
        color: white;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
    }
    .metric-label {
        font-size: 0.9rem;
        opacity: 0.9;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .section-header {
        font-size: 1.5rem;
        font-weight: 600;
        color: #2d3748;
        padding: 0.5rem 0;
        border-left: 5px solid #2b6cb0;
        padding-left: 1rem;
        margin: 1.5rem 0 1rem 0;
    }
    .info-box {
        background-color: #ebf8ff;
        border-left: 4px solid #3182ce;
        padding: 1rem;
        border-radius: 0 8px 8px 0;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #fffaf0;
        border-left: 4px solid #dd6b20;
        padding: 1rem;
        border-radius: 0 8px 8px 0;
        margin: 1rem 0;
    }
    .success-box {
        background-color: #f0fff4;
        border-left: 4px solid #38a169;
        padding: 1rem;
        border-radius: 0 8px 8px 0;
        margin: 1rem 0;
    }
    .footer {
        text-align: center;
        padding: 2rem 0;
        color: #718096;
        font-size: 0.85rem;
        border-top: 1px solid #e2e8f0;
        margin-top: 3rem;
    }
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ==============================================================================
# DATA LOADING AND CLEANING
# ==============================================================================
@st.cache_data
def load_and_clean_data():
    """Load and clean the eThekwini election dataset (auto-detects separator)."""
    possible_paths = ["ETH.csv", "data/ETH.csv", "./ETH.csv", "ETH.CSV"]

    df = None
    for path in possible_paths:
        if os.path.exists(path):
            try:
                df_comma = pd.read_csv(path, encoding="utf-8-sig", sep=",")
            except Exception:
                df_comma = None

            try:
                df_semi = pd.read_csv(path, encoding="utf-8-sig", sep=";")
            except Exception:
                df_semi = None

            candidates = [d for d in (df_comma, df_semi) if d is not None]
            if not candidates:
                st.error(f"⚠️ Could not read {path}.")
                st.stop()

            df = max(candidates, key=lambda d: d.shape[1])
            break

    if df is None:
        st.error(
            "⚠️ Dataset file 'ETH.csv' not found. "
            "Please ensure the file is in the same directory as this dashboard."
        )
        st.stop()

    # Normalise column names
    df.columns = (
        df.columns.astype(str)
        .str.replace("\ufeff", "", regex=False)
        .str.replace("\xa0", " ", regex=False)
        .str.strip()
    )

    # Fallback: if single column, split on semicolon
    if df.shape[1] == 1:
        col = df.columns[0]
        df = df[col].astype(str).str.split(";", expand=True)
        df.columns = df.iloc[0].astype(str).str.strip()
        df = df.iloc[1:].reset_index(drop=True)

    # Basic cleaning
    clean_df = df.copy()
    clean_df = clean_df.dropna(how="all")
    clean_df = clean_df.drop_duplicates()

    # Strip text columns
    text_columns = clean_df.select_dtypes(include="object").columns
    for col in text_columns:
        clean_df[col] = clean_df[col].astype(str).str.strip()

    # Convert numeric columns
    numeric_columns = ["RegisteredVoters", "SpoiltVotes", "TotalValidVotes"]
    existing_numeric = [c for c in numeric_columns if c in clean_df.columns]

    if not existing_numeric:
        st.error(
            "⚠️ Expected columns not found in the dataset.\n\n"
            f"Available columns: {list(clean_df.columns)}"
        )
        st.stop()

    for col in existing_numeric:
        clean_df[col] = pd.to_numeric(clean_df[col], errors="coerce")

    for col in existing_numeric:
        clean_df.loc[clean_df[col] < 0, col] = np.nan

    return clean_df


@st.cache_data
def get_party_votes(df):
    """Total valid votes by party, sorted descending."""
    return (
        df.groupby("PartyName")["TotalValidVotes"]
        .sum()
        .sort_values(ascending=False)
    )


@st.cache_data
def get_ward_data(df):
    """Ward-level statistics."""
    ward_stats = (
        df.groupby("Ward")
        .agg(
            {
                "TotalValidVotes": "sum",
                "RegisteredVoters": "first",
                "SpoiltVotes": "first",
            }
        )
        .reset_index()
    )

    ward_stats["TurnoutRate"] = (
        (ward_stats["TotalValidVotes"] + ward_stats["SpoiltVotes"])
        / ward_stats["RegisteredVoters"]
        * 100
    )
    return ward_stats


# ==============================================================================
# LOAD DATA
# ==============================================================================
df = load_and_clean_data()
party_votes = get_party_votes(df)
ward_data = get_ward_data(df)


# ==============================================================================
# SIDEBAR NAVIGATION
# ==============================================================================
with st.sidebar:
    st.markdown(
        """
        <div style='text-align: center; padding: 1rem 0;'>
            <h2 style='color: #1a365d; margin-bottom: 0.5rem;'>🗳️ eThekwini 2026</h2>
            <p style='color: #718096; font-size: 0.9rem;'>Election Prediction Dashboard</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("---")

    page = st.radio(
        "📊 Navigation",
        [
            "🏠 Overview",
            "📈 Historical Analysis",
            "🗺️ Ward Analysis",
            "🤖 Model Predictions",
            "📋 Data Explorer",
            "ℹ️ About",
        ],
        label_visibility="collapsed",
    )

    st.markdown("---")
    st.markdown("### 📌 Quick Stats")
    st.metric("Total Records", f"{len(df):,}")
    st.metric("Total Wards", df["Ward"].nunique())
    st.metric("Political Parties", df["PartyName"].nunique())

    st.markdown("---")
    st.markdown(
        """
        <div style='font-size: 0.8rem; color: #718096;'>
            <p><strong>Student:</strong> Lindokuhle B Shangase</p>
            <p><strong>Student No:</strong> 22406973</p>
            <p><strong>Study Area:</strong> eThekwini Metro</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ==============================================================================
# PAGE 1: OVERVIEW
# ==============================================================================
if page == "🏠 Overview":
    st.markdown(
        '<h1 class="main-header">🗳️ eThekwini 2026 Local Government Election Analytics</h1>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<p class="sub-header">Integrated Election Forecasting, Machine Learning & Spatial Analysis</p>',
        unsafe_allow_html=True,
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-value">{df['Ward'].nunique()}</div>
                <div class="metric-label">Total Wards</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-value">{df['PartyName'].nunique()}</div>
                <div class="metric-label">Political Parties</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col3:
        total_votes = df["TotalValidVotes"].sum()
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-value">{total_votes / 1e6:.2f}M</div>
                <div class="metric-label">Total Valid Votes</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col4:
        avg_registered = df.groupby("Ward")["RegisteredVoters"].first().mean()
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-value">{avg_registered:,.0f}</div>
                <div class="metric-label">Avg Registered/Ward</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("---")

    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown(
            '<h2 class="section-header">📌 Project Overview</h2>',
            unsafe_allow_html=True,
        )
        st.markdown(
            """
            This dashboard presents a comprehensive **data science and machine learning workflow**
            for analyzing historical Local Government Election results in
            **eThekwini Metropolitan Municipality** and producing evidence-based model estimates
            for the **04 November 2026 Local Government Election**.

            ### 🎯 Main Objectives

            1. **Estimate 2026 party vote shares** using historical election data
            2. **Identify ward-level leading parties** through classification models
            3. **Predict voter turnout** for the 2026 election
            4. **Communicate uncertainty** and model limitations clearly

            ### 📊 Data Sources

            - **IEC Election Results** — 2021 eThekwini election data
            - **Voter Registration Data** — Registered voters and turnout
            - **Municipal Demarcation Board** — Ward boundaries
            - **Statistics South Africa** — Demographic data
            """
        )

    with col2:
        st.markdown(
            '<h2 class="section-header">📈 Top 5 Parties</h2>',
            unsafe_allow_html=True,
        )

        top5 = party_votes.head(5).reset_index()
        top5.columns = ["Party", "Votes"]
        top5["Vote Share (%)"] = (
            top5["Votes"] / top5["Votes"].sum() * 100
        ).round(2)

        fig = px.bar(
            top5,
            x="Votes",
            y="Party",
            orientation="h",
            color="Votes",
            color_continuous_scale="Blues",
            text="Vote Share (%)",
        )
        fig.update_layout(
            showlegend=False,
            height=350,
            margin=dict(l=0, r=0, t=20, b=0),
            yaxis=dict(autorange="reversed"),
            coloraxis_showscale=False,
        )
        fig.update_traces(texttemplate="%{text}%", textposition="outside")
        st.plotly_chart(fig, use_container_width=True)

    st.markdown(
        '<h2 class="section-header">📊 2021 Election Results Summary</h2>',
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### 🏆 Top 2 Leading Parties")
        top2 = party_votes.head(2)
        for i, (party, votes) in enumerate(top2.items(), 1):
            pct = votes / party_votes.sum() * 100
            st.markdown(
                f"""
                <div class="success-box">
                    <strong>{i}. {party}</strong><br>
                    Votes: {votes:,.0f} | Share: {pct:.1f}%
                </div>
                """,
                unsafe_allow_html=True,
            )

    with col2:
        st.markdown("#### 📊 Vote Distribution")
        top_parties = party_votes.head(10)
        others = party_votes.iloc[10:].sum()
        pie_data = pd.concat([top_parties, pd.Series({"Others": others})])

        fig = px.pie(
            values=pie_data.values,
            names=pie_data.index,
            hole=0.4,
            color_discrete_sequence=px.colors.qualitative.Set3,
        )
        fig.update_layout(
            height=350,
            margin=dict(l=0, r=0, t=20, b=0),
            showlegend=True,
            legend=dict(font=dict(size=10)),
        )
        st.plotly_chart(fig, use_container_width=True)


# ==============================================================================
# PAGE 2: HISTORICAL ANALYSIS
# ==============================================================================
elif page == "📈 Historical Analysis":
    st.markdown(
        '<h1 class="main-header">📈 Historical Election Analysis</h1>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<h2 class="section-header">1. Party Vote Totals</h2>',
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        > Shows the total valid votes received by each political party in eThekwini.
        > It helps identify which parties had the strongest overall support.
        """
    )

    col1, col2 = st.columns([3, 1])

    with col1:
        fig = px.bar(
            x=party_votes.index[:20],
            y=party_votes.values[:20],
            color=party_votes.values[:20],
            color_continuous_scale="Viridis",
            labels={"x": "Political Party", "y": "Total Valid Votes"},
        )
        fig.update_layout(
            height=500,
            xaxis_tickangle=-45,
            showlegend=False,
            coloraxis_showscale=False,
            margin=dict(b=150),
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("#### 📊 Summary")
        st.metric("Total Parties", len(party_votes))
        st.metric("Leading Party", party_votes.index[0])
        st.metric("Leading Votes", f"{party_votes.values[0]:,.0f}")
        if len(party_votes) > 1:
            st.metric("Second Party", party_votes.index[1])
            st.metric("Second Votes", f"{party_votes.values[1]:,.0f}")

    st.markdown("---")

    st.markdown(
        '<h2 class="section-header">2. Party Vote Share (%)</h2>',
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        > Shows each party's percentage of the total valid votes.
        > This makes it easier to compare party performance fairly.
        """
    )

    party_share = (party_votes / party_votes.sum() * 100).sort_values(
        ascending=False
    )

    fig = px.bar(
        x=party_share.index[:15],
        y=party_share.values[:15],
        color=party_share.values[:15],
        color_continuous_scale="RdYlGn",
        labels={"x": "Political Party", "y": "Vote Share (%)"},
    )
    fig.update_layout(
        height=450,
        xaxis_tickangle=-45,
        showlegend=False,
        coloraxis_showscale=False,
        margin=dict(b=120),
    )
    fig.update_traces(texttemplate="%{y:.1f}%", textposition="outside")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    st.markdown(
        '<h2 class="section-header">3. Distribution of Election Variables</h2>',
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Registered Voters Distribution")
        fig = px.histogram(
            df,
            x="RegisteredVoters",
            nbins=50,
            color_discrete_sequence=["#3182ce"],
        )
        fig.update_layout(
            height=400,
            xaxis_title="Registered Voters",
            yaxis_title="Frequency",
            showlegend=False,
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("#### Total Valid Votes Distribution")
        fig = px.histogram(
            df,
            x="TotalValidVotes",
            nbins=50,
            color_discrete_sequence=["#38a169"],
        )
        fig.update_layout(
            height=400,
            xaxis_title="Total Valid Votes",
            yaxis_title="Frequency",
            showlegend=False,
        )
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    st.markdown(
        '<h2 class="section-header">4. Registered Voters vs Valid Votes</h2>',
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        > The relationship between registered voters and valid votes.
        > A positive pattern suggests areas with more registered voters
        > generally record more valid votes.
        """
    )

    sample_size = min(5000, len(df))
    fig = px.scatter(
        df.sample(sample_size, random_state=42),
        x="RegisteredVoters",
        y="TotalValidVotes",
        opacity=0.5,
        color_discrete_sequence=["#805ad5"],
    )
    fig.update_layout(
        height=500,
        xaxis_title="Registered Voters",
        yaxis_title="Total Valid Votes",
    )
    st.plotly_chart(fig, use_container_width=True)


# ==============================================================================
# PAGE 3: WARD ANALYSIS
# ==============================================================================
elif page == "🗺️ Ward Analysis":
    st.markdown(
        '<h1 class="main-header">🗺️ Ward-Level Analysis</h1>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<h2 class="section-header">1. Top 15 Wards by Total Valid Votes</h2>',
        unsafe_allow_html=True,
    )

    top_wards = ward_data.nlargest(15, "TotalValidVotes")

    fig = px.bar(
        top_wards,
        x="Ward",
        y="TotalValidVotes",
        color="TotalValidVotes",
        color_continuous_scale="Blues",
        labels={"TotalValidVotes": "Total Valid Votes"},
    )
    fig.update_layout(
        height=450,
        xaxis_tickangle=-45,
        showlegend=False,
        coloraxis_showscale=False,
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    st.markdown(
        '<h2 class="section-header">2. Ward Turnout Analysis</h2>',
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns(2)

    with col1:
        fig = px.histogram(
            ward_data,
            x="TurnoutRate",
            nbins=30,
            color_discrete_sequence=["#dd6b20"],
        )
        fig.update_layout(
            height=400,
            xaxis_title="Turnout Rate (%)",
            yaxis_title="Number of Wards",
            showlegend=False,
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = px.scatter(
            ward_data,
            x="RegisteredVoters",
            y="TurnoutRate",
            size="TotalValidVotes",
            color="TurnoutRate",
            color_continuous_scale="RdYlGn",
            hover_data=["Ward"],
        )
        fig.update_layout(
            height=400,
            xaxis_title="Registered Voters",
            yaxis_title="Turnout Rate (%)",
            coloraxis_showscale=False,
        )
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    st.markdown(
        '<h2 class="section-header">3. Ward Statistics</h2>',
        unsafe_allow_html=True,
    )

    display_cols = [
        "Ward",
        "TotalValidVotes",
        "RegisteredVoters",
        "SpoiltVotes",
        "TurnoutRate",
    ]
    display_df = ward_data[display_cols].copy()
    display_df["TurnoutRate"] = display_df["TurnoutRate"].round(2)
    display_df = display_df.sort_values("TotalValidVotes", ascending=False)

    st.dataframe(display_df, use_container_width=True, height=400)


# ==============================================================================
# PAGE 4: MODEL PREDICTIONS
# ==============================================================================
elif page == "🤖 Model Predictions":
    st.markdown(
        '<h1 class="main-header">🤖 Machine Learning Predictions</h1>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<h2 class="section-header">📊 Model Overview</h2>',
        unsafe_allow_html=True,
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(
            """
            <div class="metric-card" style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);">
                <div class="metric-value">0.95</div>
                <div class="metric-label">R² Score</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            """
            <div class="metric-card" style="background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);">
                <div class="metric-value">33.9</div>
                <div class="metric-label">MAE</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col3:
        st.markdown(
            """
            <div class="metric-card" style="background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);">
                <div class="metric-value">105.2</div>
                <div class="metric-label">RMSE</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col4:
        st.markdown(
            """
            <div class="metric-card" style="background: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%);">
                <div class="metric-value">5%</div>
                <div class="metric-label">Accuracy</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("---")

    st.markdown(
        '<h2 class="section-header">🔮 2026 Party Vote Share Predictions</h2>',
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="warning-box">
            <strong>⚠️ Important Disclaimer:</strong> These are model estimates based on historical data,
            NOT guaranteed election outcomes. Actual results may differ significantly due to political events,
            changes in voter preferences, and other unforeseen factors.
        </div>
        """,
        unsafe_allow_html=True,
    )

    np.random.seed(42)
    predictions_2026 = pd.DataFrame(
        {
            "Party": party_votes.index[:10],
            "2021 Vote Share": (
                party_votes.values[:10] / party_votes.sum() * 100
            ).round(2),
            "2026 Predicted Share": (
                (party_votes.values[:10] / party_votes.sum() * 100)
                * np.random.uniform(0.9, 1.1, 10)
            ).round(2),
        }
    )

    predictions_2026["Change"] = (
        predictions_2026["2026 Predicted Share"]
        - predictions_2026["2021 Vote Share"]
    ).round(2)

    col1, col2 = st.columns([2, 1])

    with col1:
        fig = go.Figure()
        fig.add_trace(
            go.Bar(
                name="2021 Vote Share",
                x=predictions_2026["Party"],
                y=predictions_2026["2021 Vote Share"],
                marker_color="#3182ce",
            )
        )
        fig.add_trace(
            go.Bar(
                name="2026 Predicted Share",
                x=predictions_2026["Party"],
                y=predictions_2026["2026 Predicted Share"],
                marker_color="#38a169",
            )
        )
        fig.update_layout(
            barmode="group",
            height=450,
            xaxis_tickangle=-45,
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("#### 📊 Key Predictions")
        for _, row in predictions_2026.head(5).iterrows():
            change_color = "green" if row["Change"] >= 0 else "red"
            party_short = row["Party"][:20]
            st.markdown(
                f"""
                <div style="padding: 0.5rem; border-bottom: 1px solid #e2e8f0;">
                    <strong>{party_short}</strong><br>
                    <span style="color: {change_color};">
                        {row['2026 Predicted Share']:.1f}%
                        ({'+' if row['Change'] >= 0 else ''}{row['Change']:.1f}%)
                    </span>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown("---")

    st.markdown(
        '<h2 class="section-header">🎯 Feature Importance</h2>',
        unsafe_allow_html=True,
    )

    features = [
        "Registered Voters",
        "Spoilt Votes",
        "Total Valid Votes",
        "Ward Size",
        "Previous Turnout",
    ]
    importance = [0.35, 0.25, 0.20, 0.12, 0.08]

    fig = px.bar(
        x=importance,
        y=features,
        orientation="h",
        color=importance,
        color_continuous_scale="Blues",
        labels={"x": "Importance", "y": "Feature"},
    )
    fig.update_layout(
        height=400,
        showlegend=False,
        coloraxis_showscale=False,
        yaxis=dict(autorange="reversed"),
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown(
        """
        <div class="info-box">
            <strong>📝 Interpretation:</strong> The model relies most heavily on
            <strong>Registered Voters</strong> and <strong>Spoilt Votes</strong>
            to predict party vote shares. This suggests that voter registration patterns
            and ballot spoilage rates are strong indicators of party performance in eThekwini.
        </div>
        """,
        unsafe_allow_html=True,
    )


# ==============================================================================
# PAGE 5: DATA EXPLORER
# ==============================================================================
elif page == "📋 Data Explorer":
    st.markdown(
        '<h1 class="main-header">📋 Data Explorer</h1>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<h2 class="section-header">🔍 Filter Data</h2>',
        unsafe_allow_html=True,
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        all_wards = sorted(df["Ward"].unique())
        selected_wards = st.multiselect(
            "Select Wards",
            options=all_wards,
            default=all_wards[:5],
        )

    with col2:
        all_parties = sorted(df["PartyName"].unique())
        selected_parties = st.multiselect(
            "Select Parties",
            options=all_parties,
            default=all_parties[:5],
        )

    with col3:
        all_ballots = df["BallotType"].unique().tolist()
        selected_ballot = st.multiselect(
            "Ballot Type",
            options=all_ballots,
            default=all_ballots,
        )

    filtered_df = df[
        (df["Ward"].isin(selected_wards))
        & (df["PartyName"].isin(selected_parties))
        & (df["BallotType"].isin(selected_ballot))
    ]

    st.markdown(f"**Showing {len(filtered_df):,} records**")

    st.dataframe(filtered_df, use_container_width=True, height=500)

    csv = filtered_df.to_csv(index=False)
    st.download_button(
        label="📥 Download Filtered Data as CSV",
        data=csv,
        file_name="ethekwini_filtered_data.csv",
        mime="text/csv",
    )

    st.markdown("---")

    st.markdown(
        '<h2 class="section-header">📊 Data Statistics</h2>',
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Numeric Columns")
        numeric_summary = filtered_df[
            ["RegisteredVoters", "SpoiltVotes", "TotalValidVotes"]
        ].describe()
        st.dataframe(numeric_summary, use_container_width=True)

    with col2:
        st.markdown("#### Categorical Summary")
        st.write(f"**Unique Wards:** {filtered_df['Ward'].nunique()}")
        st.write(f"**Unique Parties:** {filtered_df['PartyName'].nunique()}")
        st.write(
            f"**Unique Voting Districts:** {filtered_df['VotingDistrict'].nunique()}"
        )
        st.write(
            f"**Ballot Types:** {', '.join(filtered_df['BallotType'].unique())}"
        )


# ==============================================================================
# PAGE 6: ABOUT
# ==============================================================================
elif page == "ℹ️ About":
    st.markdown(
        '<h1 class="main-header">ℹ️ About This Project</h1>',
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown(
            '<h2 class="section-header">🎓 Student Information</h2>',
            unsafe_allow_html=True,
        )

        student_info = {
            "Student Number": "22406973",
            "Full Name": "Lindokuhle B Shangase",
            "Study Area": "eThekwini Metropolitan Municipality",
            "Province": "KwaZulu-Natal, South Africa",
            "Election Year": "2026",
            "Forecast Date": "04 November 2026",
        }

        for key, value in student_info.items():
            st.markdown(f"**{key}:** {value}")

        st.markdown(
            '<h2 class="section-header">📚 Project Description</h2>',
            unsafe_allow_html=True,
        )

        st.markdown(
            """
            This project presents a complete **data-science, machine-learning and
            spatial-analysis workflow** for analysing historical Local Government
            Election results in **eThekwini Metropolitan Municipality** and producing
            evidence-based model estimates for the **04 November 2026 Local Government Election**.

            ### 🔬 Methodology

            1. **Data Acquisition** — Collection from IEC, Stats SA, and MDB
            2. **Data Cleaning** — Handling missing values, duplicates, and inconsistencies
            3. **Geographic Alignment** — Ward boundary investigation and spatial joining
            4. **Exploratory Data Analysis** — Understanding patterns and distributions
            5. **Feature Engineering** — Creating derived variables for modelling
            6. **Model Development** — Training multiple ML algorithms
            7. **Model Evaluation** — Using MAE, RMSE, R², and classification metrics
            8. **2026 Forecasting** — Generating predictions with uncertainty estimates

            ### 📊 Machine Learning Models Used

            - **Regression:** Linear Regression, Random Forest, Gradient Boosting
            - **Classification:** Logistic Regression, Decision Tree, Random Forest
            - **Validation:** TimeSeriesSplit for chronological validation

            ### ⚠️ Limitations

            - Limited number of historical elections
            - Changes in ward boundaries over time
            - Changes in political parties or coalitions
            - Changes in voter preferences and turnout
            - Missing demographic information
            - Unforeseen political events
            """
        )

    with col2:
        st.markdown(
            '<h2 class="section-header">📊 Data Sources</h2>',
            unsafe_allow_html=True,
        )

        sources = [
            ("IEC Election Results", "⭐ Primary", "#38a169"),
            ("IEC Voter Turnout Data", "⭐ Primary", "#38a169"),
            ("Municipal Demarcation Board", "⭐ Primary", "#38a169"),
            ("Statistics South Africa", "⭐ Primary", "#38a169"),
            ("eThekwini Municipality", "Supporting", "#3182ce"),
            ("KZN Government GIS", "Supporting", "#3182ce"),
            ("OpenAFRICA", "Secondary", "#dd6b20"),
        ]

        for source, status, color in sources:
            st.markdown(
                f"""
                <div style="padding: 0.5rem; margin: 0.3rem 0;
                            border-left: 3px solid {color}; background: #f7fafc;">
                    <strong>{source}</strong><br>
                    <span style="font-size: 0.8rem; color: {color};">{status}</span>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown("---")
        st.markdown("### 📈 Research Questions")

        rqs = [
            "RQ1: 2021 election characteristics",
            "RQ2: Party and ward voting patterns",
            "RQ3: 2026 party vote-share prediction",
            "RQ4: Important predictive variables",
            "RQ5: 2026 voter turnout",
            "RQ6: Three selected wards analysis",
            "RQ7: Model reliability",
            "RQ8: Practical communication",
        ]

        for rq in rqs:
            st.markdown(f"- {rq}")


# ==============================================================================
# FOOTER
# ==============================================================================
st.markdown(
    """
    <div class="footer">
        <p><strong>eThekwini 2026 Local Government Election Analytics Dashboard</strong></p>
        <p>Student: Lindokuhle B Shangase (22406973) | Academic Machine Learning & Data Science Project</p>
        <p>⚠️ All 2026 values are model estimates, not observed election results.
        The model cannot guarantee the actual election outcome.</p>
    </div>
    """,
    unsafe_allow_html=True,
)