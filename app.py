"""
app.py – Commercial Due Diligence Dashboard
=============================================
Streamlit application that ties together every audit module
into an interactive 'Deal Room' for investment analysts.
"""

import sys
import os

# Ensure src/ is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

from revenue_integrity import RevenueAuditor, EBITDANormalizer
from market_moat import MarketAnalyst, SentimentAuditor

# ======================================================================
# Paths
# ======================================================================
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

RISKY_CSV = os.path.join(DATA_DIR, "risky_co.csv")
FIN_CSV = os.path.join(DATA_DIR, "financial_statements.csv")
PRICING_CSV = os.path.join(DATA_DIR, "market_pricing.csv")
REVIEWS_CSV = os.path.join(DATA_DIR, "customer_reviews.csv")


# ======================================================================
# Cache heavy computations
# ======================================================================
@st.cache_data
def run_revenue_audit():
    auditor = RevenueAuditor()
    auditor.load_data(RISKY_CSV)
    results = auditor.run_concentration_audit()
    flags = auditor.get_red_flags()
    retention = auditor.run_cohort_analysis()
    return auditor, results, flags, retention


@st.cache_data
def run_ebitda_normalisation():
    norm = EBITDANormalizer()
    norm.load_data(FIN_CSV)
    adj_df = norm.normalize_ebitda()
    return norm, adj_df


@st.cache_data
def run_market_analysis():
    analyst = MarketAnalyst()
    analyst.load_data(PRICING_CSV)
    fig = analyst.plot_price_value_matrix(save_html=False)
    return analyst, fig


@st.cache_data
def run_sentiment_audit():
    sa = SentimentAuditor()
    sa.load_data(REVIEWS_CSV)
    result = sa.audit_sentiment()
    fig = sa.visualize_sentiment(save_html=False)
    return sa, result, fig


# ======================================================================
# Load everything
# ======================================================================
auditor, conc_results, red_flags, retention_matrix = run_revenue_audit()
ebitda_norm, ebitda_df = run_ebitda_normalisation()
market_analyst, pv_fig = run_market_analysis()
sentiment_auditor, sentiment_result, sentiment_fig = run_sentiment_audit()

# ======================================================================
# Page config
# ======================================================================
st.set_page_config(
    page_title="CDD Deal Room · Risky Co",
    page_icon="🔍",
    layout="wide",
)

# ======================================================================
# Sidebar
# ======================================================================
st.sidebar.title("🔍 CDD Deal Room")
st.sidebar.markdown("**Target:** Risky Co")
st.sidebar.markdown("**Sector:** Supply Chain Software")
st.sidebar.divider()

page = st.sidebar.radio(
    "Navigate",
    ["📊 Overview", "💰 Revenue Audit", "🏰 Market Analysis", "📋 Investment Memo"],
)


# ======================================================================
# Helper: colour coding
# ======================================================================
def health_color(score: float) -> str:
    if score < 35:
        return "🔴"
    elif score < 50:
        return "🟠"
    elif score < 65:
        return "🟡"
    return "🟢"


def hhi_color(hhi: float) -> str:
    if hhi > 2500:
        return "🔴"
    elif hhi > 1500:
        return "🟠"
    return "🟢"


# ######################################################################
#  OVERVIEW
# ######################################################################
if page == "📊 Overview":
    st.title("📊 Executive Overview")
    st.markdown("High-level risk metrics for **Risky Co** at a glance.")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric(
        "Total Revenue (24m)",
        f"${auditor.total_revenue:,.0f}",
    )
    col2.metric(
        "HHI Score",
        f"{conc_results['hhi']:,.0f}",
        delta="High concentration" if conc_results["hhi"] > 2500 else "Moderate",
        delta_color="inverse",
    )
    col3.metric(
        "Brand Health",
        f"{sentiment_auditor.brand_health_score}/100",
        delta="Critical" if sentiment_auditor.brand_health_score < 35 else "OK",
        delta_color="inverse",
    )
    col4.metric(
        "Top-1 Client Exposure",
        f"{conc_results['top_1_client_pct']:.1f}%",
        delta="Above 20% threshold" if conc_results["top_1_client_pct"] > 20 else "Healthy",
        delta_color="inverse",
    )

    st.divider()

    # Red flags summary
    st.subheader("🚩 Red Flags")
    for flag in red_flags:
        if flag.startswith("✅"):
            st.success(flag)
        else:
            st.error(flag)

    st.divider()

    # EBITDA snapshot
    st.subheader("EBITDA Normalisation Snapshot")
    reported_total = ebitda_df["Reported_EBITDA"].sum()
    adjusted_total = ebitda_df["Adjusted_EBITDA"].sum()
    uplift = (adjusted_total - reported_total) / abs(reported_total) * 100

    e1, e2, e3 = st.columns(3)
    e1.metric("Reported EBITDA (24m)", f"${reported_total:,.0f}")
    e2.metric("Adjusted EBITDA (24m)", f"${adjusted_total:,.0f}")
    e3.metric("Normalisation Uplift", f"+{uplift:.1f}%")


# ######################################################################
#  REVENUE AUDIT
# ######################################################################
elif page == "💰 Revenue Audit":
    st.title("💰 Revenue Integrity Audit")

    # --- Concentration stats ---
    st.subheader("Concentration Risk")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Top-1 Client %", f"{conc_results['top_1_client_pct']:.1f}%")
    c2.metric("Top-5 Client %", f"{conc_results['top_5_client_pct']:.1f}%")
    c3.metric("HHI", f"{conc_results['hhi']:,.0f}")
    c4.metric(
        "80/20 Rule",
        f"{conc_results['clients_for_80_revenue']} clients",
        delta=f"{conc_results['pct_clients_for_80_revenue']:.0f}% of base → 80% revenue",
        delta_color="off",
    )

    st.divider()

    # --- Cohort Retention Heatmap ---
    st.subheader("Cohort Retention Matrix (Revenue %)")
    st.markdown("Rows = cohort month · Columns = months since inception · Values = retention %")

    ret = retention_matrix.copy()
    ret.index = ret.index.strftime("%Y-%m")
    ret.columns = [f"M{int(c)}" for c in ret.columns]

    # Trim to first 13 months (M0–M12) for readability
    cols_show = [c for c in ret.columns if int(c.replace("M", "")) <= 12]
    ret_trimmed = ret[cols_show]

    heatmap = go.Figure(data=go.Heatmap(
        z=ret_trimmed.values,
        x=ret_trimmed.columns,
        y=ret_trimmed.index,
        colorscale="RdYlGn",
        zmin=0,
        zmax=200,
        text=np.round(ret_trimmed.values, 1),
        texttemplate="%{text:.0f}%",
        textfont=dict(size=10),
        hovertemplate="Cohort: %{y}<br>Month: %{x}<br>Retention: %{z:.1f}%<extra></extra>",
    ))
    heatmap.update_layout(
        height=500,
        yaxis=dict(autorange="reversed"),
        template="plotly_white",
        margin=dict(l=80, r=20, t=30, b=40),
    )
    st.plotly_chart(heatmap, use_container_width=True)

    st.divider()

    # --- EBITDA Normalisation table ---
    st.subheader("EBITDA Normalisation — Affected Months")
    affected = ebitda_df[ebitda_df["Is_OneTime"]].copy()
    affected["Month"] = affected["Month"].dt.strftime("%Y-%m")
    display_cols = ["Month", "Notes", "Reported_EBITDA", "Addback", "Adjusted_EBITDA"]
    st.dataframe(
        affected[display_cols].style.format({
            "Reported_EBITDA": "${:,.0f}",
            "Addback": "+${:,.0f}",
            "Adjusted_EBITDA": "${:,.0f}",
        }),
        use_container_width=True,
        hide_index=True,
    )


# ######################################################################
#  MARKET ANALYSIS
# ######################################################################
elif page == "🏰 Market Analysis":
    st.title("🏰 Market Moat Analysis")

    # --- Price-Value Matrix ---
    st.subheader("Price-Value Matrix")
    st.plotly_chart(pv_fig, use_container_width=True)

    st.divider()

    # --- Sentiment ---
    st.subheader("Sentiment Audit — Customer Reviews")

    s1, s2, s3 = st.columns(3)
    s1.metric(
        "Brand Health Score",
        f"{sentiment_auditor.brand_health_score}/100",
        delta="Critical" if sentiment_auditor.brand_health_score < 35 else "Mixed",
        delta_color="inverse",
    )
    reviews_df = pd.read_csv(REVIEWS_CSV)
    s2.metric("Avg Rating", f"{reviews_df['Rating'].mean():.1f} / 5")
    s3.metric("Total Reviews", f"{len(reviews_df)}")

    st.plotly_chart(sentiment_fig, use_container_width=True)

    # Category breakdown
    st.markdown("**Category Net Scores**")
    cat_data = sentiment_auditor.keyword_hits.groupby("Category").agg(
        Mentions=("Review_ID", "count"),
        Net_Score=("Weight", "sum"),
    ).sort_values("Net_Score")

    cat_fig = go.Figure(go.Bar(
        x=cat_data["Net_Score"],
        y=cat_data.index,
        orientation="h",
        marker_color=["#EF553B" if v < 0 else "#00CC96" for v in cat_data["Net_Score"]],
        text=cat_data["Net_Score"].apply(lambda x: f"{x:+d}"),
        textposition="outside",
    ))
    cat_fig.update_layout(
        xaxis_title="Net Sentiment Score",
        template="plotly_white",
        height=350,
        margin=dict(l=120),
    )
    st.plotly_chart(cat_fig, use_container_width=True)


# ######################################################################
#  INVESTMENT MEMO
# ######################################################################
elif page == "📋 Investment Memo":
    st.title("📋 Investment Memo — Risky Co")
    st.markdown("*Auto-generated MECE (Mutually Exclusive, Collectively Exhaustive) assessment*")

    st.divider()

    # --- Strengths ---
    st.subheader("✅ Strengths")

    strengths = []

    # Support score
    if sentiment_auditor.keyword_hits is not None:
        support_score = sentiment_auditor.keyword_hits[
            sentiment_auditor.keyword_hits["Category"] == "Support"
        ]["Weight"].sum()
        if support_score > 0:
            strengths.append(
                f"**Strong Customer Support** — Net support sentiment is "
                f"+{support_score}, the only consistently positive category. "
                f"This is a defensible asset and a retention lever."
            )

    # Retention
    ret_display = retention_matrix.copy()
    ret_display.columns = [f"M{int(c)}" for c in ret_display.columns]
    first_cohort = ret_display.iloc[0]
    if "M12" in first_cohort.index and first_cohort["M12"] > 75:
        strengths.append(
            f"**Steady Revenue Retention** — Earliest cohort retains "
            f"{first_cohort['M12']:.0f}% of revenue at Month 12. "
            f"No cliff-drop pattern; recurring revenue base is intact."
        )

    # EBITDA normalisation uplift
    reported_total = ebitda_df["Reported_EBITDA"].sum()
    adjusted_total = ebitda_df["Adjusted_EBITDA"].sum()
    uplift = (adjusted_total - reported_total) / abs(reported_total) * 100
    if uplift > 3:
        strengths.append(
            f"**Hidden Profitability** — After stripping one-time costs, "
            f"Adjusted EBITDA is ${adjusted_total:,.0f} "
            f"(+{uplift:.1f}% vs. reported). True margins are healthier than they appear."
        )

    if not strengths:
        strengths.append("No material strengths identified.")

    for s in strengths:
        st.markdown(f"- {s}")

    st.divider()

    # --- Weaknesses ---
    st.subheader("🚩 Weaknesses")

    weaknesses = []

    if conc_results["hhi"] > 2500:
        weaknesses.append(
            f"**Dangerous Revenue Concentration** — HHI of {conc_results['hhi']:,.0f} "
            f"exceeds the 2,500 threshold. Top client holds {conc_results['top_1_client_pct']:.1f}% "
            f"of revenue; top 5 hold {conc_results['top_5_client_pct']:.1f}%. "
            f"Only {conc_results['clients_for_80_revenue']} clients "
            f"({conc_results['pct_clients_for_80_revenue']:.0f}% of base) drive 80% of revenue. "
            f"Loss of a single whale would be catastrophic."
        )

    # Pricing overvaluation
    pricing_df = pd.read_csv(PRICING_CSV)
    x = pricing_df["Feature_Score"].values.astype(float)
    y = pricing_df["Base_Price_USD"].values.astype(float)
    slope, intercept = np.polyfit(x, y, 1)
    target_row = pricing_df[pricing_df["Competitor_Name"] == "Risky Co"].iloc[0]
    fair_val = slope * target_row["Feature_Score"] + intercept
    premium_pct = (target_row["Base_Price_USD"] - fair_val) / fair_val * 100
    if premium_pct > 20:
        weaknesses.append(
            f"**Severely Overpriced** — Risky Co charges ${target_row['Base_Price_USD']:,.0f} "
            f"vs. a fair value of ${fair_val:,.0f} (Feature Score: {target_row['Feature_Score']}/100). "
            f"That's a **+{premium_pct:.0f}% premium** with no feature justification. "
            f"This is classic 'Legacy Rent Extraction' — surviving on switching costs, not merit."
        )

    if sentiment_auditor.brand_health_score < 35:
        # Category breakdown for the memo
        pricing_hits = sentiment_auditor.keyword_hits[
            sentiment_auditor.keyword_hits["Category"] == "Pricing"
        ]["Weight"].sum()
        weaknesses.append(
            f"**Collapsing Brand Health** — Brand Health Score is "
            f"{sentiment_auditor.brand_health_score}/100 (Critical). "
            f"Pricing sentiment alone scores {pricing_hits:+d}. "
            f"Customers are vocally dissatisfied and actively evaluating alternatives. "
            f"The 'support moat' is the last line of defence."
        )

    if not weaknesses:
        weaknesses.append("No material weaknesses identified.")

    for w in weaknesses:
        st.markdown(f"- {w}")

    st.divider()

    # --- Recommendation ---
    st.subheader("🎯 Recommendation")

    hhi = conc_results["hhi"]
    bhs = sentiment_auditor.brand_health_score
    top1 = conc_results["top_1_client_pct"]

    # Decision logic
    if hhi > 2500 and bhs < 35:
        verdict = "AVOID"
        verdict_color = "🔴"
        rationale = (
            f"The combination of extreme revenue concentration (HHI {hhi:,.0f}) and "
            f"critically low Brand Health ({bhs}/100) creates an untenable risk profile. "
            f"Risky Co is executing a **Legacy Rent Extraction** strategy — charging "
            f"a +{premium_pct:.0f}% premium over fair value while delivering mid-tier "
            f"features ({target_row['Feature_Score']}/100). Customers are already revolting "
            f"(avg rating {reviews_df['Rating'].mean():.1f}/5), and the only anchor is "
            f"support quality and switching costs.\n\n"
            f"**When the top {conc_results['clients_for_80_revenue']} clients renegotiate or churn, "
            f"the revenue base collapses.** This is not a question of *if*, but *when*."
        )
    elif hhi > 2500 or bhs < 50:
        verdict = "HEAVY DISCOUNT ON VALUATION"
        verdict_color = "🟠"
        rationale = (
            f"Material risks exist in concentration (HHI {hhi:,.0f}) or brand health "
            f"({bhs}/100) that warrant a significant haircut. Proceed only with a "
            f"restructuring thesis and aggressive price adjustment."
        )
    else:
        verdict = "PROCEED WITH STANDARD DILIGENCE"
        verdict_color = "🟢"
        rationale = "Metrics are within acceptable thresholds for the sector."

    st.markdown(
        f"""
        <div style="
            background: {'#2d1117' if verdict == 'AVOID' else '#2d2517' if 'DISCOUNT' in verdict else '#112d17'};
            border-left: 6px solid {'#ff4444' if verdict == 'AVOID' else '#ffaa00' if 'DISCOUNT' in verdict else '#44ff44'};
            padding: 20px;
            border-radius: 8px;
            margin: 10px 0;
        ">
            <h2 style="margin-top:0;">{verdict_color} {verdict}</h2>
            <p style="font-size: 1.05em;">{rationale}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.divider()

    # --- Summary table ---
    st.subheader("📊 Scorecard")
    scorecard = pd.DataFrame([
        {"Dimension": "Revenue Concentration (HHI)", "Score": f"{hhi:,.0f}", "Threshold": "< 2,500", "Status": hhi_color(hhi)},
        {"Dimension": "Top-1 Client Exposure", "Score": f"{top1:.1f}%", "Threshold": "< 20%", "Status": "🔴" if top1 > 20 else "🟢"},
        {"Dimension": "Brand Health", "Score": f"{bhs}/100", "Threshold": "> 50", "Status": health_color(bhs)},
        {"Dimension": "Price Premium", "Score": f"+{premium_pct:.0f}%", "Threshold": "< +20%", "Status": "🔴" if premium_pct > 20 else "🟢"},
        {"Dimension": "M12 Retention", "Score": f"{first_cohort.get('M12', 0):.0f}%", "Threshold": "> 75%", "Status": "🟢" if first_cohort.get('M12', 0) > 75 else "🔴"},
        {"Dimension": "EBITDA Uplift", "Score": f"+{uplift:.1f}%", "Threshold": "Info", "Status": "ℹ️"},
    ])
    st.dataframe(scorecard, use_container_width=True, hide_index=True)
