"""
market_moat.py – Competitive Price-Value Analysis
====================================================
Plots a Price vs. Feature scatter with a Fair Value regression line
to identify overpriced and underpriced competitors.
"""

import os
import numpy as np
import pandas as pd
import plotly.graph_objects as go


class MarketAnalyst:
    """Analyses competitive positioning via price-value benchmarking."""

    TARGET_NAME = "Risky Co"

    def __init__(self):
        self.df: pd.DataFrame | None = None

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------
    def load_data(self, file_path: str) -> pd.DataFrame:
        """Read market_pricing.csv."""
        if not os.path.isfile(file_path):
            raise FileNotFoundError(f"No file found at: {file_path}")

        self.df = pd.read_csv(file_path)

        required = {"Competitor_Name", "Base_Price_USD", "Feature_Score", "Market_Share"}
        missing = required - set(self.df.columns)
        if missing:
            raise ValueError(f"CSV is missing required columns: {missing}")

        print(f"✓ Loaded {len(self.df)} competitors")
        return self.df

    # ------------------------------------------------------------------
    # Price-Value Matrix
    # ------------------------------------------------------------------
    def plot_price_value_matrix(self, save_html: bool = True) -> go.Figure:
        """
        Scatter: X = Feature_Score, Y = Base_Price_USD
        Bubble size = Market_Share
        Includes an OLS Fair Value Line.
        """
        if self.df is None:
            raise RuntimeError("Call load_data() first.")

        df = self.df.copy()

        # --- Simple linear regression for the Fair Value Line ---
        x = df["Feature_Score"].values.astype(float)
        y = df["Base_Price_USD"].values.astype(float)
        slope, intercept = np.polyfit(x, y, 1)
        x_line = np.linspace(x.min() - 5, x.max() + 5, 100)
        y_line = slope * x_line + intercept

        # Classify each company relative to the line
        df["Fair_Value"] = slope * df["Feature_Score"] + intercept
        df["Premium_Pct"] = ((df["Base_Price_USD"] - df["Fair_Value"]) / df["Fair_Value"] * 100).round(1)
        df["Zone"] = np.where(df["Premium_Pct"] > 10, "Overpriced",
                     np.where(df["Premium_Pct"] < -10, "Good Value", "Fair Value"))

        # Colour map
        zone_colors = {"Overpriced": "#EF553B", "Good Value": "#00CC96", "Fair Value": "#636EFA"}
        df["Color"] = df["Zone"].map(zone_colors)

        # Highlight target
        is_target = df["Competitor_Name"] == self.TARGET_NAME

        # --- Build Plotly figure ---
        fig = go.Figure()

        # Fair Value Line
        fig.add_trace(go.Scatter(
            x=x_line, y=y_line,
            mode="lines",
            name="Fair Value Line",
            line=dict(dash="dash", color="grey", width=2),
        ))

        # Competitor bubbles (non-target)
        others = df[~is_target]
        fig.add_trace(go.Scatter(
            x=others["Feature_Score"],
            y=others["Base_Price_USD"],
            mode="markers+text",
            name="Competitors",
            text=others["Competitor_Name"],
            textposition="top center",
            marker=dict(
                size=others["Market_Share"] * 500,
                color=others["Color"],
                opacity=0.7,
                line=dict(width=1, color="DarkSlateGrey"),
            ),
            hovertemplate=(
                "<b>%{text}</b><br>"
                "Price: $%{y:,.0f}<br>"
                "Features: %{x}/100<br>"
                "Premium: %{customdata[0]:+.1f}%<br>"
                "Zone: %{customdata[1]}<extra></extra>"
            ),
            customdata=others[["Premium_Pct", "Zone"]].values,
        ))

        # Target bubble (Risky Co) – emphasised
        target = df[is_target]
        fig.add_trace(go.Scatter(
            x=target["Feature_Score"],
            y=target["Base_Price_USD"],
            mode="markers+text",
            name=self.TARGET_NAME,
            text=[self.TARGET_NAME],
            textposition="top center",
            textfont=dict(size=14, color="red"),
            marker=dict(
                size=target["Market_Share"].values * 500,
                color="#EF553B",
                opacity=1.0,
                line=dict(width=3, color="black"),
                symbol="star",
            ),
            hovertemplate=(
                "<b>%{text}</b> ⭐ TARGET<br>"
                "Price: $%{y:,.0f}<br>"
                "Features: %{x}/100<br>"
                "Premium: %{customdata[0]:+.1f}%<br>"
                "Zone: %{customdata[1]}<extra></extra>"
            ),
            customdata=target[["Premium_Pct", "Zone"]].values,
        ))

        # Layout
        fig.update_layout(
            title=dict(
                text="Price-Value Matrix · Supply Chain Software",
                font=dict(size=20),
            ),
            xaxis_title="Feature Score (1–100)",
            yaxis_title="Base Price (USD)",
            template="plotly_white",
            width=950,
            height=600,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )

        # Save to HTML
        if save_html:
            out_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
            html_path = os.path.join(out_dir, "price_value_matrix.html")
            fig.write_html(html_path)
            print(f"✓ Interactive chart saved → {html_path}")

        # --- Console summary ---
        target_row = target.iloc[0]
        print("\n" + "=" * 60)
        print("  PRICE-VALUE ANALYSIS SUMMARY")
        print("=" * 60)
        print(f"  Fair Value Line : Price = {slope:.1f} × FeatureScore + {intercept:,.0f}")
        print(f"\n  {'Competitor':<18} {'Price':>8} {'Features':>9} {'Fair$':>8} {'Premium':>9} {'Zone'}")
        print("  " + "-" * 56)
        for _, row in df.sort_values("Premium_Pct", ascending=False).iterrows():
            marker = " ⭐" if row["Competitor_Name"] == self.TARGET_NAME else ""
            print(
                f"  {row['Competitor_Name']:<18} "
                f"${row['Base_Price_USD']:>6,} "
                f"{row['Feature_Score']:>8} "
                f"${row['Fair_Value']:>7,.0f} "
                f"{row['Premium_Pct']:>+8.1f}% "
                f"{row['Zone']}{marker}"
            )
        print("=" * 60)

        return fig


# ======================================================================
# Sentiment Auditor
# ======================================================================
class SentimentAuditor:
    """
    Keyword-based sentiment engine for customer reviews.
    Scans Review_Text for red-flag and green-flag keywords,
    computes a Brand Health Score (1–100), and visualises results.
    """

    # Keyword → (category, sentiment_weight)
    #   weight < 0 = negative, > 0 = positive
    KEYWORD_MAP = {
        # Negative / red-flag
        "expensive":    ("Pricing",   -2),
        "overpriced":   ("Pricing",   -2),
        "cost":         ("Pricing",   -1),
        "price":        ("Pricing",   -1),
        "cash grab":    ("Pricing",   -2),
        "buggy":        ("Quality",   -2),
        "bug":          ("Quality",   -1),
        "slow":         ("Performance", -2),
        "legacy":       ("Product",   -2),
        "outdated":     ("Product",   -2),
        "clunky":       ("Product",   -1),
        "frustrating":  ("Product",   -1),
        "lacking":      ("Product",   -1),
        "locked in":    ("Switching",  -1),
        # Positive / green-flag
        "great support":("Support",    2),
        "excellent":    ("Support",    2),
        "responsive":   ("Support",    1),
        "top-notch":    ("Support",    2),
        "reliable":     ("Quality",    2),
        "solid":        ("Quality",    1),
        "powerful":     ("Product",    2),
        "fantastic":    ("Support",    2),
        "easy":         ("Product",    1),
        "helpful":      ("Support",    1),
        "stable":       ("Quality",    1),
    }

    def __init__(self):
        self.df: pd.DataFrame | None = None
        self.keyword_hits: pd.DataFrame | None = None
        self.brand_health_score: float = 0.0

    # ------------------------------------------------------------------
    def load_data(self, file_path: str) -> pd.DataFrame:
        if not os.path.isfile(file_path):
            raise FileNotFoundError(f"No file found at: {file_path}")
        self.df = pd.read_csv(file_path)
        required = {"Review_ID", "Source", "Review_Text", "Rating"}
        missing = required - set(self.df.columns)
        if missing:
            raise ValueError(f"CSV is missing required columns: {missing}")
        self.df["Review_Text"] = self.df["Review_Text"].fillna("")
        print(f"\n✓ Loaded {len(self.df)} reviews")
        return self.df

    # ------------------------------------------------------------------
    def audit_sentiment(self) -> dict:
        """
        Scan every review for keywords. Produce:
          • Per-keyword hit counts
          • Per-category sentiment tallies
          • Brand Health Score (1–100)
        """
        if self.df is None:
            raise RuntimeError("Call load_data() first.")

        records = []  # (keyword, category, weight, review_id)

        for _, row in self.df.iterrows():
            text_lower = row["Review_Text"].lower()
            for keyword, (category, weight) in self.KEYWORD_MAP.items():
                if keyword in text_lower:
                    records.append({
                        "Keyword": keyword,
                        "Category": category,
                        "Weight": weight,
                        "Review_ID": row["Review_ID"],
                    })

        hits = pd.DataFrame(records)
        self.keyword_hits = hits

        # --- Category-level summary ---
        cat_summary = (
            hits.groupby("Category")
            .agg(Mentions=("Keyword", "count"), Net_Score=("Weight", "sum"))
            .sort_values("Net_Score")
        )

        # --- Brand Health Score ---
        # Normalise: score = 50 + (net_sentiment / max_possible) * 50
        # max_possible = if every review hit the strongest keyword
        total_weight = hits["Weight"].sum()
        max_mag = len(self.df) * 2  # theoretical positive ceiling
        raw = total_weight / max_mag  # range ~ [-1, +1]
        self.brand_health_score = round(max(1, min(100, 50 + raw * 50)), 1)

        # --- Pretty-print ---
        print("\n" + "=" * 60)
        print("  SENTIMENT AUDIT – Customer Reviews")
        print("=" * 60)

        # Top keywords
        top_kw = hits["Keyword"].value_counts().head(10)
        print("\n  Top Red/Green Flag Keywords:")
        for kw, cnt in top_kw.items():
            cat, wt = self.KEYWORD_MAP[kw]
            flag = "🟢" if wt > 0 else "🔴"
            print(f"    {flag} {kw:<18} ×{cnt:<4}  ({cat}, weight {wt:+d})")

        # Category summary
        print(f"\n  {'Category':<14} {'Mentions':>9} {'Net Score':>10}")
        print("  " + "-" * 35)
        for cat, row in cat_summary.iterrows():
            indicator = "🟢" if row["Net_Score"] >= 0 else "🔴"
            print(f"  {indicator} {cat:<12} {int(row['Mentions']):>9} {int(row['Net_Score']):>+10}")

        # Health score
        print(f"\n  ──────────────────────────")
        print(f"  BRAND HEALTH SCORE: {self.brand_health_score}/100")
        if self.brand_health_score < 35:
            print("  ⚠  CRITICAL – Customer base is actively dissatisfied.")
        elif self.brand_health_score < 50:
            print("  ⚠  WARNING – Significant negative sentiment detected.")
        elif self.brand_health_score < 65:
            print("  ℹ  MIXED – Sentiment is split; monitor closely.")
        else:
            print("  ✅ HEALTHY – Positive sentiment dominates.")
        print("=" * 60)

        return {
            "brand_health_score": self.brand_health_score,
            "category_summary": cat_summary.to_dict(),
            "top_keywords": top_kw.to_dict(),
        }

    # ------------------------------------------------------------------
    def visualize_sentiment(self, save_html: bool = True) -> go.Figure:
        """Bar chart of keyword frequency, coloured by sentiment."""
        if self.keyword_hits is None or self.keyword_hits.empty:
            raise RuntimeError("Call audit_sentiment() first.")

        hits = self.keyword_hits.copy()
        kw_counts = hits.groupby("Keyword").agg(
            Count=("Review_ID", "count"),
            Weight=("Weight", "first"),
        ).sort_values("Count", ascending=True)

        colors = ["#EF553B" if w < 0 else "#00CC96" for w in kw_counts["Weight"]]

        fig = go.Figure(go.Bar(
            x=kw_counts["Count"],
            y=kw_counts.index,
            orientation="h",
            marker_color=colors,
            text=kw_counts["Count"],
            textposition="outside",
        ))

        fig.update_layout(
            title=dict(
                text="Risky Co – Keyword Sentiment Frequency",
                font=dict(size=20),
            ),
            xaxis_title="Mentions",
            yaxis_title="Keyword",
            template="plotly_white",
            width=900,
            height=550,
            margin=dict(l=140),
        )

        if save_html:
            out_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
            html_path = os.path.join(out_dir, "sentiment_analysis.html")
            fig.write_html(html_path)
            print(f"\n✓ Sentiment chart saved → {html_path}")

        return fig


# ======================================================================
# Quick run
# ======================================================================
if __name__ == "__main__":
    data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")

    # --- Price-Value Matrix ---
    pricing_path = os.path.join(data_dir, "market_pricing.csv")
    analyst = MarketAnalyst()
    analyst.load_data(pricing_path)
    analyst.plot_price_value_matrix()

    # --- Sentiment Audit ---
    reviews_path = os.path.join(data_dir, "customer_reviews.csv")
    sentiment = SentimentAuditor()
    sentiment.load_data(reviews_path)
    sentiment.audit_sentiment()
    sentiment.visualize_sentiment()
