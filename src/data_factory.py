"""
data_factory.py – Synthetic Transaction Generator
===================================================
Produces two CSV datasets in data/:
  • healthy_co.csv  → 500 clients, low concentration (no client > 2% revenue)
  • risky_co.csv    → 50 clients, top 3 hold ~70% of total revenue
"""

import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
NUM_TRANSACTIONS = 5_000
DATE_END = datetime(2026, 3, 3)
DATE_START = DATE_END - timedelta(days=730)  # 24 months back
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")

np.random.seed(42)


def _random_dates(n: int) -> pd.Series:
    """Return n random dates uniformly spread over the last 24 months."""
    start_ts = DATE_START.timestamp()
    end_ts = DATE_END.timestamp()
    timestamps = np.random.uniform(start_ts, end_ts, size=n)
    return pd.to_datetime(timestamps, unit="s").normalize()


# ---------------------------------------------------------------------------
# Healthy Company – 500 clients, very low concentration
# ---------------------------------------------------------------------------
def generate_healthy(n: int = NUM_TRANSACTIONS, num_clients: int = 500) -> pd.DataFrame:
    """
    Revenue is drawn from a tight uniform-ish distribution so that
    no single client exceeds ~2 % of total revenue.
    """
    client_ids = [f"H-{str(i).zfill(4)}" for i in range(1, num_clients + 1)]

    # Assign each transaction a client roughly uniformly
    clients = np.random.choice(client_ids, size=n)

    # Revenue per transaction: narrow band keeps concentration low
    revenue = np.round(np.random.uniform(800, 1_200, size=n), 2)

    df = pd.DataFrame({
        "Transaction_ID": [f"TXN-{str(i).zfill(6)}" for i in range(1, n + 1)],
        "Date": _random_dates(n),
        "Client_ID": clients,
        "Revenue_USD": revenue,
    })

    # Safety check – cap any client that drifts above 2 %
    total = df["Revenue_USD"].sum()
    client_totals = df.groupby("Client_ID")["Revenue_USD"].transform("sum")
    cap = 0.02 * total
    scale = np.where(client_totals > cap, cap / client_totals, 1.0)
    df["Revenue_USD"] = np.round(df["Revenue_USD"] * scale, 2)

    return df.sort_values("Date").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Risky Company – 50 clients, top 3 ≈ 70 % of revenue
# ---------------------------------------------------------------------------
def generate_risky(n: int = NUM_TRANSACTIONS, num_clients: int = 50) -> pd.DataFrame:
    """
    Pareto-style distribution: three 'whale' clients receive outsized
    revenue values; the remaining 47 share the last 30 %.
    """
    client_ids = [f"R-{str(i).zfill(4)}" for i in range(1, num_clients + 1)]
    whale_ids = client_ids[:3]
    small_ids = client_ids[3:]

    # ~70 % of transactions go to whales, rest to small clients
    whale_txn_count = int(n * 0.35)
    small_txn_count = n - whale_txn_count

    whale_clients = np.random.choice(whale_ids, size=whale_txn_count)
    small_clients = np.random.choice(small_ids, size=small_txn_count)

    # Whales get large tickets; small clients get small ones
    whale_revenue = np.round(np.random.uniform(5_000, 15_000, size=whale_txn_count), 2)
    small_revenue = np.round(np.random.uniform(200, 800, size=small_txn_count), 2)

    clients = np.concatenate([whale_clients, small_clients])
    revenue = np.concatenate([whale_revenue, small_revenue])

    df = pd.DataFrame({
        "Transaction_ID": [f"TXN-{str(i).zfill(6)}" for i in range(1, n + 1)],
        "Date": _random_dates(n),
        "Client_ID": clients,
        "Revenue_USD": revenue,
    })

    return df.sort_values("Date").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Financial Statements – 24-month P&L with one-time cost events
# ---------------------------------------------------------------------------
def generate_financial_statements(months: int = 24) -> pd.DataFrame:
    """
    Produce a monthly P&L with occasional one-time cost spikes
    tagged in a Notes column for EBITDA normalisation.
    """
    dates = pd.date_range(end=DATE_END.replace(day=1), periods=months, freq="MS")

    revenue = np.round(np.random.uniform(900_000, 1_100_000, size=months), 2)
    cogs = np.round(revenue * np.random.uniform(0.30, 0.40, size=months), 2)
    opex_base = np.round(revenue * np.random.uniform(0.20, 0.28, size=months), 2)

    # Start from base OpEx – we will add one-time spikes below
    opex = opex_base.copy()
    notes = [""] * months

    # Inject one-time events into 3 specific months
    event_months = {
        5:  ("One-time legal settlement",   180_000),
        11: ("Restructuring costs",          250_000),
        18: ("One-time legal settlement",    140_000),
    }
    for idx, (label, spike) in event_months.items():
        opex[idx] += spike
        notes[idx] = label

    reported_ebitda = np.round(revenue - cogs - opex, 2)

    df = pd.DataFrame({
        "Month": dates,
        "Revenue": revenue,
        "COGS": cogs,
        "OpEx": opex,
        "Reported_EBITDA": reported_ebitda,
        "Notes": notes,
    })
    return df


# ---------------------------------------------------------------------------
# Market Pricing – 10 competitors in Supply Chain Software
# ---------------------------------------------------------------------------
def generate_market_pricing() -> pd.DataFrame:
    """
    Produce a competitive landscape table for the 'Supply Chain Software'
    space.  Risky Co is deliberately set with a high price but only a
    moderate feature score to test overvaluation detection.
    """
    data = [
        # (Competitor_Name, Base_Price_USD, Feature_Score, Market_Share)
        ("Risky Co",         4_800,  55,  0.12),   # target – high price, mid features
        ("ChainFlow Pro",    3_200,  78,  0.18),
        ("LogiCore",         2_900,  72,  0.14),
        ("SupplyNest",       2_100,  60,  0.08),
        ("OptiRoute",        3_600,  82,  0.16),
        ("FreightMinds",     1_800,  48,  0.05),
        ("CargoSync",        2_500,  65,  0.09),
        ("WareHive",         4_200,  88,  0.11),
        ("ProcureEdge",      1_500,  42,  0.04),
        ("NexChain",         3_000,  74,  0.03),
    ]

    df = pd.DataFrame(data, columns=[
        "Competitor_Name", "Base_Price_USD", "Feature_Score", "Market_Share",
    ])
    return df


# ---------------------------------------------------------------------------
# Customer Reviews – 100 reviews for Risky Co (mixed sentiment)
# ---------------------------------------------------------------------------
def generate_customer_reviews(n: int = 100) -> pd.DataFrame:
    """
    Produce 100 synthetic customer reviews for Risky Co across
    G2, Trustpilot, and Glassdoor.  Overall pricing sentiment is
    deliberately negative; support sentiment is positive.
    """
    sources = ["G2", "Trustpilot", "Glassdoor"]

    # Template pools  (negative pricing / mixed / positive)
    negative_reviews = [
        "Way too expensive for what you get. The UI feels outdated and clunky.",
        "Pricing is outrageous compared to competitors. We’re evaluating alternatives.",
        "The platform is slow and the legacy codebase shows. Expensive licensing.",
        "Good product overall but the cost is hard to justify to leadership.",
        "Expensive and buggy. We hit critical bugs in production twice this quarter.",
        "Outdated UI, expensive renewal. Only staying because of data migration costs.",
        "Overpriced. The feature set hasn’t kept up with cheaper alternatives.",
        "Support is great but the product itself feels legacy. Not worth the price.",
        "Buggy integrations and slow response times. Too expensive for this quality.",
        "We’re locked in but already comparing. Pricing does not match value delivered.",
        "The tool is slow and the dashboard is outdated. Hard to recommend at this price.",
        "Renewal quote was 20% higher with no new features. Feels like a cash grab.",
        "Legacy system with a modern price tag. Frustrating.",
        "Expensive per-seat pricing makes it hard to roll out company-wide.",
        "The product is functional but overpriced. Competitors offer more for less.",
    ]

    positive_reviews = [
        "Great support team – always responsive and knowledgeable.",
        "Solid supply chain visibility. Support is top-notch.",
        "Our account manager is fantastic. The tool does what we need.",
        "Reliable platform with excellent customer support.",
        "Good analytics features. The support team goes above and beyond.",
        "We’ve been using it for 3 years – stable and the support is great.",
        "Best-in-class support. They resolve issues within hours.",
        "The reporting module is powerful and the team is always helpful.",
        "Easy onboarding experience. Customer success team was excellent.",
        "Training resources are solid and the support chat is very responsive.",
    ]

    neutral_reviews = [
        "Decent product. Does the job but nothing exceptional.",
        "Average experience overall. Some features are good, others need work.",
        "It works for our use case but the interface could be modernised.",
        "Meets basic requirements. The mobile experience is lacking.",
        "Standard supply chain tool. Nothing stands out.",
    ]

    # Mix: ~50% negative, ~33% positive, ~17% neutral
    neg_count = 50
    pos_count = 33
    neu_count = n - neg_count - pos_count  # 17

    review_texts = (
        list(np.random.choice(negative_reviews, size=neg_count))
        + list(np.random.choice(positive_reviews, size=pos_count))
        + list(np.random.choice(neutral_reviews, size=neu_count))
    )

    # Ratings aligned with sentiment
    ratings = (
        list(np.random.choice([1, 2, 3], size=neg_count, p=[0.3, 0.5, 0.2]))
        + list(np.random.choice([4, 5], size=pos_count, p=[0.4, 0.6]))
        + list(np.random.choice([3, 4], size=neu_count, p=[0.6, 0.4]))
    )

    # Shuffle everything together
    order = np.random.permutation(n)
    review_texts = [review_texts[i] for i in order]
    ratings = [int(ratings[i]) for i in order]

    df = pd.DataFrame({
        "Review_ID": [f"REV-{str(i).zfill(4)}" for i in range(1, n + 1)],
        "Source": np.random.choice(sources, size=n),
        "Review_Text": review_texts,
        "Rating": ratings,
    })
    return df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    healthy = generate_healthy()
    risky = generate_risky()
    financials = generate_financial_statements()
    pricing = generate_market_pricing()
    reviews = generate_customer_reviews()

    healthy_path = os.path.join(OUTPUT_DIR, "healthy_co.csv")
    risky_path = os.path.join(OUTPUT_DIR, "risky_co.csv")
    fin_path = os.path.join(OUTPUT_DIR, "financial_statements.csv")
    pricing_path = os.path.join(OUTPUT_DIR, "market_pricing.csv")
    reviews_path = os.path.join(OUTPUT_DIR, "customer_reviews.csv")

    healthy.to_csv(healthy_path, index=False)
    risky.to_csv(risky_path, index=False)
    financials.to_csv(fin_path, index=False)
    pricing.to_csv(pricing_path, index=False)
    reviews.to_csv(reviews_path, index=False)

    # Quick sanity print
    for label, df in [("Healthy Co", healthy), ("Risky Co", risky)]:
        total = df["Revenue_USD"].sum()
        top3 = df.groupby("Client_ID")["Revenue_USD"].sum().nlargest(3)
        top3_pct = top3.sum() / total * 100
        print(f"\n{'='*50}")
        print(f"  {label}")
        print(f"  Transactions : {len(df):,}")
        print(f"  Unique clients: {df['Client_ID'].nunique()}")
        print(f"  Total Revenue : ${total:,.2f}")
        print(f"  Top-3 share   : {top3_pct:.1f}%")
        print(f"  Date range    : {df['Date'].min().date()} → {df['Date'].max().date()}")
        print(f"{'='*50}")

    # Financial statements summary
    events = financials[financials["Notes"] != ""]
    print(f"\n{'='*50}")
    print(f"  Financial Statements")
    print(f"  Months         : {len(financials)}")
    print(f"  One-time events: {len(events)}")
    for _, row in events.iterrows():
        print(f"    • {row['Month'].strftime('%Y-%m')} – {row['Notes']}")
    print(f"{'='*50}")

    # Market pricing summary
    print(f"\n{'='*50}")
    print(f"  Market Pricing")
    print(f"  Competitors: {len(pricing)}")
    target = pricing[pricing["Competitor_Name"] == "Risky Co"].iloc[0]
    print(f"  Target (Risky Co): ${target['Base_Price_USD']:,} | "
          f"Features: {target['Feature_Score']}/100 | "
          f"Share: {target['Market_Share']:.0%}")
    print(f"{'='*50}")

    # Customer reviews summary
    print(f"\n{'='*50}")
    print(f"  Customer Reviews")
    print(f"  Total reviews : {len(reviews)}")
    print(f"  Sources       : {', '.join(reviews['Source'].unique())}")
    print(f"  Avg rating    : {reviews['Rating'].mean():.1f} / 5")
    print(f"  Rating dist   : {reviews['Rating'].value_counts().sort_index().to_dict()}")
    print(f"{'='*50}")

    print(f"\n✓ Files saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
