"""
revenue_integrity.py – Revenue Concentration Audit Module
==========================================================
Class-based engine that loads transaction data and surfaces
concentration risk via HHI, top-N exposure, and 80/20 analysis.
"""

import os
import pandas as pd
import numpy as np


class RevenueAuditor:
    """Audits a transaction-level CSV for revenue concentration risk."""

    def __init__(self):
        self.df: pd.DataFrame | None = None
        self.client_revenue: pd.Series | None = None
        self.total_revenue: float = 0.0
        self.audit_results: dict = {}

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------
    def load_data(self, file_path: str) -> pd.DataFrame:
        """Read a CSV with at least Client_ID and Revenue_USD columns."""
        if not os.path.isfile(file_path):
            raise FileNotFoundError(f"No file found at: {file_path}")

        self.df = pd.read_csv(file_path)

        required = {"Client_ID", "Revenue_USD"}
        missing = required - set(self.df.columns)
        if missing:
            raise ValueError(f"CSV is missing required columns: {missing}")

        # Pre-compute client-level revenue
        self.client_revenue = (
            self.df.groupby("Client_ID")["Revenue_USD"]
            .sum()
            .sort_values(ascending=False)
        )
        self.total_revenue = self.client_revenue.sum()

        print(f"✓ Loaded {len(self.df):,} transactions | "
              f"{self.client_revenue.size} clients | "
              f"${self.total_revenue:,.2f} total revenue")
        return self.df

    # ------------------------------------------------------------------
    # Concentration audit
    # ------------------------------------------------------------------
    def run_concentration_audit(self) -> dict:
        """
        Calculate key concentration metrics:
          • Top-1 and Top-5 client exposure (%)
          • Herfindahl-Hirschman Index (HHI)
          • 80/20 Rule – % of clients driving 80% of revenue
        """
        if self.client_revenue is None:
            raise RuntimeError("Call load_data() before running an audit.")

        shares = self.client_revenue / self.total_revenue  # decimal shares

        # --- Top-N exposure ---
        top1_pct = shares.iloc[0] * 100
        top5_pct = shares.iloc[:5].sum() * 100

        # --- HHI (sum of squared market-share percentages) ---
        share_pcts = shares * 100  # convert to percentage points
        hhi = float((share_pcts ** 2).sum())

        # --- 80/20 rule ---
        cumulative = shares.cumsum()
        clients_for_80 = int((cumulative <= 0.80).sum()) + 1
        pct_clients_for_80 = clients_for_80 / len(shares) * 100

        self.audit_results = {
            "top_1_client_pct": round(top1_pct, 2),
            "top_5_client_pct": round(top5_pct, 2),
            "hhi": round(hhi, 2),
            "clients_for_80_revenue": clients_for_80,
            "pct_clients_for_80_revenue": round(pct_clients_for_80, 2),
            "total_clients": len(shares),
        }

        # Pretty-print
        print("\n" + "=" * 55)
        print("  CONCENTRATION AUDIT RESULTS")
        print("=" * 55)
        print(f"  Top-1 Client Exposure  : {self.audit_results['top_1_client_pct']:.2f}%")
        print(f"  Top-5 Client Exposure  : {self.audit_results['top_5_client_pct']:.2f}%")
        print(f"  HHI Score              : {self.audit_results['hhi']:,.2f}")
        print(f"  80/20 Rule             : {clients_for_80} clients "
              f"({pct_clients_for_80:.1f}% of base) drive 80% of revenue")
        print("=" * 55)

        return self.audit_results

    # ------------------------------------------------------------------
    # Red-flag detection
    # ------------------------------------------------------------------
    def get_red_flags(self) -> list[str]:
        """
        Return a list of red-flag strings when:
          • HHI > 2500  (highly concentrated market)
          • Any single client > 20% of total revenue
        """
        if not self.audit_results:
            raise RuntimeError("Call run_concentration_audit() first.")

        flags: list[str] = []

        if self.audit_results["hhi"] > 2500:
            flags.append(
                f"🚩 HIGH HHI ({self.audit_results['hhi']:,.2f}) – "
                f"Revenue base is highly concentrated (threshold: 2,500)."
            )

        if self.audit_results["top_1_client_pct"] > 20:
            flags.append(
                f"🚩 SINGLE-CLIENT RISK – Top client accounts for "
                f"{self.audit_results['top_1_client_pct']:.2f}% of revenue (threshold: 20%)."
            )

        if not flags:
            flags.append("✅ No red flags detected. Revenue base appears diversified.")

        print("\n" + "-" * 55)
        print("  RED FLAG REPORT")
        print("-" * 55)
        for f in flags:
            print(f"  {f}")
        print("-" * 55)

        return flags

    # ------------------------------------------------------------------
    # Cohort / Vintage Analysis
    # ------------------------------------------------------------------
    def run_cohort_analysis(self) -> pd.DataFrame:
        """
        Build a revenue-retention matrix by monthly cohort.

        Steps:
          1. CohortMonth   – the calendar month of a client's *first* transaction.
          2. TransactionMonth – the calendar month of each transaction.
          3. MonthsSinceInception – integer offset from the cohort month.
          4. Pivot into a retention matrix:
               rows    = CohortMonth
               columns = MonthsSinceInception (0, 1, 2, …)
               values  = Revenue Retention %  (revenue in month X / month 0)
        """
        if self.df is None:
            raise RuntimeError("Call load_data() before running cohort analysis.")

        df = self.df.copy()

        # Ensure Date is datetime
        df["Date"] = pd.to_datetime(df["Date"])

        # Transaction month (period → back to timestamp for clean grouping)
        df["TransactionMonth"] = df["Date"].dt.to_period("M").dt.to_timestamp()

        # Cohort month = first transaction month per client
        cohort_map = (
            df.groupby("Client_ID")["TransactionMonth"]
            .min()
            .rename("CohortMonth")
        )
        df = df.merge(cohort_map, on="Client_ID")

        # Months since inception
        df["MonthsSinceInception"] = (
            (df["TransactionMonth"].dt.year - df["CohortMonth"].dt.year) * 12
            + (df["TransactionMonth"].dt.month - df["CohortMonth"].dt.month)
        )

        # Aggregate revenue by cohort × month-offset
        cohort_revenue = (
            df.groupby(["CohortMonth", "MonthsSinceInception"])["Revenue_USD"]
            .sum()
            .reset_index()
        )

        # Pivot: rows = CohortMonth, columns = MonthsSinceInception
        pivot = cohort_revenue.pivot_table(
            index="CohortMonth",
            columns="MonthsSinceInception",
            values="Revenue_USD",
        )

        # Retention % relative to Month 0
        retention = pivot.div(pivot[0], axis=0) * 100

        # Store for later use
        self.cohort_retention = retention
        self.cohort_pivot_raw = pivot

        # Pretty-print summary
        print("\n" + "=" * 55)
        print("  COHORT RETENTION MATRIX (Revenue %)")
        print("  Rows = Cohort Month | Cols = Months Since Inception")
        print("=" * 55)

        display = retention.copy()
        display.index = display.index.strftime("%Y-%m")
        display.columns = [f"M{int(c)}" for c in display.columns]
        # Show first 6 month-columns for the earliest cohort
        first_cohort_label = display.index[0]
        cols_to_show = display.columns[:6]
        snippet = display.loc[[first_cohort_label], cols_to_show]
        print(f"\n  Earliest cohort ({first_cohort_label}) – first 6 months:")
        print(snippet.to_string())

        # Month-12 spotlight
        if "M12" in display.columns:
            m12_val = display.loc[first_cohort_label, "M12"]
            print(f"\n  ➜ Month-12 retention for {first_cohort_label}: {m12_val:.1f}%")
            if m12_val < 50:
                print("    ⚠  Revenue drops below 50% by Month 12 – significant churn risk.")
            elif m12_val < 80:
                print("    ⚠  Moderate decay – worth investigating drivers.")
            else:
                print("    ✅ Strong retention through Month 12.")
        else:
            print(f"\n  (Month-12 data not yet available for cohort {first_cohort_label})")

        print("=" * 55)
        return retention


# ======================================================================
# EBITDA Normalizer
# ======================================================================
class EBITDANormalizer:
    """
    Strips one-time / non-recurring costs from Reported EBITDA
    to produce an Adjusted ("normalised") EBITDA figure.
    """

    # Keywords that signal a non-recurring cost in the Notes column
    ONE_TIME_KEYWORDS = ["one-time", "restructuring", "settlement"]

    def __init__(self):
        self.df: pd.DataFrame | None = None
        self.adjusted_df: pd.DataFrame | None = None

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------
    def load_data(self, file_path: str) -> pd.DataFrame:
        """Read a financial-statements CSV."""
        if not os.path.isfile(file_path):
            raise FileNotFoundError(f"No file found at: {file_path}")

        self.df = pd.read_csv(file_path)

        required = {"Month", "Revenue", "COGS", "OpEx", "Reported_EBITDA", "Notes"}
        missing = required - set(self.df.columns)
        if missing:
            raise ValueError(f"CSV is missing required columns: {missing}")

        self.df["Month"] = pd.to_datetime(self.df["Month"])
        self.df["Notes"] = self.df["Notes"].fillna("")

        print(f"\n✓ Loaded {len(self.df)} months of financial data")
        return self.df

    # ------------------------------------------------------------------
    # Normalisation
    # ------------------------------------------------------------------
    def normalize_ebitda(self) -> pd.DataFrame:
        """
        Identify rows whose Notes contain any of the one-time keywords,
        compute the add-back amount (= abnormal OpEx spike), and produce
        Adjusted_EBITDA = Reported_EBITDA + add-back.
        """
        if self.df is None:
            raise RuntimeError("Call load_data() first.")

        df = self.df.copy()

        # Flag one-time rows
        pattern = "|".join(self.ONE_TIME_KEYWORDS)
        df["Is_OneTime"] = df["Notes"].str.lower().str.contains(pattern, na=False)

        # Estimate 'normal' OpEx as the median of non-flagged months
        normal_opex_ratio = (
            df.loc[~df["Is_OneTime"], "OpEx"] / df.loc[~df["Is_OneTime"], "Revenue"]
        ).median()

        # Add-back = actual OpEx − estimated normal OpEx (only for flagged months)
        df["Normal_OpEx"] = np.round(df["Revenue"] * normal_opex_ratio, 2)
        df["Addback"] = np.where(
            df["Is_OneTime"],
            np.round(df["OpEx"] - df["Normal_OpEx"], 2),
            0.0,
        )
        df["Adjusted_EBITDA"] = np.round(df["Reported_EBITDA"] + df["Addback"], 2)

        self.adjusted_df = df

        # ---- Pretty-print the affected months ----
        affected = df[df["Is_OneTime"]].copy()
        affected["Month_str"] = affected["Month"].dt.strftime("%Y-%m")

        print("\n" + "=" * 72)
        print("  EBITDA NORMALISATION – Affected Months")
        print("=" * 72)
        print(f"  {'Month':<10} {'Note':<32} {'Reported':>12} {'Addback':>10} {'Adjusted':>12}")
        print("  " + "-" * 68)

        total_addback = 0.0
        for _, row in affected.iterrows():
            print(
                f"  {row['Month_str']:<10} {row['Notes']:<32} "
                f"${row['Reported_EBITDA']:>11,.2f} "
                f"${row['Addback']:>9,.2f} "
                f"${row['Adjusted_EBITDA']:>11,.2f}"
            )
            total_addback += row["Addback"]

        print("  " + "-" * 68)
        print(f"  Total add-back across all one-time events: ${total_addback:,.2f}")

        # Full-period comparison
        rep_total = df["Reported_EBITDA"].sum()
        adj_total = df["Adjusted_EBITDA"].sum()
        pct_lift = (adj_total - rep_total) / abs(rep_total) * 100

        print(f"\n  24-Month Reported EBITDA  : ${rep_total:>14,.2f}")
        print(f"  24-Month Adjusted EBITDA : ${adj_total:>14,.2f}")
        print(f"  Normalisation uplift     :  {pct_lift:>+.2f}%")
        print("=" * 72)

        return df


# ======================================================================
# Quick run
# ======================================================================
if __name__ == "__main__":
    data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")

    # --- Concentration & Cohort audit (risky_co) ---
    csv_path = os.path.join(data_dir, "risky_co.csv")
    auditor = RevenueAuditor()
    auditor.load_data(csv_path)
    auditor.run_concentration_audit()
    auditor.get_red_flags()
    auditor.run_cohort_analysis()

    # --- EBITDA Normalisation (financial_statements) ---
    fin_path = os.path.join(data_dir, "financial_statements.csv")
    normalizer = EBITDANormalizer()
    normalizer.load_data(fin_path)
    normalizer.normalize_ebitda()
