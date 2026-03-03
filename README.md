# Efficiency Play: Automated Commercial Due Diligence (CDD) Engine

## Executive Summary

An automated financial and market auditing engine designed to identify structural risks in M&A targets. The tool performs deep-dive **Revenue Integrity** checks and **Market Moat** assessments to generate an automated **Investment Memo**.

## Key Capabilities

- **Revenue Integrity:** HHI Concentration Risk, 80/20 Rule validation, and EBITDA Normalization (Add-backs).
- **Cohort Analysis:** Vintage-based retention matrices to calculate NRR and 'Leaky Bucket' risk.
- **Market Analysis:** Price-Value mapping and Keyword-based Sentiment Auditing.
- **Synthesis:** Interactive Streamlit 'Deal Room' with auto-generated MECE Investment Recommendations.

## Tech Stack

- **Backend:** Python (Pandas, NumPy)
- **Visualization:** Plotly
- **UI:** Streamlit

## Deployment

1. Install dependencies:
   ```bash
   pip install pandas numpy plotly streamlit
   ```

2. Run the engine:
   ```bash
   python src/data_factory.py
   ```

3. Launch the Deal Room:
   ```bash
   streamlit run app.py
   ```
