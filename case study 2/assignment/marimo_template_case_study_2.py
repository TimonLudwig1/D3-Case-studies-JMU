import marimo

__generated_with = "0.23.6"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Case 2 — High Season Decisions at Viador Hotels

    **Data-Driven Decisions in Practice · D3 Applications · SS 2026**

    Clara Schwarz oversees two mid-sized hotels under the Viador brand —
    a **city** location and a **resort** location. She wants your help
    refining pricing and capacity decisions for the upcoming high
    season. Your findings will brief her revenue and operations team:
    **clarity of communication is key**. Visualisations encouraged.
    *Pricing strategies should always rely on realised bookings, not
    mere requests.*

    This template provides a **structural scaffold**. Every decision
    (segmentation thresholds, buffer levels, seasonal rules, adaptive
    policy) is yours to make and defend.

    ### Method companions

    | Notebook | Focus | Feeds |
    |---|---|---|
    | [1 · EDA](./d3ip-case2-rm-1-data.html) | Booking curves, fare-class structure, EDA bridge | Q1, Q3 |
    | [2 · Dynamic control](./d3ip-case2-rm-2-dynamic.html) | Booking strips, revenue surface, S-index | Q4 baseline |
    | [3 · Prediction models](./d3ip-case2-rm-3-prediction.html) | Cancellation classifier, calibration, $q^{\text{eff}}_t$ | Q2, Q4 advanced |

    ### How to use this file

    - Each question is one section. Add as many cells as you need.
    - Use the loader in §0; do not re-download per question.
    - The final reflection section is **required**.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## §0 · Setup & data loader

    The loader below is the same one used in the three companion
    notebooks. It downloads the public mirror of the Viador booking
    dataset and adds two helper columns (`booking_date`,
    `adr_per_adult`) that the EDA companion uses throughout.
    """)
    return


@app.cell
def _():
    import numpy as np
    import pandas as pd

    DATA_URL = (
        "https://raw.githubusercontent.com/mpolinowski/hotel-booking-dataset"
        "/refs/heads/master/datasets/hotel_bookings.csv"
    )

    def load_bookings():
        df = pd.read_csv(DATA_URL)
        df["arrival_date"] = pd.to_datetime(
            df["arrival_date_year"].astype(str)
            + "-"
            + df["arrival_date_month"]
            + "-"
            + df["arrival_date_day_of_month"].astype(str),
            format="%Y-%B-%d",
            errors="coerce",
        )
        df["booking_date"] = df["arrival_date"] - pd.to_timedelta(
            df["lead_time"], unit="D"
        )
        df["adr_per_adult"] = df["adr"] / df["adults"].clip(lower=1)
        return df

    bookings = load_bookings()
    bookings.shape, bookings["arrival_date"].min(), bookings["arrival_date"].max()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## §Q1 · Calibrate Advance Sale Policies

    Clara offers discounted advance rates — but only up to a certain
    number of bookings. Selling too many in advance turns away
    spontaneous, higher-paying guests; selling too few leaves
    occupancy short.

    - Using **realised stays**, identify meaningful guest segments for
      each hotel. *When* do different guest types book? *What* prices
      do they pay?
    - Estimate how many rooms Clara could allocate to advance sales
      **for each property**. Focus on lead times and, if useful, guest
      mix.

    *Methods reference: companion NB1 §3 (univariate), §4 (per-hotel),
    §8 (booking curves), §9 (canonical-booking cut tool — drives the
    advance/late split per hotel).*
    """)
    return


@app.cell
def _():
    # Your Q1 analysis cells go here.
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## §Q2 · Derive Overbooking Parameters

    Clara wants to overbook to mitigate late cancellations and no-shows
    — but walk-aways hurt guest satisfaction and brand image.

    - Use historical data to estimate a **robust overbooking buffer**
      for each hotel. How many extra bookings can Clara accept without
      incurring frequent walk-aways?
    - Make any **trade-offs and assumptions explicit** (walk cost,
      tolerated walk frequency, independence of cancellations, …).

    *Methods reference: companion NB1 §7 (cancellation by channel /
    lead-time), NB3 §6 (cancellation model + Brier), §8.5 (buffer
    calculator with tolerated-walk slider).*
    """)
    return


@app.cell
def _():
    # Your Q2 analysis cells go here.
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## §Q3 · Seasonally Update Your Rules

    Demand patterns shift across the year. Clara wants Q1's advance
    sale limits and Q2's overbooking buffers to **adapt to seasonality**.

    - Using appropriate aggregates from past years, identify periods
      of stronger or weaker demand **for each hotel**.
    - Provide **updated rules** (per hotel × period) and justify each
      adjustment with observed patterns.

    *Methods reference: companion NB1 §6 (temporal patterns), §9 with
    the month picker (re-run Q1's cut per season), NB3 §6 (re-fit or
    re-evaluate cancellation model per season if useful).*
    """)
    return


@app.cell
def _():
    # Your Q3 analysis cells go here.
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## §Q4 · Operate on Live Booking Data

    Simulate a realistic decision environment for Viador:

    1. **Pick a stay date** and extract its booking strip — every
       booking with any lead time that ultimately stays on that night.
    2. Replay it as a **time-ordered stream** of requests (by lead
       time).
    3. Implement an **adaptive control strategy** that:
       - Makes daily decisions based on current capacity and expected
         high-/low-fare arrivals.
       - **Dynamically updates** protection limits or overbooking
         thresholds as bookings (and cancellations) arrive.
       - **Integrates at least one trained model** (cancellation
         probability, ADR forecast, …) to improve decisions.

    Report realised revenue, spillage, and walk-aways; compare against
    the static Q1 + Q2 baseline. A good recommendation is *transparent,
    robust to noise, and clearly communicated*.

    *Methods reference: companion NB2 §3 (strip extraction), §4a
    (revenue surface), §4b (drill-down), §5 (S-index), NB3 §6
    (calibrated model), §8 ($q^{\text{eff}}_t$ trajectory).*
    """)
    return


@app.cell
def _():
    # Your Q4 analysis cells go here.
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## §R · Reflection & methodology

    Briefly answer (one short paragraph each):

    1. **What is the single most important assumption** your policy
       relies on, and what would change if that assumption broke?
    2. **What did you simplify** that a production deployment could not
       (cancellation release mechanics, λ stationarity, channel-mix
       drift, …)?
    3. **What would you do with one more week** of effort?
    """)
    return


@app.cell
def _():
    # Your reflection answers go here (markdown cells are fine).
    return


if __name__ == "__main__":
    app.run()
