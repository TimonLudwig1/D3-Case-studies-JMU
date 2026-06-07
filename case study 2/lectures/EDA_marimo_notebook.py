import marimo

__generated_with = "0.23.9"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Case 2 · Notebook 1 — Exploratory Data Analysis

    **Data-Driven Decisions in Practice · D3 Applications · SS 2026**

    *Part 1 of 3* — descriptive EDA on the Viador hotel dataset,
    structured around the five question banks from the week-1 EDA deck:
    **univariate · hotel-wise · distribution channel · temporal ·
    cancellation**. Closes with the canonical-booking analysis that
    bridges from raw data to the two-class fare abstraction.

    | Notebook | Focus | Feeds |
    |---|---|---|
    | **1 · EDA** *(you are here)* | Slide-deck question banks + fare-class bridge | Q1 |
    | 2 · Dynamic control | Booking strips, static simulator, state-space S-index | Q4 baseline |
    | 3 · Prediction models | Cancellation classifier, calibration, $q^{\text{eff}}_t$ illustration | Q2, Q4 advanced |

    > **How to use this notebook.** Each section answers one of the slide-deck
    > EDA questions with a short chart and a one-line caption. Charts are
    > rendered with **Plotly Express** and composed via `mo.hstack` /
    > `mo.vstack`.

    **Static-rule recap (week 1).**

    - [`bookingLimits.py`](./bookingLimits.py) — Littlewood protection level
    - [`demand_management.py`](./demand_management.py) — full static RM playground

    ---

    ### Reading roadmap

    | If you are here for… | Read sections | Time |
    |---|---|---|
    | Just an orientation pass over the data | §1.5 (one figure) | 2 min |
    | **Q1 — advance-sale segmentation** | §3 univariate, §9 EDA bridge + cut tool | 15–20 min |
    | **Q3 — seasonal rules** | §6 temporal, §9 with the month picker | 10 min |
    | Full descriptive EDA (matches the slide-deck question banks) | §3 → §7 in order | 20–30 min |
    | The booking-curve geometry that drives Notebook 2's policy | §8 PDF + cumulative | 5 min |

    The §9 cut tool is the single most useful cell for the case write-up.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## §1 · The Revenue-Management Maturity Curve

    The week-2 lecture frames three stages of revenue management:

    | Stage | Decision logic | Data view |
    |---|---|---|
    | **Static capacity management** | One number per fare class (Littlewood, overbooking buffer) | Summary statistics |
    | **Dynamic capacity management** | Updated as the booking horizon unfolds | Live booking stream |
    | **Integrated optimization** | Joint pricing × capacity, forecast-driven | Predictive models + optimization |

    Week 1 lived in the first column. Week 2 moves you to the second and
    sketches the third. The skill you are practicing is *not* picking the
    "best" model — it is **knowing which level of sophistication is justified
    by your data and your operational reality**.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## §2 · Loading the Data
    """)
    return


@app.cell
def _():
    import numpy as np
    import pandas as pd
    import plotly.express as px
    import plotly.graph_objects as go

    DATA_URL = "https://raw.githubusercontent.com/mpolinowski/hotel-booking-dataset/refs/heads/master/datasets/hotel_bookings.csv"

    HOTEL_COLORS = {"City Hotel": "#1f77b4", "Resort Hotel": "#ff7f0e"}

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
        df["arrival_week"] = df["arrival_date"].dt.isocalendar().week.astype(int)
        df["lead_weeks"] = (df["lead_time"] // 7).astype(int)
        df["adr_per_adult"] = df["adr"] / df["adults"].clip(lower=1)
        return df

    bookings = load_bookings()
    bookings.shape, bookings["arrival_date"].min(), bookings["arrival_date"].max()

    # Disable hover tooltips, drag-zoom, pan, and modebar buttons globally
    # for every Plotly figure created downstream. Approximates passing
    # config={'staticPlot': True} in classic Plotly.show().
    import plotly.io as pio
    pio.templates["wiba_static"] = go.layout.Template(layout=go.Layout(
        hovermode=False, dragmode=False,
        modebar=dict(remove=["zoom2d", "pan2d", "select2d", "lasso2d",
                             "zoomIn2d", "zoomOut2d", "autoScale2d",
                             "resetScale2d", "toImage"]),
        xaxis=dict(fixedrange=True),
        yaxis=dict(fixedrange=True),
    ))
    pio.templates.default = "plotly+wiba_static"
    return HOTEL_COLORS, bookings, go, pd, px


@app.cell(hide_code=True)
def _(bookings, mo):
    mo.md(f"""
    **Loaded.** `{len(bookings):,}` booking records spanning
    `{bookings['arrival_date'].min().date()}` to
    `{bookings['arrival_date'].max().date()}`.

    *What is ADR?* **Average Daily Rate** — the per-room-per-night price
    the hotel actually books at, set at the moment of reservation.
    Excludes meal-plan upcharges (those live in `meal`), excludes taxes
    and fees. It is *not* total revenue: a 3-night booking at ADR €100
    contributes €300 to revenue. `adr_per_adult = adr / max(adults, 1)`
    is precomputed so we can compare bookings of different party sizes.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## §3 · Univariate Analysis
    """)
    return


@app.cell
def _(HOTEL_COLORS, bookings, mo, px):
    realised_univ = bookings[bookings["is_canceled"] == 0]

    # Top 10 booking agents.
    agents_df = (
        realised_univ["agent"].dropna().astype(int).value_counts().head(10)
        .rename_axis("agent").reset_index(name="bookings")
    )
    agents_df["agent"] = agents_df["agent"].astype(str)
    c_agents = px.bar(
        agents_df, x="bookings", y="agent", orientation="h",
        title="Top 10 booking agents",
        labels={"bookings": "Realised bookings", "agent": "Agent ID"},
    ).update_layout(yaxis={"categoryorder": "total ascending"},
                    width=440, height=300, margin=dict(l=10, r=10, t=40, b=10))
    c_agents.update_traces(marker_color="#1f77b4")

    # Room type — demand and ADR as SEPARATE panels (no dual axis).
    room_demand_df = (
        realised_univ["reserved_room_type"].value_counts()
        .rename_axis("room").reset_index(name="bookings")
        .sort_values("room")
    )
    room_adr_df = (
        realised_univ.loc[realised_univ["adr_per_adult"].between(0, 250)]
        .groupby("reserved_room_type")["adr_per_adult"].median()
        .rename_axis("room").reset_index(name="median_adr_adult")
        .sort_values("room")
    )
    c_room_demand = px.bar(
        room_demand_df, x="room", y="bookings",
        title="Room type — demand",
        labels={"room": "Reserved room type", "bookings": "Realised bookings"},
    ).update_layout(width=440, height=300, margin=dict(l=10, r=10, t=40, b=10))
    c_room_demand.update_traces(marker_color="#1f77b4")
    c_room_adr = px.bar(
        room_adr_df, x="room", y="median_adr_adult",
        title="Room type — median ADR per adult",
        labels={"room": "Reserved room type", "median_adr_adult": "Median ADR/adult (€)"},
    ).update_layout(width=440, height=300, margin=dict(l=10, r=10, t=40, b=10))
    c_room_adr.update_traces(marker_color="#d62728")

    # Top 10 source countries.
    countries_df = (
        realised_univ["country"].value_counts().head(10)
        .rename_axis("country").reset_index(name="bookings")
    )
    c_countries = px.bar(
        countries_df, x="bookings", y="country", orientation="h",
        title="Top 10 source countries",
        labels={"bookings": "Realised bookings", "country": "Country"},
    ).update_layout(yaxis={"categoryorder": "total ascending"},
                    width=440, height=300, margin=dict(l=10, r=10, t=40, b=10))
    c_countries.update_traces(marker_color="#2ca02c")

    # Meal preference per hotel (grouped bar).
    meals_df = (
        realised_univ.groupby(["hotel", "meal"]).size().reset_index(name="bookings")
    )
    meal_order = ["BB", "HB", "SC", "FB", "Undefined"]
    c_meals = px.bar(
        meals_df, x="meal", y="bookings", color="hotel", barmode="group",
        category_orders={"meal": meal_order},
        color_discrete_map=HOTEL_COLORS,
        title="Meal preference per hotel",
        labels={"meal": "Meal plan", "bookings": "Realised bookings", "hotel": "Hotel"},
    ).update_layout(width=440, height=300, margin=dict(l=10, r=10, t=40, b=10))

    mo.vstack([
        mo.hstack([c_agents, c_countries], justify="start"),
        mo.hstack([c_room_demand, c_room_adr], justify="start"),
        mo.hstack([c_meals], justify="start"),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Answers.** Most bookings come through a small handful of agents (#9
    and #240 dominate). **Room A** is by far the most demanded, but the
    upper room categories (C, F, G, L) earn substantially more per adult
    — A and B are the price floor. **PRT** (Portugal) is the largest
    source country — both hotels are Portuguese. **BB (bed & breakfast)**
    is the dominant meal plan everywhere.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## §4 · Hotel-wise Analysis
    """)
    return


@app.cell
def _(HOTEL_COLORS, bookings, go, mo, pd, px):
    realised_h = bookings[bookings["is_canceled"] == 0].copy()
    realised_h["total_nights"] = (
        realised_h["stays_in_weekend_nights"] + realised_h["stays_in_week_nights"]
    )
    realised_h["revenue"] = realised_h["adr"] * realised_h["total_nights"]

    summary_rows = []
    for _h in ["City Hotel", "Resort Hotel"]:
        mask = bookings["hotel"] == _h
        rmask = realised_h["hotel"] == _h
        summary_rows.append({
            "hotel": _h,
            "volume_share": mask.mean() * 100,
            "revenue_M": realised_h.loc[rmask, "revenue"].sum() / 1e6,
            "cancel_rate": bookings.loc[mask, "is_canceled"].mean() * 100,
            "repeat_rate": realised_h.loc[rmask, "is_repeated_guest"].mean() * 100,
        })
    summary_df = pd.DataFrame(summary_rows)
    metric_labels = {
        "volume_share": "Volume share (%)",
        "revenue_M":    "Revenue (€M)",
        "cancel_rate":  "Cancellation rate (%)",
        "repeat_rate":  "Repeat-guest rate (%)",
    }
    summary_long = summary_df.melt(
        id_vars="hotel", value_vars=list(metric_labels.keys()),
        var_name="metric_key", value_name="value",
    )
    summary_long["metric"] = summary_long["metric_key"].map(metric_labels)

    c_hotel_metrics = px.bar(
        summary_long, x="hotel", y="value", color="hotel",
        facet_col="metric",
        category_orders={"metric": list(metric_labels.values())},
        color_discrete_map=HOTEL_COLORS,
        labels={"hotel": "", "value": ""},
    ).update_layout(
        width=760, height=240, showlegend=False,
        margin=dict(l=10, r=10, t=40, b=10),
    )
    c_hotel_metrics.update_yaxes(matches=None, showticklabels=True)
    c_hotel_metrics.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))

    # Lead-time box-like (precomputed quartiles → go.Box).
    lt_box = go.Figure()
    for _h in ["City Hotel", "Resort Hotel"]:
        _sub = bookings.loc[bookings["hotel"] == _h, "lead_time"]
        lt_box.add_trace(go.Box(
            name=_h, x=[_h],
            q1=[float(_sub.quantile(0.25))],
            median=[float(_sub.median())],
            q3=[float(_sub.quantile(0.75))],
            lowerfence=[float(_sub.quantile(0.05))],
            upperfence=[float(_sub.quantile(0.95))],
            marker_color=HOTEL_COLORS[_h],
        ))
    lt_box.update_layout(
        title="Lead time per hotel",
        yaxis_title="Lead time (days)",
        width=380, height=260, showlegend=False,
        margin=dict(l=10, r=10, t=40, b=10),
    )
    c_lead_time = lt_box

    mo.vstack([c_hotel_metrics, c_lead_time])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Answers.** City Hotel takes ~60% of volume but City and Resort earn
    comparable total revenue because Resort's ADR is higher *and* its
    stays are longer. City has substantially **longer lead times and a
    higher cancellation rate** (~42% vs ~28%). Repeat-guest rates are
    low everywhere (~3% City, ~6% Resort).
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## §5 · Distribution Channel Analysis
    """)
    return


@app.cell
def _(bookings, go, mo, px):
    realised_ch = bookings[bookings["is_canceled"] == 0]
    chan_order = realised_ch["distribution_channel"].value_counts().index.tolist()

    chan_vol_df = (
        realised_ch["distribution_channel"].value_counts()
        .rename_axis("channel").reset_index(name="bookings")
    )
    c_chan_volume = px.bar(
        chan_vol_df, x="channel", y="bookings",
        category_orders={"channel": chan_order},
        title="Channel volume",
        labels={"channel": "Channel", "bookings": "Realised bookings"},
    ).update_layout(width=280, height=240, margin=dict(l=10, r=10, t=40, b=10))
    c_chan_volume.update_traces(marker_color="#1f77b4")

    c_chan_lead = go.Figure()
    for ch in chan_order:
        _sub = bookings.loc[bookings["distribution_channel"] == ch, "lead_time"]
        c_chan_lead.add_trace(go.Box(
            name=ch, x=[ch],
            q1=[float(_sub.quantile(0.25))], median=[float(_sub.median())],
            q3=[float(_sub.quantile(0.75))],
            lowerfence=[float(_sub.quantile(0.05))],
            upperfence=[float(_sub.quantile(0.95))],
            marker_color="#1f77b4",
        ))
    c_chan_lead.update_layout(
        title="Lead time per channel", yaxis_title="Lead time (days)",
        width=300, height=240, showlegend=False,
        margin=dict(l=10, r=10, t=40, b=10),
    )

    chan_adr_df = (
        realised_ch[realised_ch["adr_per_adult"].between(0, 250)]
        .groupby("distribution_channel")["adr_per_adult"].median()
        .rename_axis("channel").reset_index(name="median_adr_adult")
    )
    c_chan_pricing = px.bar(
        chan_adr_df, x="channel", y="median_adr_adult",
        category_orders={"channel": chan_order},
        title="Channel pricing",
        labels={"channel": "Channel", "median_adr_adult": "Median ADR/adult (€)"},
    ).update_layout(width=280, height=240, margin=dict(l=10, r=10, t=40, b=10))
    c_chan_pricing.update_traces(marker_color="#2ca02c")

    mo.hstack([c_chan_volume, c_chan_lead, c_chan_pricing], justify="start")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Answers.** **TA/TO** (Travel Agent / Tour Operator) is by far the
    biggest channel — they also book *earliest* (longest lead times) and
    sit in the middle of the price band. **Direct** bookings have shorter
    lead times. **Corporate** sits at the highest per-adult ADR but with
    smaller volume.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## §6 · Temporal Analysis
    """)
    return


@app.cell
def _(HOTEL_COLORS, bookings, pd, px):
    realised_t = bookings[bookings["is_canceled"] == 0]
    month_order = [
        "January", "February", "March", "April", "May", "June",
        "July", "August", "September", "October", "November", "December",
    ]

    monthly_vol_df = (
        realised_t.groupby(["arrival_date_month", "hotel"]).size()
        .reset_index(name="value")
        .assign(metric="Realised bookings")
    )
    monthly_adr_df = (
        realised_t[realised_t["adr_per_adult"].between(0, 250)]
        .groupby(["arrival_date_month", "hotel"])["adr_per_adult"].median()
        .reset_index(name="value")
        .assign(metric="Median ADR per adult (€)")
    )
    monthly_long = pd.concat([monthly_vol_df, monthly_adr_df], ignore_index=True)
    # Plotly draws line segments in DataFrame row order — must sort by
    # chronological month BEFORE passing in, otherwise `category_orders`
    # only fixes the axis labels and the line zig-zags between unsorted rows.
    monthly_long["arrival_date_month"] = pd.Categorical(
        monthly_long["arrival_date_month"], categories=month_order, ordered=True,
    )
    monthly_long = monthly_long.sort_values(
        ["metric", "hotel", "arrival_date_month"]
    ).reset_index(drop=True)

    c_temporal = px.line(
        monthly_long, x="arrival_date_month", y="value",
        color="hotel", markers=True,
        facet_col="metric",
        category_orders={
            "arrival_date_month": month_order,
            "metric": ["Realised bookings", "Median ADR per adult (€)"],
        },
        color_discrete_map=HOTEL_COLORS,
        labels={"arrival_date_month": "Arrival month", "value": "", "hotel": "Hotel"},
    ).update_layout(
        width=820, height=300,
        margin=dict(l=10, r=10, t=40, b=10),
    )
    c_temporal.update_yaxes(matches=None, showticklabels=True)
    c_temporal.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
    c_temporal
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Answers.** Both hotels peak in **August**, with City showing strong
    shoulder seasons in May–June and September–October. **Resort prices
    swing harder by season** (€~35 in low season → €~60 in peak summer);
    City prices are flatter. Pricing and volume rise together — typical
    leisure-season dynamics, especially pronounced at Resort.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## §7 · Booking-Cancellation Analysis
    """)
    return


@app.cell
def _(bookings, go, mo, pd, px):
    chan_order_c = bookings["distribution_channel"].value_counts().index.tolist()
    cancel_chan_df = (
        bookings.groupby("distribution_channel")["is_canceled"].mean() * 100
    ).rename_axis("channel").reset_index(name="cancel_pct")
    c_cancel_by_channel = px.bar(
        cancel_chan_df, x="channel", y="cancel_pct",
        category_orders={"channel": chan_order_c},
        title="Cancellation by channel",
        labels={"channel": "Channel", "cancel_pct": "% cancelled"},
    ).update_layout(width=280, height=240, margin=dict(l=10, r=10, t=40, b=10))
    c_cancel_by_channel.update_traces(marker_color="#d62728")

    lt_bins = [-1, 7, 30, 90, 180, 365, 999]
    lt_labels = ["0–7", "8–30", "31–90", "91–180", "181–365", "365+"]
    cancel_lt_tmp = bookings[["lead_time", "is_canceled"]].copy()
    cancel_lt_tmp["lt_bucket"] = pd.cut(cancel_lt_tmp["lead_time"], bins=lt_bins, labels=lt_labels)
    cancel_lt_df = (
        cancel_lt_tmp.groupby("lt_bucket", observed=True)["is_canceled"].mean() * 100
    ).rename_axis("lt_bucket").reset_index(name="cancel_pct")
    c_cancel_by_lead = px.bar(
        cancel_lt_df, x="lt_bucket", y="cancel_pct",
        category_orders={"lt_bucket": lt_labels},
        title="Cancellation by lead time",
        labels={"lt_bucket": "Lead-time bucket (days)", "cancel_pct": "% cancelled"},
    ).update_layout(width=320, height=240, margin=dict(l=10, r=10, t=40, b=10))
    c_cancel_by_lead.update_traces(marker_color="#d62728")

    realised_match = bookings[bookings["is_canceled"] == 0].copy()
    realised_match["room_match"] = (
        realised_match["reserved_room_type"] == realised_match["assigned_room_type"]
    )
    match_sub = realised_match[realised_match["adr_per_adult"].between(0, 250)].copy()
    match_sub["match_label"] = match_sub["room_match"].map(
        {True: "Reserved = assigned", False: "Reserved ≠ assigned"}
    )

    c_mismatch_adr = go.Figure()
    for label in ["Reserved = assigned", "Reserved ≠ assigned"]:
        _sub = match_sub.loc[match_sub["match_label"] == label, "adr_per_adult"]
        c_mismatch_adr.add_trace(go.Box(
            name=label, x=[label],
            q1=[float(_sub.quantile(0.25))], median=[float(_sub.median())],
            q3=[float(_sub.quantile(0.75))],
            lowerfence=[float(_sub.quantile(0.05))],
            upperfence=[float(_sub.quantile(0.95))],
            marker_color="#1f77b4",
        ))
    c_mismatch_adr.update_layout(
        title="Room mismatch vs ADR", yaxis_title="ADR per adult (€)",
        width=320, height=240, showlegend=False,
        margin=dict(l=10, r=10, t=40, b=10),
    )

    mo.hstack([c_cancel_by_channel, c_cancel_by_lead, c_mismatch_adr], justify="start")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Answers.** **TA/TO** has the highest cancellation rate among the
    high-volume channels (~41%). Cancellation rises sharply with lead
    time — the 181–365 day bucket cancels in over **55%** of cases, the
    365+ bucket over **65%**. **Room mismatches** are associated with
    *lower* realised ADR (median €76 vs €95 when reserved = assigned) —
    consistent with cheap rooms being overbooked and reassigned at the
    original (cheap) reservation price, not with discretionary upgrades.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## §8 · Booking Curves — How Requests Accumulate

    Two complementary views on lead time as an *ordered* axis (so lines
    are appropriate here):

    - **Left (density)** — share of bookings landing in each lead-week
      bucket. Shows *where decisions cluster*.
    - **Right (cumulative)** — share of the final book already on the
      books W weeks before arrival. Shows *how much demand is still to
      come*, and where the 50% mark sits per hotel.
    """)
    return


@app.cell
def _(HOTEL_COLORS, bookings, pd, px):
    realised_curves = bookings[bookings["is_canceled"] == 0]
    curve_records = []
    for _h, _grp in realised_curves.groupby("hotel"):
        counts = _grp.groupby("lead_weeks").size().sort_index()
        counts = counts[counts.index <= 52]
        total = counts.sum()
        pdf_s = counts / total * 100
        cdf_s = (counts.iloc[::-1].cumsum() / total * 100).iloc[::-1]
        for w in counts.index:
            curve_records.append({
                "hotel": _h, "lead_weeks": int(w),
                "pdf_pct": float(pdf_s.loc[w]),
                "cdf_pct": float(cdf_s.loc[w]),
            })
    curve_df = pd.DataFrame(curve_records)
    curve_long = pd.concat([
        curve_df[["hotel", "lead_weeks", "pdf_pct"]]
        .rename(columns={"pdf_pct": "value"})
        .assign(metric="Density: % in week-bucket"),
        curve_df[["hotel", "lead_weeks", "cdf_pct"]]
        .rename(columns={"cdf_pct": "value"})
        .assign(metric="Cumulative: % of book on the books"),
    ], ignore_index=True).sort_values(["metric", "hotel", "lead_weeks"]).reset_index(drop=True)

    c_curves = px.line(
        curve_long, x="lead_weeks", y="value", color="hotel",
        markers=True, facet_col="metric",
        category_orders={"metric": ["Density: % in week-bucket",
                                    "Cumulative: % of book on the books"]},
        color_discrete_map=HOTEL_COLORS,
        labels={"lead_weeks": "Weeks before arrival", "value": "", "hotel": "Hotel"},
    ).update_layout(
        width=820, height=300,
        margin=dict(l=10, r=10, t=40, b=10),
    )
    c_curves.update_xaxes(autorange="reversed")
    c_curves.update_yaxes(matches=None, showticklabels=True)
    c_curves.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
    c_curves
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Static rules are the **expected value** of these curves. Dynamic rules
    respond to how the *actual* curve unfolds for a specific stay night —
    which is the topic of Notebook 2.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## §9 · EDA bridge — where do the "fare classes" actually come from?

    Week 1 used a two-class abstraction (low fare €159, high fare €225)
    borrowed from the static-RM example. The data has no such labels.
    The slides hint at the answer: **ADR combined with lead time is a
    workable proxy for booking class.**

    Before reading any signal off ADR we homogenise the bookings. Raw
    ADR is per-room-per-night and conflates couples with solos, suites
    with standard rooms, room-only with half-board. We define a
    **canonical booking** = room type `A` + meal `BB` + exactly **1
    adult** (the most common combination, ~12 000 records). On that
    subset, `adr` is a directly comparable per-unit price.

    *One more slice.* Clara's case is about the **upcoming high season**.
    The temporal panel in §6 showed Resort ADR swinging from €~35 in
    January to €~60 in August — pooling all months hides exactly the
    regime she cares about. The picker below restricts the EDA bridge
    to a subset of arrival months; default is summer (Jun–Sep). Switch
    to off-season months to see how the City–Resort gap narrows.
    """)
    return


@app.cell(hide_code=True)
def _(bookings, mo):
    months_all = [
        "January", "February", "March", "April", "May", "June",
        "July", "August", "September", "October", "November", "December",
    ]
    month_pick = mo.ui.multiselect(
        options=months_all,
        value=["June", "July", "August", "September"],
        label="Arrival months included in the EDA bridge",
    )
    # Canonical-booking definition. Defaults reproduce the original
    # ("A" + "BB" + 1 adult) but students can widen it to test sensitivity
    # — e.g. include room "D" or 2 adults — and see whether the
    # lead-time / ADR structure survives.
    room_types_canon = sorted(bookings["reserved_room_type"].dropna().unique().tolist())
    meal_types_canon = sorted(bookings["meal"].dropna().unique().tolist())
    adults_options = sorted(
        int(a) for a in bookings["adults"].dropna().unique() if 0 <= a <= 4
    )
    canon_room = mo.ui.multiselect(
        options=room_types_canon,
        value=["A"],
        label="Canonical room type(s)",
    )
    canon_meal = mo.ui.multiselect(
        options=meal_types_canon,
        value=["BB"],
        label="Canonical meal plan(s)",
    )
    canon_adults = mo.ui.multiselect(
        options=[str(a) for a in adults_options],
        value=["1"],
        label="Canonical party size (adults)",
    )
    # Controls are rendered together with the upper viz in the next
    # cell so the four widgets sit directly above the chart pair.
    return canon_adults, canon_meal, canon_room, month_pick


@app.cell
def _(
    HOTEL_COLORS,
    bookings,
    canon_adults,
    canon_meal,
    canon_room,
    go,
    mo,
    month_pick,
    pd,
    px,
):
    # Fallback: if nothing is selected, show all months instead of an empty
    # chart. The mo.md warning below makes this fallback visible.
    selected_months = list(month_pick.value) or [
        "January", "February", "March", "April", "May", "June",
        "July", "August", "September", "October", "November", "December",
    ]
    empty_picker = not list(month_pick.value)
    # Apply the user-defined canonical filter; fall back to the original
    # ("A" / "BB" / 1 adult) if a control is left empty.
    rooms_used = list(canon_room.value) or ["A"]
    meals_used = list(canon_meal.value) or ["BB"]
    adults_used = [int(a) for a in (canon_adults.value or ["1"])]
    canon_df = bookings.loc[
        (bookings["is_canceled"] == 0)
        & (bookings["reserved_room_type"].isin(rooms_used))
        & (bookings["meal"].isin(meals_used))
        & (bookings["adults"].isin(adults_used))
        & (bookings["adr"].between(0, 250))
        & (bookings["lead_time"] <= 365)
        & (bookings["arrival_date_month"].isin(selected_months))
    ].copy()

    # Pre-bin the joint (lead_time, adr) density so the chart payload is
    # a tiny per-cell count table instead of thousands of raw points.
    canon_density = canon_df[["hotel", "lead_time", "adr"]].copy()
    canon_density["lt_bin"] = (canon_density["lead_time"] // 15) * 15
    canon_density["adr_bin"] = (canon_density["adr"] // 10) * 10
    density_df = (
        canon_density.groupby(["hotel", "lt_bin", "adr_bin"])
        .size().reset_index(name="count")
    )

    c_canon_scatter = px.density_heatmap(
        density_df, x="lt_bin", y="adr_bin", z="count",
        facet_col="hotel", histfunc="sum",
        color_continuous_scale="Blues",
        title="Density (binned)",
        labels={"lt_bin": "Lead (d)", "adr_bin": "ADR (€)", "count": ""},
    ).update_layout(
        width=340, height=240, coloraxis_showscale=False,
        margin=dict(l=10, r=10, t=40, b=10),
    )
    c_canon_scatter.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))

    eda_bins = [0, 7, 14, 30, 60, 120, 240, 365]
    eda_labels = ["0–7", "8–14", "15–30", "31–60", "61–120", "121–240", "241–365"]
    canon_df["lt_bucket"] = pd.cut(
        canon_df["lead_time"], bins=eda_bins, labels=eda_labels, include_lowest=True,
    )
    bucket_rows = []
    for (_h, _b), _grp in canon_df.groupby(["hotel", "lt_bucket"], observed=True):
        bucket_rows.append({
            "hotel": _h, "lt_bucket": str(_b),
            "q1": float(_grp["adr"].quantile(0.25)),
            "median": float(_grp["adr"].median()),
            "q3": float(_grp["adr"].quantile(0.75)),
        })
    bucket_df = pd.DataFrame(bucket_rows)

    band_rgba = {
        "City Hotel":   "rgba(31, 119, 180, 0.18)",
        "Resort Hotel": "rgba(255, 127, 14, 0.18)",
    }
    c_canon_buckets = go.Figure()
    for _h in ["City Hotel", "Resort Hotel"]:
        _sub = (
            bucket_df[bucket_df["hotel"] == _h]
            .set_index("lt_bucket").reindex(eda_labels).reset_index()
        )
        col = HOTEL_COLORS[_h]
        c_canon_buckets.add_trace(go.Scatter(
            x=_sub["lt_bucket"], y=_sub["q3"],
            mode="lines", line=dict(width=0),
            showlegend=False, hoverinfo="skip",
        ))
        c_canon_buckets.add_trace(go.Scatter(
            x=_sub["lt_bucket"], y=_sub["q1"],
            mode="lines", line=dict(width=0),
            fill="tonexty", fillcolor=band_rgba[_h],
            showlegend=False, hoverinfo="skip",
        ))
        c_canon_buckets.add_trace(go.Scatter(
            x=_sub["lt_bucket"], y=_sub["median"],
            mode="lines+markers", line=dict(color=col),
            name=f"{_h} (median)",
        ))
    c_canon_buckets.update_layout(
        title="Median ADR by lead-time bucket (shaded = IQR)",
        xaxis_title="Lead-time bucket (days)",
        yaxis_title="ADR (€), canonical",
        width=560, height=320,
        margin=dict(l=10, r=10, t=60, b=10),
    )

    warn = (
        mo.md("> ⚠️ **No months selected** — falling back to all 12 months.")
        if empty_picker else None
    )
    _controls = mo.vstack([
        month_pick,
        mo.hstack([canon_room, canon_meal, canon_adults], justify="start"),
    ])
    _chart = mo.hstack([c_canon_scatter, c_canon_buckets], justify="start")
    mo.vstack([p for p in [_controls, warn, _chart] if p is not None])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Pick your own early/late cut per hotel

    The IQR chart suggests one threshold for City and a different one
    for Resort. Each column below is one hotel: choose the months you
    care about, **define the canonical booking that hotel should be
    measured on** (room type, meal plan, party size), set the lead-time
    cut (in days), and see the resulting **early-booking** (lead ≥ cut)
    vs **late-booking** (lead < cut) median canonical ADR. The two
    hotels' canon settings are independent — useful when, say, City's
    A/BB/1 has plenty of records but Resort's most representative
    canonical is A/HB/2.

    *A "–" in either table means the selected slice contains no canonical
    bookings in that bucket — usually because the filter is too narrow,
    the month filter is empty, or the cut sits outside the strip's
    lead-time range.*
    """)
    return


@app.cell(hide_code=True)
def _(bookings, mo):
    _all_months = [
        "January", "February", "March", "April", "May", "June",
        "July", "August", "September", "October", "November", "December",
    ]
    _room_opts = sorted(bookings["reserved_room_type"].dropna().unique().tolist())
    _meal_opts = sorted(bookings["meal"].dropna().unique().tolist())
    _adult_opts = [str(int(a)) for a in sorted(
        bookings["adults"].dropna().unique()
    ) if 0 <= a <= 4]

    city_month_pick = mo.ui.multiselect(
        options=_all_months,
        value=["June", "July", "August", "September"],
        label="City Hotel — months",
    )
    resort_month_pick = mo.ui.multiselect(
        options=_all_months,
        value=["June", "July", "August", "September"],
        label="Resort Hotel — months",
    )
    city_canon_room = mo.ui.multiselect(
        options=_room_opts, value=["A"], label="City — room type(s)",
    )
    city_canon_meal = mo.ui.multiselect(
        options=_meal_opts, value=["BB"], label="City — meal plan(s)",
    )
    city_canon_adults = mo.ui.multiselect(
        options=_adult_opts, value=["1"], label="City — party size",
    )
    resort_canon_room = mo.ui.multiselect(
        options=_room_opts, value=["A"], label="Resort — room type(s)",
    )
    resort_canon_meal = mo.ui.multiselect(
        options=_meal_opts, value=["BB"], label="Resort — meal plan(s)",
    )
    resort_canon_adults = mo.ui.multiselect(
        options=_adult_opts, value=["1"], label="Resort — party size",
    )
    city_cut = mo.ui.slider(
        start=1, stop=200, value=30, step=1,
        label="City Hotel — early/late cut (days)",
        full_width=True, show_value=True, debounce=True,
    )
    resort_cut = mo.ui.slider(
        start=1, stop=200, value=60, step=1,
        label="Resort Hotel — early/late cut (days)",
        full_width=True, show_value=True, debounce=True,
    )
    return (
        city_canon_adults,
        city_canon_meal,
        city_canon_room,
        city_cut,
        city_month_pick,
        resort_canon_adults,
        resort_canon_meal,
        resort_canon_room,
        resort_cut,
        resort_month_pick,
    )


@app.cell
def _(
    bookings,
    city_canon_adults,
    city_canon_meal,
    city_canon_room,
    city_cut,
    city_month_pick,
    mo,
    resort_canon_adults,
    resort_canon_meal,
    resort_canon_room,
    resort_cut,
    resort_month_pick,
):
    _all_months = [
        "January", "February", "March", "April", "May", "June",
        "July", "August", "September", "October", "November", "December",
    ]

    def _hotel_cut_table(
        _hotel, _months_picked, _cut_val,
        _rooms_pick, _meals_pick, _adults_pick,
    ):
        months_used = list(_months_picked) or _all_months
        rooms_used = list(_rooms_pick) or ["A"]
        meals_used = list(_meals_pick) or ["BB"]
        adults_used = [int(a) for a in (_adults_pick or ["1"])]
        _sub = bookings.loc[
            (bookings["is_canceled"] == 0)
            & (bookings["reserved_room_type"].isin(rooms_used))
            & (bookings["meal"].isin(meals_used))
            & (bookings["adults"].isin(adults_used))
            & (bookings["adr"].between(0, 250))
            & (bookings["lead_time"] <= 365)
            & (bookings["arrival_date_month"].isin(months_used))
            & (bookings["hotel"] == _hotel)
        ]
        late = _sub.loc[_sub["lead_time"] < _cut_val, "adr"]
        early = _sub.loc[_sub["lead_time"] >= _cut_val, "adr"]

        def _fmt(series):
            if not len(series):
                return ("–", "–", 0)
            return (f"€{series.median():.1f}", f"€{series.mean():.1f}", int(len(series)))

        l_med, l_mean, l_n = _fmt(late)
        e_med, e_mean, e_n = _fmt(early)
        canon_caption = (
            f"*Canonical = room {'/'.join(rooms_used)} · meal {'/'.join(meals_used)} · "
            f"{'/'.join(str(a) for a in adults_used)} adult(s)*"
        )
        return mo.md(
            f"**{_hotel} — cut at {int(_cut_val)} days**  \n"
            f"{canon_caption}\n\n"
            "| Group | Median ADR | Mean ADR | n |\n"
            "|---|---:|---:|---:|\n"
            f"| Late (lead &lt; cut) | {l_med} | {l_mean} | {l_n} |\n"
            f"| Early (lead ≥ cut) | {e_med} | {e_mean} | {e_n} |"
        )

    city_table = _hotel_cut_table(
        "City Hotel", city_month_pick.value, int(city_cut.value),
        city_canon_room.value, city_canon_meal.value, city_canon_adults.value,
    )
    resort_table = _hotel_cut_table(
        "Resort Hotel", resort_month_pick.value, int(resort_cut.value),
        resort_canon_room.value, resort_canon_meal.value, resort_canon_adults.value,
    )

    mo.hstack([
        mo.vstack([
            city_month_pick,
            mo.hstack([city_canon_room, city_canon_meal, city_canon_adults],
                      justify="start"),
            city_cut, city_table,
        ]),
        mo.vstack([
            resort_month_pick,
            mo.hstack([resort_canon_room, resort_canon_meal, resort_canon_adults],
                      justify="start"),
            resort_cut, resort_table,
        ]),
    ], justify="start")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **What the experiments are telling you.** Three structural patterns
    that emerge once you actually play with the controls — none of them
    decisions yet:

    - **The seasonal level shift dominates the hotel comparison.**
      All-year canonical medians are City €83 vs Resort €42 (a 2× gap).
      Restricted to Jun–Sep, Resort nearly **doubles** to ~€80 while
      City rises more modestly to ~€95. The high-season market is
      structurally different — *not* a louder version of the off-season
      market.
    - **Lead-time elasticity is highly conditional**, not a fixed hotel
      trait. With *all* months pooled and a moderate cut (~45 days),
      City separates by only a few euros (€85 vs €81) — barely a class
      signal at all. But **Resort in summer with a long cut** (e.g. 150+
      days) splits clearly: late-booking guests pay near rack rate
      (~€100+) while deep-advance bookers got real deals (~€60). The
      classic "late = expensive" pattern *does* live in this data — for
      Resort, in summer, and only once you push the cut far enough out
      that the late bucket is essentially the high season itself.
    - **IQR width is the other half of the class signal.** A wide IQR
      band means there is room to discriminate even when medians barely
      move; a narrow one means the segmentation has little to bite on.
      Whichever lead-time and month slice you settle on for Q1, check
      the IQR too — not just the medians.

    The decisions — threshold levels per hotel, which months count as
    "high season", how aggressive a cut to set — are yours to make in Q1.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Next steps

    - **Notebook 2 — Dynamic control.** Booking-strip extraction, a static
      protection-level simulator, and the state-space $S$-index.
    - **Notebook 3 — Prediction models for RM.** A leakage-aware
      cancellation classifier with calibration, plus a worked
      illustration of cancellation-adjusted occupancy $q^{\text{eff}}_t$.
    """)
    return


if __name__ == "__main__":
    app.run()