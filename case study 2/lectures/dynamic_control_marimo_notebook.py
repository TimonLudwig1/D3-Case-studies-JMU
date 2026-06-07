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
    # Case 2 · Notebook 2 — Dynamic Control

    **Data-Driven Decisions in Practice · D3 Applications · SS 2026**

    *Part 2 of 3* — what happens **inside a single stay night**. Extract
    the booking strip, run a static-protection simulator against it, then
    introduce the state-space index $S = q / (\tau\,\lambda)$ as the
    bridge from static to dynamic.

    | Notebook | Focus | Feeds |
    |---|---|---|
    | 1 · Data | Booking curves, fare-class structure, EDA bridge | Q1 |
    | **2 · Dynamic control** *(you are here)* | Booking strips, static simulator, state-space S-index | Q4 baseline |
    | 3 · Prediction models | Cancellation classifier, calibration, $q^{\text{eff}}_t$ illustration | Q2, Q4 advanced |

    *If you have not seen Notebook 1 yet, start there for the demand and
    pricing context. This notebook stands alone, but its policy choices
    only make sense once the empirical structure from NB1 is in your head.*

    > **Where does dynamic policy matter most?** NB1's §9 cut analysis is
    > the direct answer: dynamic protection is only worth the operational
    > cost where **late and early bookings price differently**.
    >
    > - **City Hotel pooled all-year:** late vs early medians are ~€85 vs
    >   ~€81 — barely a class signal. A static rule is fine.
    > - **Resort Hotel in summer** with a long cut (~150+ days): late
    >   ~€100+, early ~€60. *This* is where dynamic protection earns its
    >   keep — protect rooms for late high-fare bookings; release them
    >   only after the deep-advance discount window closes.
    >
    > The default below is Resort Hotel on a peak-summer night for exactly
    > this reason. Switch to City to see how flat the simulator looks when
    > the underlying class signal is weak.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Loading the data
    """)
    return


@app.cell
def _():
    import numpy as np
    import pandas as pd
    import plotly.express as px
    import plotly.graph_objects as go

    DATA_URL = "https://raw.githubusercontent.com/mpolinowski/hotel-booking-dataset/refs/heads/master/datasets/hotel_bookings.csv"

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
        df["lead_weeks"] = (df["lead_time"] // 7).astype(int)
        df["adr_per_adult"] = df["adr"] / df["adults"].clip(lower=1)
        return df

    bookings = load_bookings()
    bookings.shape

    # Global static-plot defaults: no hover, no drag, no modebar, no
    # axis zoom — approximates Plotly's config={'staticPlot': True}.
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
    return bookings, go, np, pd, px


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## §3 · Extracting Booking Strips from Real Data

    A **booking strip** is the slide-4 idea: focus on *one* arrival date,
    extract every booking whose stay covers that night, and sort them by
    booking date. The result is a synthetic time-series of decision moments
    — exactly the kind of stream Clara would see in real time.

    Pick a stay date below. The widget filters the dataset and renders the
    strip as a sortable table plus an arrival-timing chart. This is the
    raw material for every dynamic decision rule we build later.
    """)
    return


@app.cell(hide_code=True)
def _(bookings, mo):
    valid_dates = sorted(bookings["arrival_date"].dropna().dt.date.unique())
    # Default to a known peak-summer night for Resort — this is where
    # NB1's cut analysis surfaced the strongest case for dynamic protection.
    import datetime as _dt
    default_date = _dt.date(2016, 8, 13)
    if default_date not in valid_dates:
        default_date = valid_dates[len(valid_dates) // 2]
    stay_picker = mo.ui.dropdown(
        options=[str(d) for d in valid_dates],
        value=str(default_date),
        label="Pick a stay night",
    )
    hotel_picker = mo.ui.dropdown(
        options=["City Hotel", "Resort Hotel", "Both"],
        value="Resort Hotel",
        label="Hotel",
    )
    room_types = sorted(bookings["reserved_room_type"].dropna().unique().tolist())
    room_picker = mo.ui.dropdown(
        options=room_types,
        value="A" if "A" in room_types else room_types[0],
        label="Room type",
    )
    mo.hstack([stay_picker, hotel_picker, room_picker])
    return hotel_picker, room_picker, stay_picker


@app.cell
def _(bookings, hotel_picker, pd, room_picker, stay_picker):
    stay_date = pd.to_datetime(stay_picker.value)

    # A booking covers a stay night if arrival ≤ night < arrival + n_nights.
    nights = bookings["stays_in_weekend_nights"] + bookings["stays_in_week_nights"]
    covers = (
        (bookings["arrival_date"] <= stay_date)
        & (stay_date < bookings["arrival_date"] + pd.to_timedelta(nights, unit="D"))
    )
    if hotel_picker.value != "Both":
        covers = covers & (bookings["hotel"] == hotel_picker.value)
    # Restrict to a single room type so the capacity slider and the fare
    # threshold both refer to a homogeneous inventory pool.
    covers = covers & (bookings["reserved_room_type"] == room_picker.value)

    strip_full = bookings.loc[covers].sort_values("booking_date").copy()
    strip_full["orig_idx"] = strip_full.index
    strip_full = strip_full.reset_index(drop=True)

    strip = strip_full[[
        "hotel", "booking_date", "arrival_date", "lead_time",
        "market_segment", "distribution_channel", "adr",
        "is_canceled", "reserved_room_type",
    ]].copy()
    strip
    return stay_date, strip


@app.cell(hide_code=True)
def _(mo, stay_date, strip):
    mo.md(f"""
    **Strip summary for {stay_date.date()}** —
    `{len(strip)}` requests,
    `{int(strip['is_canceled'].sum())}` cancellations,
    realised ADR = €`{strip.loc[strip['is_canceled'] == 0, 'adr'].mean():.1f}`.

    Two things to notice in the table above:

    1. The strip has **many more rows than rooms in any one hotel** —
       that is precisely the room count plus cancellations plus rejected
       requests' shadow data. Real-world strips look chunkier (more arrival
       clustering, more weekday/weekend signal); the dataset's curated form
       still preserves the temporal shape.
    2. Each row is a **decision moment**. With static rules, the answer
       "accept at fare *x*" only depends on the fare class and the booking
       limit. With dynamic rules, it depends on *which row we are at* and
       *what has happened before*.
    """)
    return


@app.cell
def _(px, strip):
    def _build_strip():
        if len(strip) == 0:
            return None
        plot_df = strip.copy()
        plot_df["status"] = plot_df["is_canceled"].map(
            {0: "Realised stay", 1: "Later cancelled"}
        )
        fig = px.scatter(
            plot_df, x="booking_date", y="adr", color="status",
            color_discrete_map={"Realised stay": "#1f77b4",
                                "Later cancelled": "#d62728"},
            hover_data=["market_segment", "distribution_channel"],
            opacity=0.7,
            title="Booking strip — colour = eventual outcome, dashed line = arrival",
            labels={"booking_date": "Booking date", "adr": "ADR (€)",
                    "status": "Status"},
        )
        fig.add_vline(
            x=plot_df["arrival_date"].iloc[0],
            line_dash="dash", line_color="black",
        )
        fig.update_traces(marker=dict(size=8))
        fig.update_layout(width=760, height=300,
                          margin=dict(l=10, r=10, t=40, b=10))
        return fig

    fig_strip = _build_strip()
    fig_strip
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## §4 · Implementing Adaptive Booking Logic

    Adaptive logic, in the slide's terms, is four moves:

    1. **Track available inventory** every day along the booking horizon.
    2. **Prioritise high-paying** customers — protect rooms for them.
    3. **Stop accepting low-fare requests** when remaining-demand risk is high.
    4. **Allow overbooking** when cancellations are likely.

    The trade-off is named on the slide: **spoilage vs. spillage**.

    - *Spoilage*: an empty room you should have sold (too restrictive).
    - *Spillage*: a paying guest you turned away because you protected
      too aggressively, or a walk-away because you overbooked too much.

    Capacity is **given** — it is the room count of the chosen type on
    the chosen night. Pick it once from the dropdown below; the same
    value drives both the §4a decision surface and the §4b drill-down.
    Your decision variables are the **ADR threshold** splitting low- vs
    high-fare and the **protection level** for the high-fare class.
    §4a shows the entire decision surface (revenue over the
    *split × protection* plane); §4b lets you drill into one point and
    watch the inventory trajectory request-by-request.

    > **Known simplification — cancellations.** The simulator below
    > processes the strip in chronological booking order and accepts
    > each booking exactly once; it never *releases* rooms when a
    > booking later cancels. That ignores the cancellation column
    > entirely. A realistic policy has to choose: release rooms at the
    > cancellation moment? Hold them as overbooking buffer? Use them
    > to feed a waitlist? **Q2 explicitly invites you to design that
    > release mechanism** — the simulator here is the no-release
    > baseline that any improvement should beat.

    ### §4a · Revenue surface over the decision plane
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    cap_pick = mo.ui.dropdown(
        options=["100", "110", "120", "130", "140"],
        value="120",
        label="Capacity (rooms of this type for this night)",
    )
    cap_pick
    return (cap_pick,)


@app.cell
def _(cap_pick, go, mo, pd, strip):
    def _build_surface():
        if len(strip) == 0:
            return mo.md("*Strip is empty — pick a different stay night.*")
        cap = int(cap_pick.value)

        adr_arr = strip["adr"].to_numpy()
        n = len(adr_arr)
        adr_grid = list(range(40, 201, 10))
        prot_grid = list(range(0, 51, 2))

        rows = []
        for split in adr_grid:
            is_high = adr_arr >= split
            for prot in prot_grid:
                rooms = cap
                rev = 0.0
                for i in range(n):
                    a = adr_arr[i]
                    if is_high[i]:
                        if rooms > 0:
                            rooms -= 1
                            rev += a
                    else:
                        if rooms > prot:
                            rooms -= 1
                            rev += a
                rows.append({"split": split, "protection": prot,
                             "revenue": rev})
        surf_df = pd.DataFrame(rows)

        piv = surf_df.pivot(index="protection", columns="split",
                            values="revenue")
        zmax = float(surf_df["revenue"].max())
        zmin = float(surf_df["revenue"].min())
        opt = surf_df.loc[surf_df["revenue"].idxmax()]

        fig = go.Figure()
        fig.add_trace(go.Heatmap(
            x=piv.columns.tolist(), y=piv.index.tolist(),
            z=piv.values, zmin=zmin, zmax=zmax,
            colorscale="Viridis",
            colorbar=dict(title="Revenue (€)"),
            hovertemplate=(
                f"capacity = {cap}<br>"
                "ADR threshold = €%{x}<br>"
                "Protection = %{y}<br>"
                "Revenue = €%{z:,.0f}<extra></extra>"
            ),
        ))
        fig.add_trace(go.Scatter(
            x=[opt["split"]], y=[opt["protection"]], mode="markers",
            marker=dict(symbol="star", size=18, color="red",
                        line=dict(width=1.5, color="white")),
            showlegend=False,
            hovertemplate=(
                f"<b>Optimum · capacity {cap}</b><br>"
                "ADR threshold = €%{x}<br>"
                "Protection = %{y}<br>"
                f"Revenue = €{opt['revenue']:,.0f}<extra></extra>"
            ),
        ))
        fig.update_layout(
            width=620, height=420,
            margin=dict(l=10, r=10, t=60, b=40),
            title=(f"Revenue over (ADR threshold, protection)  ·  "
                   f"capacity = {cap}  ·  red ★ = optimum"),
            xaxis_title="ADR threshold (€)",
            yaxis_title="Protection",
            # Override the static template locally so students can hover.
            hovermode="closest",
        )
        fig.update_xaxes(fixedrange=False)
        fig.update_yaxes(fixedrange=False)
        return fig

    _build_surface()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **How to read it.** Each tile is one full pass of the §4 admission
    policy over the booking strip; colour is the revenue that policy
    would have realised. The **red ★** marks the optimum *given this
    single demand realisation*. What matters more than the absolute
    optimum:

    - **How flat is the plateau around the optimum?** A wide flat
      region means the policy is robust to mis-specifying split or
      protection by a few units. A sharp peak means the calibration
      has to be precise. Real-world Q4 policies should prefer the flat
      regions.
    - **How does the optimum migrate when you switch capacity?** Flip
      the capacity dropdown above between 100 and 140 and watch the
      star travel — as capacity grows, protection becomes less critical
      and the threshold matters more. This is the structural reason
      revenue management is a "tight-capacity" problem.

    The drill-down below uses the **same capacity** picked above; the
    split and protection sliders let you inspect the inventory
    trajectory for any single cell of the surface.

    ### §4b · Drill into a single (split, protection) point
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    # debounce=True so the simulator + S-trajectory cells only re-run on
    # slider release, not on every intermediate value during a drag.
    # Capacity is reused from the §4a dropdown above; only split and
    # protection are drill-down knobs.
    protection = mo.ui.slider(
        start=0, stop=50, value=15, step=1,
        label="Rooms protected for high-fare",
        full_width=True, show_value=True, debounce=True,
    )
    high_low_split = mo.ui.slider(
        start=40, stop=200, value=100, step=5,
        label="ADR threshold splitting low vs high fare for this room type (€)",
        full_width=True, show_value=True, debounce=True,
    )
    return high_low_split, protection


@app.cell
def _(cap_pick, high_low_split, mo, np, pd, protection, px, strip):
    def _simulate():
        rooms_total = int(cap_pick.value)
        protect = int(protection.value)
        fare_cut = float(high_low_split.value)

        df = strip.copy()
        df["fare_class"] = np.where(df["adr"] >= fare_cut, "high", "low")
        df = df.sort_values("booking_date").reset_index(drop=True)

        rooms = rooms_total
        hi = lo = rej_hi = rej_lo = 0
        rev = 0.0
        traj = []

        for _, r in df.iterrows():
            if r["fare_class"] == "high":
                if rooms > 0:
                    rooms -= 1; hi += 1; rev += r["adr"]
                else:
                    rej_hi += 1
            else:
                if rooms > protect:
                    rooms -= 1; lo += 1; rev += r["adr"]
                else:
                    rej_lo += 1
            traj.append(rooms)

        traj_df = pd.DataFrame({"request_index": range(1, len(traj) + 1),
                                "rooms_remaining": traj})
        chart = px.line(
            traj_df, x="request_index", y="rooms_remaining",
            title=f"Inventory under static protection (red dash = protection level {protect})",
            labels={"request_index": "Request index (chronological)",
                    "rooms_remaining": "Rooms remaining"},
        )
        chart.update_traces(line_color="#1f77b4")
        chart.add_hline(y=protect, line_dash="dash", line_color="#d62728")
        chart.update_layout(width=600, height=280,
                            margin=dict(l=10, r=10, t=40, b=10))
        return chart, hi, lo, rej_hi, rej_lo, rev, rooms

    fig_sim, high_sold, low_sold, rejected_high, rejected_low, revenue, rooms_left = _simulate()
    sim_table = mo.md(f"""
    | Result | Value |
    |---|---:|
    | High-fare accepted | {high_sold} |
    | Low-fare accepted | {low_sold} |
    | High-fare rejected (**spillage**) | {rejected_high} |
    | Low-fare rejected | {rejected_low} |
    | Rooms unsold (**spoilage** if > 0) | {max(rooms_left, 0)} |
    | Realised revenue | €{revenue:,.0f} |
    """)
    mo.vstack([
        mo.vstack([protection, high_low_split]),
        mo.hstack([fig_sim, sim_table], widths=[2, 1], align="center"),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Move the protection slider. You will see:

    - Low protection → **spillage** when too many low-fare bookings consume
      capacity before the high-fare requests arrive.
    - High protection → **spoilage** because low-fare requests are rejected
      that you cannot replace with high-fare demand.

    The static optimum is *somewhere in between* and depends on the demand
    realisation. A dynamic rule replaces the constant protection number
    with a **state-dependent** one — which is the topic of the next section.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## §4.5 · Calibrating the arrival rate

    Before any dynamic decision rule, the single most useful number you
    can read off the strip is **how fast bookings arrive on average**.
    That number, $\hat\lambda$, is what the state-space index in §5
    uses as a yardstick for "balanced" vs "starved" vs "drowning".

    Two questions to ask the data:

    1. **Point estimate.** Average arrivals per day across the booking
       horizon:

       $$
       \hat\lambda \;=\; \frac{\text{arrivals in the strip}}{\text{lead-time horizon (days)}}
       $$

    2. **Variation.** Is the rate stable, or does it have bursts and lulls?
       Bin the strip by week and look at the spread. If the standard
       deviation of weekly arrivals is roughly $\sqrt{\text{mean}}$ (the
       Poisson reference), a single $\hat\lambda$ is a defensible
       summary. If it is much larger, your rate is **overdispersed** —
       clustering by channel, weekday, or season dominates, and Q4 needs
       a more nuanced forecast than the constant we use below.
    """)
    return


@app.cell
def _(mo, pd, px, strip):
    def _build_calibration():
        if len(strip) == 0:
            return mo.md("*Strip is empty for this selection.*")

        n_total = len(strip)
        horizon_days = max(int(strip["lead_time"].max()) + 1, 1)
        lam_point = n_total / horizon_days

        weekly = (
            pd.to_datetime(strip["booking_date"])
            .dt.to_period("W").dt.start_time
            .value_counts().sort_index()
            .rename_axis("week").reset_index(name="arrivals")
        )
        mean_w = float(weekly["arrivals"].mean())
        sd_w = float(weekly["arrivals"].std(ddof=0))
        poisson_sd = mean_w ** 0.5

        c_weekly = px.bar(
            weekly, x="week", y="arrivals",
            title="Arrivals per booking-week (this strip)",
            labels={"week": "Booking week", "arrivals": "Bookings landed"},
        )
        c_weekly.update_traces(marker_color="#1f77b4")
        c_weekly.add_hline(y=mean_w, line_dash="dash", line_color="black")
        c_weekly.update_layout(width=600, height=240,
                               margin=dict(l=10, r=10, t=40, b=10))

        od_ratio = sd_w / poisson_sd if poisson_sd > 0 else float("nan")
        # Rule of thumb: ratios under ~1.5 are close enough to Poisson that
        # a single λ̂ is operationally fine; over 2 means real bursts
        # dominate and a constant rate will mis-time the policy.
        if od_ratio < 1.5:
            od_verdict = "≈ Poisson — a single λ̂ is a defensible summary"
        elif od_ratio < 2.0:
            od_verdict = "moderate overdispersion — λ̂ is usable, but expect noisy timing"
        else:
            od_verdict = "**bursty** — real clustering dominates; Q4 needs a richer forecast"
        summary = mo.md(
            f"**Calibration for this strip**\n\n"
            f"- Total arrivals: **{n_total}**, lead-time horizon: **{horizon_days} days**\n"
            f"- Point estimate **λ̂ = {lam_point:.3f}** bookings/day\n"
            f"- Weekly mean: **{mean_w:.1f}** · Weekly SD: **{sd_w:.1f}** "
            f"(Poisson reference √mean = **{poisson_sd:.1f}**)\n"
            f"- Overdispersion ratio SD/√mean = **{od_ratio:.2f}** — {od_verdict}"
        )
        return mo.vstack([c_weekly, summary])

    _build_calibration()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## §5 · A State Space for Dynamic Control

    The slide gives you a scalar **booking-pressure index** that compresses
    the two relevant axes (rooms left, time left) into one number:

    $$
    S \;=\; \frac{\text{rooms left}}{\text{remaining days} \;\cdot\; \text{daily demand}}
    $$

    Interpretation:

    - $S > 1$: **oversupplied** — accept almost everything; risk is spoilage.
    - $S \approx 1$: **balanced** — apply the static rule.
    - $S < 1$: **undersupplied** — start protecting aggressively for high fare.

    This is a *blunt* but powerful device. The slide warns it is non-linear
    in practice (high-value demand clusters late, channel mixes shift). It
    is your job in Q4 to decide *how* to use $S$: as a soft signal, a hard
    threshold, or a multiplier on the static protection level.

    **$S$ is a diagnostic, not a policy artefact.** It reads the
    *observed demand-vs-capacity balance* off the booking strip — it
    does **not** depend on the ADR threshold or protection level you
    happen to pick in §4. You then *use* $S$ to design the policy.
    Below we therefore compute $S(t)$ along the natural booking pace
    (every request accepted until rooms run out), driven only by the
    structural input **capacity**.
    """)
    return


@app.cell
def _(cap_pick, mo, np, pd, px, stay_date, strip):
    def _build_s():
        if len(strip) == 0:
            return mo.md("*Strip is empty for this selection.*")
        cap = int(cap_pick.value)
        horizon_days = max(int(strip["lead_time"].max()) + 1, 1)
        lam = max(len(strip) / horizon_days, 1e-3)

        # Natural pace: every request accepted, rooms drain monotonically
        # until full. Floor at 0 once exhausted.
        sim = strip.sort_values("booking_date").reset_index(drop=True).copy()
        sim["request_index"] = np.arange(1, len(sim) + 1)
        sim["rooms_left"] = np.maximum(cap - sim["request_index"], 0)
        sim["days_to_arrival"] = (
            pd.to_datetime(stay_date) - sim["booking_date"]
        ).dt.days
        sim["S_index"] = sim["rooms_left"] / (
            np.maximum(sim["days_to_arrival"], 1) * lam
        )

        s_max = float(sim["S_index"].max())
        y_top = max(min(s_max * 1.1, 6.0), 1.5)

        fig = px.line(
            sim, x="booking_date", y="S_index",
            hover_data={"rooms_left": True, "days_to_arrival": True,
                        "S_index": ":.2f"},
            labels={"booking_date": "Booking date", "S_index": "S index"},
            title=(
                f"S(t) on the natural booking pace  ·  capacity = {cap}  ·  "
                f"empirical λ̂ = {lam:.2f} bookings/day"
            ),
        )
        fig.update_traces(line_color="#2ca02c", line_width=2)
        fig.update_yaxes(range=[0, y_top])
        fig.add_hrect(y0=0, y1=1, fillcolor="#d62728",
                      opacity=0.10, line_width=0)
        fig.add_hline(y=1.0, line_dash="dash", line_color="black")
        crossing = sim[sim["S_index"] < 1].head(1)
        if len(crossing):
            fig.add_vline(x=crossing["booking_date"].iloc[0],
                          line_width=1.5, line_color="black")
        fig.update_layout(width=760, height=300,
                          margin=dict(l=10, r=10, t=60, b=10),
                          hovermode="closest")
        fig.update_xaxes(fixedrange=False)
        fig.update_yaxes(fixedrange=False)
        return fig

    _build_s()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **What this chart actually shows.** The green line is the
    booking-pressure index under a "natural" trajectory — every request
    accepted until rooms run out. It is a property of *capacity vs
    observed demand*, not of any specific admission policy. Three
    readings:

    - **Initial S level.** Very high at the start (lots of time, lots
      of capacity). If S stays well above 1 for most of the horizon,
      the room type is structurally oversupplied for this night — the
      policy can stay generous.
    - **The black rule** is the first booking date at which $S < 1$.
      That timestamp is your natural candidate for "when to stop
      accepting low fare and start protecting" — i.e., a time-based
      analogue of §4's fixed protection level. Different capacities
      shift this crossing earlier or later.
    - **Where S ends up.** A trajectory that dives steeply through 1
      and stays low means demand will saturate well before arrival —
      protection should kick in *much* earlier than the crossing
      suggests, because high-fare requests typically cluster late.

    *Why the §4 sliders don't show up here.* They drove the previous
    version of this chart and made $S$ look policy-dependent, which
    inverted the causality. The right mental model is: **S informs
    policy, not the other way round.** Use the §4 surface above to
    pick a (split, protection) pair *given* what S tells you about
    structural pressure.

    *Methodological caution* (from the slide): the rule
    "$S<1 \Rightarrow \text{reject low}$" is too coarse — high-value
    demand clusters near arrival, so a Q4 write-up should justify a
    softer translation (e.g. protection level scales with $1/S$, or
    triggers only after S stays below 1 for $k$ consecutive days).
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Next step

    The $S$-index uses *naive* inventory (raw room count) and *average*
    daily demand. Both can be replaced with predictions — that is the
    subject of **Notebook 3**, which trains a cancellation classifier
    and shows what the cancellation-adjusted occupancy
    $q^{\text{eff}}_t$ looks like along a strip. Wiring those
    predictions back into the control rule is the case-work payoff
    (Q2 and Q4).
    """)
    return


if __name__ == "__main__":
    app.run()