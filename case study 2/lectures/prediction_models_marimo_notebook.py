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
    # Case 2 · Notebook 3 — Prediction Models for RM

    **Data-Driven Decisions in Practice · D3 Applications · SS 2026**

    *Part 3 of 3* — the prediction layer of the slide-2 maturity curve:
    a leakage-aware **cancellation classifier**, calibration as the
    bridge from probability to decision, and one worked illustration
    ($q^{\text{eff}}_t$, slide 9) of how a calibrated prediction plugs
    into the dynamic state space from NB2.

    | Notebook | Focus | Feeds |
    |---|---|---|
    | 1 · Data | Booking curves, fare-class structure, EDA bridge | Q1 |
    | 2 · Dynamic control | Booking strips, static simulator, state-space S-index | Q4 baseline |
    | **3 · Prediction models** *(you are here)* | Cancellation classifier, calibration, $q^{\text{eff}}_t$ illustration | Q2, Q4 advanced |

    *This notebook trains and explains the prediction pipeline. **It does
    not wire predictions into a closed-loop control policy** — that
    integration is Q4 territory. What this notebook gives you is a
    calibrated $\hat{P}(\text{cancel})$ you can plug into your own buffer
    or admission rule, plus one worked example showing what the resulting
    trajectory looks like.*
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Loading the data and extracting a strip

    Same loader as NB1/NB2, plus a self-contained strip extractor so this
    notebook runs independently.
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
    return bookings, np, pd, px


@app.cell(hide_code=True)
def _(bookings, mo):
    valid_dates = sorted(bookings["arrival_date"].dropna().dt.date.unique())
    default_idx = len(valid_dates) // 2
    stay_picker = mo.ui.dropdown(
        options=[str(d) for d in valid_dates],
        value=str(valid_dates[default_idx]),
        label="Pick a stay night (used in §8 augmented occupancy)",
    )
    hotel_picker = mo.ui.dropdown(
        options=["City Hotel", "Resort Hotel", "Both"],
        value="City Hotel",
        label="Hotel",
    )
    mo.hstack([stay_picker, hotel_picker])
    return hotel_picker, stay_picker


@app.cell
def _(bookings, hotel_picker, pd, stay_picker):
    stay_date = pd.to_datetime(stay_picker.value)
    nights = bookings["stays_in_weekend_nights"] + bookings["stays_in_week_nights"]
    covers = (
        (bookings["arrival_date"] <= stay_date)
        & (stay_date < bookings["arrival_date"] + pd.to_timedelta(nights, unit="D"))
    )
    if hotel_picker.value != "Both":
        covers = covers & (bookings["hotel"] == hotel_picker.value)
    strip_full = bookings.loc[covers].sort_values("booking_date").reset_index(drop=True).copy()
    strip_full.shape
    return (strip_full,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## §6 · Predictive Modelling for Revenue Management

    Slide 7 turns three operational questions into supervised-learning problems:

    | Question | Target | Task |
    |---|---|---|
    | Will this booking cancel? | `is_canceled` | Classification |
    | What is the likely ADR of the next request? | `adr` | Regression |
    | How many more requests will arrive? | counts by date | Forecasting |

    We focus on the first because it has the cleanest pay-off in your case:
    a calibrated cancellation probability **directly** feeds the
    overbooking buffer. The pipeline below is the standard scikit-learn
    pattern you saw with the Airbnb-rent case — `ColumnTransformer` for
    numeric/categorical, fit, evaluate.

    The pipeline below is provided as a finished artefact — this part of
    the course is not about tuning prediction pipelines, it is about
    deploying their output as part of a decision rule. Read it, run it,
    and use the resulting cancellation probabilities downstream.

    ### Why these features, and not others

    Several columns in the raw CSV are **only populated after the booking
    decision has been made**. Using them looks brilliant on a hold-out set
    and is useless in production. We drop them:

    | Column | Why it leaks |
    |---|---|
    | `reservation_status`, `reservation_status_date` | Direct labels of the outcome — these *are* the target |
    | `assigned_room_type` | Determined at check-in, not at booking |
    | `booking_changes` | Cumulative count over the booking's lifetime |
    | `days_in_waiting_list` | Only known once the waiting period has ended |
    | `country` (optional) | High-cardinality and proxy for distribution channel; we drop it for generality |

    The remaining features are split into a numeric and a categorical
    group, each with its own preprocessing pipeline (median-impute + scale
    for numerics, constant-impute + one-hot for categoricals). The two
    pipelines are wired together with `ColumnTransformer` and fed into a
    Random Forest classifier inside one `Pipeline`. The whole thing fits
    in one `.fit()` call and scores new requests in one `.predict_proba()`.
    """)
    return


@app.cell
def _(bookings):
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression  # noqa: F401
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import (
        brier_score_loss, matthews_corrcoef, recall_score, roc_auc_score,
    )
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    # Subsample for snappy interactive fitting (full dataset works locally,
    # but in molab/WASM staying around 20k rows keeps things responsive).
    # Sample size and tree count tuned for browser/WASM execution
    # (Pyodide is ~3–5× slower than native CPython). Lower these further
    # if students report slow first-load times on molab.
    df_model = bookings.sample(n=min(10_000, len(bookings)), random_state=42).copy()

    # Numeric features: all observable the moment the request arrives.
    num_features = [
        "lead_time", "stays_in_weekend_nights", "stays_in_week_nights",
        "adults", "children", "babies", "is_repeated_guest",
        "previous_cancellations", "previous_bookings_not_canceled",
        "required_car_parking_spaces", "total_of_special_requests", "adr",
    ]
    # Categorical features: the channel/segment/contract metadata the
    # booking system already carries when the request lands.
    cat_features = [
        "hotel", "arrival_date_month", "meal", "market_segment",
        "distribution_channel", "reserved_room_type", "deposit_type",
        "customer_type",
    ]

    X = df_model[num_features + cat_features].copy()
    X[num_features] = X[num_features].fillna(0)
    y = df_model["is_canceled"].astype(int)

    pre = ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                          ("sc", StandardScaler())]), num_features),
        ("cat", Pipeline([("imp", SimpleImputer(strategy="constant", fill_value="UNK")),
                          ("oh", OneHotEncoder(handle_unknown="ignore"))]), cat_features),
    ])

    pipe = Pipeline([("pre", pre),
                     ("clf", RandomForestClassifier(
                         n_estimators=100, n_jobs=-1, random_state=42,
                         min_samples_leaf=20, class_weight="balanced_subsample"))])

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=42
    )
    pipe.fit(X_tr, y_tr)
    p_te = pipe.predict_proba(X_te)[:, 1]
    auc = roc_auc_score(y_te, p_te)
    brier = brier_score_loss(y_te, p_te)
    # Hit rate = recall on the positive class at the 0.5 threshold,
    # i.e. share of actual cancellations the model would catch under
    # a "flag if p_hat >= 0.5" rule.
    y_pred = (p_te >= 0.5).astype(int)
    hit_rate = recall_score(y_te, y_pred)
    mcc = matthews_corrcoef(y_te, y_pred)
    return auc, brier, hit_rate, mcc, p_te, pipe, y_te


@app.cell(hide_code=True)
def _(auc, brier, hit_rate, mcc, mo):
    mo.md(f"""
    **Cancellation model trained.**

    - ROC-AUC = `{auc:.3f}` (random baseline = 0.5; well-calibrated models
      on this dataset typically sit between 0.83 and 0.90 depending on the
      feature set)
    - Brier score = `{brier:.3f}` (lower is better; this captures
      calibration, which matters more than AUC for overbooking decisions)
    - Hit rate = `{hit_rate:.3f}` (share of actual cancellations caught
      when flagging at $\hat{{p}} \\geq 0.5$; threshold-dependent)
    - MCC = `{mcc:.3f}` (Matthews correlation coefficient; balanced
      single-number quality at the 0.5 threshold)

    > **Hit rate in one line.** Of the bookings that *did* cancel, what
    > fraction did the model flag with $\hat{{p}} \\geq 0.5$? Formally
    > $\\text{{TP}} / (\\text{{TP}} + \\text{{FN}})$ — same as recall /
    > sensitivity / true-positive rate. It is the metric you care about
    > when the cost of a missed cancellation is large (you held the room
    > for nothing). Unlike AUC and Brier it depends on a chosen decision
    > threshold (0.5 here); shifting that threshold trades hit rate
    > against false alarms.

    > **MCC in one line.** A single number summarising the whole
    > confusion matrix at the chosen threshold:
    > $\\text{{MCC}} = \\dfrac{{\\text{{TP}}\\cdot\\text{{TN}} - \\text{{FP}}\\cdot\\text{{FN}}}}{{\\sqrt{{(\\text{{TP}}+\\text{{FP}})(\\text{{TP}}+\\text{{FN}})(\\text{{TN}}+\\text{{FP}})(\\text{{TN}}+\\text{{FN}})}}}}$.
    > Ranges in $[-1, 1]$: 1 perfect, 0 random, negative means worse than
    > random. It is the only common single-number metric that stays
    > honest under heavy class imbalance — accuracy and F1 can both
    > look high while MCC reveals the model is barely doing anything.

    > **Brier score in one line.** It is the mean squared error between the
    > predicted probability and the 0/1 outcome:
    > $\\text{{Brier}} = \\tfrac{{1}}{{n}}\\sum_i (\\hat{{p}}_i - y_i)^2$.
    > A coin flip predicting 0.5 for everything scores 0.25; a model that
    > always says 1 when the outcome is 1 and 0 when it isn't scores 0. So
    > **lower is better**, with 0 being perfect and ~0.25 being "no
    > better than the base rate guess". Unlike AUC, Brier punishes a model
    > that is confidently wrong about the *magnitude* of the probability,
    > not just the ordering — which is exactly what an overbooking
    > calculation cares about.

    Pure discrimination (AUC) is not enough. To translate a probability into
    a decision (*"how many to overbook?"*) the probabilities must be
    **calibrated** — when the model says 30% it should cancel ~30% of the time.
    """)
    return


@app.cell
def _(p_te, pd, px, y_te):
    from sklearn.calibration import calibration_curve
    frac_pos, mean_pred = calibration_curve(y_te, p_te, n_bins=10, strategy="quantile")

    cal_df = pd.DataFrame({"predicted": mean_pred, "observed": frac_pos})

    fig_cal = px.line(
        cal_df, x="predicted", y="observed", markers=True,
        labels={"predicted": "Predicted cancellation probability",
                "observed": "Observed cancellation rate"},
        title="Calibration plot — do the probabilities mean what they say?",
    )
    fig_cal.update_traces(line_color="#1f77b4")
    fig_cal.add_shape(type="line", x0=0, y0=0, x1=1, y1=1,
                      line=dict(color="black", dash="dash"))
    fig_cal.update_layout(width=460, height=380,
                          margin=dict(l=10, r=10, t=50, b=10))
    fig_cal
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Playground — feel which features carry the signal

    Toggle features below to see how much each one contributes, and try a
    different classifier to see how *calibration* changes even when AUC
    stays similar.

    > **Scratch fit, not the reference model.** This playground trains a
    > separate pipeline on a smaller subsample (8 000 rows, random seed 7)
    > with a faster classifier configuration. The §6 reference model
    > used downstream in §8 is the one shown two cells above (10 000
    > rows, 100 trees) and is *unaffected* by anything you do here. The
    > AUC and Brier numbers below will not exactly match the reference.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    NUM_ALL = [
        "lead_time", "stays_in_weekend_nights", "stays_in_week_nights",
        "adults", "children", "babies", "is_repeated_guest",
        "previous_cancellations", "previous_bookings_not_canceled",
        "required_car_parking_spaces", "total_of_special_requests", "adr",
    ]
    CAT_ALL = [
        "hotel", "arrival_date_month", "meal", "market_segment",
        "distribution_channel", "reserved_room_type", "deposit_type",
        "customer_type",
    ]

    num_pick = mo.ui.multiselect(
        options=NUM_ALL,
        value=["lead_time", "adr", "total_of_special_requests",
               "previous_cancellations"],
        label="Numeric features",
    )
    cat_pick = mo.ui.multiselect(
        options=CAT_ALL,
        value=["hotel", "market_segment", "deposit_type"],
        label="Categorical features",
    )
    model_pick = mo.ui.dropdown(
        options=["Logistic Regression", "Random Forest", "Gradient Boosting"],
        value="Random Forest",
        label="Classifier",
    )
    mo.vstack([num_pick, cat_pick, model_pick])
    return cat_pick, model_pick, num_pick


@app.cell
def _(bookings, cat_pick, mo, model_pick, num_pick, pd, px):
    from sklearn.calibration import calibration_curve as _cal_curve
    from sklearn.compose import ColumnTransformer as _CT
    from sklearn.ensemble import (
        GradientBoostingClassifier as _GBM,
        RandomForestClassifier as _RF,
    )
    from sklearn.impute import SimpleImputer as _SI
    from sklearn.linear_model import LogisticRegression as _LR
    from sklearn.metrics import (
        brier_score_loss as _brier,
        matthews_corrcoef as _mcc,
        recall_score as _recall,
        roc_auc_score as _auc,
    )
    from sklearn.model_selection import train_test_split as _split
    from sklearn.pipeline import Pipeline as _Pipe
    from sklearn.preprocessing import OneHotEncoder as _OHE, StandardScaler as _SS

    chosen_num = list(num_pick.value)
    chosen_cat = list(cat_pick.value)

    if not chosen_num and not chosen_cat:
        fig_play = mo.md("> **Pick at least one feature above.**")
    else:
        df_play = bookings.sample(n=min(8_000, len(bookings)), random_state=7).copy()
        X_play = df_play[chosen_num + chosen_cat].copy()
        if chosen_num:
            X_play[chosen_num] = X_play[chosen_num].fillna(0)
        y_play = df_play["is_canceled"].astype(int)

        transformers = []
        if chosen_num:
            transformers.append(("num", _Pipe([("i", _SI(strategy="median")),
                                                ("s", _SS())]), chosen_num))
        if chosen_cat:
            transformers.append(("cat", _Pipe([
                ("i", _SI(strategy="constant", fill_value="UNK")),
                ("o", _OHE(handle_unknown="ignore"))]), chosen_cat))
        pre_play = _CT(transformers)

        if model_pick.value == "Logistic Regression":
            clf = _LR(max_iter=1000, class_weight="balanced")
        elif model_pick.value == "Gradient Boosting":
            clf = _GBM(n_estimators=120, max_depth=3, random_state=7)
        else:
            clf = _RF(n_estimators=80, n_jobs=-1, random_state=7,
                      min_samples_leaf=25, class_weight="balanced_subsample")

        pipe_play = _Pipe([("pre", pre_play), ("clf", clf)])
        Xtr, Xte, ytr, yte = _split(X_play, y_play, test_size=0.25,
                                    stratify=y_play, random_state=7)
        pipe_play.fit(Xtr, ytr)
        pte = pipe_play.predict_proba(Xte)[:, 1]
        ypred = (pte >= 0.5).astype(int)
        auc_play = _auc(yte, pte)
        brier_play = _brier(yte, pte)
        hit_play = _recall(yte, ypred)
        mcc_play = _mcc(yte, ypred)

        frac_pos_p, mean_pred_p = _cal_curve(yte, pte, n_bins=10, strategy="quantile")
        cal_play_df = pd.DataFrame({"predicted": mean_pred_p, "observed": frac_pos_p})

        fig_play_chart = px.line(
            cal_play_df, x="predicted", y="observed", markers=True,
            labels={"predicted": "Predicted cancellation probability",
                    "observed": "Observed cancellation rate"},
            title=f"Playground calibration · {model_pick.value}",
        )
        fig_play_chart.update_traces(line_color="#2ca02c")
        fig_play_chart.add_shape(type="line", x0=0, y0=0, x1=1, y1=1,
                                 line=dict(color="black", dash="dash"))
        fig_play_chart.update_layout(width=440, height=380,
                                     margin=dict(l=10, r=10, t=50, b=10))

        # Lower-is-better for Brier; readable verdict to anchor the eye.
        if brier_play < 0.10:
            brier_verdict = "well calibrated"
        elif brier_play < 0.20:
            brier_verdict = "acceptable"
        else:
            brier_verdict = "poorly calibrated"
        # Standard rough cutoffs for ROC AUC.
        if auc_play >= 0.85:
            auc_verdict = "strong"
        elif auc_play >= 0.75:
            auc_verdict = "useful"
        elif auc_play >= 0.65:
            auc_verdict = "weak"
        else:
            auc_verdict = "near-random"

        # MCC verdicts: rough cutoffs commonly used in clinical/ML lit.
        if mcc_play >= 0.5:
            mcc_verdict = "strong"
        elif mcc_play >= 0.3:
            mcc_verdict = "useful"
        elif mcc_play >= 0.1:
            mcc_verdict = "weak"
        else:
            mcc_verdict = "near-random"
        metrics_table = mo.md(
            "**Playground metrics**\n\n"
            "| Metric | Value | Read |\n"
            "|---|---:|:--|\n"
            f"| Classifier | — | {model_pick.value} |\n"
            f"| Numeric features | {len(chosen_num)} | of 12 |\n"
            f"| Categorical features | {len(chosen_cat)} | of 8 |\n"
            f"| ROC-AUC | **{auc_play:.3f}** | {auc_verdict} |\n"
            f"| Brier score | **{brier_play:.3f}** | {brier_verdict} |\n"
            f"| Hit rate (p̂≥0.5) | **{hit_play:.3f}** | "
            f"recall on cancellations |\n"
            f"| MCC (p̂≥0.5) | **{mcc_play:.3f}** | {mcc_verdict} |\n\n"
            "*AUC ranks; Brier scores the probability magnitude; hit rate "
            "and MCC depend on the 0.5 threshold. Brier is the one that "
            "matters for an overbooking rule; MCC is the honest single-"
            "number summary if you commit to a hard classify-or-not cut.*"
        )
        fig_play = mo.hstack([fig_play_chart, metrics_table],
                             widths=[2, 1], align="center")
    fig_play
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Things worth trying:

    - **Strip everything except `lead_time` and `deposit_type`.** Watch
      how much of the AUC survives. The result tells you which signals
      are actually doing the work.
    - **Switch from Random Forest to Logistic Regression.** AUC drops a
      little; calibration often improves. For decision rules that *use*
      the probability, the trade may be worth it.
    - **Add `market_segment` and `distribution_channel`.** Channel mix
      carries a lot of cancellation risk in this dataset.
    - **Drop `adr`.** Notice how the model still works — the strongest
      cancellation signals are not price-related.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Calibration matters more than AUC for any decision that uses the
    probability as a *number* rather than just as a ranking. A weaker
    model that says what it means tends to beat a stronger model that is
    over-confident.

    ### Translating $\hat{P}(\text{cancel})$ into operational quantities

    The model returns one number per booking: the predicted probability of
    cancellation. Three things you can do with it:

    1. **Expected check-ins from a current book of business.** If you have
       $n$ bookings on the books with predicted cancellation probabilities
       $\hat{p}_1, \dots, \hat{p}_n$, the expected number of *arriving*
       guests is

       $$\mathbb{E}[\text{check-ins}] = \sum_{i=1}^n (1 - \hat{p}_i)$$

       This is the $q^{\text{eff}}_t$ quantity from slide 9, which we
       plot live in §8 below.

    2. **Static overbooking buffer.** The simplest rule: accept extra
       bookings until the expected check-ins equal capacity. If the
       average cancellation probability on a representative night is
       $\bar{p}$, then to fill $c$ rooms in expectation you sell
       $$n^\star = \frac{c}{1 - \bar{p}}.$$
       The buffer is $n^\star - c$. Worked example: 100 rooms, $\bar{p} =
       0.25 \Rightarrow n^\star = 133$, buffer of 33. Whether 33 is the
       *right* buffer depends on the walk-cost, the spread of $\hat{p}$,
       and your tolerance for walk-aways — Q2 territory.

    3. **Per-request decision rule.** When a low-fare request arrives at
       time $t$, you can ask: does accepting it push expected check-ins
       above the capacity-plus-tolerated-walk-rate threshold? This is a
       fully dynamic version of (2) and is exactly the kind of policy
       Q4 invites you to design.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Using the model on a new request

    You score a new request with one call:

    ```python
    p_cancel = pipe.predict_proba(new_request_df[feat_cols])[:, 1]
    ```

    where `new_request_df` is a DataFrame with the same columns the model
    was trained on. `pipe` carries its own preprocessing, so you do *not*
    one-hot or impute by hand. In §8 we use exactly this call against the
    booking strip and turn the per-request probabilities into a
    cancellation-adjusted occupancy trajectory.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## §7 · Bringing It Together: Raw → Model → Decision

    The pipeline you are now wiring up is:

    ```
        hotel_bookings.csv
              │
              ▼
        feature engineering           ──►  segment / season / channel cuts
              │
              ▼
        prediction model              ──►  P(cancel), P(high-fare arrival), …
              │
              ▼
        decision logic                ──►  protection level(t), overbooking buffer
              │
              ▼
        booking-strip simulation      ──►  realised revenue, rejections
    ```

    Each arrow is a place where you make **a methodological choice**.
    The week-1 work fixed the last arrow; the work for week 2 is to make
    the middle two transparent and defensible.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## §8 · Prediction-Augmented State Space

    Slide 9 closes the loop. The naive state index uses *room count* and
    *time × average demand*. Both can be replaced by predictions:

    $$\text{Booking pressure} = \frac{q_t^{\text{eff}}}{\widehat{D}_{\text{remaining}}} \quad \text{where} \quad q_t^{\text{eff}} = \sum_i q_i (1 - \hat{P}(\text{cancel}_i))$$

    - $q_t^{\text{eff}}$ — **cancellation-adjusted occupancy**: discount
      each currently-on-the-books reservation by its predicted survival
      probability.
    - $\widehat{D}_{\text{remaining}}$ — **forecast of remaining arrivals**:
      not the historical mean, but a prediction conditioned on the current
      booking pace, day-of-week, season, etc.

    Below we compute $q_t^{\text{eff}}$ on the live strip using the
    cancellation model from §6 and contrast the *raw* inventory
    trajectory with the *risk-adjusted* one.
    """)
    return


@app.cell
def _(np, pd, pipe, px, strip_full):
    s_aug = strip_full.sort_values("booking_date").reset_index(drop=True).copy()

    # Pull the columns the trained model expects directly from the full
    # booking row carried along in strip_full — no leakage worries because
    # we only use features that were available at booking time.
    feat_cols = list(pipe.named_steps["pre"].feature_names_in_)
    p_cancel = pipe.predict_proba(s_aug[feat_cols])[:, 1]
    s_aug["p_cancel"] = p_cancel
    s_aug["expected_show"] = 1 - p_cancel
    s_aug["q_eff"] = s_aug["expected_show"].cumsum()
    s_aug["request_index"] = np.arange(1, len(s_aug) + 1)
    s_aug["naive_cum"] = s_aug["request_index"].astype(float)

    aug_long = pd.concat([
        s_aug[["request_index", "naive_cum"]]
        .rename(columns={"naive_cum": "value"})
        .assign(series="Naive cumulative bookings"),
        s_aug[["request_index", "q_eff"]]
        .rename(columns={"q_eff": "value"})
        .assign(series="q_eff (cancel-adjusted)"),
    ], ignore_index=True).sort_values(["series", "request_index"]).reset_index(drop=True)

    fig_aug = px.line(
        aug_long, x="request_index", y="value", color="series",
        line_dash="series",
        category_orders={"series": ["Naive cumulative bookings",
                                    "q_eff (cancel-adjusted)"]},
        color_discrete_map={"Naive cumulative bookings": "#1f77b4",
                            "q_eff (cancel-adjusted)": "#2ca02c"},
        labels={"request_index": "Request index",
                "value": "Effective rooms committed",
                "series": ""},
        title="Naive vs. cancellation-adjusted occupancy along the strip",
    )
    fig_aug.update_layout(width=760, height=340,
                          margin=dict(l=10, r=10, t=50, b=10))
    fig_aug
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The gap between the dotted and the solid line **is your overbooking
    headroom**. At any point along the strip, the naive curve says "we have
    sold this many rooms"; the solid curve says "we expect this many guests
    to actually show up". The difference grows as more risky bookings (high
    cancellation probability) enter the strip.

    This is *exactly* the data structure you need to drive Q4: at every
    request, check $q_t^{\text{eff}}$ against capacity, not the raw count.

    *Forecast caveat.* We did not implement $\widehat{D}_{\text{remaining}}$
    above — that requires an arrival-rate model conditioned on calendar
    position and current booking pace, which is **a deliberate exercise
    left to you**. Hints:

    - Empirical booking curves (Notebook 1) give you a base rate.
    - Day-of-week and month effects are visible in the data.
    - The simplest forecast: scale historical bookings-per-remaining-day
      by a seasonal multiplier.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### §8.5 · From P(cancel) to an overbooking buffer (Q2 bridge)

    The cancellation probabilities can be turned into a concrete buffer
    number using the formula from §6:
    $$n^\star = \frac{c}{1 - \bar{p}}, \qquad \text{buffer} = n^\star - c$$
    where $c$ is capacity and $\bar{p}$ is the mean predicted cancellation
    probability for the current strip. The sliders below let you set the
    capacity for this night and inspect the implied buffer; the expected
    walk-aways follow from the variance of the per-row probabilities.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    buf_capacity = mo.ui.slider(
        start=10, stop=200, value=80, step=2,
        label="Capacity for this night (rooms)",
        full_width=True, show_value=True, debounce=True,
    )
    buf_walk_tol = mo.ui.slider(
        start=0.0, stop=0.10, value=0.02, step=0.005,
        label="Tolerated walk rate (fraction of capacity)",
        full_width=True, show_value=True, debounce=True,
    )
    mo.vstack([buf_capacity, buf_walk_tol])
    return buf_capacity, buf_walk_tol


@app.cell
def _(buf_capacity, buf_walk_tol, mo, pipe, strip_full):
    feat_cols_b = list(pipe.named_steps["pre"].feature_names_in_)
    p_cancel_strip = pipe.predict_proba(strip_full[feat_cols_b])[:, 1]
    p_bar = float(p_cancel_strip.mean())
    p_var = float(p_cancel_strip.var())

    cap_b = int(buf_capacity.value)
    walk_tol = float(buf_walk_tol.value)

    # Point estimate: n* = c / (1 - p_bar). Expected check-ins from
    # selling n* bookings = n* (1 - p_bar) = c, by construction.
    n_star = int(round(cap_b / max(1 - p_bar, 1e-3)))
    buffer = n_star - cap_b

    # Variance of check-ins under independent Bernoulli model:
    # Var(checkins) = sum p_i (1 - p_i). Approximate using the strip's
    # empirical Bernoulli variance and scale by n*.
    expected_var_per_row = float((p_cancel_strip * (1 - p_cancel_strip)).mean())
    walks_sd = (n_star * expected_var_per_row) ** 0.5

    # check-ins ~ N(cap_b, walks_sd) by construction (E[checkins] = cap_b).
    # walks = max(checkins - cap_b, 0). For a normal centred at cap_b:
    #   E[walks] = walks_sd / sqrt(2*pi)  (half-normal expectation)
    #   P(walks > tol) = P(checkins > cap_b + tol) = 1 - Phi(tol / walks_sd)
    from math import erf, pi, sqrt
    tol_walks = walk_tol * cap_b
    mean_walks = walks_sd / sqrt(2 * pi)
    if walks_sd > 0:
        z = tol_walks / walks_sd
        prob_exceed = 0.5 * (1 - erf(z / sqrt(2)))
    else:
        prob_exceed = 0.0

    mo.md(
        f"**Strip-derived inputs**\n"
        f"- Mean predicted cancellation probability $\\bar{{p}}$ = **{p_bar:.3f}**\n"
        f"- Per-row variance of $\\hat{{p}}$ = **{p_var:.3f}**\n\n"
        f"**Buffer recommendation**\n\n"
        f"| Quantity | Value |\n"
        f"|---|---:|\n"
        f"| Capacity $c$ | {cap_b} rooms |\n"
        f"| Mean cancel rate $\\bar{{p}}$ | {p_bar:.1%} |\n"
        f"| Bookings to sell $n^\\star = c / (1 - \\bar{{p}})$ | **{n_star}** |\n"
        f"| **Buffer $n^\\star - c$** | **{buffer}** rooms |\n"
        f"| Expected walk-aways | {mean_walks:.1f} (sd of check-ins ≈ {walks_sd:.1f}) |\n"
        f"| Tolerated walk count (at {walk_tol:.1%} of capacity) | {tol_walks:.1f} |\n"
        f"| Approximate P(walks > tolerance) | {prob_exceed:.1%} |\n\n"
        f"*Move the capacity slider to see how the buffer scales with the room "
        f"count.* The tolerated walk rate determines how risky a buffer you "
        f"are willing to run; the probability column tells you how often this "
        f"buffer would breach the tolerance under a Poisson-binomial check-in "
        f"model. **None of these numbers are case answers** — they are what "
        f"the strip's prediction layer gives you when you ask it for an "
        f"overbooking number. Justify your final Q2 choice with the case's "
        f"walk-cost economics, not just this calculator."
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## §9 · Scikit-Learn Workflow Refresher

    The closing slide reminds you of the canonical pipeline:

    1. **Select** rows and feature columns (mind leakage).
    2. **Preprocess** — impute missing values, encode categoricals.
    3. **Split** into train and test with `train_test_split`.
    4. **Fit** a classifier or regressor.
    5. **Evaluate** with the right metric for your downstream decision.
    6. **Wrap** preprocessing + model in a `Pipeline` so you can score
       new data with one call.
    7. **Validate** with cross-validation; check for overfitting.

    Compare against the §6 cell above — every step is there. Three
    practical tips you will not regret following in your write-up:

    - **Stratify** the train/test split when classes are imbalanced.
    - **Report the metric that matches the cost structure.** For
      overbooking, *Brier score* and calibration matter more than accuracy.
    - **Document feature exclusions.** A one-paragraph "what we did not
      use, and why" is worth more than another 0.01 of AUC.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Methods Recap (across all three notebooks)

    You have now seen each method the week-2 deck introduces, executed on
    the actual Viador data:

    - empirical booking curves and joint ADR × lead-time structure *(NB1)*;
    - booking-strip extraction for a chosen stay night *(NB2)*;
    - inventory simulation with a fixed protection level, surfaced as a
      spoilage/spillage trade-off *(NB2)*;
    - the state-space index $S = \dfrac{q}{\tau\,\lambda}$ *(NB2)*;
    - a leakage-aware sklearn pipeline for cancellation probability,
      evaluated by AUC *and* calibration *(NB3)*;
    - the cancellation-adjusted occupancy $q^{\text{eff}}_t$ *(NB3)*.

    None of these *are* answers. They are the moving parts. The case
    write-up is about which moving parts you choose, how you parameterise
    them, and how you defend the choices given the data — including the
    pieces of analysis (per-hotel, per-season, channel-level) that these
    notebooks deliberately keep pooled.

    Good luck. If you get stuck, post in the channel — I will read and
    reply asynchronously.
    """)
    return


if __name__ == "__main__":
    app.run()