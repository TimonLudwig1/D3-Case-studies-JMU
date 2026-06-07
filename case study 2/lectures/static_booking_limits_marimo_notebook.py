import marimo

__generated_with = "0.23.9"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _():
    import marimo as mo
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.stats import poisson
    from pulp import LpMaximize, LpProblem, LpVariable, lpSum, value
    from scipy.optimize import linprog
    from scipy.optimize import Bounds, LinearConstraint, milp
    import pulp

    # UI elements for parameters
    total_rooms_slider = mo.ui.slider(
        start=50, stop=200, value=118, label="Total Rooms", full_width=True
    )
    low_fare_slider = mo.ui.slider(
        start=50, stop=300, value=159, label="Low Fare", full_width=True
    )
    high_fare_slider = mo.ui.slider(
        start=100, stop=400, value=225, label="High Fare", full_width=True
    )
    mean_demand_high_fare_slider = mo.ui.slider(
        start=1, stop=50, value=27.3, label="Mean Demand High Fare", full_width=True
    )
    return (
        LpMaximize,
        LpProblem,
        LpVariable,
        high_fare_slider,
        linprog,
        low_fare_slider,
        lpSum,
        mean_demand_high_fare_slider,
        mo,
        np,
        poisson,
        pulp,
        total_rooms_slider,
    )


@app.cell(hide_code=True)
def _(np, pulp):
    def pulp_to_scipy_linprog(prob):
        variables = prob.variables()

        # Objective coefficients
        constant_offset = prob.objective.constant

        objective = np.array([prob.objective.get(v,0) for v in variables])
        if prob.sense == pulp.LpMaximize:
            c = -objective
            constant_offset = -constant_offset  # important: flip the sign of constant too!
        else:
            c = objective

        # Constraints
        A_ub, b_ub, A_eq, b_eq = [], [], [], []
        for constraint in prob.constraints.values():
            coeffs = [constraint.get(v,0) if constraint.get(v,0) is not None else 0 for v in variables]
            const_term = constraint.constant

            if constraint.sense == pulp.LpConstraintLE:
                A_ub.append(coeffs)
                b_ub.append(-const_term)
            elif constraint.sense == pulp.LpConstraintGE:
                A_ub.append([-coef for coef in coeffs])
                b_ub.append(const_term)
            elif constraint.sense == pulp.LpConstraintEQ:
                A_eq.append(coeffs)
                b_eq.append(-const_term)

        # Bounds
        bounds = []
        for v in variables:
            lb = v.lowBound if v.lowBound is not None else -np.inf
            ub = v.upBound if v.upBound is not None else np.inf
            bounds.append((lb, ub))

        # Integrality
        integrality = np.array([1 if v.cat == pulp.LpInteger or v.cat == pulp.LpBinary else 0 for v in variables])

        return {
            'c': c,
            'A_ub': np.array(A_ub) if A_ub else None,
            'b_ub': np.array(b_ub) if b_ub else None,
            'A_eq': np.array(A_eq) if A_eq else None,
            'b_eq': np.array(b_eq) if b_eq else None,
            'bounds': bounds,
            'integrality': integrality,  # key addition
            'variable_names': [v.name for v in variables],
            'constraint_names': [con.name for con in prob.constraints.values()],
            'objective_sense': 'max' if prob.sense == pulp.LpMaximize else 'min',
            'problem_name': prob.name,
            'constant_offset': constant_offset
        }

    return (pulp_to_scipy_linprog,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Revenue Management: Static Booking Limits

    This notebook demonstrates a static booking limit model for revenue management, specifically for a hotel with two fare classes: high and low. The goal is to determine the optimal number of rooms to protect for the high fare class in order to maximize expected revenue.

    ## Model Description

    The model considers a hotel with a fixed number of rooms (`total_rooms`).  Rooms can be sold at either a high fare (`high_fare`) or a low fare (`low_fare`).  The demand for high fare rooms is assumed to follow a Poisson distribution with a mean of `mean_demand_high_fare`.

    The core idea is to protect a certain number of rooms (`protected_high_fare_rooms`) for potential high-fare customers.  If the actual demand for high-fare rooms exceeds this protection level, the excess demand is turned away (or, in this simplified model, assumed to be accommodated at the low fare).

    The booking limit for the low fare class is then implicitly determined as `total_rooms - protected_high_fare_rooms`.  This means we will sell no more than `total_rooms - protected_high_fare_rooms` low fare tickets because we are protecting `protected_high_fare_rooms` seats for high fare passengers.

    ## Parameters

    *   `total_rooms`: The total number of rooms available in the hotel.
    *   `low_fare`: The price of a low-fare room.
    *   `high_fare`: The price of a high-fare room.
    *   `mean_demand_high_fare`: The average demand for high-fare rooms (lambda parameter of the Poisson distribution).
    *   `protected_high_fare_rooms`: The number of rooms to protect for high-fare customers.  This is the decision variable we want to optimize.

    ## Expected Revenue Calculation

    The expected revenue is calculated by considering all possible demand scenarios for high-fare rooms (from 0 to `total_rooms`). For each scenario, we calculate the probability of that demand occurring (using the Poisson distribution) and the resulting revenue. The revenue is calculated as follows:

    *   If the high-fare demand is less than or equal to `protected_high_fare_rooms`, then all high-fare demand is accommodated at the high fare, and the remaining rooms are sold at the low fare.
    *   If the high-fare demand exceeds `protected_high_fare_rooms`, then only `protected_high_fare_rooms` are sold at the high fare, and the rest of the rooms are sold at the low fare.

    The expected revenue is the sum of the revenue for each scenario, weighted by the probability of that scenario.

    ## Optimization

    The goal is to find the value of `protected_high_fare_rooms` that maximizes the expected revenue.  This is done by calculating the expected revenue for all possible values of `protected_high_fare_rooms` (from 0 to `total_rooms`) and selecting the value that yields the highest expected revenue.

    ## Interactive Exploration

    Use the sliders below to adjust the parameters of the model and observe the impact on the expected revenue and the optimal number of protected high-fare rooms.
    """)
    return


@app.cell(hide_code=True)
def _(
    optimal_protected_high_fare_rooms,
    protected_high_fare_rooms_range,
    revenues,
    total_rooms_slider,
):
    import altair as alt
    import pandas as pd

    # Create a dataframe for Altair
    df = pd.DataFrame({
        'protected_high_fare_rooms': protected_high_fare_rooms_range,
        'expected_revenue': revenues
    })

    # Create the Altair plot
    chart = alt.Chart(df).mark_line(point=True).encode(
        x=alt.X('protected_high_fare_rooms', title='Number of High Fare Rooms Protected', scale=alt.Scale(domain=[0, total_rooms_slider.value])),
        y=alt.Y('expected_revenue', title='Expected Revenue ($)'),
        tooltip=['protected_high_fare_rooms', 'expected_revenue']
    ).properties(
        title='Effect of Protected High Fare Rooms on Expected Revenue'
    )

    # Create a dataframe for the rule
    rule_df = pd.DataFrame({'x': [optimal_protected_high_fare_rooms]})

    # Add a rule to highlight the optimal point
    rule = alt.Chart(rule_df).mark_rule(color='red').encode(
        x=alt.X('x:Q', title='Number of High Fare Rooms Protected')
    )


    # Combine the line chart and the rule
    final_chart = (chart + rule)
    return (final_chart,)


@app.cell
def _(
    final_chart,
    high_fare_slider,
    low_fare_slider,
    mean_demand_high_fare_slider,
    mo,
    total_rooms_slider,
):
    mo.hstack([
    mo.vstack(
        [
            total_rooms_slider,
            low_fare_slider,
            high_fare_slider,
            mean_demand_high_fare_slider,

        ]),
    final_chart],
              widths=[0.5,0.5])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Full Enumeration
    """)
    return


@app.cell
def _(
    high_fare_slider,
    low_fare_slider,
    mean_demand_high_fare_slider,
    np,
    poisson,
    total_rooms_slider,
):
    # Revenue function (using UI element values)
    def expected_revenue(protected_high_fare_rooms):
        low_fare_rooms = total_rooms_slider.value - protected_high_fare_rooms
        revenue = 0
        for d in range(
            total_rooms_slider.value + 1
        ):  # Potential high fare demand scenarios
            prob = poisson.pmf(d, mean_demand_high_fare_slider.value)
            if d <= protected_high_fare_rooms:
                # All high fare demand can be accommodated
                revenue += prob * (
                    d * high_fare_slider.value + low_fare_rooms * low_fare_slider.value
                )
            else:
                # Only protect_high_fare_rooms can be accommodated, rest get low fare
                revenue += prob * (
                    protected_high_fare_rooms * high_fare_slider.value
                    + (total_rooms_slider.value - protected_high_fare_rooms)
                    * low_fare_slider.value
                )
        return revenue


    # Optimization and plotting data
    protected_high_fare_rooms_range = range(total_rooms_slider.value + 1)
    revenues = [expected_revenue(ph) for ph in protected_high_fare_rooms_range]

    optimal_protected_high_fare_rooms = np.argmax(revenues)
    max_revenue = revenues[optimal_protected_high_fare_rooms]
    return (
        max_revenue,
        optimal_protected_high_fare_rooms,
        protected_high_fare_rooms_range,
        revenues,
    )


@app.cell(hide_code=True)
def _(max_revenue, mo, optimal_protected_high_fare_rooms):
    mo.md(
        rf"""
    Optimal number of high fare rooms to protect: {optimal_protected_high_fare_rooms}

    Maximum expected revenue: {max_revenue:.2f}
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Linear Program
    """)
    return


@app.cell
def _(
    LpMaximize,
    LpProblem,
    LpVariable,
    high_fare_slider,
    low_fare_slider,
    lpSum,
    mean_demand_high_fare_slider,
    poisson,
    total_rooms_slider,
):
    # Poisson probabilities
    probabilities = [poisson.pmf(d, mean_demand_high_fare_slider.value) for d in range(total_rooms_slider.value + 1)]

    # Define the LP problem
    problem = LpProblem("Maximize_Expected_Revenue", LpMaximize)

    # Decision variables
    protect = LpVariable("protect", 0, total_rooms_slider.value, cat="Integer")
    sales_high = LpVariable.dicts("sales_high", range(total_rooms_slider.value + 1), 0, total_rooms_slider.value, cat="Continuous")

    # Objective function
    revenue_terms = []

    for d in range(total_rooms_slider.value + 1):
        revenue_terms.append(probabilities[d] * (sales_high[d] * high_fare_slider.value + (total_rooms_slider.value - protect) * low_fare_slider.value))

    problem += lpSum(revenue_terms), "Expected_Revenue"

    # Constraints
    for d in range(total_rooms_slider.value + 1):
        problem += sales_high[d] <= protect, f"High_Fare_Sales_Limit_{d}"
        problem += sales_high[d] <= d, f"Demand_Limit_{d}"
    return (problem,)


@app.cell(hide_code=True)
def _(linprog, problem, pulp_to_scipy_linprog):
    linprog_data = pulp_to_scipy_linprog(problem)

    res = linprog(
        c=linprog_data['c'],
        A_ub=linprog_data['A_ub'],
        b_ub=linprog_data['b_ub'],
        A_eq=linprog_data['A_eq'],
        b_eq=linprog_data['b_eq'],
        bounds=linprog_data['bounds'],
        integrality=linprog_data['integrality'],
        method='highs'
    )

    rev = -res.fun - linprog_data['constant_offset']
    dec = res.x[0]
    return dec, rev


@app.cell
def _(dec, mo, rev):
    mo.hstack([
        mo.stat(round(rev,2), bordered=True, label="Revenue"),
        mo.stat(dec, bordered=True, label="Protected High Fare Rooms")],justify="start")
    return


if __name__ == "__main__":
    app.run()