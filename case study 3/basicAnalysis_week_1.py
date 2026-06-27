import marimo

__generated_with = "0.23.9"
app = marimo.App()


@app.cell(hide_code=True)
def _():
    import folium
    import numpy as np
    import marimo as mo
    import pandas as pd
    import seaborn as sns
    import pulp as pl
    import highspy

    return folium, mo, pd, pl, sns


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # CashLog Basic Analysis

    We will use this notebook to implement a basic version of CashLog's decision problem by performing the following steps:

    1. Define and load relevent model parameters
    2. Define and initialize the decision variables
    3. Define and implement the objective function
    4. Define and implement the relevant constraints
    5. Solve the problem and anlyse the results
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### The decision problem can be modeled in the sense of a MIP
    """)
    return


@app.cell
def _(pl):
    prob = pl.LpProblem('CashLog_BasicAnalysis', pl.LpMinimize) # using pulp to solve the optimization problem, braucht name und die Art des Problems (Minimierung oder Maximierung)
    return (prob,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Define and load model parameters

    $W:$ Set of warehouses<br>
    $R:$ Set of customer regions<br>
    $S:$ Set of links between warehouses and regions<br>

    $f_i:$ Fixed costs of warehouse $i$<br>
    $c_{ij}:$ Costs if region $j$ is served by warehouse $i$<br>
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    test
    """)
    return


@app.cell
def _(mo, pd):
    warehouses = pd.read_csv('https://raw.githubusercontent.com/D3IP-SS25/data-driven-scm-dataset/refs/heads/main/data/warehouses.csv', index_col='warehouseID')
    W = warehouses.index.values

    regions = pd.read_csv('https://raw.githubusercontent.com/D3IP-SS25/data-driven-scm-dataset/refs/heads/main/data/regions.csv', index_col='regionID')
    R = regions.index.values

    shifts = pd.read_csv('https://raw.githubusercontent.com/D3IP-SS25/data-driven-scm-dataset/refs/heads/main/data/shifts.csv', index_col=['warehouseID', 'regionID'])
    S = shifts.index.values

    # Create interactive tabs to display the dataframes
    tab1 = mo.ui.table(warehouses, label="Warehouses", selection=None, show_column_summaries=False, show_download=False, page_size= 10)
    tab2 = mo.ui.table(regions, label="Regions", selection=None, show_column_summaries=False, show_download=False)
    tab3 = mo.ui.table(shifts, label="Shifts", selection=None, show_column_summaries=False, show_download=False)

    # Create tabbed interface
    tabs = mo.ui.tabs({
        "Warehouses": tab1,
        "Regions": tab2,
        "Shifts": tab3
    })

    tabs
    return R, S, W, regions, shifts, warehouses


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Define and initialize the decision variables

    $x_{ij}:$ Binary variable indicating if region $j$ is served by warehouse $i$<br>
    $y_{i}:$ Binary variable indicating if warehouse $i$ is opened<br>
    """)
    return


@app.cell
def _(S, W, pl):
    x = pl.LpVariable.dicts(name='x', indices=S, cat=pl.LpBinary)
    y = pl.LpVariable.dicts(name='y', indices=W, cat=pl.LpBinary)
    return x, y


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Define and implement the objective function

    We want to minimize the total network costs (variable costs + fixed costs):

    $$\min \sum_{i\in W}\sum_{j\in R} x_{ij} c_{ij} + \sum_{i\in W} f_i y_i$$
    """)
    return


@app.cell
def _(S, W, pl, prob, shifts, warehouses, x, y):
    variableCosts = pl.lpSum([x[i,j] * shifts.loc[i,j].transportationCosts for i,j in S]) 
    fixedCosts = pl.lpSum([y[i] * warehouses.loc[i].fixedCosts for i in W])

    prob.setObjective(variableCosts + fixedCosts)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Define and implement the relevant constraints

    Regions can only be served by open warehouses:<br>
    $$x_{ij} \leq y_{i} \quad \forall i\in W, j\in R$$

    Each region has to be served by exactly one warehouse:<br>
    $$\sum_{i\in W} x_{ij} = 1 \quad \forall j\in R$$
    """)
    return


@app.cell
def _(R, W, pl, prob, x, y):
    for i in W:
        for j in R:
            prob.addConstraint(x[i,j] <= y[i])

    for j in R:
        prob.addConstraint(pl.lpSum([x[i,j] for i in W]) == 1)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Solve the problem and analyze the results

    Next, we use a solver to determine an optimal solution for the optimization problem.
    """)
    return


@app.cell
def _(mo):
    solve_button = mo.ui.run_button(label='Solve')
    show_details = mo.ui.switch(label="Show details")
    solve_button
    return show_details, solve_button


@app.cell
def _(S, W, mo, pl, prob, shifts, solve_button, warehouses, x, y):
    mo.stop(not solve_button.value)
    with mo.status.spinner(title='Optimizing...') as spinner:
        solver = pl.HiGHS(msg=True)
        probSolved = prob.solve(solver)

        fixedCostsOpt = sum([y[i].varValue * warehouses.loc[i].fixedCosts for i in W])
        variableCostsOpt = sum([x[i,j].varValue * shifts.loc[i,j].transportationCosts for i,j in S])
    return fixedCostsOpt, variableCostsOpt


@app.cell
def _(
    W,
    fixedCostsOpt,
    mo,
    prob,
    show_details,
    variableCostsOpt,
    warehouses,
    y,
):
    cost_output = mo.md(f"""
    **Minimal total cost**: {prob.objective.value():0,.2f} Euro

    **Variable costs**: {variableCostsOpt:0,.0f} € <br>
    **Fixed costs**:   {fixedCostsOpt:0,.0f} €
    """)

    closed_warehouses = [
        warehouses.loc[idx].city
        for idx in W
        if getattr(y[idx], "varValue", 0.0) <= 0.1
    ]

    opened_warehouses = [
        warehouses.loc[idx].city
        for idx in W
        if getattr(y[idx], "varValue", 0.0) >= 0.9
    ]


    closed_md = "\n".join(f"- {warehouse}\n" for warehouse in closed_warehouses)
    opened_md = "\n".join(f"- {warehouse}\n" for warehouse in opened_warehouses)
    closed_output = mo.md(f"""
    **🏭 Warehouses to close:** 

    {closed_md}
    """)

    opened_output = mo.md(f"""
    **🏭 Warehouses to keep:** 

    {opened_md}
    """)

    warehouse_panel = mo.hstack([closed_output, opened_output], widths=(0.8, 1))

    mo.vstack([
        cost_output,
        show_details,
        warehouse_panel if show_details.value else ""  # nothing rendered when off
    ])
    return


@app.cell
def _(S, W, folium, mo, pd, regions, sns, solve_button, warehouses, x, y):
    mo.stop(not solve_button.value)

    region_results = []

    for w, r in S:
        v = x[w, r].varValue
        if v >= 0.1:
            region_results.append(
                {
                    'regionID': r,
                    'warehouseID': w,
                    'serviced': v,
                    'zipCode': regions.loc[r].zipCode,
                    'lat': regions.loc[r].lat,
                    'lon': regions.loc[r].lon,
                    'city': regions.loc[r].city,
                }
            )

    warehouse_results = [
        {
            'warehouseID': w,
            'city': warehouses.loc[w].city,
            'open': y[w].varValue,
            'lat': warehouses.loc[w].lat,
            'lon': warehouses.loc[w].lon,
        }
        for w in W
    ]

    plot_df_regions = pd.DataFrame(region_results)
    plot_df_warehouses = pd.DataFrame(warehouse_results)
    plot_df_warehouses = plot_df_warehouses[plot_df_warehouses.open == 1]

    # Define a color palette with a fixed number of colors
    num_warehouses = len(W)
    palette = sns.color_palette('tab20', n_colors=num_warehouses).as_hex()
    color_map = {W[i]: palette[i % 20] for i in range(num_warehouses)}

    # Create the map
    m = folium.Map(location=[41, -4], zoom_start=6)

    # Add regions (customers) to the map
    for index, row in plot_df_regions.iterrows():
        color = color_map[row['warehouseID']]
        folium.CircleMarker(
            location=[row['lat'], row['lon']],
            radius=4,  # pixels, not meters
            color=color,
            fill=True,
            fill_opacity=0.8,
            weight=0.5,
            popup=row['city']
        ).add_to(m)

    # Add warehouses to the map with a warehouse icon
    for index, row in plot_df_warehouses.iterrows():
        color = color_map[row['warehouseID']]  # Get fixed color for the warehouse
        folium.Marker(
            location=[row['lat'], row['lon']],
            popup=row['city'],
            icon=folium.DivIcon(html='<div style="font-size:30px;">🏦</div>'),
        ).add_to(m)

    m
    return (plot_df_warehouses,)


@app.cell(hide_code=True)
def _(plot_df_warehouses):
    plot_df_warehouses
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
