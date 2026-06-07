# Case: High Season Decisions at Viador Hotels

*Course: Data-Driven Decisions*  
*Topic: Revenue Management & Tactical Adaptation*

---

Clara Schwarz oversees two mid-sized hotels under the Viador brand — one located in a lively city center and the other in a popular leisure destination. During the course, you have already explored historical booking data from both of these hotels. Clara now wants to apply the insights you've gained to refine pricing and capacity decisions for the upcoming high season.

The city hotel often sees short-notice corporate and business travelers, while the resort hotel experiences strong fluctuations in demand due to holidays and weekends. Clara has access to detailed booking data for both locations, including lead time, stay duration, channel, cancellation behavior, and price. She is particularly interested in aligning her decisions across the two contexts — learning from their differences while using consistent logic.

Your job is to help Clara design and justify data-driven strategies for advance sales, overbooking rules, and demand adaptation over time. Your findings will be used to brief her revenue and operations team, so clarity of communication is key. Visualizations are encouraged to highlight important insights. And don’t forget: pricing strategies should always rely on realized bookings, not mere requests.

---

## Q1 – Calibrate Advance Sale Policies

Clara offers discounted advance rates — but only up to a certain number of bookings. Selling too many rooms in advance may lead to turning away spontaneous, higher-paying guests. Selling too few could leave occupancy short.

Using realized stays identify meaningful guest segments for each hotel. When do different types of guests book? What prices do they pay?

Estimate how many rooms Clara could allocate to advance sales for each property. Focus on lead times and potentially guest mix.

---

## Q2 – Derive Overbooking Parameters

Clara wants to implement an overbooking policy to mitigate late cancellations and no-shows. But overbooking comes with risk: walk-aways hurt guest satisfaction and brand image.

Use historical data to estimate a robust overbooking buffer for each hotel. How many extra bookings can Clara accept without incurring frequent walk-aways?

Explain any trade-offs and assumptions you make.

---

## Q3 – Seasonally Update Your Rules

Demand patterns change throughout the year, and Clara would like to **adapt** her advance sale and overbooking thresholds to seasonality.

Using appropriate aggregates from past years, identify periods of stronger or weaker demand for each hotel. How would your advance booking limits and overbooking buffers from Q1 and Q2 change across these periods?

Provide updated rules for both city and resort hotels and justify your logic using observed patterns.

---

## Q4 – Operate on Live Booking Data

Imagine Clara gives you access to a daily stream of individual bookings coming in for the next season. For each booking, you see the date, stay period, channel, price.

Design a **monitoring system** that updates Clara’s advance sale ceilings and overbooking parameters as real bookings accumulate. What metrics would you track? How would you flag when adjustments are needed?

Use a small mock dataset or simulated example to demonstrate how this dynamic process could work. Highlight how **static rules** (from Q1 and Q2) evolve when applied in real time.

---

Remember: A good recommendation is transparent, robust to noise, and clearly communicated. Visuals can go a long way in making your insights actionable. And if you ever wonder whether a pricing rule is fair — put yourself in the shoes of a guest.
