# Dynamic Pricing in Hotel Revenue Management
## Key Insights & Logic

---

## 1. The Core Problem

Hotels need to decide: **how much to charge for a room, and when to stop selling cheap rooms?**

The challenge is that two types of customers exist, but you cannot identify them individually:
- **Low-fare customers** — price-sensitive, plan ahead, book early to get a deal
- **High-fare customers** — price-insensitive, book late, willing to pay whatever the market demands

Since customers just see a price on a website and decide to book or not, the hotel cannot give individual prices. Instead, it must decide *how many rooms to offer at what price* at any given point in time.

---

## 2. Building the Fare Class Signal from Data

### Why Raw ADR is Unusable

ADR (Average Daily Rate) is contaminated by other variables:
- Room type (suite vs. standard)
- Party size (1 adult vs. 2 adults)
- Meal plan (room-only vs. half-board)

A high ADR might just mean a suite was booked, not that the customer paid a premium fare.

### The Fix: Canonical Bookings

Filter down to one consistent booking type — e.g. **Room A + Bed & Breakfast + 1 adult** — so that ADR becomes a clean, comparable price signal across bookings.

### Why Filter by Season Too

Pooling all months hides the seasonal structure. Resort ADR swings from ~€35 in January to ~€60 in August. A threshold calibrated on all-year data would be meaningless for summer pricing decisions.

---

## 3. Finding the Lead-Time Cutoff

### The Hypothesis

Early bookers are price-sensitive → they represent **low-fare class**.  
Late bookers don't care about price → they represent **high-fare class**.

### How to Find the Cutoff

Split bookings into early (lead ≥ cut) and late (lead < cut) groups and compare median ADR. The cutoff is valid where the **ADR gap between groups is largest and most stable**.

### Key Finding: The Signal is Conditional

The low/high fare signal only emerges clearly under specific conditions:
- **Resort Hotel** (not City — business travellers book at stable rates regardless of lead time)
- **Summer months** (not all year — seasonal leisure demand drives the pattern)
- **Long cutoff** (~150+ days) — pushes the early group far enough out that they are genuinely price-planning

### Why 2-Adult Bookings Show a Stronger Signal

Couples and families:
- Are **forced** into summer by school holidays
- Pay for two people, so total cost is high → strong motivation to book early
- Show a ~€44 ADR gap (€85 early vs. €129 late) vs. ~€30 for solo bookings

### The Two-Way Relationship

The ADR pattern is not purely customer behaviour — it also reflects the **hotel's own pricing strategy**. The hotel sets lower early rates to stimulate early demand, then raises prices as arrival approaches. Customers respond to those prices. Each side is reacting to the other.

---

## 4. The State Index S

### The Intuition

At any point, the hotel needs to know: *how urgently do I need to protect rooms for high-fare customers?*

Two things determine this together:
- How many rooms are still available?
- How much time (and demand) is left?

### The Basic Formula

$$S = \frac{\text{Rooms left}}{\text{Remaining days} \times \text{Daily demand}}$$

| S Value | Situation | Action |
|---------|-----------|--------|
| S > 1 | Oversupplied | Keep selling low-fare — spoilage risk |
| S ≈ 1 | Balanced | Monitor closely |
| S < 1 | Undersupplied | Close low-fare — spillage risk |

### Important: S is Not a Fixed Day Cutoff

The switch from "sell low-fare" to "protect for high-fare" is **not** triggered by a fixed number of days before arrival. It triggers when S drops below your threshold — which could happen:
- 60 days out in a hot summer
- 4 days out in a slow period

Time alone doesn't trigger the switch. It is the **combination of remaining inventory and remaining demand** that matters.

---

## 5. Spoilage vs. Spillage

These are the two failure modes, pulling in opposite directions:

- **Spoilage** — a room goes unsold because you were too selective. Revenue = €0.
- **Spillage** — you turned away a high-fare customer because you already sold the room cheaply.

The entire pricing logic is about balancing these two risks using S as the guide.

---

## 6. Connecting the Cutoff to S: The Missing Link

The basic S uses total daily demand. But what you actually care about is **high-fare demand specifically** — because S is meant to tell you whether to protect rooms *for high-fare customers*.

### The Smarter S

$$S = \frac{\text{Rooms left}}{\text{Forecasted remaining HIGH-FARE demand}}$$

**This is where the cutoff feeds directly into S:**

- Your EDA found that high-fare Resort summer customers typically book within the cutoff window (e.g. 10–154 days before arrival)
- So "remaining high-fare demand" = all bookings still expected to arrive **after the cutoff point**
- If you are 160 days out, almost all high-fare demand is still incoming
- If you are 20 days out, most high-fare demand has already arrived

Without the fare class cutoff, S is just a blunt inventory counter. With it, S becomes a genuinely smart tool that knows *who* is still coming.

---

## 7. The Prediction-Augmented State Space

### Effective Inventory: Accounting for Cancellations

Not every booking that exists will actually show up. A booking with a 40% cancellation probability should only count as 0.6 of a room:

$$q_t^{eff} = \sum_i q_i (1 - P(\text{cancel}_i))$$

Each booking gets its **own individual cancellation probability** — not a flat rate — based on its specific features.

### Predicted Booking Pressure

The full, smart version of S:

$$\text{Booking Pressure} = \frac{q_t^{eff}}{\hat{D}_{remaining}}$$

Where:
- **Numerator** $q_t^{eff}$ = effective inventory already sold (cancellation-adjusted)
- **Denominator** $\hat{D}_{remaining}$ = forecasted remaining HIGH-FARE arrivals from today until arrival day

High pressure → close low-fare, raise prices  
Low pressure → open low-fare, avoid spoilage

---

## 8. The ML Models and What They Do

### Cancellation Classifier

**Problem:** Will this booking cancel? (yes/no → binary classification)

**Features:** lead time, channel, deposit type, party size, season, room type, previous cancellations...

**Output:** Individual $P(\text{cancel}_i)$ for each booking → feeds into $q_t^{eff}$

**Models to try:** LogisticRegression, RandomForestClassifier, XGBClassifier

### ADR Regressor

**Problem:** What price will this booking likely pay? (continuous → regression)

**Features:** lead time, channel, party size, room type, season, current booking pressure S...

**Output:** Predicted ADR for this specific booking → compared against price floor

**Decision rule:**
- Predicted ADR ≥ price floor → accept
- Predicted ADR < price floor → reject or counter-offer

### Why Individual Probabilities Matter

Using a flat cancellation rate for everyone would:
- Underestimate cancellations for risky bookings (far out, OTA, no deposit)
- Overestimate cancellations for solid bookings (close in, direct, prepaid)

The ML classifier gives every booking its own probability based on its unique feature combination — that is what makes the effective inventory calculation accurate.

---

## 9. The Full Integrated Workflow

```
Historical Data
      ↓
Clean ADR (canonical bookings) + Filter by season
      ↓
Find lead-time cutoff (where ADR gap is largest)
      ↓
Label bookings: low-fare vs. high-fare
      ↓
Train cancellation classifier → P(cancel_i) per booking
Train ADR regressor → predicted ADR per booking
Build demand forecast → D_remaining (high-fare only, after cutoff)
      ↓
── LIVE OPERATION (per future night, updated daily) ──
      ↓
Calculate q_t^eff = Σ q_i(1 - P(cancel_i))
Calculate Booking Pressure = q_t^eff / D_remaining
      ↓
New booking arrives with its features
      ↓
Predict its ADR and cancellation probability
Compare to pressure-driven price floor
      ↓
Accept → update q_t^eff → recalculate pressure → update price
Reject → hold room for higher-fare incoming demand
      ↓
Repeat every day for every future night independently
```

---

## 10. Key Takeaways

1. **Fare class labels are constructed, not observed** — you derive them from ADR + lead time analysis on clean canonical data.

2. **The cutoff is not arbitrary** — it is empirically grounded in where the ADR gap is largest. For Resort summer 2-adult bookings, this was ~154 days.

3. **S only makes sense with fare-class-aware demand** — using total demand in the denominator ignores *who* is still coming.

4. **The cutoff defines D_remaining** — bookings expected after the cutoff = high-fare demand = the denominator of booking pressure.

5. **Cancellation classification makes inventory real** — raw room counts overstate true inventory; $q_t^{eff}$ corrects for expected no-shows.

6. **City and Resort need different models** — City Hotel has stable business-travel demand with no strong lead-time signal. Resort has seasonal leisure demand with a clear early/late fare class pattern.

7. **The two risks always trade off** — every pricing decision is a bet between spoilage (too selective) and spillage (too generous). Booking pressure is the instrument for making that bet rationally.
