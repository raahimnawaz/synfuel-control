# The chemistry behind the reactor model

This document is the "study the hydrocarbons" deliverable. It explains the chemistry the
Phase 1 simulator actually models, why thermal runaway happens, and — critically —
**where every physical constant comes from**. The goal is that someone who knows
Fischer-Tropsch (FT) chemistry can read this and find nothing invented.

## 1. What we're making and from what

We feed **CO₂ + H₂** and make **liquid hydrocarbons**. This is the *CO₂-based*
Fischer-Tropsch route (power-to-liquid / e-fuels): renewable H₂ plus captured CO₂ →
synthetic fuel. It proceeds in two coupled steps:

**Step 1 — reverse water-gas-shift (RWGS).** CO₂ is not hydrogenated to fuel directly;
it is first reduced to CO:

$$\text{CO}_2 + \text{H}_2 \rightleftharpoons \text{CO} + \text{H}_2\text{O}
\qquad \Delta H^\circ_{298} = +41.2~\text{kJ/mol}$$

Mildly **endothermic**, and equilibrium-limited at low temperature — at room temperature
almost no CO forms. RWGS is endothermic, so its equilibrium constant *rises* with
temperature (van't Hoff), which is why higher T produces more CO to feed Step 2.

**Step 2 — Fischer-Tropsch chain growth.** CO + H₂ polymerise on the catalyst surface
into hydrocarbon chains. Per `-CH₂-` unit added:

$$\text{CO} + 2\,\text{H}_2 \rightarrow (\text{-CH}_2\text{-}) + \text{H}_2\text{O}
\qquad \Delta H \approx -165~\text{kJ/mol CO}$$

Strongly **exothermic**. This single number is the source of every control headache in
the project: more reaction → more heat → (via Arrhenius) more reaction. (Full paraffin
stoichiometry: $n\,\text{CO} + (2n{+}1)\,\text{H}_2 \rightarrow \text{C}_n\text{H}_{2n+2}
+ n\,\text{H}_2\text{O}$.)

## 2. Anderson-Schulz-Flory: how the carbon-number spread works

FT does not make one molecule — it makes a *distribution* of chain lengths, set by a
single parameter: the **chain-growth probability α**. At each surface step a growing
chain either adds another carbon (probability α) or terminates and desorbs (1−α). This
is a geometric process, giving the **Anderson-Schulz-Flory (ASF)** distribution.

Mole fraction of chains of length *n*:

$$x_n = (1-\alpha)\,\alpha^{\,n-1}$$

Mass (weight) fraction — what matters for fuel yield, since heavier chains carry more mass:

$$w_n = n\,(1-\alpha)^2\,\alpha^{\,n-1}$$

The **C5+ fraction** (gasoline/diesel-range liquid fuel, the valuable product) is
everything heavier than C4:

$$f_{C5+} = 1 - \sum_{n=1}^{4} n\,(1-\alpha)^2\,\alpha^{\,n-1}$$

This is implemented in [`reactor.py`](reactor.py) as `asf_weight_fraction` and
`c5plus_fraction`. With α = 0.90 it gives **f_C5+ ≈ 0.92** — high α strongly favours
liquids. *Higher α ⇒ more liquid fuel.* And α **falls as temperature rises**, which sets
up the central trade-off below.

## 3. Why it runs away (and why that's a control problem)

The FT rate follows Arrhenius kinetics, $k = A\,e^{-E_a/RT}$, with a large activation
energy (~100 kJ/mol). Combine that exponential temperature sensitivity with the −165
kJ/mol exothermicity and you get a positive feedback loop:

> hotter bed → exponentially faster FT → more heat released → hotter bed …

The jacket coolant removes heat *linearly* in temperature. As long as cooling can keep
up, the reactor sits on a stable, cool, high-selectivity branch (~227 °C in our model).
But if heat generation outpaces removal — e.g. a **cooling failure** (the jacket
heat-transfer coefficient drops) — the bed ignites and jumps to a hot branch (>320 °C in
our model) where the cobalt catalyst sinters and methane selectivity spikes. That jump is
the **thermal runaway** the downstream controller exists to prevent. See it directly by
running `python -m sim.run_sim`.

## 4. The central trade-off the controller must manage

| Push temperature **up** | Push temperature **down** |
|---|---|
| ✅ faster kinetics, higher CO₂ conversion | ✅ higher α → more C5+ liquid (better selectivity) |
| ✅ RWGS equilibrium shifts toward CO | ✅ safe, far from runaway, catalyst lasts |
| ❌ α drops → more useless CH₄ | ❌ low conversion, low throughput |
| ❌ approaches runaway, catalyst sinters | |

Maximising **C5+ yield = conversion × C5+ selectivity** subject to **T < runaway limit**
is a constrained optimisation in temperature (and feed ratio, pressure, catalyst) — i.e.
exactly the control problem of Phases 3–4.

## 5. Source of every constant

Defined in [`params.py`](params.py). **Literature** values govern the dynamics; only the
two **tuned** pre-exponentials and the lumped reactor transport/geometry are fitted —
and those are genuinely reactor-specific quantities that are always fitted per unit.

| Symbol | Value | Status | Source |
|---|---|---|---|
| ΔH (FT, per CO) | −165 kJ/mol | literature | Dry (2002), *Catal. Today* 71:227 |
| ΔH (RWGS) | +41.2 kJ/mol | literature | NIST WebBook (standard formation enthalpies) |
| Eₐ (FT) | 100 kJ/mol | literature | Yates & Satterfield (1991); van der Laan & Beenackers (1999) |
| Eₐ (RWGS) | 70 kJ/mol | literature | van der Laan & Beenackers (1999) |
| α (ASF) | 0.90 | literature | Dry (2002); typical low-T cobalt FT |
| A_FT, A_RWGS | fitted | **tuned** | calibrated to give ~33% conversion at ~227 °C, 25 bar (in the documented cobalt LTFT window) |
| ρ·cp, UA/V, geometry | — | **illustrative** | lumped CSTR transport values chosen for realistic time constants |

Operating window (nominal): **227 °C, 25 bar, H₂:CO₂ = 3** — inside the standard
low-temperature cobalt FT range (200–240 °C, 20–30 bar).

## References

- M. E. Dry, "The Fischer-Tropsch process: 1950–2000," *Catalysis Today* **71** (2002) 227–241.
- G. P. van der Laan & A. A. C. M. Beenackers, "Kinetics and Selectivity of the
  Fischer-Tropsch Synthesis: A Literature Review," *Catalysis Reviews* **41** (1999) 255–318.
- I. C. Yates & C. N. Satterfield, "Intrinsic kinetics of the Fischer-Tropsch synthesis
  on a cobalt catalyst," *Energy & Fuels* **5** (1991) 168–173.
- NIST Chemistry WebBook, standard reference data (RWGS enthalpy at 298 K).
