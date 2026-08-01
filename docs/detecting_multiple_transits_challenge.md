# Case Study: Detecting Multiple Transits

## The problem

A light curve represents a **star**, not one specific planet. A star such as
Kepler-186 can therefore produce several transit patterns in the same flux
series:

| Planet | Period | Signal in the shared light curve |
| --- | ---: | --- |
| Kepler-186b | 3.89 days | Frequent, shallow dips |
| Kepler-186c | 7.27 days | A second repeating pattern |
| Kepler-186f | 129.94 days | Rare, long-period dips |

Our current pipeline finds only the strongest BLS period. It then describes
that same signal regardless of which planet name was originally used to fetch
the star's light curve. This is why the old Kepler-186 extracted files were
identical.

```text
Current result

One stellar light curve ──► strongest period ──► one candidate row

Required result

One stellar light curve ──► period A ─┬─► candidate A
                                     ├─► candidate B
                                     └─► candidate C
```

The challenge is not just finding several high periodogram peaks. A strong
signal and its aliases can hide weaker planets, so each accepted signal must be
removed from the next search.

## How to deal with it

### Iterative search

The basic approach is **detect, mask, and search again**:

```text
Prepare light curve
       │
       ▼
Find strongest credible transit
       │
       ▼
Mask its in-transit cadences
       │
       ▼
Search the remaining cadences
       │
       ├── another signal ──► repeat
       │
       └── no credible signal ──► finish
```

The original measurements should remain available. The mask only prevents an
already detected signal from influencing the next BLS search.

### Masking versus subtraction

| Strategy | How it works | Main trade-off |
| --- | --- | --- |
| **Masking** | Ignore predicted in-transit points during the next search | Robust, but removes some searchable data |
| **Model subtraction** | Fit a transit and remove its model from the flux | Preserves data, but fitting errors create residual artifacts |

Masking is the safer first version for this project. Model subtraction may be
useful later, especially for overlapping transits, but it requires a reliable
transit model.

### How Kepler handled it

Kepler's Transiting Planet Search (TPS) combined repeated events into a
Multiple Event Statistic (MES). A signal passing its criteria became a
Threshold-Crossing Event (TCE). Its in-transit cadences were then
gapped/deweighted and the search was repeated, allowing additional planets to
surface. The cycle stopped when no new event passed the detection criteria.
This process is described by the
[NASA Exoplanet Archive's Kepler candidate documentation](https://exoplanetarchive.ipac.caltech.edu/docs/API_kepcandidate_columns.html).

Kepler's Data Validation stage also fitted transit models and generated
residual-flux products. The
[DV time-series documentation](https://exoplanetarchive.ipac.caltech.edu/docs/KeplerDV.html)
distinguishes these fitted residuals from the gapped inputs used when searching
for later TCEs.

For this pipeline, the equivalent final flow would be:

```text
Discover candidates with cumulative masks
                    │
                    ▼
Fit one shared stellar trend while masking every candidate
                    │
                    ▼
Measure each candidate while masking the other candidates
                    │
                    ▼
Write one row per candidate, sorted by period
```

## Problems and complexity

| Issue | Why it matters | Likely approach |
| --- | --- | --- |
| **Stopping** | BLS always returns a best peak, even for noise | Require real observed events plus calibrated significance and SNR checks |
| **Mask width** | A narrow mask leaves transit edges; a wide mask hides useful data | Start near 1.5 transit durations and track the removed fraction |
| **Aliases and harmonics** | The same planet can reappear at `P/2`, `2P`, or another multiple | Compare event overlap and phase, not period ratio alone |
| **Secondary eclipses** | An eclipsing binary's secondary can look like another planet | Associate phase-0.5 signals with the existing candidate |
| **Detrending** | Rerunning the whole pipeline gives each planet a different trend | Discover first, then fit one final trend using the union of all masks |
| **Feature contamination** | Other planets can distort depth, noise, odd/even, and shape metrics | Measure one candidate while masking all the others |
| **Overlapping transits** | Masking one event may remove part of another | Keep original flux and limit mask padding; consider model subtraction later |
| **Transit timing variations** | Real events may drift outside a fixed periodic mask | Use padded masks first; support event-specific timing later |
| **TESS coverage** | One sector often contains fewer than three events | Keep the first version periodic; treat single/double transits separately |
| **False positives** | Pre-masking deep eclipses removes useful negative examples | Detect astrophysical eclipses and describe them with vetting features |
| **Runtime** | Every extra candidate requires another global search | Reuse preparation and grids; defer TLS and detailed fitting until acceptance |
| **Evaluation** | One missed or extra row breaks comparison by position | Keep sorted output, but match benchmark rows by nearest period |

Three data views will be needed:

| View | What is masked? |
| --- | --- |
| Next-candidate search | Previously accepted candidates |
| Final trend fitting | All accepted candidates |
| Candidate measurement | Every candidate except the one being measured |

The main architectural change is therefore splitting the current
`detrend_with_bls_mask()` operation into reusable preparation, single-candidate
search, shared detrending, and candidate measurement steps.

## To-do list

- [ ] Separate light-curve preparation, BLS search, detrending, and feature
  measurement without changing current single-candidate results.
- [ ] Create an internal candidate record containing period, epoch, duration,
  detection diagnostics, observed events, and masks.
- [ ] Add the iterative search with cumulative padded masks, candidate and mask
  limits, and period-sorted output.
- [ ] Define and validate stopping, alias, duplicate, and secondary-eclipse
  rules using real and injected signals.
- [ ] Fit one shared final trend and calculate each candidate's features while
  excluding the other candidates.
- [ ] Defer expensive TLS/model refinement until after candidate acceptance and
  profile the cost of multi-candidate searches.
- [ ] Improve comparison to report matched, missed, and spurious periods rather
  than relying only on row position.
- [ ] Add regression cases for single planets, Kepler-186, pure noise,
  eclipsing binaries, overlaps, and gaps; add TTV and sparse TESS events later.
