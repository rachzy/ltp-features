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

### Knowing when to stop

BLS always returns a highest peak, even when the remaining light curve is only
noise. Without a stopping rule, the loop would keep inventing candidates until
it reached an arbitrary limit.

**Possible solution:** Accept a signal only when it has enough actually
observed events, in-transit samples, depth, periodogram significance, and
signal-to-noise. These thresholds must be calibrated for this pipeline rather
than copied directly from Kepler's MES threshold.

### Choosing the mask width

A mask that matches exactly one fitted duration may leave ingress, egress, or
small timing errors behind. BLS can then rediscover those leftovers. A very
wide mask avoids that problem but throws away more data and may hide another
planet.

**Possible solution:** Start with a padded mask around 1.5 transit durations,
include a cadence-based minimum margin, and stop if the cumulative masked
fraction becomes excessive.

### Aliases and harmonics

The same signal can appear at half, twice, or another multiple of its real
period. However, period ratio alone is not enough to reject a candidate because
different planets can genuinely orbit near resonances such as 2:1.

**Possible solution:** Compare the predicted event times and mask overlap as
well as the period ratio. A new candidate should be supported by transit events
that were not already explained by an earlier candidate.

### Secondary eclipses

After the primary eclipse of a binary is masked, its secondary eclipse may
become the strongest remaining periodic dip. Treating it as another planet
would create a duplicate candidate for the same binary system.

**Possible solution:** Check whether a new signal has the same period and lies
near phase 0.5 of an existing signal. If so, attach it to that candidate's
secondary-eclipse and false-positive features instead of creating another row.

### Keeping detrending consistent

Calling the complete current pipeline once per candidate would fit a different
spline trend on every iteration. The resulting depths and noise measurements
would no longer share the same baseline.

**Possible solution:** Use one prepared flux series during discovery. After all
provisional candidates are found, fit one final stellar trend while excluding
the union of every candidate mask, then refine all candidates on that common
detrended curve.

### Preventing feature contamination

Transits from another planet can distort the current candidate's depth,
stability, local noise, odd/even ratio, secondary-eclipse measurement, and
shape. This is especially relevant for shallow signals.

**Possible solution:** When measuring candidate A, temporarily mask candidates
B, C, and so on while preserving A. Compute star-level noise metrics once with
all detected transits excluded and share those values across candidate rows.

### Overlapping transits

Two planets can cross the star at the same time. Masking the first candidate
will then remove part of the second candidate's event, reducing the evidence
available in the next search.

**Possible solution:** Never destroy or interpolate the original flux, keep
mask padding controlled, and require support from several events rather than
one overlap. Model subtraction can be explored later if overlapping systems
remain difficult.

### Transit-timing variations

A periodic mask assumes each transit occurs exactly at
`t0 + epoch × period`. Gravitational interactions can shift individual events,
leaving part of the transit outside the mask and making the same planet appear
again.

**Possible solution:** Use padded periodic masks in the first version. Later,
measure each transit center separately and build event-specific masks for
systems with significant timing variations.

### Sparse TESS coverage

Kepler usually observed stars for years, while one TESS sector covers roughly
27 days. Many real TESS planets therefore produce only one or two visible
transits and cannot pass a standard three-event periodic search.

**Possible solution:** Keep the first implementation focused on periodic
signals with at least three observed events. Treat single- and double-transit
detection as a separate future search mode.

### Preserving false positives

The pipeline has an option to mask deep eclipses before BLS. This can remove
eclipsing binaries—the exact objects needed as negative examples when training
a planet-versus-false-positive model.

**Possible solution:** Do not pre-mask astrophysical eclipses during normal
candidate discovery. Reserve early masking for confidently instrumental
artifacts and let the candidate's shape, secondary, and odd/even features
describe an eclipsing binary.

### Controlling runtime

Every additional candidate requires another period search. Repeating
preparation, global BLS, detailed duration fitting, de-aliasing, and TLS from
scratch would make multi-candidate systems much slower.

**Possible solution:** Prepare and bin the light curve once, reuse period grids,
run a cheaper BLS search inside the loop, and defer TLS and detailed model fits
until a candidate passes the acceptance checks. Also enforce candidate and
compute-budget limits.

### Matching detections during evaluation

Sorting rows by period is deterministic, but comparison by row position is
fragile. If one planet is missed or one false signal is inserted, every later
candidate can be matched against the wrong literature planet.

**Possible solution:** Keep production CSVs sorted and label-free, but make the
evaluation code pair extracted and confirmed candidates by nearest compatible
period. Report unmatched rows separately as missed or spurious detections.

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
