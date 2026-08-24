# Admissions rat race

Simulation labs for a quota-constrained university admissions game: students can pay for tutoring that changes what the university observes, without changing underlying ability. Numbers come from the tested `admissions_simulation` engine.

| Surface | What it is | How to open |
|---|---|---|
| Notebooks | Full Simulation lab exhibits (equilibria, figures, finite-population checks) | Jupyter |
| Learn page | Guided walk through the same illustration presets | `python -m learn.server` |
| Explorer | Free-play knobs on `Q` and `w` | `python -m explorer.server` |

## Setup

From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install jupyter         # interactive notebooks only; not needed for learn/explorer/tests
```

Learn and Explorer use the stdlib HTTP server only; their runtime needs no extra packages beyond the repo itself. `requirements.txt` covers notebook plotting (`matplotlib`, `numpy`) and the test runner (`pytest`).

## Notebooks

Run from the repo root so `admissions_simulation` imports resolve:

```bash
jupyter notebook
```

Then open, in this order:

1. **`one_stage_admissions_simulation.ipynb`** — Lee–Suen one-stage continuum model (paper Figure 1). Nested case for the two-criterion lab when diversity weight \(w = 0\).
2. **`two_criterion_admissions_simulation.ipynb`** — writeup §§1–5: types \((a,d)\), university value \(v=(1-w)a+wd\), tutoring thresholds \(S_d\).

Both follow the Simulation lab flow: one parameter source → every equilibrium → select the locally stable root used for comparative statics → distinct outcome metrics → finite-population validation.

**Illustration parameters (two-criterion / learn):** shares \(\lambda=(0.448,0.252,0.192,0.108)\), \(B=1\), uniform \(F(c)=c/B\), \(D_1=0.36\), quotas \(Q\in\{0.33,0.40\}\), weights \(w\in\{0,1/4,3/4\}\).

| Notebook section | What to look for |
|---|---|
| Authoritative parameters | Fixed \(B\), shares, \(Q\), \(w\) — do not invent a second source |
| Metric definitions | Keep \(S_d\), mass, \(C_d\), \(\mu_{ad}\), \(x_{ad}\), admitted shares distinct |
| Equilibria / regimes | All roots shown; comparative statics use the selected stable equilibrium only |
| Stratified (\(w=3/4\)) | Ranking is tutoring-independent; diversity chooses *who* races |
| Mixed (\(w=1/4\)) | Diversity premium neutralized at the margin by credibility loss |
| Finite-population validation | Monte Carlo check — not a separate equilibrium concept |

## Learn page

Guided teaching surface over the same pinned illustration population and engine as the notebooks.

```bash
python -m learn.server
```

Open [http://127.0.0.1:8766/](http://127.0.0.1:8766/). Optional: `--host` / `--port` (default **8766**).

Work the stations top to bottom; use the quota toggles to compare \(Q=0.33\) vs \(Q=0.40\):

1. **Setup** — physical meaning of \(Q\), \(w\), \(S_d\), mass, \(C_d\), \(\mu\), \(x\), admitted shares.
2. **One-stage** (\(w=0\)) — capacity-expansion paradox: a larger quota can raise tutoring intensity.
3. **Stratified** (\(w=3/4\)) — diversity chooses who races; ranking is tutoring-independent.
4. **Mixed** (\(w=1/4\)) — diversity premium equals extra credibility loss (\(\mu_{10}=\mu_{11}\), \(C_0-C_1=1/3\)).
5. **Takeaway** — valuing diversity in the objective alone does not eliminate the rat race.

More detail: [`learn/README.md`](learn/README.md).

## Interactive explorer

Free-play on university quota and diversity weight (same pinned population):

```bash
python -m explorer.server
```

Open [http://127.0.0.1:8765/](http://127.0.0.1:8765/) (default port **8765**).

