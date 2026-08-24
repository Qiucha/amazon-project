# Admissions learn

Guided teaching surface for the one-stage and two-criterion Simulation labs.
Presets match the notebook illustration population; numbers come from the same
tested `admissions_simulation` engine as the notebooks and Interactive explorer.

## Run

From the repository root:

```bash
python -m learn.server
```

Open [http://127.0.0.1:8766/](http://127.0.0.1:8766/). Default port is **8766**
(explorer uses 8765).

## Learning path

1. **Setup** — physical meaning of `Q`, `w`, `S_d`, mass, `C_d`, `μ`, `x`, admitted shares.
2. **One-stage** (`w=0`) — capacity-expansion paradox: larger quota can raise tutoring intensity.
3. **Stratified** (`w=3/4`) — diversity chooses *who* races; ranking is tutoring-independent.
4. **Mixed** (`w=1/4`) — diversity premium equals extra credibility loss (`μ₁₀=μ₁₁`, `C₀−C₁=1/3`).
5. **Takeaway** — valuing diversity in the objective alone does not eliminate the rat race.

Free-play `Q`/`w` knobs: `python -m explorer.server`.
