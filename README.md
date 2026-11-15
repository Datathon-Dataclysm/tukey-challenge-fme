# tukey-challenge

Live dashboard for our RawTree recommender (Datathon FME 2026). Read-only: it never writes to RawTree.

| Tab | Question it answers | Source |
|---|---|---|
| **En directo** | Are we answering on time right now? (the organisers' status light, latency, fallbacks, the requests flowing in, what the store buys) | RawTree |
| **Versiones** | Which version converts best, and by how much? (official metric, A/B comparisons) | public leaderboard API |
| **Diagnóstico** | Which shoppers convert less? How much does each carousel slot weigh? Is the store changing? | RawTree feedback |

## Honest uncertainty

- **Official metric, not luck.** Versions are compared on the organisers' *expected* purchases per 100 requests
  (each shopper's purchase probability), not on the realised purchases: with ~2 purchases per 100 requests,
  realised counts carry ±0.4 points of pure luck on 5,000 requests.
- **Same 15 minutes, same reference.** The whole store drifts about ±10 % from one hour to the next, so each version
  is measured relative to the organisers' reference in the same 15-minute piece. Its 95 % interval comes from how
  much that ratio varies between pieces (cluster interval with Student t), so it includes the drift that a binomial
  interval ignores.
- **Clean A/B first.** Versions that ran at the same time (requests split at random) are compared piece by piece;
  versions that ran at different times are only compared through the reference, and the page says so.
- **Many groups, corrected p-values.** Segments and store trends compare many groups at once; their p-values are
  Holm-adjusted, otherwise one group in twenty looks different by pure chance.
- **No causal claims from observation.** The carousel-position block says what it can't tell: position and "the
  model puts its best guess first" are mixed unless the order is randomised.

The statistics live in `tools/decision_stats.py` (pure pandas/numpy, with self-checks: `python tools/decision_stats.py`).

## Run

```bash
pip install -r requirements.txt
cp .env.example .env                      # then fill in TEAM_ID and RAWTREE_API_KEY
python -m streamlit run tools/dashboard.py
```

Run it from this folder: `.env` and `.streamlit/config.toml` (theme) are read from here.

## Files

| File | What it is |
|---|---|
| `tools/dashboard.py` | the Streamlit app |
| `tools/decision_stats.py` | the statistics and the wording of every conclusion |
| `tools/leaderboard.py` | the public leaderboard per team and per model version, from the terminal |
| `tools/offline_results.csv` | optional (`model_version,<offline metric>`): adds an offline-vs-online check |
| `.streamlit/config.toml` | colours and font |
