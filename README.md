# tukey-challenge

Live dashboard for our RawTree recommender (Datathon FME 2026): what flows between the shop, the recommender and the
shoppers' feedback, turned into decisions. Read-only: it never writes to RawTree.

- **En directo**: requests → our answer → what the shopper did (operations, reliability).
- **Decisiones**: one block per decision (which version to ship, where the model fails, how much the order matters,
  whether to follow live trends, whether offline evaluation can be trusted), each with a chart, its uncertainty and a
  one-line conclusion.

## Run

```bash
pip install -r requirements.txt
cp .env.example .env            # then fill in TEAM_ID and RAWTREE_API_KEY
streamlit run tools/dashboard.py
```

Run it from this folder: `.env` and `.streamlit/config.toml` (theme) are read from here.

## Files

| File | What it is |
|---|---|
| `tools/dashboard.py` | the Streamlit app |
| `tools/decision_stats.py` | the statistics behind the «Decisiones» tab |
| `tools/leaderboard.py` | the public leaderboard per team and per model version, from the terminal |
| `tools/offline_results.csv` | optional (`model_version,<offline metric>`), feeds the offline-vs-online block |
| `.streamlit/config.toml` | colours and font |
