"""HKJC production runner.

Trains the current best config on each test year (2015-2024) and generates
a Vega-Lite HTML visualization with Kelly-level tabs and year tabs.

Current best config:
  - Offset deep_slow model (market as base_margin) + 5-seed bagging
  - Feature set: prune2 (18 edge) + body_wt_mkt_residual (19 edge) + 4 race card + 2 market
  - Bagging seeds: [7, 17, 42, 123, 256]
  - Kelly fractions swept: {0.01, 0.02, 0.03, 0.05}

Output: results/bankroll_viz.html — interactive, open in browser.

See FINDINGS.md at repo root for the full session report.
"""
import sys, os, json
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss as sk_log_loss

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../data-collection")))
from model import split_data
from model_classifier import OffsetModel
from betting import simulate_bankroll
from utilities import conv_dict, load_pool_dividends



INFRA_COLS = ['Year', 'Win Odds', 'y.status_place']
MARKET_COLS = ['Implied_Prob', 'log_implied_prob']
RACE_CARD = ['Racecourse', 'Track_Turf', 'Draw', 'Act_Wt']
EDGE = [
    'class_change', 'wt_z', 'recent_form', 'trainer_track_spec',
    'tt_gap', 'tr_gap', 'draw_outside_ST', 'prev_win',
    'fav_field_size', 'form_vs_career', 'setup_weight_z', 'career_beat_odds',
    'bt_avg_early', 'bt_last_behind',
    'trail_sect_closing', 'trail_sect_peak_at', 'trail_sect_avg', 'trail_sect_best_gain',
    'body_wt_mkt_residual',
]
BAG_SEEDS = [7, 17, 42, 123, 256]
KELLY_SWEEP = [0.01, 0.02, 0.03, 0.05]
INITIAL_BANKROLL = 10_000_000.0


def bagged_offset_predict(X_train, Y_train, X_test, seeds=BAG_SEEDS):
    preds_list = []
    for seed in seeds:
        model = OffsetModel(params={'seed': seed, 'nthread': 10}, rounds=400)
        model.fit(X_train, Y_train)
        preds_list.append(model.predict_proba(X_test)['Y_prob'])
    stacked = pd.concat(preds_list, axis=1)
    mean_pred = stacked.mean(axis=1)
    mean_pred = mean_pred / mean_pred.groupby(level=0).transform('sum')
    return mean_pred


def simulate_and_collect(ret, alpha, initial_bankroll, race_dates):
    _, _, br_wo, br_wr = simulate_bankroll(
        ret, alpha, initial_bankroll=initial_bankroll,
        probability_col='pred_prob', kelly_odds_col='Win Odds',
        payoff_col='Returns', result_col='y.status_win',
    )
    rows = []
    race_ids = br_wo.index.tolist()
    for i, rid in enumerate(race_ids):
        dt = race_dates.get(rid)
        if pd.isna(dt):
            continue
        rows.append({
            'race_idx': i + 1,
            'race_id': str(rid),
            'date': pd.to_datetime(dt).strftime('%Y-%m-%d'),
            'br_no': float(br_wo.iloc[i]),
            'br_wr': float(br_wr.iloc[i]),
        })
    return rows, \
           float(br_wo.iloc[-1]) if len(br_wo) > 0 else initial_bankroll, \
           float(br_wr.iloc[-1]) if len(br_wr) > 0 else initial_bankroll


def main():
    # Load the single production features parquet (built by data-processing/prep-data.py).
    # Must contain all 25 features incl. body_wt_mkt_residual.
    df = pd.read_parquet("../data/processed/race_features.parquet")

    # Dates for the visualization timeline
    ext = pd.read_parquet("../data/processed/race_features_extended.parquet")
    race_dates = ext.reset_index()[['race_id', 'Date']].drop_duplicates('race_id').set_index('race_id')['Date']

    odds_all = df['Win Odds']
    dividends = conv_dict(load_pool_dividends())
    dividends_df = dividends.reset_index() if dividends.index.names != [None] else dividends.copy()

    cols = MARKET_COLS + RACE_CARD + EDGE
    df_sub = df[cols + INFRA_COLS].drop(columns=['Win Odds'])

    CONFIGS = [f'Bagged_K{int(k*100):02d}' for k in KELLY_SWEEP]
    yearly_data = {c: {} for c in CONFIGS}
    yearly_summary = {c: {} for c in CONFIGS}
    all_years_data = {c: [] for c in CONFIGS}
    running = {c: {'no': INITIAL_BANKROLL, 'wr': INITIAL_BANKROLL} for c in CONFIGS}

    for test_year in range(2015, 2027):
        train_years = list(range(2010, test_year))
        print(f"\n{'='*60}\nTest year: {test_year}\n{'='*60}")

        X_train, X_test, Y_train, Y_test = split_data(
            df_sub, train_years, [test_year], 'y.status_place',
            remove_ent=True, shuffle=False,
        )
        odds = odds_all.reindex(Y_test.index).dropna()

        # Bagged offset (one per seed, 5 total) — same predictions used for all Kelly levels
        print(f"  Training {len(BAG_SEEDS)} bagged models...")
        bagged = bagged_offset_predict(X_train, Y_train, X_test, BAG_SEEDS)

        Y_test_win = (Y_test == 1).astype(int).rename('y.status_win')
        common = bagged.index.intersection(Y_test_win.index).intersection(odds.index)
        ret = pd.DataFrame({
            'y.status_win': Y_test_win.loc[common].values,
            'pred_prob': bagged.loc[common].values,
        }, index=common).reset_index()
        race_id_name = Y_test.index.names[0]
        ret = ret.merge(dividends_df,
                        left_on=race_id_name,
                        right_on=dividends_df.columns[0] if race_id_name not in dividends_df.columns else race_id_name,
                        how='left', suffixes=('', '_div'))
        odds_flat = odds.loc[common].reset_index()
        ret = ret.merge(odds_flat, on=list(Y_test.index.names), how='left', suffixes=('', '_odds'))
        if 'race_id' not in ret.columns and race_id_name in ret.columns:
            ret = ret.rename(columns={race_id_name: 'race_id'})

        for kelly in KELLY_SWEEP:
            conf = f'Bagged_K{int(kelly*100):02d}'

            # Per-year reset starting at $10M
            rows_reset, fn_reset, fw_reset = simulate_and_collect(
                ret, kelly, INITIAL_BANKROLL, race_dates)
            yearly_data[conf][test_year] = rows_reset
            yearly_summary[conf][test_year] = {
                'final_no': fn_reset, 'final_wr': fw_reset,
                'roi_no': (fn_reset - INITIAL_BANKROLL) / INITIAL_BANKROLL,
                'roi_wr': (fw_reset - INITIAL_BANKROLL) / INITIAL_BANKROLL,
                'n_races': len(rows_reset),
            }
            # Compounded running — simulate no-rebate and with-rebate separately
            # (each starts from the respective running bankroll)
            rows_run_no, fn_run, _ = simulate_and_collect(
                ret, kelly, running[conf]['no'], race_dates)
            _, _, fw_run = simulate_and_collect(
                ret, kelly, running[conf]['wr'], race_dates)
            # Combine: use no-rebate's per-race dates but assemble bankroll from both sims
            # Re-run one more time for with-rebate trajectory from running['wr']
            rows_run_wr, _, _ = simulate_and_collect(
                ret, kelly, running[conf]['wr'], race_dates)
            # Merge: for each race, take br_no from first sim, br_wr from second
            merged = []
            for r_no, r_wr in zip(rows_run_no, rows_run_wr):
                merged.append({
                    'race_idx': r_no['race_idx'],
                    'race_id': r_no['race_id'],
                    'date': r_no['date'],
                    'br_no': r_no['br_no'],
                    'br_wr': r_wr['br_wr'],
                })
            all_years_data[conf].extend(merged)
            running[conf]['no'] = fn_run
            running[conf]['wr'] = fw_run

            print(f"    {conf}: ROI_no={yearly_summary[conf][test_year]['roi_no']:+.2%}  "
                  f"ROI_wr={yearly_summary[conf][test_year]['roi_wr']:+.2%}")

    # Overall summary
    overall = {}
    for conf in CONFIGS:
        fn = running[conf]['no']; fw = running[conf]['wr']
        overall[conf] = {
            'final_no': fn, 'final_wr': fw,
            'roi_no': (fn - INITIAL_BANKROLL) / INITIAL_BANKROLL,
            'roi_wr': (fw - INITIAL_BANKROLL) / INITIAL_BANKROLL,
            'n_races': len(all_years_data[conf]),
        }
        print(f"\n  {conf} TOTAL: no_rebate={overall[conf]['roi_no']:+.2%}  "
              f"with_rebate={overall[conf]['roi_wr']:+.2%}")

    # Summary table
    print(f"\n{'='*70}\nKELLY SWEEP SUMMARY (bagged Offset, seeds={BAG_SEEDS})\n{'='*70}")
    print(f"  {'Config':<15s}{'cum_no':>12s}{'cum_wr':>12s}{'pos_no':>10s}{'pos_wr':>10s}")
    for conf in CONFIGS:
        years = sorted(yearly_summary[conf].keys())
        pos_no = sum(1 for y in years if yearly_summary[conf][y]['roi_no'] > 0)
        pos_wr = sum(1 for y in years if yearly_summary[conf][y]['roi_wr'] > 0)
        print(f"  {conf:<15s}{overall[conf]['roi_no']*100:>+11.2f}%"
              f"{overall[conf]['roi_wr']*100:>+11.2f}%"
              f"{pos_no:>6d}/{len(years):<3d}"
              f"{pos_wr:>6d}/{len(years):<3d}")

    # Build HTML
    html = build_html(CONFIGS, yearly_data, yearly_summary, all_years_data, overall)
    out_path = 'results/bankroll_viz.html'
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(html)
    print(f"\nSaved viz to {out_path}")
    print(f"Open: file://{os.path.abspath(out_path).replace(os.sep, '/')}")


def build_html(CONFIGS, yearly_data, yearly_summary, all_years_data, overall):
    years = sorted(next(iter(yearly_data.values())).keys())
    data_blob = json.dumps({
        'configs': CONFIGS,
        'years': years,
        'yearly_data': {c: {str(y): yearly_data[c][y] for y in years} for c in CONFIGS},
        'yearly_summary': {c: {str(y): yearly_summary[c][y] for y in years} for c in CONFIGS},
        'all_years_data': {c: all_years_data[c] for c in CONFIGS},
        'overall': overall,
        'initial_bankroll': INITIAL_BANKROLL,
    })

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<title>HKJC Bankroll Viz — Bagged Offset Kelly Sweep</title>
<script src="https://cdn.jsdelivr.net/npm/vega@5"></script>
<script src="https://cdn.jsdelivr.net/npm/vega-lite@5"></script>
<script src="https://cdn.jsdelivr.net/npm/vega-embed@6"></script>
<style>
  body {{ font-family: -apple-system, sans-serif; margin: 20px; background: #fafafa; color: #222; }}
  h1 {{ font-size: 18px; margin-bottom: 4px; }}
  .sub {{ color: #666; font-size: 13px; margin-bottom: 12px; }}
  .row {{ margin-bottom: 12px; }}
  .row .label {{ display: inline-block; width: 90px; font-size: 12px; color: #555; }}
  .pill-group {{ display: inline-flex; gap: 4px; flex-wrap: wrap; }}
  .pill {{
    background: white; border: 1px solid #ccc; padding: 5px 12px; cursor: pointer;
    font-size: 13px; border-radius: 16px; color: #333;
  }}
  .pill:hover {{ background: #eef; }}
  .pill.active {{ background: #0077cc; color: white; border-color: #0077cc; font-weight: bold; }}
  .summary {{
    display: flex; gap: 18px; margin-bottom: 12px; padding: 10px;
    background: white; border: 1px solid #eee; border-radius: 4px; flex-wrap: wrap;
  }}
  .stat {{ font-size: 13px; }}
  .stat .label {{ color: #666; }}
  .stat .val {{ font-weight: bold; font-size: 15px; }}
  .pos {{ color: #0a7; }}
  .neg {{ color: #c33; }}
  #chart {{ background: white; padding: 10px; border: 1px solid #eee; border-radius: 4px; }}
</style>
</head>
<body>
<h1>HKJC Bagged Offset — Kelly Sweep</h1>
<p class="sub">prune2 + body_wt_mkt_residual (19 edge), 5-seed bagging [7, 17, 42, 123, 256]. Per-year tab resets to $10M. &quot;All Years&quot; is compounded.</p>

<div class="row">
  <span class="label">Kelly:</span>
  <span class="pill-group" id="config-tabs"></span>
</div>
<div class="row">
  <span class="label">Year:</span>
  <span class="pill-group" id="year-tabs"></span>
</div>

<div class="summary" id="summary"></div>
<div id="chart"></div>

<script>
const DATA = {data_blob};
let currentConfig = DATA.configs[0];
let currentYear = 'all';

function fmtPct(x) {{
  const sign = x >= 0 ? '+' : '';
  const cls = x >= 0 ? 'pos' : 'neg';
  return `<span class="${{cls}}">${{sign}}${{(x*100).toFixed(2)}}%</span>`;
}}
function fmtMoney(x) {{ return '$' + Math.round(x).toLocaleString(); }}

function render() {{
  let data, title, summaryHTML;
  if (currentYear === 'all') {{
    data = DATA.all_years_data[currentConfig];
    title = `${{currentConfig}} — All Years (compounded)`;
    const s = DATA.overall[currentConfig];
    summaryHTML = `
      <div class="stat"><span class="label">Races:</span> <span class="val">${{s.n_races}}</span></div>
      <div class="stat"><span class="label">Final (no rebate):</span> <span class="val">${{fmtMoney(s.final_no)}}</span></div>
      <div class="stat"><span class="label">Final (with rebate):</span> <span class="val">${{fmtMoney(s.final_wr)}}</span></div>
      <div class="stat"><span class="label">Cum ROI (no rebate):</span> <span class="val">${{fmtPct(s.roi_no)}}</span></div>
      <div class="stat"><span class="label">Cum ROI (with rebate):</span> <span class="val">${{fmtPct(s.roi_wr)}}</span></div>
    `;
  }} else {{
    data = DATA.yearly_data[currentConfig][currentYear];
    title = `${{currentConfig}} — ${{currentYear}} (starting from $10M)`;
    const s = DATA.yearly_summary[currentConfig][currentYear];
    summaryHTML = `
      <div class="stat"><span class="label">Races:</span> <span class="val">${{s.n_races}}</span></div>
      <div class="stat"><span class="label">Final (no rebate):</span> <span class="val">${{fmtMoney(s.final_no)}}</span></div>
      <div class="stat"><span class="label">Final (with rebate):</span> <span class="val">${{fmtMoney(s.final_wr)}}</span></div>
      <div class="stat"><span class="label">ROI (no rebate):</span> <span class="val">${{fmtPct(s.roi_no)}}</span></div>
      <div class="stat"><span class="label">ROI (with rebate):</span> <span class="val">${{fmtPct(s.roi_wr)}}</span></div>
    `;
  }}
  document.getElementById('summary').innerHTML = summaryHTML;

  const rows = [];
  data.forEach(d => {{
    rows.push({{ race_idx: d.race_idx, date: d.date, bankroll: d.br_no, series: 'no_rebate' }});
    rows.push({{ race_idx: d.race_idx, date: d.date, bankroll: d.br_wr, series: 'with_rebate' }});
  }});

  const spec = {{
    $schema: 'https://vega.github.io/schema/vega-lite/v5.json',
    title, width: 1100, height: 450,
    data: {{ values: rows }},
    mark: {{ type: 'line', interpolate: 'monotone' }},
    encoding: {{
      x: {{ field: 'race_idx', type: 'quantitative', axis: {{ title: 'Race # (chronological)' }} }},
      y: {{ field: 'bankroll', type: 'quantitative', axis: {{ title: 'Bankroll ($)', format: '$,.0f' }} }},
      color: {{ field: 'series', type: 'nominal',
                scale: {{ domain: ['no_rebate', 'with_rebate'], range: ['#c33', '#0077cc'] }},
                legend: {{ title: 'Series' }} }},
      tooltip: [
        {{ field: 'date', type: 'temporal', title: 'Date' }},
        {{ field: 'series', type: 'nominal' }},
        {{ field: 'bankroll', type: 'quantitative', format: '$,.0f' }}
      ]
    }}
  }};
  vegaEmbed('#chart', spec, {{ actions: false }});
}}

function buildTabs() {{
  const cfgEl = document.getElementById('config-tabs');
  const cfgBtns = [];
  DATA.configs.forEach(c => {{
    const b = document.createElement('button');
    b.textContent = c;
    b.className = 'pill';
    b.onclick = () => {{
      cfgBtns.forEach(x => x.classList.remove('active'));
      b.classList.add('active');
      currentConfig = c;
      render();
    }};
    cfgEl.appendChild(b); cfgBtns.push(b);
  }});

  const yrEl = document.getElementById('year-tabs');
  const yrBtns = [];
  DATA.years.forEach(y => {{
    const b = document.createElement('button');
    b.textContent = y;
    b.className = 'pill';
    b.onclick = () => {{
      yrBtns.forEach(x => x.classList.remove('active'));
      b.classList.add('active');
      currentYear = y;
      render();
    }};
    yrEl.appendChild(b); yrBtns.push(b);
  }});
  const all = document.createElement('button');
  all.textContent = 'All Years';
  all.className = 'pill';
  all.onclick = () => {{
    yrBtns.forEach(x => x.classList.remove('active'));
    all.classList.add('active');
    currentYear = 'all';
    render();
  }};
  yrEl.appendChild(all); yrBtns.push(all);

  cfgBtns[0].click();
  yrBtns[yrBtns.length - 1].click();
}}

buildTabs();
</script>
</body>
</html>
"""
    return html


if __name__ == '__main__':
    main()
