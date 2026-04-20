#!/usr/bin/env python3
"""
Statistical rigor analysis for the research paper.
Computes: Welch's t-tests, effect sizes, CIs, normality tests, Kruskal-Wallis.
"""

import csv
import numpy as np
from scipy import stats
from pathlib import Path

RESULTS_DIR = Path('/Users/tom/Documents/Research Paper/results')

def read_csv(path):
    with open(path, 'r', encoding='utf-8-sig') as f:
        return list(csv.DictReader(f))

def mean(v): return np.mean(v)
def std(v): return np.std(v, ddof=1)
def ci95(v):
    """95% confidence interval"""
    n = len(v)
    m = np.mean(v)
    se = stats.sem(v)
    h = se * stats.t.ppf(0.975, n - 1)
    return m - h, m + h

def cohens_d(a, b):
    """Cohen's d effect size"""
    na, nb = len(a), len(b)
    pooled_std = np.sqrt(((na-1)*np.var(a, ddof=1) + (nb-1)*np.var(b, ddof=1)) / (na + nb - 2))
    return (np.mean(a) - np.mean(b)) / pooled_std if pooled_std > 0 else 0

def effect_label(d):
    d = abs(d)
    if d < 0.2: return 'negligible'
    if d < 0.5: return 'small'
    if d < 0.8: return 'medium'
    return 'large'

# ─── Load E1 Data ──────────────────────────────────────────────────────

e1_file = sorted(RESULTS_DIR.glob('e1_baseline_device1*.csv'))[0]
e1_rows = read_csv(e1_file)
models = ['tinyllama', 'phi3:mini', 'qwen2.5:3b', 'mistral:7b']
model_labels = {'tinyllama': 'TinyLlama-1.1B', 'phi3:mini': 'Phi-3-mini-3.8B',
                'qwen2.5:3b': 'Qwen2.5-3B', 'mistral:7b': 'Mistral-7B'}

# Extract warm-run throughput per model
e1_data = {}
for m in models:
    warm = [r for r in e1_rows if r['model'] == m and int(r['run']) > 1]
    e1_data[m] = {
        'tps': [float(r['tokens_per_sec']) for r in warm if float(r['tokens_per_sec']) > 0],
        'latency': [float(r['total_time_s']) for r in warm],
        'ttft': [float(r['ttft_ms']) for r in warm],
        'ram': [float(r['peak_ram_mb']) for r in warm],
    }

# ─── Load E3 Data ──────────────────────────────────────────────────────

e3_file = sorted(RESULTS_DIR.glob('e3_overhead_device1*.csv'))[0]
e3_rows = read_csv(e3_file)
e3_data = {}
for m in models:
    warm = [r for r in e3_rows if r['model'] == m and int(r['run']) > 1]
    e3_data[m] = {
        'step1': [float(r['step1_s']) for r in warm],
        'step2': [float(r['step2_s']) for r in warm],
        'step3': [float(r['step3_s']) for r in warm],
        'overhead': [float(r['overhead_ratio']) for r in warm],
        'chain': [float(r['agent_chain_s']) for r in warm],
    }

print('╔══════════════════════════════════════════════════════════════════════╗')
print('║  Statistical Rigor Analysis                                        ║')
print('╚══════════════════════════════════════════════════════════════════════╝')

# ═══════════════════════════════════════════════════════════════════════
# 1. NORMALITY TESTS (Shapiro-Wilk)
# ═══════════════════════════════════════════════════════════════════════
print('\n' + '='*70)
print('1. NORMALITY TESTS (Shapiro-Wilk, H0: data is normally distributed)')
print('='*70)
print(f'  {"Model":<20} {"Metric":<12} {"W-stat":>8} {"p-value":>10} {"Normal?":>8}')
print(f'  {"─"*20} {"─"*12} {"─"*8} {"─"*10} {"─"*8}')

for m in models:
    for metric_name, metric_key in [('Throughput', 'tps'), ('Latency', 'latency'), ('TTFT', 'ttft')]:
        vals = e1_data[m][metric_key]
        if len(vals) >= 3:
            w, p = stats.shapiro(vals)
            normal = 'Yes' if p > 0.05 else 'No'
            print(f'  {model_labels[m]:<20} {metric_name:<12} {w:>8.4f} {p:>10.4f} {normal:>8}')

# ═══════════════════════════════════════════════════════════════════════
# 2. PAIRWISE WELCH'S T-TESTS (Throughput)
# ═══════════════════════════════════════════════════════════════════════
print('\n' + '='*70)
print('2. PAIRWISE WELCH\'S T-TESTS (Throughput, tok/s)')
print('='*70)
print(f'  {"Comparison":<35} {"t-stat":>8} {"df":>6} {"p-value":>10} {"Cohen d":>8} {"Effect":>12} {"Sig?":>5}')
print(f'  {"─"*35} {"─"*8} {"─"*6} {"─"*10} {"─"*8} {"─"*12} {"─"*5}')

pairs = [
    ('tinyllama', 'phi3:mini'),
    ('tinyllama', 'qwen2.5:3b'),
    ('tinyllama', 'mistral:7b'),
    ('phi3:mini', 'qwen2.5:3b'),
    ('phi3:mini', 'mistral:7b'),
    ('qwen2.5:3b', 'mistral:7b'),
]

for m1, m2 in pairs:
    a, b = e1_data[m1]['tps'], e1_data[m2]['tps']
    t_stat, p_val = stats.ttest_ind(a, b, equal_var=False)
    d = cohens_d(a, b)
    df = len(a) + len(b) - 2
    sig = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else 'ns'
    label = f'{model_labels[m1]} vs {model_labels[m2]}'
    print(f'  {label:<35} {t_stat:>8.2f} {df:>6} {p_val:>10.6f} {d:>8.2f} {effect_label(d):>12} {sig:>5}')

# ═══════════════════════════════════════════════════════════════════════
# 3. KRUSKAL-WALLIS H-TEST (Non-parametric overall)
# ═══════════════════════════════════════════════════════════════════════
print('\n' + '='*70)
print('3. KRUSKAL-WALLIS H-TEST (Non-parametric, all models)')
print('='*70)

for metric_name, metric_key in [('Throughput', 'tps'), ('Latency', 'latency'), ('TTFT', 'ttft')]:
    groups = [e1_data[m][metric_key] for m in models]
    h_stat, p_val = stats.kruskal(*groups)
    sig = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else 'ns'
    print(f'  {metric_name:<15} H={h_stat:.2f}, p={p_val:.6f} {sig}')

# ═══════════════════════════════════════════════════════════════════════
# 4. 95% CONFIDENCE INTERVALS
# ═══════════════════════════════════════════════════════════════════════
print('\n' + '='*70)
print('4. 95% CONFIDENCE INTERVALS')
print('='*70)
print(f'  {"Model":<20} {"Metric":<12} {"Mean":>8} {"95% CI Low":>12} {"95% CI High":>12} {"±":>8}')
print(f'  {"─"*20} {"─"*12} {"─"*8} {"─"*12} {"─"*12} {"─"*8}')

for m in models:
    for name, key in [('tok/s', 'tps'), ('Latency(s)', 'latency'), ('TTFT(ms)', 'ttft')]:
        vals = e1_data[m][key]
        m_val = np.mean(vals)
        lo, hi = ci95(vals)
        margin = (hi - lo) / 2
        print(f'  {model_labels[m]:<20} {name:<12} {m_val:>8.2f} {lo:>12.2f} {hi:>12.2f} {margin:>8.2f}')

# ═══════════════════════════════════════════════════════════════════════
# 5. AGENT STEP ANALYSIS — Paired t-test Step1 vs Step3
# ═══════════════════════════════════════════════════════════════════════
print('\n' + '='*70)
print('5. PAIRED T-TEST: Step 1 vs Step 3 (Context Accumulation)')
print('='*70)
print(f'  {"Model":<20} {"S1 mean":>8} {"S3 mean":>8} {"t-stat":>8} {"p-value":>10} {"Cohen d":>8} {"Sig?":>5}')
print(f'  {"─"*20} {"─"*8} {"─"*8} {"─"*8} {"─"*10} {"─"*8} {"─"*5}')

for m in models:
    s1, s3 = e3_data[m]['step1'], e3_data[m]['step3']
    t_stat, p_val = stats.ttest_rel(s1, s3)
    d = cohens_d(s1, s3)
    sig = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else 'ns'
    print(f'  {model_labels[m]:<20} {np.mean(s1):>8.2f} {np.mean(s3):>8.2f} {t_stat:>8.2f} {p_val:>10.6f} {d:>8.2f} {sig:>5}')

# ═══════════════════════════════════════════════════════════════════════
# 6. OVERHEAD RATIO — One-sample t-test vs 1.0
# ═══════════════════════════════════════════════════════════════════════
print('\n' + '='*70)
print('6. ONE-SAMPLE T-TEST: Overhead Ratio vs 1.0 (Linear baseline)')
print('='*70)
print(f'  {"Model":<20} {"Mean OR":>8} {"95% CI":>18} {"t-stat":>8} {"p-value":>10} {"Sig?":>5}')
print(f'  {"─"*20} {"─"*8} {"─"*18} {"─"*8} {"─"*10} {"─"*5}')

for m in models:
    ors = e3_data[m]['overhead']
    t_stat, p_val = stats.ttest_1samp(ors, 1.0)
    lo, hi = ci95(ors)
    sig = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else 'ns'
    print(f'  {model_labels[m]:<20} {np.mean(ors):>8.3f} [{lo:.3f}, {hi:.3f}] {t_stat:>8.2f} {p_val:>10.6f} {sig:>5}')

# ═══════════════════════════════════════════════════════════════════════
# 7. SUMMARY TABLE FOR PAPER
# ═══════════════════════════════════════════════════════════════════════
print('\n' + '='*70)
print('7. PUBLICATION-READY SUMMARY TABLE')
print('='*70)
print('\n| Model | tok/s (mean ± SD) [95% CI] | Latency p50 (s) [95% CI] | TTFT (ms) [95% CI] |')
print('|---|---|---|---|')

for m in models:
    tps_vals = e1_data[m]['tps']
    lat_vals = e1_data[m]['latency']
    ttft_vals = e1_data[m]['ttft']
    
    tps_m, tps_s = np.mean(tps_vals), np.std(tps_vals, ddof=1)
    tps_lo, tps_hi = ci95(tps_vals)
    
    lat_p50 = np.median(lat_vals)
    lat_lo, lat_hi = ci95(lat_vals)
    
    ttft_m = np.mean(ttft_vals)
    ttft_lo, ttft_hi = ci95(ttft_vals)
    
    print(f'| {model_labels[m]} | {tps_m:.2f} ± {tps_s:.2f} [{tps_lo:.2f}, {tps_hi:.2f}] | '
          f'{lat_p50:.2f} [{lat_lo:.2f}, {lat_hi:.2f}] | '
          f'{ttft_m:.0f} [{ttft_lo:.0f}, {ttft_hi:.0f}] |')

print('\n' + '='*70)
print('STATISTICAL ANALYSIS COMPLETE')
print('='*70)
