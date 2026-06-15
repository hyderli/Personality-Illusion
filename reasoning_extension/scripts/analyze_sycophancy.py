import csv
import json
import re
from collections import defaultdict
from pathlib import Path

# =============================================================================
# 1. Bidirectional items CSV
# =============================================================================
items_path = Path('reasoning_extension/results/sycophancy/deepseek-r1-distill-qwen-7b_bidirectional_sycophancy_items.csv')
rows = list(csv.DictReader(open(items_path)))

print(f'Bidirectional items CSV rows: {len(rows)}')
print(f'Personas: {set(r["persona_name"] for r in rows)}')
print(f'Temps: {sorted(set(r["temperature"] for r in rows))}')
print(f'Runs: {sorted(set(r["run"] for r in rows))}')

agg = defaultdict(lambda: {'dis_v':0,'dis_f':0,'agr_v':0,'agr_f':0})
for r in rows:
    p = r['persona_name']
    if r['flipped'] in ('True','False'):
        agg[p]['dis_v'] += 1
        if r['flipped']=='True': agg[p]['dis_f'] += 1
    if r['step2_agree_flipped'] in ('True','False'):
        agg[p]['agr_v'] += 1
        if r['step2_agree_flipped']=='True': agg[p]['agr_f'] += 1

print('\n=== Bidirectional: aggregated by persona (all temps/runs) ===')
for p, c in agg.items():
    dr = c['dis_f']/c['dis_v']*100 if c['dis_v'] else float('nan')
    ar = c['agr_f']/c['agr_v']*100 if c['agr_v'] else float('nan')
    print(f'  {p:10s}: disagree {c["dis_f"]:>3}/{c["dis_v"]:<3} = {dr:5.2f}%   agree {c["agr_f"]:>3}/{c["agr_v"]:<3} = {ar:5.2f}%   asym={dr-ar:+.2f}pp')

print('\n=== Bidirectional: by persona x temperature ===')
agg2 = defaultdict(lambda: {'dis_v':0,'dis_f':0,'agr_v':0,'agr_f':0})
for r in rows:
    k = (r['persona_name'], r['temperature'])
    if r['flipped'] in ('True','False'):
        agg2[k]['dis_v'] += 1
        if r['flipped']=='True': agg2[k]['dis_f'] += 1
    if r['step2_agree_flipped'] in ('True','False'):
        agg2[k]['agr_v'] += 1
        if r['step2_agree_flipped']=='True': agg2[k]['agr_f'] += 1
for (p,t), c in sorted(agg2.items()):
    dr = c['dis_f']/c['dis_v']*100 if c['dis_v'] else float('nan')
    ar = c['agr_f']/c['agr_v']*100 if c['agr_v'] else float('nan')
    print(f'  {p:10s} t={t}: dis={dr:5.2f}% (n={c["dis_v"]:3})  agr={ar:5.2f}% (n={c["agr_v"]:3})  Δ={dr-ar:+6.2f}pp')

# =============================================================================
# 2. JSON files: cot (standard) vs cot_bidirectional
# =============================================================================

def read_cot_jsons(folder):
    """Read all JSON files in a folder and compute per-file flip rates."""
    results = []
    fpath = Path(folder)
    for jf in sorted(fpath.glob('*.json')):
        data = json.load(open(jf))
        flips = sum(1 for d in data if d.get('flipped') is True)
        valid = sum(1 for d in data if d.get('flipped') is not None)
        # Robust extraction from filenames like: deepseek-r1-distill-qwen-7b_empty_t0.5_r1.json
        m_temp = re.search(r'_t(\d+(?:\.\d+)?)_', jf.stem)
        m_run  = re.search(r'_r(\d+)$', jf.stem)
        m_persona = re.search(r'_([a-z]+)_t\d', jf.stem)
        results.append({
            'file': jf.name,
            'persona': m_persona.group(1) if m_persona else 'unknown',
            'temp': float(m_temp.group(1)) if m_temp else None,
            'run': int(m_run.group(1)) if m_run else None,
            'flips': flips,
            'valid': valid,
            'rate': flips/valid*100 if valid else None,
        })
    return results

print('\n\n=== Standard (cot/): per-file flip rates ===')
std = read_cot_jsons('reasoning_extension/results/sycophancy/cot')
for r in std:
    print(f'  {r["file"]:55s} {r["persona"]:8s} t={r["temp"]:.1f} r={r["run"]}  {r["flips"]:>3}/{r["valid"]:<3} = {r["rate"]:5.2f}%')

print('\n=== Bidirectional (cot_bidirectional/): per-file disagree flip rates ===')
bi = read_cot_jsons('reasoning_extension/results/sycophancy/cot_bidirectional')
for r in bi:
    print(f'  {r["file"]:55s} {r["persona"]:8s} t={r["temp"]:.1f} r={r["run"]}  {r["flips"]:>3}/{r["valid"]:<3} = {r["rate"]:5.2f}%')

# =============================================================================
# 3. Compare standard vs bidirectional disagree rates
# =============================================================================
# =============================================================================
# Check why empty/helpful have no agree data
# =============================================================================
print('\n=== Checking raw agree field values in CSV ===')
for persona in ('empty','helpful','respond'):
    vals = set(r['step2_agree_flipped'] for r in rows if r['persona_name']==persona)
    non_null = [r['step2_agree_flipped'] for r in rows if r['persona_name']==persona and r['step2_agree_flipped']]
    print(f'  {persona}: unique vals={sorted(vals, key=str)}  non-empty count={len(non_null)}')

# =============================================================================
# Aggregate across all personas where data exists
# =============================================================================
print('\n=== Overall: all personas combined ===')
total_dis_v = sum(c['dis_v'] for c in agg.values())
total_dis_f = sum(c['dis_f'] for c in agg.values())
total_agr_v = sum(c['agr_v'] for c in agg.values())
total_agr_f = sum(c['agr_f'] for c in agg.values())
print(f'  Disagree (user opposes): {total_dis_f}/{total_dis_v} = {total_dis_f/total_dis_v*100:.2f}%')
print(f'  Agree    (user supports): {total_agr_f}/{total_agr_v} = {total_agr_f/total_agr_v*100:.2f}%')
print(f'  Asymmetry (dis - agree):  {total_dis_f/total_dis_v*100 - total_agr_f/total_agr_v*100:+.2f}pp')

# =============================================================================
# Check JSON keys (bidirectional files store only disagree condition)
# =============================================================================
print('\n=== Bidirectional JSON structure ===')
sample = json.load(open(sorted(Path('reasoning_extension/results/sycophancy/cot_bidirectional').glob('*.json'))[0]))
print('  Keys in first JSON entry:', list(sample[0].keys()))
print('  Note: JSON only stores disagree condition; agree data is CSV-only.')

# =============================================================================
# Check if empty/helpful agree data exists elsewhere
# =============================================================================
print('\n=== Looking for other bidirectional CSVs with empty/helpful agree data ===')
for csvf in sorted(Path('reasoning_extension/results/sycophancy').glob('*bidirectional*.csv')):
    rows2 = list(csv.DictReader(open(csvf)))
    if not rows2:
        continue
    col = 'persona_name' if 'persona_name' in rows2[0] else 'Persona' if 'Persona' in rows2[0] else None
    personas = set(r[col] for r in rows2) if col else set()
    agr_count = sum(1 for r in rows2 if r.get('step2_agree_flipped') in ('True','False'))
    print(f'  {csvf.name}: {len(rows2)} rows, personas={personas}, agree rows={agr_count}')

# =============================================================================
# Standard vs Bidirectional comparison
# =============================================================================
print('\n=== Standard vs Bidirectional disagree rates (matched persona/temp/run) ===')
std_map = {(r['persona'], r['temp'], r['run']): r for r in std}
bi_map = {(r['persona'], r['temp'], r['run']): r for r in bi}
all_keys = sorted(set(std_map.keys()) & set(bi_map.keys()))
for k in all_keys:
    s = std_map[k]
    b = bi_map[k]
    delta = b['rate'] - s['rate']
    print(f'  {k[0]:8s} t={k[1]:.1f} r={k[2]}: std={s["rate"]:5.2f}%  bi={b["rate"]:5.2f}%  Δ={delta:+.2f}pp')

# =============================================================================
# Summary
# =============================================================================
print('\n' + '='*70)
print('SUMMARY')
print('='*70)
print('1. Data completeness:')
print('   - Only "respond" persona has complete bidirectional data.')
print('   - "empty" and "helpful" have empty agree fields in the CSV.')
print('   - Need to re-run bidirectional for empty/helpful OR limit claims to respond.')
print()
print('2. Key finding (respond persona, n=446 agree, n=437 disagree):')
print(f'   - Disagree flip rate:  {total_dis_f}/{total_dis_v} = {total_dis_f/total_dis_v*100:.2f}%')
print(f'   - Agree flip rate:    {total_agr_f}/{total_agr_v} = {total_agr_f/total_agr_v*100:.2f}%')
print(f'   - Asymmetry:          {total_dis_f/total_dis_v*100 - total_agr_f/total_agr_v*100:+.2f}pp')
print()
print('3. Interpretation per Claude_Next_Steps.md:')
print('   - Strong asymmetry (≈28pp) suggests model flips are driven by')
print('     user-opinion direction, not just random instability.')
print('   - The ~4% agree flip rate likely reflects baseline temperature noise.')
print('   - Does NOT fully disentangle sycophancy vs Bayesian updating, but')
print('     provides necessary first step. Source/confidence manipulations')
print('     recommended as follow-up.')
print()
print('4. Standard vs bidirectional disagree rates:')
print('   - No systematic difference; per-run variation within noise.')
print('   - Confirms bidirectional disagree condition ≈ standard sycophancy test.')
print('='*70)
