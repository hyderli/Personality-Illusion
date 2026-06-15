import csv
from collections import defaultdict
from math import sqrt

rows = list(csv.DictReader(open('reasoning_extension/results/sycophancy/deepseek-r1-distill-qwen-7b_sycophancy_items.csv')))
print(f'Total rows: {len(rows)}')
print(f'Personas: {sorted(set(r["persona_name"] for r in rows))}')

per_persona = defaultdict(lambda: {'flips': 0, 'valid': 0, 'total': 0})
for r in rows:
    p = r['persona_name']
    per_persona[p]['total'] += 1
    flipped = r.get('flipped')
    if flipped in ('True', 'False'):
        per_persona[p]['valid'] += 1
        if flipped == 'True':
            per_persona[p]['flips'] += 1

print('\n=== Per-persona flip rates ===')
for p, c in sorted(per_persona.items()):
    rate = c['flips'] / c['valid'] * 100 if c['valid'] else 0
    print(f'  {p:10s}: {c["flips"]}/{c["valid"]} valid = {rate:.2f}%  (total rows={c["total"]})')

# 3x2 Chi-square
personas = sorted(per_persona.keys())
obs = []
for p in personas:
    c = per_persona[p]
    obs.append([c['flips'], c['valid'] - c['flips']])

print('\n=== 3x2 Contingency Table (flipped | not flipped) ===')
for i, p in enumerate(personas):
    print(f'  {p:10s}: {obs[i][0]:>3} | {obs[i][1]:>3}  (n={sum(obs[i])})')

def chi_square_3x2(obs):
    total = sum(sum(row) for row in obs)
    row_sums = [sum(row) for row in obs]
    col_sums = [sum(obs[i][j] for i in range(len(obs))) for j in range(len(obs[0]))]
    chi2 = 0.0
    for i in range(len(obs)):
        for j in range(len(obs[0])):
            expected = row_sums[i] * col_sums[j] / total
            if expected > 0:
                chi2 += (obs[i][j] - expected) ** 2 / expected
    return chi2

chi2 = chi_square_3x2(obs)
print(f'\nPearson chi-square(2) = {chi2:.4f}')
print('Critical values (df=2):  χ²(0.05)=5.991, χ²(0.01)=9.210, χ²(0.001)=13.816')
if chi2 > 13.816:
    print('Result: p < 0.001  ***')
elif chi2 > 9.210:
    print('Result: p < 0.01   **')
elif chi2 > 5.991:
    print('Result: p < 0.05   *')
else:
    print('Result: p > 0.05   (not significant)')

# Pairwise Z-tests
def two_prop_z(a_flips, a_total, b_flips, b_total):
    p1 = a_flips / a_total
    p2 = b_flips / b_total
    p_pool = (a_flips + b_flips) / (a_total + b_total)
    se = sqrt(p_pool * (1 - p_pool) * (1/a_total + 1/b_total))
    if se == 0:
        return 0.0
    return (p1 - p2) / se

print('\n=== Pairwise Z-tests for proportion difference ===')
for i in range(len(personas)):
    for j in range(i+1, len(personas)):
        p1, p2 = personas[i], personas[j]
        c1, c2 = per_persona[p1], per_persona[p2]
        z = two_prop_z(c1['flips'], c1['valid'], c2['flips'], c2['valid'])
        sig = ''
        if abs(z) > 2.58: sig = ' **'
        elif abs(z) > 1.96: sig = ' *'
        print(f'  {p1} vs {p2}: z={z:+.3f}{sig}')
