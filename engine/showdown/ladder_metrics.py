"""Descriptive ladder outcome metrics; Elo change is the primary batch endpoint."""
from collections import Counter


def summarize_games(rows, target):
    unique = {r['battle_tag']:r for r in rows}
    rows = list(unique.values())
    rated = [r for r in rows if r.get('rating')]
    start = rated[0]['rating']['before'] if rated else None
    end = rated[-1]['rating']['after'] if rated else None
    trajectory = ([start] + [r['rating']['after'] for r in rated]) if rated else []
    results = Counter(r['result'] for r in rows)
    n = len(rows)
    return dict(primary_metric='ladder_elo_change', target=target, completed=n,
                complete=n>=target, results=dict(results), elo_start=start, elo_end=end,
                elo_change=end-start if rated else None,
                elo_min=min(trajectory) if rated else None, elo_max=max(trajectory) if rated else None,
                rated_games=len(rated), win_rate=results['win']/n if n else None,
                comparison_note='Descriptive public ladder result; different starting ratings and opponents prevent a controlled causal comparison')
