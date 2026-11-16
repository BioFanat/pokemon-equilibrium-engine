import json, numpy as np, sys
d = json.load(open(sys.argv[1]))
for tag, streams in d.items():
    if not isinstance(streams, list): continue
    lhs = np.array([s['lhs'] for s in streams])
    def bnd(key):
        per = np.array([sum(1/(1-v) for v in s[key].values()) for s in streams])
        # bound from stream-averaged meanBC
        styles = streams[0][key].keys()
        avgbc = {k: np.mean([s[key][k] for s in streams]) for k in styles}
        pooled = sum(1/(1-v) for v in avgbc.values())
        return per, pooled
    pa, poola = bnd('meanBC'); pn, pooln = bnd('meanBC_nf')
    rp = np.array([s['realprod'] for s in streams])
    print(tag, 'LHS mean %.2f' % lhs.mean(), 'per-stream LHS', np.round(lhs,1))
    print('  iid-all: mean of per-stream %.2f, from avg BC %.2f; holds %d/10' % (pa.mean(), poola, (lhs<=pa).sum()))
    print('  iid-nf : mean of per-stream %.2f, from avg BC %.2f; holds %d/10' % (pn.mean(), pooln, (lhs<=pn).sum()))
    print('  realprod mean %.2f holds %d/10' % (rp.mean(), (lhs<=rp).sum()))
    ll = {k: np.mean([s['ll'][k] for s in streams]) for k in streams[0]['ll']}
    lln = {k: np.mean([s['ll_nf'][k] for s in streams]) for k in streams[0]['ll_nf']}
    print('  ll all', {k: round(v,4) for k,v in ll.items()})
    print('  ll nf ', {k: round(v,4) for k,v in lln.items()})
    if 'ent_true_nf' in streams[0]:
        print('  ent_true_nf mean %.4f' % np.mean([s['ent_true_nf'] for s in streams]))
