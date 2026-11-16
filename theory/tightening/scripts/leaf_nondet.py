import random, numpy as np, randbats as rb, poke_engine as pe, nash, pkbattle as pb, collections, time, sys
rng=random.Random(int(sys.argv[1]) if len(sys.argv)>1 else 0)
rows=[]
for trial in range(12):
    if trial%2==0:
        t=rb.sample_team(rng,n=2); st=pe.State(side_one=pe.Side(pokemon=[rb.to_pokemon(t[0])]), side_two=pe.Side(pokemon=[rb.to_pokemon(t[1])])); kind='1v1'
    else:
        st=rb.random_state(rng); kind='6v6'
        for _ in range(rng.randint(0,6)):   # advance a few random turns
            o1,o2=pb.options(st); st=pb.step(st,rng.choice(o1),rng.choice(o2),rng)
            if pb.alive(st.side_one)==0 or pb.alive(st.side_two)==0: break
    s1,s2=pb.options(st)
    kids=[]
    for a in s1:
        for b in s2:
            for br in pe.generate_instructions(st,a,b):
                kids.append(st.apply_instructions(br))
    rng.shuffle(kids); kids=kids[:40]
    for c in kids:
        vals=[nash._leaf(c,'mid') for _ in range(5)]
        r=pe.id(c,0); M=np.array([np.nan if v is None else v for v in r.matrix],float)
        rng_=np.nanmax(M)-np.nanmin(M) if np.isfinite(M).any() else np.nan
        rows.append((kind, len(set(np.round(vals,6))), max(vals)-min(vals), rng_, pb.alive(c.side_one)+pb.alive(c.side_two)))
rows_by={}
for r in rows: rows_by.setdefault(r[0],[]).append(r)
for k,g in rows_by.items():
    nd=[r for r in g if r[1]>1]
    sp=[r[2] for r in nd]; d1=[r[3] for r in g if r[3]==r[3]]
    print(k,'children',len(g),'nondeterministic',len(nd),'(%.1f%%)'%(100*len(nd)/len(g)),'spread median/max among those', (np.median(sp) if sp else 0), (max(sp) if sp else 0), '| depth-1 range median', np.median(d1))
    print('  nondet spreads:', sorted(np.round(sp,1))[-10:])
