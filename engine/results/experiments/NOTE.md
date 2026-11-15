# E1-E3 (run 2026-11-13) used the buggy matrix

E1, E2, E3 here were played with the original `nash.engine_matrix`: pe.id's root search uses
alpha-beta pruning and returns 55-80% of cells as NaN ("pruned"), which were set to 0. Every
`nash:*` and `safest:*` agent in these files therefore solved a largely invented game (side two
worst affected). Their spec strings lack the `:legacy` suffix only because the flag did not exist
yet; to reproduce them, append `:legacy`. `E1fix` reruns E1 on the unpruned depth-2 matrix.
Valid from these files: random-vs-random sanity, the runner itself; E3 only as "solver choice
does not matter on that (wrong) game".
