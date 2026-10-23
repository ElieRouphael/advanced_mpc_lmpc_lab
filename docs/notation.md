# Notation

| Symbol | Meaning |
|---|---|
| `x_k` | state at discrete time `k` |
| `u_k` | control input |
| `A,B` | LTI state/input matrices |
| `N` | prediction horizon |
| `Q,R` | stage-cost state/input weights |
| `P` | terminal quadratic weight / Riccati solution |
| `K` | feedback gain with convention `u=-Kx` |
| `X,U` | state and input constraint sets |
| `X_f` | terminal set |
| `Pre(S)` | one-step predecessor of set `S` |
| `O_inf` | maximal positively invariant set under fixed dynamics/controller |
| `C_inf` | maximal control-invariant set |
| `SS^j` | sampled safe set through iteration `j` |
| `CS^j` | convex safe set `conv(SS^j)` |
| `S` | matrix whose rows are the stored safe states, so `x_N = S' lambda` |
| `lambda` | barycentric weights for safe-set interpolation |
| `V_N` | finite-horizon optimal value |
| `Q^j` | LMPC empirical terminal value approximation |
| `kappa(s)` | track centerline curvature |
| `s` | arc length along the track centerline |
| `e_y,e_psi` | Frenet lateral and heading errors (`e_y > 0` to the left) |
| `z = v^2` | speed squared, the decision variable of the spatial speed planner |
