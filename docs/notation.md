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
| `lambda` | barycentric weights for safe-set interpolation |
| `V_N` | finite-horizon optimal value |
| `Q^j` | LMPC empirical terminal value approximation |
| `kappa(s)` | track centerline curvature |
| `e_y,e_psi` | Frenet lateral and heading errors |
