# Technical Oral-Defense Guide

This is a compact checklist for explaining the repository at a whiteboard.

## MPC in 90 seconds

> I start from a discrete dynamic model and optimize a finite sequence of future inputs. The objective balances state/tracking error and control effort while state/input constraints are enforced explicitly. At each sample I solve the finite-horizon problem, apply only the first input, measure/estimate the new state, and solve again. For formal recursive feasibility and stability, I can use a terminal invariant set, a local terminal controller, and a terminal Lyapunov cost.

## Why terminal ingredients?

- `X_f`: gives a region from which a known admissible tail exists.
- `kappa_f`: supplies the appended input in the shifted-sequence feasibility proof.
- `V_f`: upper-bounds / approximates the tail and supports cost decrease.

## Learning MPC in 90 seconds

> LMPC targets repeated tasks. A successful iteration gives state/input trajectories that are known to complete the task, plus realized costs-to-go. Those states form a sampled safe set. For linear systems with convex constraints I can convexify it and express the terminal state as a barycentric combination of stored safe states. I use the same barycentric weights to interpolate the stored costs-to-go. As more successful iterations are collected, the terminal safe set/value information improves and the iteration cost can decrease under the standard assumptions.

## Collision avoidance

Be ready to explain:

- state `[position, speed]`, input `acceleration`;
- friction limit `|a| <= mu g`;
- disjoint safe strategies (yield/pass);
- why the logical choice can be encoded by a binary variable;
- why lower friction reduces the feasible set;
- why “remaining before the intersection” is not identical to “zero terminal speed.”

## Invariance

**Positive invariant:** fixed closed-loop dynamics keep every point inside.

**Control invariant:** for every point there exists an admissible control that keeps the next point inside.

Do not interchange these two definitions.

## LQR sign/stability trap

With `u=-Kx`,

```text
Acl = A - B K
```

and for discrete time stability means

```text
rho(Acl) < 1
```

not `< 0`.

## Racing / local model learning

A local affine model has form

$$
x^+\approx A_qx+B_qu+c_q.
$$

It is fitted from nearby transitions, typically with weighted least squares and regularization. It is a local predictor inside the optimizer, not a statement that the nonlinear vehicle is globally linear.

## Limitations to volunteer before being cornered

- reduced-order simulation is not hardware validation;
- deterministic LMPC guarantees do not automatically survive stochastic/nonlinear uncertainty;
- global convex hulls can be unsafe for nonlinear/nonconvex systems;
- production MPC uses dedicated real-time solvers and explicit fallback behavior;
- estimator uncertainty and delays should enter the prediction/control architecture on a real platform.
