# Key Ideas in Brief

A one-page summary of the main ideas in the notebooks, useful as a refresher after working through them.

## MPC in one paragraph

Start from a discrete-time model and optimize a finite sequence of future inputs. The objective balances state or tracking error against control effort, and state and input constraints are enforced explicitly. At each sample, solve the finite-horizon problem, apply only the first input, measure or estimate the new state, and solve again. Recursive feasibility and stability come from the terminal ingredients: a terminal invariant set, a local terminal controller, and a terminal cost that acts as a local Lyapunov function (notebook 03).

## Why terminal ingredients?

- `X_f`: a region from which a known admissible continuation exists.
- `kappa_f`: the local controller that supplies the appended input in the shifted-sequence feasibility argument.
- `V_f`: bounds the cost of the tail; it gives the decrease of the optimal value along the closed loop and, near the origin, the upper bound `V_N(x) <= V_f(x)` that the Lyapunov argument also needs.

## Learning MPC in one paragraph

LMPC targets repeated tasks. A successful iteration gives state and input trajectories known to complete the task, together with their realized costs-to-go. The visited states form a sampled safe set. For linear systems with convex constraints, its convex hull is also a safe set: with the stored states as the rows of `S`, the terminal state is written `x_N = S' lambda` with `lambda >= 0` and `1' lambda = 1`, and the same barycentric weights interpolate the stored costs-to-go into a terminal cost. As successful iterations accumulate, the safe set grows, the terminal cost improves, and the iteration cost is non-increasing under the standard assumptions (notebook 05).

## Collision avoidance

The main points of notebook 04:

- state `[position, speed]`, input `acceleration`;
- friction limit `|a| <= mu g`;
- two disjoint safe strategies (yield or pass), so the safe set is nonconvex;
- the logical choice between them can be encoded by a binary variable (mixed-integer formulation) or by solving one convex problem per strategy;
- lower friction shrinks the feasible set;
- "stopping before the intersection" is not the same constraint as "zero terminal speed".

## Invariance

**Positive invariant:** under fixed closed-loop dynamics, every point of the set stays in the set.

**Control invariant:** for every point of the set there exists an admissible input that keeps the next state in the set.

The two definitions are easy to mix up; the first is about a given controller, the second about the existence of one.

## LQR sign convention and stability

With `u = -Kx`, the closed loop is

```text
Acl = A - B K
```

and discrete-time stability means

```text
rho(Acl) < 1
```

(spectral radius below one), not eigenvalues with negative real part.

## Racing and local model learning

A local affine model has the form

$$
x^+\approx A_qx+B_qu+c_q.
$$

It is fitted from transitions recorded near the query point, with kernel-weighted least squares and regularization (notebook 07). It is a local predictor used inside the optimizer, not a claim that the nonlinear vehicle is globally linear.

## Dynamic programming and reinforcement learning

MPC is a truncated dynamic-programming problem: the terminal cost stands in for the unknown infinite-horizon value function. LMPC estimates that value from realized costs of successful iterations, and value-based reinforcement learning estimates it from sampled transitions (notebook 08).

## Known limitations

- Reduced-order simulation is not hardware validation.
- Deterministic LMPC guarantees do not automatically survive stochastic or nonlinear uncertainty.
- Global convex hulls of safe states can be unsafe for nonlinear dynamics or nonconvex constraints.
- Production MPC uses dedicated real-time solvers and explicit fallback behaviour.
- Estimation uncertainty and delays should enter the prediction and control architecture on a real platform.
