# Mathematical Notes

## 1. Condensed LTI prediction

For

$$
x_{k+1}=Ax_k+Bu_k,
$$

we have

$$
\begin{aligned}
x_1 &= Ax_0+Bu_0,\\
x_2 &= A^2x_0+ABu_0+Bu_1,\\
&\vdots\\
x_N &= A^Nx_0+\sum_{j=0}^{N-1}A^{N-1-j}Bu_j.
\end{aligned}
$$

Stacking the states yields

$$
X=\mathcal A x_0+\mathcal B U.
$$

This converts all polyhedral state constraints into affine constraints in the stacked input `U`.

## 2. Quadratic cost condensing

With

$$
J=X^\top\bar QX+U^\top\bar RU,
$$

substitute `X = Abar x0 + Bbar U`:

$$
J=U^\top(\mathcal B^\top\bar Q\mathcal B+\bar R)U
+2x_0^\top\mathcal A^\top\bar Q\mathcal BU+\text{constant}.
$$

Thus a linear MPC problem with quadratic objective and polyhedral constraints is a convex QP when `Q >= 0` and `R > 0`.

## 3. Discrete LQR and the sign convention

The infinite-horizon problem

$$
\min \sum_{k=0}^{\infty}(x_k^\top Qx_k+u_k^\top Ru_k)
$$

gives

$$
u_k=-Kx_k,
\quad
K=(R+B^\top PB)^{-1}B^\top PA,
$$

with `P` solving the DARE

$$
P=A^\top PA-A^\top PB(R+B^\top PB)^{-1}B^\top PA+Q.
$$

Therefore the closed loop is `A-BK`. Discrete stability requires `rho(A-BK)<1`.

## 4. Recursive feasibility

Assume the optimal input sequence at time `k` is

$$
(u_0^\star,\ldots,u_{N-1}^\star)
$$

and its terminal state belongs to `X_f`. If `X_f` is invariant under a terminal controller `kappa_f`, the candidate

$$
(u_1^\star,\ldots,u_{N-1}^\star,\kappa_f(x_N^\star))
$$

is feasible one sample later. The proof is constructive: **shift the old plan, append the terminal action**.

## 5. MPC stability through the value function

If

$$
V_f(f(x,\kappa_f(x)))-V_f(x)\le -\ell(x,\kappa_f(x)),
$$

then evaluating the next finite-horizon problem on the shifted candidate yields

$$
V_N(x^+)-V_N(x)\le-\ell(x,u_0^\star).
$$

Hence the finite-horizon optimal value is a Lyapunov function under the usual positivity assumptions.

## 6. Predecessor and invariant sets

For `x+ = Ax+Bu`,

$$
\mathrm{Pre}(S)=\{x:\exists u\in U,\ Ax+Bu\in S\}.
$$

A controlled-invariant set `C` satisfies `C subset Pre(C)` (together with state constraints). Fixed-point iteration gives a maximal controlled-invariant subset of a constraint set in many linear/polyhedral settings.

## 7. LMPC terminal ingredients

For repeated successful trajectories, define a sampled safe set

$$
SS^j=\bigcup_{i=0}^j\bigcup_{t=0}^{T_i}\{x_t^i\}.
$$

For linear systems with convex constraints one can use

$$
CS^j=\operatorname{conv}(SS^j).
$$

If `S` contains stored safe states as columns, the terminal equality is

$$
x_N=S\lambda,\qquad \lambda\ge0,\qquad \mathbf 1^\top\lambda=1.
$$

If `q` contains realized cost-to-go samples, the barycentric terminal value is

$$
V_f(x_N)\approx q^\top\lambda.
$$

The LMPC QP jointly optimizes the finite-horizon inputs and the barycentric coordinates.

## 8. Why local convexification is safer than global convexification in racing

For linear dynamics and convex constraints, convex combinations preserve feasibility under matching convex combinations of inputs. Nonlinear dynamics and nonconvex track geometry do not generally preserve this property. A practical racing implementation therefore uses local subsets, local models, and additional validation/robustification.

## 9. Spatial speed planning

With `ds/dt=v` and `dv/dt=a`,

$$
\frac{d(v^2)}{ds}=2a.
$$

Defining `z=v^2` gives a simple spatial dynamic relation. A lateral acceleration cap

$$
v^2|\kappa|\le \mu g
$$

creates a curvature-dependent pointwise speed limit. Forward/backward passes enforce longitudinal acceleration/braking limits.

## 10. Bellman and terminal value

Dynamic programming writes

$$
V^\star(x)=\min_u\{\ell(x,u)+\gamma V^\star(f(x,u))\}.
$$

MPC truncates the horizon and approximates the unknown tail with `V_f`. LMPC estimates selected tail values from repeated closed-loop data. This is the cleanest conceptual connection between MPC and value-based learning.
