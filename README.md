# Advanced MPC & Learning MPC Lab

A compact but mathematically serious educational repository on **constrained optimal control, Model Predictive Control (MPC), invariant-set geometry, Learning MPC (LMPC), autonomous-driving examples, and the connection to dynamic programming / reinforcement learning**.

The repository is designed around one principle:

> **Do not treat the optimizer as a black box.** Understand the model, decision variables, objective, constraints, feasibility argument, stability argument, and the assumptions behind every learned component.

## What is inside?

| Notebook | Main topic | Key mathematical objects |
|---|---|---|
| `00_roadmap_and_math_primer.ipynb` | roadmap + stability refresher | Schur stability, Lyapunov functions, invariant sets |
| `01_convex_optimization_and_cftoc.ipynb` | optimization + finite-time optimal control | convexity, QP, KKT, condensing |
| `02_polytopes_reachability_invariance.ipynb` | constraint geometry | H/V polytopes, predecessor sets, controlled invariant sets |
| `03_mpc_stability_and_terminal_sets.ipynb` | classical MPC | DARE, LQR terminal controller, recursive feasibility, Lyapunov decrease |
| `04_collision_avoidance_cftoc.ipynb` | autonomous-driving CFTOC | friction-limited actuation, logical strategies, feasibility maps |
| `05_learning_mpc_double_integrator.ipynb` | Learning MPC | sampled/convex safe set, barycentric coordinates, empirical value function |
| `06_frenet_minimum_time_racing.ipynb` | racing / Frenet coordinates | spatial dynamics, curvature-limited speed, minimum-time intuition |
| `07_data_driven_local_models_for_racing.ipynb` | learning local dynamics | weighted local affine regression, LTV prediction models |
| `08_dynamic_programming_to_rl.ipynb` | DP → RL | Bellman recursion, value iteration, Q-values |
| `09_capstone_questions_and_extensions.ipynb` | synthesis | technical-defense questions + research extensions |

The `src/advanced_mpc_lmpc/` package contains original, small-scale numerical tools used by the notebooks. They favor transparency over production-level solver performance.

---

## Learning path

The notebooks are intentionally ordered. A useful mental map is

```text
constrained optimization
        ↓
finite-time optimal control
        ↓
reachability / invariant sets
        ↓
receding-horizon MPC
        ↓
terminal set + terminal cost
        ↓
recursive feasibility + stability
        ↓
Learning MPC
        ↓
safe sets + learned value from repeated data
        ↓
data-driven local models / racing extensions
        ↓
connection to Bellman values and RL
```

## Installation

Python 3.10+ is recommended.

```bash
python -m venv .venv
# Linux/macOS
source .venv/bin/activate
# Windows PowerShell
# .venv\Scripts\Activate.ps1

pip install -r requirements.txt
pip install -e .
jupyter lab
```

No commercial optimizer is required. The examples use SciPy-based optimization so the repository is easy to run. For high-rate or large-scale MPC, a dedicated QP/NLP solver is strongly preferable.

## Quick smoke test

```bash
python -m unittest discover -s tests -v
```

To execute every notebook non-interactively:

```bash
python scripts/run_all_notebooks.py
```

## A first MPC example

```python
import numpy as np
from advanced_mpc_lmpc.linear import dlqr
from advanced_mpc_lmpc.mpc import LinearMPC, simulate_receding_horizon

A = np.array([[1., 1.], [0., 1.]])
B = np.array([[0.5], [1.]])
Q = np.diag([1., 0.2])
R = np.array([[0.15]])

K, P, eig = dlqr(A, B, Q, R)

mpc = LinearMPC(
    A, B, Q, R, N=8, P=P,
    x_lower=[-5, -3], x_upper=[5, 3],
    u_lower=[-1], u_upper=[1],
)

X, U, V = simulate_receding_horizon(mpc, x0=[-4, 0])
```

## Mathematical scope

The core classical MPC design follows the standard structure

$$
\begin{aligned}
V_N(x)=\min_{u_0,\ldots,u_{N-1}}\quad &
\sum_{k=0}^{N-1}\ell(x_k,u_k)+V_f(x_N)\\
\text{s.t.}\quad &x_{k+1}=Ax_k+Bu_k,\\
&x_k\in\mathcal X,\quad u_k\in\mathcal U,\\
&x_N\in\mathcal X_f.
\end{aligned}
$$

The LMPC notebook replaces hand-designed terminal information by successful closed-loop data. If stored safe states are arranged as columns of `S`, the terminal point is represented using barycentric weights

$$
x_N=S\lambda,
\qquad
\lambda\ge0,
\qquad
\mathbf 1^\top\lambda=1,
$$

and the terminal value is approximated by the same convex combination of realized costs-to-go.

## What this repository does *not* claim

This is an educational laboratory, not a certified autonomous-driving stack.

- The collision example uses a low-order longitudinal model.
- The racing speed planner is a reduced spatial model, not a full tire-force raceline optimizer.
- The local vehicle model notebook demonstrates data-driven linearization; it is not a complete online racing LMPC implementation.
- The small QP solver wrapper is chosen for readability, not hard real-time guarantees.

Those limitations are discussed explicitly because understanding them is part of control engineering.

## Recommended way to study

For every notebook, write down the answer to these four questions before running code:

1. **Model:** What are the state, input, and dynamics?
2. **Optimization:** What exactly are the decision variables, objective, and constraints?
3. **Guarantees:** What supports feasibility, stability, or safety?
4. **Failure modes:** Which assumptions are likely to fail on a physical platform?

The file [`docs/interview_oral_defense.md`](docs/interview_oral_defense.md) contains a concise technical-defense checklist.

## References and acknowledgements

The exposition is original, while the subject progression and several motivating examples were informed by advanced control teaching material on optimization, MPC, invariant sets, Learning MPC, dynamic programming, and racing. See [`docs/references.md`](docs/references.md) for full acknowledgement and recommended reading.

## License

Original code and repository text: MIT License. See `LICENSE`.

External references and course materials retain their own licenses and copyrights; this repository does not redistribute those source files.


## VS Code / Jupyter rendering

The notebooks use standard Markdown math delimiters:

- inline mathematics: `$x_{k+1}=Ax_k+Bu_k$`
- display mathematics:

```text
$$
x_{k+1}=Ax_k+Bu_k
$$
```

For the best experience in VS Code:

1. Install the **Python** extension from Microsoft.
2. Install the **Jupyter** extension from Microsoft.
3. Open an `.ipynb` file as a Jupyter Notebook, not as raw JSON.
4. Select the project Python environment/kernel.
5. Run a Markdown cell or press `Esc` after editing it to render the mathematics.

The same `$...$` / `$$...$$` syntax also renders on GitHub.
