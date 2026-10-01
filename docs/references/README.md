# Downloaded reference material

This folder contains full-text copies of the published academic references cited across
`notebooks/00`–`09`, collected for offline reading. Every file here was fetched from a copy
the author/publisher has explicitly made freely available on the open web (author homepage,
institutional repository, or arXiv) — nothing here was sourced from a paywalled or pirated
mirror.

See `docs/references.md` for the repository's consolidated citation list; this folder is the
downloadable subset of it, with full per-notebook provenance below.

## Books

| File | Citation | Cited in | Source |
|---|---|---|---|
| `boyd_vandenberghe_2004_convex_optimization.pdf` | S. Boyd, L. Vandenberghe, *Convex Optimization*, Cambridge University Press, 2004 | 01, 04 | [web.stanford.edu/~boyd/cvxbook](https://web.stanford.edu/~boyd/cvxbook/) — Cambridge UP "kindly agreed to allow us to keep the book available on the web" |
| `rawlings_mayne_diehl_2022_mpc_theory_computation_design_2e.pdf` | J. B. Rawlings, D. Q. Mayne, M. M. Diehl, *Model Predictive Control: Theory, Computation, and Design*, 2nd ed., Nob Hill Publishing, 2022 | 04, 05 | [sites.engineering.ucsb.edu/~jbraw/mpc](https://sites.engineering.ucsb.edu/~jbraw/mpc/) — author's own faculty page |
| `hastie_tibshirani_friedman_2009_elements_of_statistical_learning_2e.pdf` | T. Hastie, R. Tibshirani, J. Friedman, *The Elements of Statistical Learning*, 2nd ed., Springer, 2009 | 07 | [hastie.su.domains/ElemStatLearn](https://hastie.su.domains/ElemStatLearn/) — Springer "agreed to allow us to keep the book available on the web" |
| `bertsekas_dp_and_optimal_control_vol1_short_view.pdf` | D. P. Bertsekas, *Dynamic Programming and Optimal Control* | 01 | [web.mit.edu/dimitrib](https://www.mit.edu/~dimitrib/dpbook.html) — author's freely-posted "short view" (front matter / condensed edition, **not** the full current 4th-edition book, which is sold commercially) |

## Papers

| File | Citation | Cited in | Source |
|---|---|---|---|
| `rosolia_borrelli_2016_lmpc_iterative_tasks.pdf` | U. Rosolia, F. Borrelli, "Learning Model Predictive Control for Iterative Tasks. A Data-Driven Control Framework," *IEEE Transactions on Automatic Control*, 63(7), 2018 | 05 | [arXiv:1609.01387](https://arxiv.org/abs/1609.01387) |
| `rosolia_carvalho_borrelli_2016_autonomous_racing_lmpc.pdf` | U. Rosolia, A. Carvalho, F. Borrelli, "Autonomous Racing using Learning Model Predictive Control," *American Control Conference*, 2017 | 07 | [arXiv:1610.06534](https://arxiv.org/abs/1610.06534) |
| `rosolia_borrelli_2019_learning_to_autonomously_race.pdf` | U. Rosolia, F. Borrelli, "Learning How to Autonomously Race a Car: A Predictive Control Approach," *IEEE Transactions on Control Systems Technology*, 28(6), 2020 | 07 | [arXiv:1901.08184](https://arxiv.org/abs/1901.08184) |
| `perantoni_limebeer_2014_optimal_control_f1_variable_parameters.pdf` | G. Perantoni, D. J. N. Limebeer, "Optimal control for a Formula One car with variable parameters," *Vehicle System Dynamics*, 52(5), 2014 | 06 | [Oxford University Research Archive](https://ora.ox.ac.uk/objects/uuid:ce1a7106-0a2c-41af-8449-41541220809f) (institutional open-access repository) |

## Cited but not included here (no legitimate free copy found)

These remain cited by title/publisher in the notebooks and in `docs/references.md`, but no
official free distribution exists, so they are not duplicated in this folder:

- F. Borrelli, A. Bemporad, M. Morari, *Predictive Control for Linear and Hybrid Systems*, Cambridge University Press, 2017 — cited in 01, 04, 05, 06
- R. Rajamani, *Vehicle Dynamics and Control*, 2nd ed., Springer, 2012 — cited in 04, 06, 07
- H. B. Pacejka, *Tire and Vehicle Dynamics*, 3rd ed., Butterworth-Heinemann, 2012 — cited in 06
- L. Ljung, *System Identification: Theory for the User*, 2nd ed., Prentice Hall, 1999 — cited in 07
- A. Bemporad, M. Morari, "Control of systems integrating logic, dynamics, and constraints," *Automatica*, 35(3), 1999 — cited in 04 (Elsevier-paywalled; no open-access copy found)

## Deliberately excluded: the course's own source material

The notebooks are built from proprietary course slides and workshop notebooks that
`NOTICE.md` and `docs/references.md` explicitly state are **not redistributed** in this
repository — their authors retain all rights. This folder does not attempt to source or
include copies of:

- Luigi Glielmo's course slide sets (L-MPC course introduction, optimization, MPC, polytopes/
  invariant sets, safe sets and value functions, DP-to-RL, CFTOC examples)
- the *Raceline Computation in Frenet Coordinates* slides
- Olivier Sename's stability/Lyapunov notes
- the double-integrator LMPC workshop, vehicle–bike CFTOC workshop, and autonomous-racing
  LMPC workshop notebooks referenced as the basis for notebooks 04, 05, and 07

Respecting that boundary was intentional, not an oversight.
