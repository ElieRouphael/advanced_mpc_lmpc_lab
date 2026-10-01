# References and Acknowledgements

This repository contains original explanatory text and original small-scale Python implementations. Its organization and motivating examples were informed by advanced teaching material supplied with the project, including:

- Luigi Glielmo, **L-MPC: Learning Model Predictive Control — Course Introduction**, Università di Napoli Federico II / Grenoble INP, 2025–2026.
- Luigi Glielmo, **An Introduction to Optimization**, December 2025.
- Luigi Glielmo, **MPC — An Introduction**, January 2026.
- Luigi Glielmo, **Polytopes, Reachability, and Invariant Sets**, January 2026.
- Luigi Glielmo, **Learning Safe Sets and Value Functions from Data — Mathematical foundations for Learning MPC**, January 2026.
- Luigi Glielmo, **From Dynamic Programming to Reinforcement Learning**, December 2025.
- Luigi Glielmo, **Examples of Constrained Finite-Time Optimal Control (CFTOC)**, January 2026.
- **Raceline Computation in Frenet Coordinates — Spatial vehicle model and minimum-time optimal control**, January 2026.
- Olivier Sename, **Background notes on stability / Lyapunov stability**, Grenoble INP / GIPSA-lab.

Some supplied course material explicitly credits Ugo Rosolia, Charlott Vallon, Francesco Borrelli, Andy Packard, Hansung Kim, and Roya Firoozi for earlier notes/examples. Readers interested in research-grade LMPC should consult the primary literature by Rosolia and Borrelli and standard MPC texts such as Borrelli, Bemporad & Morari, *Predictive Control for Linear and Hybrid Systems*.

## Licensing note

The supplied source materials retain their original copyright/licensing conditions. They are **not redistributed in this repository**. The code and prose in this repository are newly written for this educational implementation and are licensed under the repository's MIT license.

## Additional references used for explanatory extensions

The repository also adds derivations and implementation details beyond the supplied teaching
material. The following standard references are useful for those extensions:

- Stephen Boyd and Lieven Vandenberghe, *Convex Optimization*, Cambridge University Press, 2004 — convex quadratic programming, duality, and numerical formulation.
- Francesco Borrelli, Alberto Bemporad, and Manfred Morari, *Predictive Control for Linear and Hybrid Systems*, Cambridge University Press, 2017 — constrained MPC, hybrid/logical models, and mixed-integer predictive control.
- Alberto Bemporad and Manfred Morari, “Control of systems integrating logic, dynamics, and constraints,” *Automatica*, 35(3), 1999 — mixed logical dynamical systems and mixed-integer optimization in control.
- Rajesh Rajamani, *Vehicle Dynamics and Control*, 2nd ed., Springer, 2012 — vehicle longitudinal/lateral dynamics and tire-road force limitations.
- James B. Rawlings, David Q. Mayne, and Moritz M. Diehl, *Model Predictive Control: Theory, Computation, and Design*, 2nd ed., Nob Hill Publishing, 2022 — MPC theory and computation.

## Learning Model Predictive Control

- Ugo Rosolia and Francesco Borrelli, “Learning Model Predictive Control for Iterative Tasks. A Data-Driven Control Framework,” *IEEE Transactions on Automatic Control*, 63(7), 1883–1896, 2018. DOI: 10.1109/TAC.2017.2753460.
- Francesco Borrelli, Alberto Bemporad, and Manfred Morari, *Predictive Control for Linear and Hybrid Systems*, Cambridge University Press, 2017.
- James B. Rawlings, David Q. Mayne, and Moritz M. Diehl, *Model Predictive Control: Theory, Computation, and Design*, 2nd ed., Nob Hill Publishing, 2022.
