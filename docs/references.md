# References and Acknowledgements

## Acknowledgements

The text and code in this repository are original. The choice and order of topics, and several motivating examples (the vehicle–bike collision scenario, the repeated double-integrator task, and the racing setting), were informed by the 2025–2026 course *L-MPC: Learning Model Predictive Control* taught by Luigi Glielmo (Università di Napoli Federico II, with the Université Franco-Italienne and Grenoble INP / ENSE3), and by Olivier Sename's background notes on Lyapunov stability (Grenoble INP / GIPSA-lab). The course material credits earlier notes and examples by Ugo Rosolia, Charlott Vallon, Francesco Borrelli, Andy Packard, Hansung Kim and Roya Firoozi.

That material keeps its own copyright and licence and is not redistributed here. Everything in this repository is licensed under the MIT license (see `LICENSE`).

## References

Links point to copies made freely available by the authors or publishers, where such copies exist. No reference material is redistributed in this repository.

### Optimization and MPC

- S. Boyd and L. Vandenberghe, *Convex Optimization*, Cambridge University Press, 2004. Convex sets and functions, quadratic programming, duality and KKT conditions (notebooks 01, 04). Free copy: [web.stanford.edu/~boyd/cvxbook](https://web.stanford.edu/~boyd/cvxbook/).
- F. Borrelli, A. Bemporad and M. Morari, *Predictive Control for Linear and Hybrid Systems*, Cambridge University Press, 2017. Constrained finite-time optimal control, invariant sets, MPC stability, hybrid and mixed-integer MPC (notebooks 01–06, 09).
- J. B. Rawlings, D. Q. Mayne and M. M. Diehl, *Model Predictive Control: Theory, Computation, and Design*, 2nd ed., Nob Hill Publishing, 2022. MPC theory and computation (notebooks 03–05, 09). Free copy: [sites.engineering.ucsb.edu/~jbraw/mpc](https://sites.engineering.ucsb.edu/~jbraw/mpc/).
- D. Q. Mayne, J. B. Rawlings, C. V. Rao and P. O. M. Scokaert, "Constrained model predictive control: Stability and optimality", *Automatica*, 36(6), 2000 (notebooks 03, 09).
- A. Bemporad and M. Morari, "Control of systems integrating logic, dynamics, and constraints", *Automatica*, 35(3), 1999. Mixed logical dynamical systems (notebook 04).

### Learning MPC and racing

- U. Rosolia and F. Borrelli, "Learning Model Predictive Control for Iterative Tasks. A Data-Driven Control Framework", *IEEE Transactions on Automatic Control*, 63(7), 1883–1896, 2018. DOI: 10.1109/TAC.2017.2753460 (notebooks 05, 08, 09). Preprint: [arXiv:1609.01387](https://arxiv.org/abs/1609.01387).
- U. Rosolia, A. Carvalho and F. Borrelli, "Autonomous Racing using Learning Model Predictive Control", *American Control Conference*, 2017 (notebook 07). Preprint: [arXiv:1610.06534](https://arxiv.org/abs/1610.06534).
- U. Rosolia and F. Borrelli, "Learning How to Autonomously Race a Car: A Predictive Control Approach", *IEEE Transactions on Control Systems Technology*, 28(6), 2020. Local linear regression of the vehicle dynamics (notebook 07). Preprint: [arXiv:1901.08184](https://arxiv.org/abs/1901.08184).
- G. Perantoni and D. J. N. Limebeer, "Optimal control for a Formula One car with variable parameters", *Vehicle System Dynamics*, 52(5), 2014. Spatial (Frenet) minimum-time formulation (notebook 06). Open access: [Oxford University Research Archive](https://ora.ox.ac.uk/objects/uuid:ce1a7106-0a2c-41af-8449-41541220809f).

### Vehicle dynamics and identification

- R. Rajamani, *Vehicle Dynamics and Control*, 2nd ed., Springer, 2012 (notebooks 04, 06, 07).
- H. B. Pacejka, *Tire and Vehicle Dynamics*, 3rd ed., Butterworth-Heinemann, 2012 (notebooks 06, 07).
- L. Ljung, *System Identification: Theory for the User*, 2nd ed., Prentice Hall, 1999 (notebook 07).
- T. Hastie, R. Tibshirani and J. Friedman, *The Elements of Statistical Learning*, 2nd ed., Springer, 2009. Kernel smoothing and local regression (notebook 07). Free copy: [hastie.su.domains/ElemStatLearn](https://hastie.su.domains/ElemStatLearn/).

### Dynamic programming and reinforcement learning

- R. Bellman, *Dynamic Programming*, Princeton University Press, 1957 (notebook 08).
- D. P. Bertsekas, *Dynamic Programming and Optimal Control*, Athena Scientific (notebooks 01, 08). The author's page: [mit.edu/~dimitrib/dpbook.html](https://www.mit.edu/~dimitrib/dpbook.html).
- R. S. Sutton and A. G. Barto, *Reinforcement Learning: An Introduction*, 2nd ed., MIT Press, 2018 (notebook 08).
- V. Mnih et al., "Human-level control through deep reinforcement learning", *Nature*, 518, 2015 (notebook 08).
