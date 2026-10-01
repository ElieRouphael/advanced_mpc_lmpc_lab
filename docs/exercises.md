# Exercises

## Optimization / CFTOC
1. Derive the condensed QP matrices for a horizon `N=3` double integrator by hand.
2. Add a constraint `|u_k-u_{k-1}| <= Delta_u_max`.
3. Verify a KKT point numerically for a 2-variable QP.

## Invariant sets
1. Change the input bound from `1` to `0.5`. How does `C_inf` change?
2. Compare the maximal invariant set for two feedback gains.
3. Explain why an unstable open-loop `A` can still admit a nonempty controlled-invariant set.

## MPC
1. Compare horizons `N=2,4,8,16` with and without a terminal set.
2. Remove the terminal constraint. Find an initial condition for which short-horizon behavior degrades.
3. Add reference tracking and derive the steady-state target equations.

## Collision avoidance
1. Vary `Qv/Ru` and map how aggressive the chosen maneuver becomes.
2. Add an explicit full-stop requirement for the yield strategy.
3. Treat bicycle arrival time as an interval and enforce robust avoidance.

## LMPC
1. Initialize with two different feasible seed controllers.
2. Disable safe-set convexification and require the terminal state to equal one stored state.
3. Add measurement noise and discuss why deterministic guarantees are no longer exact.

## Racing
1. Vary `mu` and plot lap-time sensitivity.
2. Vary braking capability and identify which corners become limiting.
3. Replace the synthetic track curvature with a spline-generated map.

## DP/RL
1. Change the discount factor and explain the policy change.
2. Implement policy iteration and compare with value iteration.
3. Write the equivalent reward-maximization formulation of the cost-minimization example.
