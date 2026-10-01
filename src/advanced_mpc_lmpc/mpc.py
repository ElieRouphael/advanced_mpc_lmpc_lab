from dataclasses import dataclass
import numpy as np
from .linear import prediction_matrices
from .qp import solve_qp


@dataclass
class MPCSolution:
    success: bool
    U: np.ndarray
    X: np.ndarray
    objective: float
    message: str


class LinearMPC:
    """Condensed finite-horizon linear MPC with quadratic cost and polyhedral constraints."""
    def __init__(self, A, B, Q, R, N, P=None,
                 x_lower=None, x_upper=None, u_lower=None, u_upper=None,
                 terminal_H=None, terminal_h=None):
        self.A=np.asarray(A,float); self.B=np.asarray(B,float)
        self.Q=np.asarray(Q,float); self.R=np.atleast_2d(np.asarray(R,float))
        self.N=int(N); self.P=self.Q if P is None else np.asarray(P,float)
        self.n=self.A.shape[0]; self.m=self.B.shape[1]
        self.Sx,self.Su=prediction_matrices(self.A,self.B,self.N)
        self.x_lower=None if x_lower is None else np.asarray(x_lower,float)
        self.x_upper=None if x_upper is None else np.asarray(x_upper,float)
        self.u_lower=None if u_lower is None else np.asarray(u_lower,float)
        self.u_upper=None if u_upper is None else np.asarray(u_upper,float)
        self.terminal_H=terminal_H; self.terminal_h=terminal_h

    def solve(self, x0, x_ref=None, extra_A_ub=None, extra_b_ub=None, warm=None):
        x0=np.asarray(x0,float).reshape(self.n)
        x_ref=np.zeros(self.n) if x_ref is None else np.asarray(x_ref,float).reshape(self.n)
        Qbar=np.kron(np.eye(self.N),self.Q)
        Qbar[-self.n:,-self.n:]=self.P
        Rbar=np.kron(np.eye(self.N),self.R)
        rstack=np.tile(x_ref,self.N)
        base=self.Sx@x0-rstack
        H=2*(self.Su.T@Qbar@self.Su+Rbar)
        f=2*self.Su.T@Qbar@base
        Aubs=[]; bubs=[]
        if self.x_upper is not None:
            Aubs.append(self.Su); bubs.append(np.tile(self.x_upper,self.N)-self.Sx@x0)
        if self.x_lower is not None:
            Aubs.append(-self.Su); bubs.append(-np.tile(self.x_lower,self.N)+self.Sx@x0)
        if self.terminal_H is not None:
            SuN=self.Su[-self.n:]; SxN=self.Sx[-self.n:]
            Aubs.append(np.asarray(self.terminal_H)@SuN)
            bubs.append(np.asarray(self.terminal_h)-np.asarray(self.terminal_H)@SxN@x0)
        if extra_A_ub is not None:
            Aubs.append(np.asarray(extra_A_ub,float)); bubs.append(np.asarray(extra_b_ub,float))
        Aub=np.vstack(Aubs) if Aubs else None
        bub=np.concatenate(bubs) if bubs else None
        lb=np.tile(self.u_lower,self.N) if self.u_lower is not None else None
        ub=np.tile(self.u_upper,self.N) if self.u_upper is not None else None
        res=solve_qp(H,f,A_ub=Aub,b_ub=bub,lb=lb,ub=ub,x0=warm)
        U=res.x.reshape(self.N,self.m)
        X=(self.Sx@x0+self.Su@res.x).reshape(self.N,self.n)
        return MPCSolution(res.success,U,X,res.objective,res.message)


def simulate_receding_horizon(controller: LinearMPC, x0, steps=50, tol=1e-3):
    x=np.asarray(x0,float).copy(); xs=[x.copy()]; us=[]; costs=[]
    warm=None
    for _ in range(steps):
        sol=controller.solve(x,warm=warm)
        if not sol.success: break
        u=sol.U[0]
        us.append(u.copy()); costs.append(sol.objective)
        x=controller.A@x+controller.B@u
        xs.append(x.copy())
        warm=np.r_[sol.U[1:].reshape(-1), sol.U[-1].reshape(-1)]
        if np.linalg.norm(x)<tol: break
    return np.asarray(xs), np.asarray(us), np.asarray(costs)
