import unittest
import numpy as np
from advanced_mpc_lmpc.linear import dlqr, prediction_matrices
from advanced_mpc_lmpc.polytopes import HPolytope, maximal_control_invariant
from advanced_mpc_lmpc.mpc import LinearMPC, simulate_receding_horizon
from advanced_mpc_lmpc.collision import choose_collision_strategy
from advanced_mpc_lmpc.racing import synthetic_track, minimum_time_speed_profile

class TestCore(unittest.TestCase):
    def setUp(self):
        self.A=np.array([[1.,1.],[0.,1.]])
        self.B=np.array([[0.5],[1.]])

    def test_prediction(self):
        Sx,Su=prediction_matrices(self.A,self.B,3)
        x0=np.array([1.,0.]); U=np.array([1.,0.,-1.])
        X=(Sx@x0+Su@U).reshape(3,2)
        x=x0.copy(); manual=[]
        for u in U:
            x=self.A@x+self.B[:,0]*u; manual.append(x.copy())
        np.testing.assert_allclose(X,np.array(manual))

    def test_lqr_stable(self):
        K,P,e=dlqr(self.A,self.B,np.eye(2),np.array([[0.2]]))
        self.assertLess(np.max(np.abs(e)),1.0)

    def test_control_invariant_nonempty(self):
        X=HPolytope.box([-4,-3],[4,3])
        C,_=maximal_control_invariant(self.A,self.B,X,-1,1,max_iter=20)
        self.assertGreaterEqual(len(C.vertices_2d()),3)
        self.assertTrue(C.contains([0,0]))

    def test_mpc_regulates(self):
        Q=np.diag([1.,0.2]); R=np.array([[0.2]])
        K,P,_=dlqr(self.A,self.B,Q,R)
        mpc=LinearMPC(self.A,self.B,Q,R,8,P=P,x_lower=[-5,-3],x_upper=[5,3],u_lower=[-1],u_upper=[1])
        X,U,V=simulate_receding_horizon(mpc,[-4,0],20)
        self.assertLess(np.linalg.norm(X[-1]),0.05)

    def test_collision_has_solution(self):
        best,_=choose_collision_strategy(np.array([60.,5.]),N=8,mu=0.1)
        self.assertIsNotNone(best)

    def test_speed_profile_limits(self):
        s=np.linspace(0,300,101); k=synthetic_track(s)
        sol=minimum_time_speed_profile(s,k)
        self.assertTrue(sol['success'])
        self.assertLessEqual(np.max(sol['a']),3.0+1e-8)
        self.assertGreaterEqual(np.min(sol['a']),-6.0-1e-8)

if __name__=='__main__': unittest.main()
