import numpy as np
from scipy.optimize import minimize


def synthetic_track(s, length=300.0):
    """Smooth periodic curvature profile for an educational closed track."""
    s=np.asarray(s,float)
    kappa=(0.010*np.sin(2*np.pi*s/length)
           +0.018*np.sin(4*np.pi*s/length+0.5)
           +0.008*np.sin(6*np.pi*s/length-0.8))
    return kappa


def minimum_time_speed_profile(s, kappa, mu=1.0, g=9.81, vmax=35.0,
                               amin=-6.0, amax=3.0, v0=12.0):
    """
    Minimum-time centerline speed planning in the spatial domain.

    Decision z_i=v_i^2. Spatial longitudinal dynamics:
        z_{i+1} = z_i + 2 ds a_i.
    With a pointwise lateral limit z_i |kappa_i| <= mu g, the minimum-time
    solution is the largest feasible speed-squared profile. A forward pass
    enforces acceleration capability and a backward pass enforces braking.
    """
    s=np.asarray(s,float); kappa=np.asarray(kappa,float)
    ds=np.diff(s); n=len(s)
    lat_cap=np.where(np.abs(kappa)>1e-7, mu*g/np.abs(kappa), vmax**2)
    cap=np.minimum(vmax**2,lat_cap)
    z=np.maximum(2.0**2, cap.copy())
    z[0]=min(float(cap[0]), max(v0**2, 2.0**2))
    # Forward: z_{i+1}-z_i <= 2 ds amax
    for i,dsi in enumerate(ds):
        z[i+1]=min(z[i+1], z[i]+2*dsi*amax)
    # Backward: z_{i+1}-z_i >= 2 ds amin -> z_i <= z_{i+1}-2 ds amin
    for i in range(n-2,-1,-1):
        z[i]=min(z[i], z[i+1]-2*ds[i]*amin)
    # Re-enforce initial speed and forward acceleration once more.
    z[0]=min(float(cap[0]), max(v0**2, 2.0**2))
    for i,dsi in enumerate(ds):
        z[i+1]=min(z[i+1], z[i]+2*dsi*amax, cap[i+1])
    v=np.sqrt(np.maximum(z,1e-9))
    a=np.diff(z)/(2*ds)
    time=float(np.sum(ds*2.0/(v[:-1]+v[1:])))
    feasible=bool(np.all(a<=amax+1e-8) and np.all(a>=amin-1e-8) and np.all(z<=cap+1e-8))
    return {"success":feasible,"v":v,"a":a,"time":time,"z":z,
            "zmax":cap,"message":"forward-backward spatial speed planning"}

def local_affine_regression(X,U,Y,xq,uq,k=30,ridge=1e-6):
    """Fit y=A x+B u+c from nearest stored transitions around query (xq,uq)."""
    X=np.asarray(X,float); U=np.asarray(U,float); Y=np.asarray(Y,float)
    xq=np.asarray(xq,float); uq=np.asarray(uq,float)
    Z=np.hstack([X,U])
    zq=np.r_[xq,uq]
    scale=np.std(Z,axis=0)+1e-6
    d=np.linalg.norm((Z-zq)/scale,axis=1)
    idx=np.argsort(d)[:min(k,len(d))]
    w=np.exp(-(d[idx]/(d[idx].mean()+1e-9))**2)
    Phi=np.hstack([X[idx],U[idx],np.ones((len(idx),1))])
    W=np.diag(w)
    Theta=np.linalg.solve(Phi.T@W@Phi+ridge*np.eye(Phi.shape[1]),Phi.T@W@Y[idx])
    nx=X.shape[1]; nu=U.shape[1]
    A=Theta[:nx].T; B=Theta[nx:nx+nu].T; c=Theta[-1]
    return A,B,c,idx


def nonlinear_vehicle_step(x,u,kappa,dt=0.1,L=2.7):
    """Reduced kinematic Frenet model: x=[e_y,e_psi,v], u=[delta,a]."""
    ey,epsi,v=np.asarray(x,float); delta,a=np.asarray(u,float)
    den=max(0.25,1-kappa*ey)
    ey_dot=v*np.sin(epsi)
    epsi_dot=v/L*np.tan(delta)-kappa*v*np.cos(epsi)/den
    v_dot=a
    return np.array([ey+dt*ey_dot, epsi+dt*epsi_dot, max(0.1,v+dt*v_dot)])


def collect_vehicle_data(n=1500,seed=2):
    """Generate diverse safe-ish transitions for local model identification demos."""
    rng=np.random.default_rng(seed)
    X=[];U=[];Y=[]
    x=np.array([0.0,0.0,15.0])
    for k in range(n):
        kap=0.015*np.sin(2*np.pi*k/220)+0.008*np.sin(2*np.pi*k/73)
        delta=np.clip(-0.35*x[0]-0.7*x[1]+2.7*kap+rng.normal(0,0.015),-0.35,0.35)
        a=np.clip(0.8*(18-x[2])+rng.normal(0,0.25),-3,2)
        u=np.array([delta,a])
        y=nonlinear_vehicle_step(x,u,kap)
        X.append(x.copy());U.append(u);Y.append(y.copy())
        x=y
        if abs(x[0])>2.5: x[0]*=0.3
    return np.asarray(X),np.asarray(U),np.asarray(Y)
