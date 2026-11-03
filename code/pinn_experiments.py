"""Fourier-structured PINN; training uses the radial PDE residual only."""
import time
import numpy as np
import torch
from fd_experiments import BETA, QFIELD

CASES = [(BETA,64,0),(BETA,64,1),(BETA,64,2),
         (BETA,16,0),(BETA,256,0),(1.,64,0)]
DTYPE = torch.float64


def radial(net,y,beta):
    return y+beta**2*y*(1-y)*net(2*y-1)


def evaluate(net,y,beta):
    y = torch.as_tensor(y,dtype=DTYPE).reshape(-1,1).detach().requires_grad_(True)
    s = radial(net,y,beta)
    sp = torch.autograd.grad(s.sum(),y,create_graph=True)[0]
    spp = torch.autograd.grad(sp.sum(),y,create_graph=True)[0]
    return s,sp,-spp+beta**2*s


def design(c,gamma,Q):
    q = np.clip(c/gamma,-Q,Q)
    lam = max(0.,(abs(c)/Q-gamma)/2)
    assert abs((gamma+2*lam)*q-c) < 1e-12
    assert abs(lam*(q*q-Q*Q)) < 1e-12 and abs(q)<=Q
    return float(q),float(lam)


def train(beta,nc,seed,folder):
    torch.manual_seed(seed)
    net = torch.nn.Sequential(torch.nn.Linear(1,16),torch.nn.Tanh(),
                              torch.nn.Linear(16,16),torch.nn.Tanh(),
                              torch.nn.Linear(16,1)).to(dtype=DTYPE)
    nodes = (np.arange(nc)+.5)/nc
    adam = torch.optim.Adam(net.parameters(),lr=.005)
    lbfgs = torch.optim.LBFGS(net.parameters(),max_iter=1200,max_eval=1800,
                             tolerance_grad=1e-12,tolerance_change=1e-15,
                             history_size=50,line_search_fn='strong_wolfe')
    history=[]
    def step(opt,stage):
        opt.zero_grad()
        loss = (evaluate(net,nodes,beta)[2]/beta**2).square().mean()
        loss.backward()
        history.append((stage,len(history)+1,float(loss.detach())))
        return loss
    tick=time.perf_counter()
    for _ in range(500):
        step(adam,'Adam')
        adam.step()
    lbfgs.step(lambda: step(lbfgs,'L-BFGS'))
    seconds=time.perf_counter()-tick
    tag=f'beta_{beta:.8f}_n{nc}_seed{seed}'
    torch.save(net.state_dict(),folder/f'{tag}.pt')

    def sample(y):
        return [v.detach().numpy().ravel() for v in evaluate(net,y,beta)]

    endpoints, endpoint_d, _ = sample([0.,1.])
    np.testing.assert_array_equal(endpoints,[0.,1.])
    scale = np.sinh(beta)/beta
    c = scale*endpoint_d[0]

    def quadrature(n):
        x,w = np.polynomial.legendre.leggauss(n)
        y,w = (x+1)/2,w/2
        s,sp,r = sample(y)
        p,dy_p,_ = sample(1-y)
        exact=lambda z: np.sinh(beta*z)/np.sinh(beta)
        derivative=lambda z: beta*np.cosh(beta*z)/np.sinh(beta)
        d,q = QFIELD
        dc = 1/np.cosh(beta)+d
        ef,eg = p-exact(1-y)+dc*(s-exact(y)),q*(s-exact(y))
        df,dg = -dy_p+derivative(1-y)+dc*(sp-derivative(y)),q*(sp-derivative(y))
        f,g = exact(1-y)+dc*exact(y),q*exact(y)
        fp,gp = -derivative(1-y)+dc*derivative(y),q*derivative(y)
        field = np.sqrt((w@(beta**2*(ef**2+eg**2)+df**2+dg**2))/
                        (w@(beta**2*(f**2+g**2)+fp**2+gp**2)))
        return np.sqrt(w@r**2),np.sqrt(w@(ef**2+eg**2)/2),field

    res,err,b_err = quadrature(256)
    res512,_,_ = quadrature(512)
    y = (np.arange(4096)+.37)/4096
    s,sp,r = sample(y)
    p,dp,_ = sample(1-y)
    cq = scale*(p*sp+dp*s)
    yy = np.array([.17,.36,.59,.81])
    h = 2e-4
    second_fd = (sample(yy+h)[0]-2*sample(yy)[0]+sample(yy-h)[0])/h**2
    sy,_,ry = sample(yy)
    audit = np.max(abs(second_fd-(beta**2*sy-ry)))
    assert audit < 1e-6 and abs(res-res512) < max(1e-12,res*1e-5)
    params=list(net.parameters())
    loss=lambda: (evaluate(net,nodes,beta)[2]/beta**2).square().mean()
    grad=torch.autograd.grad(loss(),params)
    base=[p.detach().clone() for p in params]
    rng=torch.Generator().manual_seed(724)
    directions=[torch.randn(p.shape,dtype=DTYPE,generator=rng) for p in params]
    length=torch.sqrt(sum(v.square().sum() for v in directions))
    directions=[v/length for v in directions]
    directional=sum((g*v).sum().item() for g,v in zip(grad,directions))
    vals=[]
    for sign in (1,-1,0):
        with torch.no_grad():
            for param,original,v in zip(params,base,directions):
                param.copy_(original+sign*1e-5*v)
        if sign:
            vals.append(loss().item())
    grad_gap=abs((vals[0]-vals[1])/2e-5-directional)
    assert grad_gap < 1e-7
    qhat,_ = design(c,2.,.8)
    qstar,_ = design(1.,2.,.8)
    kb = np.sqrt(np.sinh(2*beta)/(4*beta)-.5)/beta
    c_bound = kb*res
    assert abs(c-1) <= c_bound+1e-12
    assert abs(qhat-qstar) <= c_bound/2+1e-12
    assert (qhat-qstar)**2 <= c_bound**2/4+1e-24
    assert err < (1e-6 if beta<.1 else 1e-4)
    result = dict(case=tag,beta=beta,nc=nc,seed=seed,potential_L2=err,relative_B=b_err,
                  residual_L2_256=res,residual_L2_512=res512,residual_peak_shifted=np.max(abs(r)),
                  torque_coefficient=c,coefficient_abs_error=abs(c-1),seconds=seconds,
                  lbfgs_evaluations=len(history)-500,second_derivative_audit=audit,
                  gradient_audit=grad_gap)
    decision = dict(case=tag,beta=beta,seed=seed,qd=0.,qq=qhat,qq_reference=qstar,
                    design_error=abs(qhat-qstar),regret=(qhat-qstar)**2,
                    estimated_coefficient_bound=c_bound,estimated_regret_bound=c_bound**2/4)
    return result,decision,history,zip(y,r,cq)
