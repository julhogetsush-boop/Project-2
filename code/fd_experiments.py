"""Periodic finite differences and independently derived reference solutions."""
import time
import numpy as np
from scipy import sparse
from scipy.fft import fft, ifft, dst, idst
from scipy.sparse.linalg import spsolve, cg

A, B, P, A0, LZ, MU = .05, .051, 4, 1e-3, .1, 4*np.pi*1e-7
DELTA = np.log(B/A)
BETA = P*DELTA
KAPPA = BETA/(2*np.pi)
T0 = np.pi*LZ*A0**2*P**2/(MU*np.sinh(BETA))
QFIELD = (.25, .5)


def physical(x, y, q=QFIELD):
    s = np.sinh(BETA*y)/np.sinh(BETA)
    p = np.sinh(BETA*(1-y))/np.sinh(BETA)
    sp = BETA*np.cosh(BETA*y)/np.sinh(BETA)
    pp = -BETA*np.cosh(BETA*(1-y))/np.sinh(BETA)
    c, t = np.cos(2*np.pi*x), np.sin(2*np.pi*x)
    d, e = q
    f, g = p+(1/np.cosh(BETA)+d)*s, e*s
    return (f*c+g*t, 2*np.pi*(-f*t+g*c),
            (pp+(1/np.cosh(BETA)+d)*sp)*c+e*sp*t)


def manufactured(x, y):
    c = (1+y)*np.cos(2*np.pi*x)
    s = np.sin(2*np.pi*x)*np.sin(np.pi*y)
    t = .2*np.cos(6*np.pi*x)*np.sin(2*np.pi*y)
    return c+s+t, BETA**2*c+(BETA**2+np.pi**2)*s+(9*BETA**2+4*np.pi**2)*t


def matrix(n):
    # U[j,i]: radial index first, periodic angular index second.
    i = np.arange(n)
    shift = sparse.csr_matrix((np.ones(n), (i, (i+1)%n)), shape=(n,n))
    c = 2*sparse.eye(n)-shift-shift.T
    t = sparse.diags([-1,2,-1],[-1,0,1],shape=(n-1,n-1),dtype=float)
    return (n*n*(sparse.kron(t,sparse.eye(n))+
                 KAPPA**2*sparse.kron(sparse.eye(n-1),c))).tocsr()


def data(n, case, q=QFIELD):
    x, y = np.arange(n)[None,:]/n, np.arange(n+1)[:,None]/n
    if case == 'physical':
        u, _, _ = physical(x,y,q)
        f = np.zeros_like(u)
    else:
        u, f = manufactured(x,y)
    b = f[1:-1].copy()
    b[0] += n*n*u[0]
    b[-1] += n*n*u[-1]
    return x, y, u, b.ravel()


def spectrum(n):
    return 4*n*n*(np.sin(np.pi*np.arange(1,n)[:,None]/(2*n))**2
                  +KAPPA**2*np.sin(np.pi*np.arange(n)[None,:]/n)**2)


def oracle(b, n):
    coeff = fft(dst(b.reshape(n-1,n),type=1,axis=0),axis=1)
    return idst(ifft(coeff/spectrum(n),axis=1).real,type=1,axis=0).ravel()


def diagnostics(k,b,u,n):
    r = np.linalg.norm(b-k@u)
    lam = spectrum(n)
    return dict(relative_residual=r/np.linalg.norm(b),
                backward_error=r/(lam.max()*np.linalg.norm(u)+np.linalg.norm(b)),
                condition_2=lam.max()/lam.min(), forward_l2h_bound=r/(n*lam.min()))


def field(k_u, exact, n):
    u = exact.copy()
    u[1:-1] = k_u.reshape(n-1,n)
    ux = n/2*(np.roll(u,-1,axis=1)-np.roll(u,1,axis=1))[1:-1]
    uy = n/2*(u[2:]-u[:-2])
    torque = -LZ*P*A0**2/(MU*DELTA)*np.mean(ux*uy,axis=1)
    return ux, uy, torque


def solve_cg(k,b,rtol):
    steps = []
    u, info = cg(k,b,x0=np.zeros_like(b),rtol=rtol,atol=0,
                 callback=lambda _: steps.append(1))
    if info != 0:
        raise RuntimeError(f'CG failed: info={info}, rtol={rtol}')
    return u,len(steps)


def check():
    # Three angular nodes and two radial rows: all six entries audited by hand.
    c = np.array([[2,-1,-1],[-1,2,-1],[-1,-1,2]])*KAPPA**2
    d = 2*np.eye(3)+c
    np.testing.assert_allclose(matrix(3).toarray()/9,
                               np.block([[d,-np.eye(3)],[-np.eye(3),d]]))
    np.testing.assert_allclose(np.linalg.eigvalsh(matrix(3).toarray()),
                               np.sort(spectrum(3).ravel()),rtol=1e-13)
    _,_,_,b = data(3,'physical')
    np.testing.assert_allclose(b.reshape(2,3)[0],9*np.cos(2*np.pi*np.arange(3)/3))
    np.testing.assert_allclose(physical(0.,1.,(0,0))[2],0,atol=1e-14)


def experiments():
    check()
    refinement, residuals, sweep, costs, fits = [], [], [], [], {}
    for case in ('manufactured','physical'):
        previous = None
        rows = []
        for n in (16,32,64,128):
            x,y,exact,b = data(n,case)
            k = matrix(n)
            assert k.nnz == 5*n*(n-1)-2*n and (k-k.T).nnz == 0
            u = spsolve(k,b)
            ref = oracle(b,n)
            e = u-exact[1:-1].ravel()
            err = np.linalg.norm(e)/n
            diag = diagnostics(k,b,u,n)
            gap = np.linalg.norm(u-ref)/n
            assert gap <= 1e-3*err and diag['forward_l2h_bound'] <= 1e-3*err
            energy = np.sqrt(e@(k@e))/n
            row = dict(case=case,n=n,h=1/n,dofs=k.shape[0],nnz=k.nnz,
                       l2h=err,linf=np.max(np.abs(e)),energy=energy,
                       pairwise_order=None if previous is None else np.log2(previous[0]/err),
                       energy_order=None if previous is None else np.log2(previous[1]/energy))
            previous = err,energy
            residuals.append(dict(case=case,n=n,oracle_l2h_gap=gap,**diag))
            if case == 'physical':
                ux,uy,t = field(u,exact,n)
                _,ex,ey = physical(x,y[1:-1])
                # In the area-weighted relative B norm the radial factors cancel.
                norm = lambda a,b: np.linalg.norm(np.stack([KAPPA*a,b]))
                row.update(relative_B=norm(ux-ex,uy-ey)/norm(ex,ey),
                           torque_over_T0=t[n//2-1]/T0,
                           relative_torque_error=abs(t[n//2-1]/(T0*QFIELD[1])-1))
                # Closed-form discrete harmonic torque (report Eq. 43).
                beta_h = 2*KAPPA*n*np.sin(np.pi/n)
                rho = 2*n*np.arcsinh(beta_h/(2*n))
                ratio_exact = (np.sinc(2/n)*np.sinh(BETA)/BETA
                               *n*np.sinh(rho/n)/np.sinh(rho))
                np.testing.assert_allclose(t/(T0*QFIELD[1]),ratio_exact,
                                           rtol=1e-10,atol=1e-12)
                assert np.ptp(t) < 1e-8
                _,_,rev_exact,rev_b = data(n,case,(QFIELD[0],-QFIELD[1]))
                np.testing.assert_allclose(field(spsolve(k,rev_b),rev_exact,n)[2],-t,atol=1e-9)
                if n == 64:
                    for tol in (1e-2,1e-4,1e-6,1e-8,1e-10,1e-12):
                        v,it = solve_cg(k,b,tol)
                        sweep.append(dict(n=n,rtol=tol,iterations=it,
                            actual_residual=diagnostics(k,b,v,n)['relative_residual'],
                            potential_l2h=np.linalg.norm(v-exact[1:-1].ravel())/n,
                            algebraic_over_discretisation=np.linalg.norm(v-ref)/(n*err),
                            torque_over_T0=field(v,exact,n)[2][n//2-1]/T0))
                    assert sweep[0]['algebraic_over_discretisation'] > 1
                    assert max(s['algebraic_over_discretisation'] for s in sweep[1:]) < 1e-3
                for rep in range(5):
                    tick = time.perf_counter()
                    assembled = matrix(n)
                    asm = time.perf_counter()-tick
                    for method in ('LU','CG'):
                        tick = time.perf_counter()
                        v,it = (spsolve(assembled,b),0) if method=='LU' else solve_cg(assembled,b,1e-10)
                        seconds = time.perf_counter()-tick
                        ratio = np.linalg.norm(v-ref)/(n*err)
                        torque_gap = abs(field(v,exact,n)[2][n//2-1]-field(ref,exact,n)[2][n//2-1])
                        assert ratio <= 1e-3 and torque_gap <= 1e-8
                        costs.append(dict(n=n,dofs=k.shape[0],repeat=rep+1,method=method,
                                          assembly_seconds=asm,solve_seconds=seconds,
                                          iterations=it,algebraic_ratio=ratio,torque_gap_Nm=torque_gap))
            rows.append(row)
        lh, le = np.log([r['h'] for r in rows]),np.log([r['l2h'] for r in rows])
        fit = np.polyfit(lh,le,1)
        fits[case] = dict(order=fit[0],max_log_deviation=np.max(abs(le-np.polyval(fit,lh))))
        assert 1.95 < fit[0] < 2.05
        refinement.extend(rows)
    return refinement,residuals,sweep,costs,fits


