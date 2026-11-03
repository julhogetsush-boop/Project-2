"""Rebuild the Member 3 numerical evidence, from the repository root."""
import os
for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
    os.environ[name]='1'
import argparse
import csv
import json
import platform
from pathlib import Path
import numpy as np
import scipy
import matplotlib
import fd_experiments as fd

ROOT=Path(__file__).resolve().parents[1]


def write_csv(path,rows,columns=None):
    rows=list(rows)
    columns=columns or list(dict.fromkeys(key for row in rows for key in row))
    with path.open('w',newline='',encoding='utf-8') as handle:
        writer=csv.writer(handle)
        writer.writerow(columns)
        writer.writerows([row.get(k) for k in columns] if isinstance(row,dict) else row
                         for row in rows)


def write_json(path,obj):
    path.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fd-only',action='store_true',help='Run the FDM evidence without PyTorch.')
    args=parser.parse_args()
    out=ROOT/'results'
    f=out/'fd'
    f.mkdir(parents=True,exist_ok=True)
    env=dict(python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__,
             matplotlib=matplotlib.__version__,platform=platform.platform(),
             processor=platform.processor(),numerical_threads=1)
    data,res,sweep,costs,fits=fd.experiments()
    for name,rows in [('convergence',data),('residuals',res),('cg_tolerance',sweep),('costs_raw',costs)]:
        write_csv(f/f'{name}.csv',rows)
    summary=[]
    for n in (16,32,64,128):
        for method in ('LU','CG'):
            subset=[r for r in costs if r['n']==n and r['method']==method]
            a,b,c=np.quantile([r['solve_seconds'] for r in subset],[.25,.5,.75])
            summary.append(dict(n=n,method=method,solve_q25_s=a,solve_median_s=b,solve_q75_s=c,
               assembly_median_s=np.median([r['assembly_seconds'] for r in subset]),
               max_algebraic_ratio=max(r['algebraic_ratio'] for r in subset),
               max_torque_gap_Nm=max(r['torque_gap_Nm'] for r in subset),
               iterations=subset[0]['iterations']))
    write_csv(f/'costs.csv',summary)
    write_json(f/'results.json',dict(environment=env,fits=fits,beta=fd.BETA,kappa=fd.KAPPA,T0_Nm=fd.T0))
    print('FDM checks passed; fits:',fits,flush=True)
    import figures
    fig=out/'figures'
    fig.mkdir(exist_ok=True)
    figures.finite_difference(fig,data,sweep,summary)
    if args.fd_only:
        return
    import torch
    import pinn_experiments as pinn
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    p=out/'pinn'
    p.mkdir(exist_ok=True)
    env['torch']=torch.__version__
    runs,designs=[],[]
    for beta,nc,seed in pinn.CASES:
        r,d,h,shift=pinn.train(beta,nc,seed,p)
        runs.append(r)
        designs.append(d)
        write_csv(p/(r['case']+'_loss.csv'),h,['stage','evaluation','scaled_mse'])
        write_csv(p/(r['case']+'_shifted.csv'),shift,['y','physical_residual','torque_coefficient'])
        print(r['case'], 'checks passed; potential L2 =',r['potential_L2'],flush=True)
    write_csv(p/'runs.csv',runs)
    write_csv(p/'design.csv',designs)
    active=pinn.design(1.,.8,.8)
    np.testing.assert_allclose(active,[.8,.225],atol=1e-14)
    write_json(p/'results.json',dict(environment=env,runs=runs,designs=designs,
                                   active_design=dict(qq=active[0],multiplier=active[1])))
    figures.neural(fig,runs,p/(runs[0]['case']+'_shifted.csv'))
    print('All numerical checks passed.',flush=True)


if __name__=='__main__':
    main()


