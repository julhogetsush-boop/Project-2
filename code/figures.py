"""Generate compact verification figures from measured results."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False,
                     'pdf.fonttype':42,'svg.fonttype':'none'})
COLORS=['#0072B2','#D55E00','#009E73']


def save(fig,folder,name):
    for extension in ('pdf','png'):
        fig.savefig(folder/f'{name}.{extension}',dpi=300)
    plt.close(fig)


def finite_difference(folder,rows,sweep,costs):
    fig,axes=plt.subplots(1,3,figsize=(10,3),layout='constrained')
    for color,case in zip(COLORS,('manufactured','physical')):
        selected=[r for r in rows if r['case']==case]
        axes[0].loglog([r['h'] for r in selected],[r['l2h'] for r in selected],
                       'o-',color=color,label=case.capitalize())
    axes[0].set(xlabel='Grid spacing h',ylabel='Potential error (discrete L²)',title='a   Grid refinement')
    axes[0].set_xticks([1/128,1/64,1/32,1/16],['1/128','1/64','1/32','1/16'])
    axes[0].tick_params(axis='x',which='minor',labelbottom=False)
    axes[0].legend(frameon=False)
    axes[1].semilogx([r['rtol'] for r in sweep],[r['torque_over_T0'] for r in sweep],'o-',color=COLORS[0])
    axes[1].axhline(.5,color='.4',linestyle='--',label='Continuous reference')
    axes[1].set(xlabel='Requested CG relative tolerance',ylabel='Torque / reference scale',title='b   Solver tolerance')
    axes[1].legend(frameon=False,fontsize=7)
    for color,method in zip(COLORS,('LU','CG')):
        r=[q for q in costs if q['method']==method]
        y=np.array([q['solve_median_s'] for q in r])*1000
        a=np.array([q['solve_q25_s'] for q in r])*1000
        b=np.array([q['solve_q75_s'] for q in r])*1000
        axes[2].errorbar([q['n']*(q['n']-1) for q in r],y,yerr=[y-a,b-y],fmt='o-',
                         capsize=3,color=color,label=method)
    axes[2].set(xscale='log',yscale='log',xlabel='Unknowns',ylabel='Solve time (ms)',title='c   Matched accuracy')
    axes[2].legend(frameon=False)
    save(fig,folder,'fd_verification')


def neural(folder,runs,shifted):
    y,r,c=np.loadtxt(shifted,delimiter=',',skiprows=1,unpack=True)
    fig,axes=plt.subplots(1,3,figsize=(10,3),layout='constrained')
    axes[0].plot(y,r,color=COLORS[0],lw=1)
    axes[0].set(xlabel='Radial coordinate y',ylabel='Physical residual',title='a   Off-grid residual')
    axes[1].plot(y,(c-1)*1e9,color=COLORS[1],lw=1)
    axes[1].axhline(0,color='.4',linestyle='--')
    axes[1].set(xlabel='Radial coordinate y',ylabel='Torque coefficient error (×10⁹)',title='b   Radial torque check')
    labels=['64/0','64/1','64/2','16/0','256/0','64/0*']
    axes[2].scatter(labels,[t['potential_L2'] for t in runs],color=COLORS[0])
    axes[2].set(yscale='log',xlabel='Collocation count / seed (* stress)',
                ylabel='Potential error (continuous L²)',title='c   All prescribed runs')
    save(fig,folder,'pinn_verification')

