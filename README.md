# Member 5 Verification Package

This folder contains an independent verification oracle and automated checks for the PMSM Project 2 finite-difference work. Member 5's assigned role is testing, validation, and reproducibility; this package is not a replacement for the team's production FDM/PINN implementation.

## Included checks

- Four-mesh P5 manufactured-solution refinement at `n = 16, 32, 64, 128`, with grid-weighted discrete L2 errors and a fitted slope.
- CG tolerance sweep on `n = 128`, comparing algebraic error against an independent same-grid Fourier/Thomas direct reference and checking the `1e-3` target.
- Cross-check between CG and the independent direct reference.
- Explicit left and right non-zero Dirichlet contributions in the first/last interior rows.
- Automated checks for boundary RHS transfer, periodic stencil wraparound, and decreasing MMS error under refinement.

## Run

Requires Python 3.10+ and NumPy.

```powershell
python -m pip install -r requirements.txt
python verify_mms.py
python -m unittest discover -s tests -v
```

The verification script writes `results/mms_refinement.csv`, `results/cg_tolerance_n128.csv`, and `results/verification_summary.json`.

## Explicit assumptions and limits

The demonstration uses `-kappa^2*u_xx - u_yy = f`, `u=exp(x) sin(pi*y)`, `kappa=1`, `x in [0,1]`, and a periodic `y` interval of length 2. These values make a reproducible P5 manufactured test; replace or reconcile them with the team's agreed model parameters before reporting them as project results.

The team's physical boundary values, PINN model/API, magnetic-field definition, and torque functional were not supplied. Consequently, this package does not claim to validate the physical case, PINN generalisation, magnetic field, or torque. Once the team solver API is available, add an adapter that runs the same checks against the production implementation and record the exact commit and environment.

## Reproducibility record

Record the actual Python and NumPy versions, command, date, repository commit, return code, and any deviations from the assumptions above alongside the generated CSV files. Do not mark a result as reproduced until it has been run from the repository checkout using the documented command.
