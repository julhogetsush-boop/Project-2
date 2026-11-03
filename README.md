# PMSM air-gap field and optimal excitation

Member 3 reproducibility package for Project 2: periodic finite differences, a Fourier-structured physics-informed neural network (PINN), and verification of field and excitation results. Implementation follows the supplied technical report, §§3.1, 3.4, 5.1 and 6, with outputs for Tables 3–9. This is a fresh implementation from the report; the original source and training checkpoints were not supplied.

## Reproduce

Use Python 3.13 on a CPU. In a clean Python environment, run from this directory:

```sh
python -m pip install -r requirements.txt
python code/run_all.py
```

For just the finite-difference evidence, use `python code/run_all.py --fd-only`. Do not use Python's `-O` switch; numerical acceptance checks use assertions. A full run trains all six prescribed networks. It replaces generated files of the same name under `results/`; it does not edit the source. Timings vary with machine load and hardware. Training trajectories may vary across Python/PyTorch builds even with fixed seeds.

| Evidence in the project report | Generated evidence |
|---|---|
| Tables 3–4: manufactured and physical refinement | `results/fd/convergence.csv`; fitted orders and maximum log-fit deviations in `results/fd/results.json` |
| Table 5: direct solver diagnostics | `results/fd/residuals.csv`: relative residual, backward error, spectral condition number, algebraic error bound and FFT–DST difference |
| Table 6: CG tolerance sweep | `results/fd/cg_tolerance.csv`: requested tolerance, actual residual, iteration count, potential error and torque |
| Table 7: matched accuracy timing | `results/fd/costs_raw.csv` (five repeats) and `results/fd/costs.csv` (median and quartiles) |
| Table 8: neural experiments | `results/pinn/runs.csv`, loss history, shifted-grid CSV and state dictionary per run |
| Table 9: excitation design | `results/pinn/design.csv`; active-constraint test and environment in `results/pinn/results.json` |
| Verification figures | `results/figures/`: editable vector PDF and 300 dpi PNG from measured outputs |

## Model and checks

- The angular grid is periodic with no duplicate endpoint. The radial grid excludes prescribed Dirichlet endpoints from the unknown vector. The known traces enter the right-hand side with a positive coefficient. The code checks an independently written six-unknown matrix, its spectrum and a nonzero boundary row.
- The two continuous reference problems are the analytic physical solution and the independent multimode manufactured solution. The FFT–DST solve uses a separate computational path for the same discrete operator. LU forward-error bounds are compared with measured discretisation error; fresh LU and CG timings are accepted only when the algebraic difference is at most 0.1% of the discretisation error and the midpoint torque agrees with the discrete reference to 1e-8 N·m.
- The nominal field uses equivalent boundary amplitudes (0.25, 0.50). The design uses penalty 2 and constraint radius 0.8, giving the analytical optimum (0, 0.50). The model is a smooth, current-free, nonsalient air gap. The amplitudes are dimensionless, not calibrated currents.
- The PINN imposes endpoint values by construction and uses only the scaled radial PDE residual during training. It uses CPU, one numerical thread, fixed seeds and float64 calculations. PyTorch default linear-layer initialisation is seeded, then the model is converted to float64. All six runs are retained; nominal seed 0 is fixed before validation. Independent 256/512-point quadrature, shifted-grid samples, derivative checks and KKT checks support the result. Numerical integration estimates of the continuous bounds are not interval-certified enclosures. The excitation objective is strongly convex; network training is not.
- The code implements the model, discretisation, boundary elimination, references, PINN ansatz and diagnostics. NumPy handles arrays and quadrature, SciPy provides sparse LU/CG and transforms, PyTorch provides automatic differentiation and optimisation, and Matplotlib renders the figures.

## Availability and attribution

The source, dependency specification, numerical results and generation commands are included in this package. The tables report simulated benchmark data; no experimental or private inputs are required. The supplied PJ2 report and slides are not redistributed in this package. Before public distribution, the team should confirm its repository license and contributor attribution.

This reconstruction was generated with AI assistance from the team's supplied technical report. Its included checks are automated checks, not a claim of independent human review or of student-authored historical work. Record Member 2's formula review and Member 5's independent reproduction against the actual Git commit, and report individual contributions factually.

## Verification of the supplied outputs

The full command completed twice on Windows 11, Python 3.13.4 and CPU PyTorch 2.15.0. The supplied results are from the final run; all numerical acceptance checks passed. The six PINN potential errors were unchanged across these two runs in this environment.

| Quantity | Supplied report | This reconstruction |
|---|---:|---:|
| Manufactured FDM potential convergence order | 2.0018 | 2.00184696 |
| Physical FDM potential convergence order | 1.9977 | 1.99766970 |
| Physical FDM midpoint torque / T₀, n = 128 | 0.499799363 | 0.499799363379 |
| Nominal seed 0 PINN potential L² error | 1.127e-9 | 4.569e-9 |
| Nominal seed 0 PINN physical residual L² | 4.205e-7 | 8.568e-7 |
| Nominal seed 0 PINN design q_q | 0.499999998037 | 0.499999999789 |

The neural values are independently measured for the reconstructed network and should replace the corresponding old numbers if this implementation becomes the team's final version. Original initial weights and optimizer history were unavailable; this comparison does not establish why the two runs differ. Timings in this package are new measurements.

**Figure source data.** In `fd_verification`, panels a–c use the refinement, CG sweep and timing CSVs respectively; timing bars span the 25th to 75th percentiles of five repeats. In `pinn_verification`, panels a–b use the prespecified nominal seed 0 shifted-grid CSV; panel c uses all rows of `runs.csv` and marks the separate β=1 stress geometry with an asterisk.
