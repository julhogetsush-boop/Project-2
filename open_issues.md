# Open Verification Issues

| Issue | Needed from | Next verification step |
|---|---|---|
| Project-specific `kappa`, radial interval, angular period, and physical boundary data are unavailable | Member 2 / project model | Replace demonstration settings and rerun physical and manufactured cases on `n=16, 32, 64, 128` |
| Production FDM module and callable interface are unavailable | Member 3 | Add an adapter that runs the independent checks against the team's matrix assembly and solver |
| Torque definition and discretisation are unavailable | Member 2 | Compare torque across CG tolerances and mesh levels using the agreed formula and units |
| PINN implementation, checkpoints, and training seeds are unavailable | Member 3 | Evaluate PDE residual and magnetic-field error away from training points across recorded seeds |

These are open integration items, not failed checks in the standalone manufactured test.
