# AI-Generated Code Audit Record

This package includes AI-assisted code. Project 2 requires an audit before AI-generated solver code is used. The checks below record the available evidence; the student's own review and sign-off are still required.

| Gate | Evidence in this package | Status |
|---|---|---|
| Transcription | `verify_mms.py` states the operator, derives `f=(pi^2-kappa^2)u`, and computes both Dirichlet contributions explicitly. | Reviewed against the stated P5 test; student sign-off pending |
| Structure and indexing | Shape convention is documented; periodic `y` uses `np.roll`; radial boundary terms are checked at both ends; automated seam and boundary tests pass. | Automated checks pass |
| Linearisation | The manufactured problem is linear and the operator is applied directly; no nonlinear solve or hidden linearisation is used. | Pass for this test problem |
| Independent reproduction | CG is compared to a structurally different Fourier-mode plus Thomas tridiagonal reference; the test suite checks their agreement. | Automated check pass |
| Interrogation | The student should be able to explain the stencil, boundary transfer, Fourier eigenvalues, Thomas recurrence, norms, and each assumption. | Student review/sign-off pending |

This audit does not certify the team's production solver, physical PMSM case, PINN, magnetic field, or torque.
