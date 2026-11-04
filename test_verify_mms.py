import unittest
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from verify_mms import (
    KAPPA,
    X_LENGTH,
    Y_PERIOD,
    apply_operator,
    conjugate_gradient,
    manufactured_system,
    spectral_direct_reference,
    verify_nonzero_boundary_row,
    weighted_l2,
)


class ManufacturedVerificationTests(unittest.TestCase):
    def test_nonzero_dirichlet_terms_are_moved_to_rhs(self):
        check = verify_nonzero_boundary_row(32)
        self.assertAlmostEqual(
            check["left_expected_RHS_contribution"],
            check["left_assembled_RHS_contribution"],
            places=10,
        )
        self.assertAlmostEqual(
            check["right_expected_RHS_contribution"],
            check["right_assembled_RHS_contribution"],
            places=10,
        )
        self.assertGreater(check["left_assembled_RHS_contribution"], 0.0)

    def test_periodic_stencil_wraps_across_y_seam(self):
        n = 16
        hx = X_LENGTH / n
        hy = Y_PERIOD / n
        y = np.arange(n) * hy
        u = np.tile(np.sin(np.pi * y), (n - 1, 1))
        actual = apply_operator(u, hx, hy, KAPPA)
        # The x-constant mode contributes zero on an interior radial row;
        # check y=0, where the stencil must wrap from the last periodic point.
        i = 1
        expected_y = 4.0 * np.sin(np.pi / n) ** 2 / hy**2 * u[i, 0]
        self.assertAlmostEqual(actual[i, 0], expected_y, places=9)

    def test_mms_error_decreases_under_refinement(self):
        errors = []
        for n in (16, 32):
            rhs, exact, _, hx, hy = manufactured_system(n)
            solution, _, residual = conjugate_gradient(rhs, hx, hy, 1e-12)
            errors.append(weighted_l2(solution - exact, hx, hy))
            self.assertLess(residual / np.linalg.norm(rhs.ravel()), 2e-12)
        self.assertLess(errors[1], errors[0])

    def test_cg_agrees_with_independent_spectral_reference(self):
        n = 16
        rhs, _, _, hx, hy = manufactured_system(n)
        direct = spectral_direct_reference(rhs, hx, hy)
        iterative, _, residual = conjugate_gradient(rhs, hx, hy, 1e-10)
        self.assertLess(weighted_l2(iterative - direct, hx, hy), 1e-10)
        self.assertLess(residual / np.linalg.norm(rhs.ravel()), 1.1e-10)


if __name__ == "__main__":
    unittest.main()
