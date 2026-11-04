"""Independent FDM verification tools for PMSM Project 2, Member 5.

This standalone verification oracle uses the P5 manufactured solution
u(x, y) = exp(x) sin(pi*y) for -kappa^2 u_xx - u_yy = f.

Explicit demonstration assumptions (replace with the team's agreed values
before treating these results as project data): kappa=1, x in [0, 1], and a
periodic y interval of length 2. The script does not implement the team's
physical boundary-value problem, PINN, or torque functional.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np


KAPPA = 1.0
X_LENGTH = 1.0
Y_PERIOD = 2.0
MESH_SIZES = (16, 32, 64, 128)
TOLERANCES = (1e-4, 1e-5, 1e-6, 1e-7)
TARGET_RATIO = 1e-3
RESULTS_DIR = Path(__file__).resolve().parent / "results"


def exact_solution(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.exp(x) * np.sin(np.pi * y)


def apply_operator(u: np.ndarray, hx: float, hy: float, kappa: float = KAPPA) -> np.ndarray:
    """Apply the five-point operator to interior-x, periodic-y unknowns."""
    result = (2.0 * kappa**2 / hx**2 + 2.0 / hy**2) * u
    x_neighbors = np.zeros_like(u)
    x_neighbors[1:, :] += u[:-1, :]
    x_neighbors[:-1, :] += u[1:, :]
    y_neighbors = np.roll(u, 1, axis=1) + np.roll(u, -1, axis=1)
    result -= (kappa**2 / hx**2) * x_neighbors
    result -= y_neighbors / hy**2
    return result


def manufactured_system(
    n: int, kappa: float = KAPPA
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
    """Return RHS, exact interior values, y nodes, hx, hy for the P5 test."""
    hx = X_LENGTH / n
    hy = Y_PERIOD / n
    x = np.arange(1, n, dtype=float) * hx
    y = np.arange(n, dtype=float) * hy
    xx, yy = np.meshgrid(x, y, indexing="ij")
    exact = exact_solution(xx, yy)

    # Since u_xx=u and u_yy=-pi^2*u, the manufactured forcing is:
    # f = -kappa^2*u_xx - u_yy = (pi^2-kappa^2)*u.
    forcing = (np.pi**2 - kappa**2) * exact
    rhs = forcing.copy()
    left_boundary = np.sin(np.pi * y)
    right_boundary = np.exp(X_LENGTH) * np.sin(np.pi * y)
    rhs[0, :] += kappa**2 / hx**2 * left_boundary
    rhs[-1, :] += kappa**2 / hx**2 * right_boundary
    return rhs, exact, forcing, hx, hy


def conjugate_gradient(
    rhs: np.ndarray,
    hx: float,
    hy: float,
    rtol: float,
    kappa: float = KAPPA,
    max_iter: int = 100_000,
) -> tuple[np.ndarray, int, float]:
    """Solve A u=b and return solution, iterations, and explicitly recomputed residual."""
    solution = np.zeros_like(rhs)
    residual = rhs.copy()
    direction = residual.copy()
    rr = float(np.vdot(residual, residual).real)
    rhs_norm = float(np.linalg.norm(rhs.ravel()))
    threshold = rtol * rhs_norm

    for iteration in range(max_iter + 1):
        if np.sqrt(rr) <= threshold:
            true_residual = rhs - apply_operator(solution, hx, hy, kappa)
            return solution, iteration, float(np.linalg.norm(true_residual.ravel()))
        applied = apply_operator(direction, hx, hy, kappa)
        denominator = float(np.vdot(direction, applied).real)
        if denominator <= 0.0:
            raise RuntimeError("CG lost positive definiteness; check operator/sign convention")
        alpha = rr / denominator
        solution += alpha * direction
        residual -= alpha * applied
        rr_new = float(np.vdot(residual, residual).real)
        direction = residual + (rr_new / rr) * direction
        rr = rr_new

    raise RuntimeError(f"CG failed to reach rtol={rtol:g} in {max_iter} iterations")


def spectral_direct_reference(
    rhs: np.ndarray, hx: float, hy: float, kappa: float = KAPPA
) -> np.ndarray:
    """Independent discrete solve: Fourier modes in y + Thomas solve in x."""
    n_y = rhs.shape[1]
    n_x = rhs.shape[0]
    modes = np.arange(n_y, dtype=float)
    lambda_y = 4.0 * np.sin(np.pi * modes / n_y) ** 2 / hy**2
    diagonal = np.broadcast_to(
        (2.0 * kappa**2 / hx**2 + lambda_y[:, None]), (n_y, n_x)
    ).copy()
    off_diagonal = -kappa**2 / hx**2
    transformed_rhs = np.fft.fft(rhs, axis=1).T.copy()

    for j in range(1, n_x):
        factor = off_diagonal / diagonal[:, j - 1]
        diagonal[:, j] -= factor * off_diagonal
        transformed_rhs[:, j] -= factor * transformed_rhs[:, j - 1]

    transformed_solution = np.empty_like(transformed_rhs)
    transformed_solution[:, -1] = transformed_rhs[:, -1] / diagonal[:, -1]
    for j in range(n_x - 2, -1, -1):
        transformed_solution[:, j] = (
            transformed_rhs[:, j] - off_diagonal * transformed_solution[:, j + 1]
        ) / diagonal[:, j]
    return np.fft.ifft(transformed_solution.T, axis=1).real


def weighted_l2(values: np.ndarray, hx: float, hy: float) -> float:
    return float(np.sqrt(hx * hy) * np.linalg.norm(values.ravel()))


def run_refinement() -> tuple[list[dict[str, Any]], float]:
    table: list[dict[str, Any]] = []
    references: dict[int, tuple[np.ndarray, np.ndarray, float, float]] = {}
    for n in MESH_SIZES:
        rhs, exact, _, hx, hy = manufactured_system(n)
        numerical = spectral_direct_reference(rhs, hx, hy)
        residual = float(
            np.linalg.norm((rhs - apply_operator(numerical, hx, hy)).ravel())
        )
        error_l2 = weighted_l2(numerical - exact, hx, hy)
        error_linf = float(np.max(np.abs(numerical - exact)))
        references[n] = (numerical, exact, hx, hy)
        table.append(
            {
                "n": n,
                "hx": hx,
                "hy": hy,
                "error_L2": error_l2,
                "error_Linf": error_linf,
                "reference_method": "Fourier modes + Thomas tridiagonal solve",
                "reference_residual_L2": residual,
                "observed_order_L2": None,
            }
        )

    for previous, current in zip(table, table[1:]):
        previous["observed_order_L2"] = float(
            np.log(previous["error_L2"] / current["error_L2"])
            / np.log(previous["hx"] / current["hx"])
        )
    fitted_slope = float(
        np.polyfit(
            np.log([row["hx"] for row in table]),
            np.log([row["error_L2"] for row in table]),
            1,
        )[0]
    )
    return table, fitted_slope


def run_tolerance_study() -> tuple[list[dict[str, Any]], float]:
    n = MESH_SIZES[-1]
    rhs, exact, _, hx, hy = manufactured_system(n)
    reference = spectral_direct_reference(rhs, hx, hy)
    reference_residual = float(
        np.linalg.norm((rhs - apply_operator(reference, hx, hy)).ravel())
    )
    discretisation_error = weighted_l2(reference - exact, hx, hy)
    rows: list[dict[str, Any]] = []
    for tolerance in TOLERANCES:
        solution, iterations, residual = conjugate_gradient(rhs, hx, hy, tolerance)
        algebraic_error = weighted_l2(solution - reference, hx, hy)
        ratio = algebraic_error / discretisation_error
        rows.append(
            {
                "n": n,
                "rtol": tolerance,
                "iterations": iterations,
                "explicit_residual_L2": residual,
                "algebraic_error_L2": algebraic_error,
                "discretisation_error_L2": discretisation_error,
                "Ealg_over_Edisc": ratio,
                "passes_1e-3_target": bool(ratio <= TARGET_RATIO),
            }
        )
    return rows, reference_residual


def verify_nonzero_boundary_row(n: int = 128) -> dict[str, float]:
    rhs, _, forcing, hx, hy = manufactured_system(n)
    y = np.arange(n, dtype=float) * hy
    j = n // 4  # y=0.5, where sin(pi*y)=1.
    expected_left = KAPPA**2 / hx**2 * np.sin(np.pi * y[j])
    expected_right = KAPPA**2 / hx**2 * np.exp(X_LENGTH) * np.sin(np.pi * y[j])
    measured_left = float(rhs[0, j] - forcing[0, j])
    measured_right = float(rhs[-1, j] - forcing[-1, j])
    if not np.isclose(measured_left, expected_left, rtol=1e-13, atol=1e-13):
        raise AssertionError("left non-zero Dirichlet contribution is incorrect")
    if not np.isclose(measured_right, expected_right, rtol=1e-13, atol=1e-13):
        raise AssertionError("right non-zero Dirichlet contribution is incorrect")
    return {
        "n": float(n),
        "y_check": float(y[j]),
        "left_expected_RHS_contribution": float(expected_left),
        "left_assembled_RHS_contribution": measured_left,
        "right_expected_RHS_contribution": float(expected_right),
        "right_assembled_RHS_contribution": measured_right,
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    refinement, fitted_slope = run_refinement()
    tolerance_rows, reference_residual = run_tolerance_study()
    boundary_check = verify_nonzero_boundary_row()
    finest_rhs, _, _, finest_hx, finest_hy = manufactured_system(MESH_SIZES[-1])
    direct_reference = spectral_direct_reference(finest_rhs, finest_hx, finest_hy)
    cg_crosscheck, cg_crosscheck_iterations, cg_crosscheck_residual = conjugate_gradient(
        finest_rhs, finest_hx, finest_hy, 1e-10
    )
    crosscheck_error = weighted_l2(
        cg_crosscheck - direct_reference, finest_hx, finest_hy
    )
    passing = [row for row in tolerance_rows if row["passes_1e-3_target"]]
    loosest_passing = passing[0] if passing else None

    write_csv(RESULTS_DIR / "mms_refinement.csv", refinement)
    write_csv(RESULTS_DIR / "cg_tolerance_n128.csv", tolerance_rows)
    summary = {
        "status": "standalone manufactured-solution verification demonstration",
        "assumptions": {
            "equation": "-kappa^2*u_xx - u_yy = f",
            "exact_solution": "exp(x)*sin(pi*y)",
            "kappa": KAPPA,
            "x_interval": [0.0, X_LENGTH],
            "periodic_y_length": Y_PERIOD,
            "meshes": list(MESH_SIZES),
            "norm": "grid-weighted discrete L2",
        },
        "refinement_fitted_slope_L2": fitted_slope,
        "finest_discretisation_error_L2": tolerance_rows[0]["discretisation_error_L2"],
        "direct_reference_method": "Fourier modes + Thomas tridiagonal solve",
        "direct_reference_residual_L2_n128": reference_residual,
        "independent_cg_crosscheck_n128": {
            "cg_rtol": 1e-10,
            "iterations": cg_crosscheck_iterations,
            "explicit_residual_L2": cg_crosscheck_residual,
            "weighted_L2_difference_from_direct_reference": crosscheck_error,
        },
        "loosest_tested_rtol_passing_1e-3": loosest_passing,
        "nonzero_boundary_row_check": boundary_check,
        "not_verified": ["team physical boundary data", "PINN", "magnetic field", "torque functional"],
    }
    (RESULTS_DIR / "verification_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    print("P5 manufactured-solution verification; assumptions:")
    print(f"  kappa={KAPPA:g}, x=[0,{X_LENGTH:g}], periodic y length={Y_PERIOD:g}")
    print("Four-mesh refinement (grid-weighted discrete L2):")
    print("n     hx           error_L2       order     direct_residual_L2")
    for row in refinement:
        order = "-" if row["observed_order_L2"] is None else f"{row['observed_order_L2']:.6f}"
        print(
            f"{row['n']:<5d} {row['hx']:<12.6g} {row['error_L2']:<14.6e} "
            f"{order:<9} {row['reference_residual_L2']:.6e}"
        )
    print(f"fitted slope={fitted_slope:.6f}")
    print("CG tolerance study on n=128:")
    print("rtol       iterations  residual_L2    Ealg/Edisc       pass")
    for row in tolerance_rows:
        print(
            f"{row['rtol']:<10.0e} {row['iterations']:<11d} "
            f"{row['explicit_residual_L2']:<13.6e} "
            f"{row['Ealg_over_Edisc']:<15.6e} {row['passes_1e-3_target']}"
        )
    print("Boundary row check at y=0.5:", json.dumps(boundary_check))
    print(
        "Independent CG/direct-reference difference (weighted L2):",
        f"{crosscheck_error:.6e}",
    )
    print("Results written to", RESULTS_DIR)
    print("Physical case, PINN field, and torque remain unverified without team inputs.")


if __name__ == "__main__":
    main()
