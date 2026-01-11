#ifndef NAIVE_CG_H
#define NAIVE_CG_H

#include "tensor.h"
#include "ops.h"
#include <iostream>
#include <cmath>

namespace naive {
namespace solver {

    // ==========================================
    // Preconditioned Conjugate Gradient (PCG)
    // Solves Ax = b with preconditioner M^-1
    // ==========================================
    template <typename T>
    void pcg_solve(const Tensor<T>& A, const Tensor<T>& b, const Tensor<T>& inv_diag, 
                   Tensor<T>& x, int max_iter, double tol) {
        
        // --- Initialization ---
        // r0 = b - A * x0. Assuming x0 = 0, so r0 = b.
        Tensor<T> r = b; 
        
        // z0 = M^-1 * r0
        Tensor<T> z(r.shape());
        ops::elwise_mult(inv_diag, r, z);

        // p0 = z0
        Tensor<T> p = z;
        Tensor<T> Ap(b.shape());

        // rho_0 = r0^T * z0
        double rho = ops::dot(r, z);

        // --- Iteration Loop ---
        for (int k = 0; k < max_iter; ++k) {
            // Check convergence using Euclidean norm of residual
            if (ops::norm(r) < tol) {
                std::cout << "PCG Converged at iter " << k << std::endl;
                return;
            }

            // 1. Matrix-Vector Multiplication: Ap = A * p
            ops::gemv(A, p, Ap);

            // 2. Step length: alpha = rho / (p^T * A * p)
            double pAp = ops::dot(p, Ap);
            double alpha = rho / pAp;

            // 3. Update solution: x = x + alpha * p
            ops::axpy(static_cast<T>(alpha), p, x);

            // 4. Update residual: r = r - alpha * Ap
            ops::axpy(static_cast<T>(-alpha), Ap, r);

            // 5. Preconditioning: z = M^-1 * r
            ops::elwise_mult(inv_diag, r, z);

            // 6. Update search direction
            double rho_new = ops::dot(r, z);
            double beta = rho_new / rho;
            
            // p = z + beta * p
            ops::scal(static_cast<T>(beta), p);      // p = beta * p
            ops::axpy(static_cast<T>(1.0), z, p);    // p = p + z

            // Prepare for next iteration
            rho = rho_new;
        }
        
        std::cout << "PCG Reached max iterations without full convergence." << std::endl;
    }

    // ==========================================
    // Standard Conjugate Gradient (CG)
    // Wrapper around PCG with Identity Preconditioner
    // ==========================================
    template <typename T>
    void cg_solve(const Tensor<T>& A, const Tensor<T>& b, Tensor<T>& x, int max_iter, double tol) {
        // Construct Identity Preconditioner (all 1s)
        Tensor<T> identity_precond(b.shape());
        T* ptr = identity_precond.data();
        size_t size = identity_precond.size();
        
        // Fill with 1.0
        for (size_t i = 0; i < size; ++i) {
            ptr[i] = static_cast<T>(1.0);
        }

        pcg_solve(A, b, identity_precond, x, max_iter, tol);
    }

} // namespace solver
} // namespace naive

#endif // NAIVE_CG_H