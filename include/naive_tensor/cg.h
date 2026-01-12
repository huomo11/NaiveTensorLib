#ifndef NAIVE_CG_H
#define NAIVE_CG_H

#include "tensor.h"
#include "ops.h"
#include <iostream>
#include <cmath>
#include <vector>
#include <deque>
#include <iomanip>

namespace naive
{
    namespace solver
    {
        // ==========================================
        // Standard PCG (FP64 Baseline)
        // ==========================================
        template <typename T>
        void pcg_solve(const Tensor<T> &A, const Tensor<T> &b, const Tensor<T> &inv_diag,
                       Tensor<T> &x, int max_iter, double tol)
        {
            Tensor<T> r = b; // r0 = b (assuming x0 = 0)
            Tensor<T> z(r.shape());
            ops::elwise_mult(inv_diag, r, z);
            Tensor<T> p = z;
            Tensor<T> Ap(b.shape());
            double rho = ops::dot(r, z);

            for (int k = 0; k < max_iter; ++k)
            {
                if (ops::norm(r) < tol)
                {
                    std::cout << "PCG Converged at iter " << k << std::endl;
                    return;
                }

                ops::gemv(A, p, Ap);
                double pAp = ops::dot(p, Ap);
                double alpha = rho / pAp;

                ops::axpy(static_cast<T>(alpha), p, x);
                ops::axpy(static_cast<T>(-alpha), Ap, r);

                ops::elwise_mult(inv_diag, r, z);
                double rho_new = ops::dot(r, z);
                double beta = rho_new / rho;

                ops::scal(static_cast<T>(beta), p);
                ops::axpy(static_cast<T>(1.0), z, p);

                rho = rho_new;
            }
            std::cout << "PCG Reached max iterations." << std::endl;
        }

        // ==========================================
        // Standard CG (Identity Preconditioner)
        // ==========================================
        template <typename T>
        void cg_solve(const Tensor<T> &A, const Tensor<T> &b, Tensor<T> &x, int max_iter, double tol)
        {
            Tensor<T> identity_precond(b.shape());
            identity_precond.fill(static_cast<T>(1.0));
            pcg_solve(A, b, identity_precond, x, max_iter, tol);
        }

        // ==========================================
        // FP16 Simulation (Crash Demo)
        // ==========================================
        template <typename T>
        void pcg_solve_fp16_simulation(const Tensor<T> &A, const Tensor<T> &b, const Tensor<T> &inv_diag,
                                       Tensor<T> &x, int max_iter, double tol)
        {
            std::cout << ">>> SIMULATION: FP16 Quantization Enabled <<<" << std::endl;
            Tensor<T> r = b;
            ops::quantize_fp16(r);
            Tensor<T> z(r.shape());
            ops::elwise_mult(inv_diag, r, z);
            ops::quantize_fp16(z);
            Tensor<T> p = z;
            Tensor<T> Ap(b.shape());
            double rho = ops::dot(r, z);
            double rho_old = rho;

            for (int k = 0; k < max_iter; ++k)
            {
                if (std::abs(rho) < 1e-50 || std::isnan(rho))
                {
                    std::cout << "CRASH: rho instability at iter " << k << std::endl;
                    break;
                }
                if (ops::norm(r) < tol)
                {
                    std::cout << "Converged at iter " << k << std::endl;
                    break;
                }

                ops::gemv(A, p, Ap);
                ops::quantize_fp16(Ap);

                double pAp = ops::dot(p, Ap);
                double alpha = rho / pAp;

                ops::axpy(static_cast<T>(alpha), p, x);
                ops::axpy(static_cast<T>(-alpha), Ap, r);
                ops::quantize_fp16(r);

                ops::elwise_mult(inv_diag, r, z);
                ops::quantize_fp16(z);

                double rho_new = ops::dot(r, z);
                double beta = rho_new / rho_old;

                ops::scal(static_cast<T>(beta), p);
                ops::axpy(static_cast<T>(1.0), z, p);
                ops::quantize_fp16(p);

                rho_old = rho_new;
                rho = rho_new;

                if (k % 50 == 0)
                    std::cout << "Iter " << k << " |r|=" << ops::norm(r) << std::endl;
            }
        }

        // ==========================================
        // AMP-PCG (Algorithm 4.1)
        // ==========================================

        enum class Precision
        {
            FP64,
            FP32,
            FP16
        };

        struct AmpParams
        {
            double tol = 1e-10;
            double tau_z_s = 1e-4;
            double tau_z_h = 1e-6;
            int d = 10;
            double C = 1.0;
        };

        inline double calculate_eta_k(const std::deque<double> &r_norms, int k, const AmpParams &params)
        {
            if (k < params.d)
                return 0.0;
            if (r_norms.size() < 2)
                return 0.0;

            double sum = 0.0;
            const double eps32 = 1.19209290e-07;

            for (size_t i = 1; i < r_norms.size(); ++i)
            {
                double r_prev = r_norms[i - 1];
                double r_curr = r_norms[i];
                sum += eps32 * ((3.0 + params.C) * r_prev + (2.0 + params.C) * r_curr);
            }
            return sum;
        }

        template <typename T>
        void amp_pcg_solve(const Tensor<T> &A, const Tensor<T> &b, const Tensor<T> &inv_diag,
                           Tensor<T> &x, int max_iter, AmpParams params)
        {
            std::cout << ">>> AMP-PCG: Algorithm 4.1 Started <<<" << std::endl;

            // Initialize
            Tensor<T> r = b;
            Tensor<T> Ax(b.shape());
            ops::gemv(A, x, Ax);
            ops::axpy(static_cast<T>(-1.0), Ax, r); // r = b - Ax

            double delta_k = ops::norm(r);
            double b_norm = ops::norm(b);
            std::deque<double> r_norms;
            r_norms.push_back(delta_k);

            Tensor<T> y(r.shape());
            Tensor<T> z(r.shape());
            Tensor<T> p(r.shape());
            Tensor<T> q(b.shape());

            double rho_prev = 1.0;
            double rho_curr = 1.0;
            Precision u_z = Precision::FP64;
            Precision u_r = Precision::FP64;
            Precision last_u_z = Precision::FP64;

            for (int k = 0; k < max_iter; ++k)
            {
                // 1. Precision Selection
                double nu_k = delta_k / b_norm;
                if (r_norms.size() > (size_t)params.d + 1)
                    r_norms.pop_front();
                double eta_k = calculate_eta_k(r_norms, k, params);

                if (nu_k >= params.tau_z_s)
                    u_z = Precision::FP64;
                else if (nu_k >= params.tau_z_h)
                    u_z = Precision::FP32;
                else
                    u_z = Precision::FP16;

                if (k > params.d && eta_k < params.tol * b_norm)
                    u_r = Precision::FP32;
                else
                    u_r = Precision::FP64;

                // 2. Dynamic Scaling & Preconditioning
                double scale_factor = (delta_k < 1e-30) ? 1.0 : (1.0 / delta_k);
                ops::copy(r, y);
                ops::scal(static_cast<T>(scale_factor), y);

                if (u_z == Precision::FP32)
                    ops::quantize_fp32(y);
                else if (u_z == Precision::FP16)
                    ops::quantize_fp16(y);

                ops::elwise_mult(inv_diag, y, z);

                if (u_z == Precision::FP32)
                    ops::quantize_fp32(z);
                else if (u_z == Precision::FP16)
                    ops::quantize_fp16(z);

                // 3. Search Direction
                rho_curr = ops::dot(r, z);
                double beta = 0.0;

                // Soft restart if precision changed or instability detected
                if (k > 0 && u_z == last_u_z && std::abs(rho_prev) > 1e-50)
                {
                    beta = rho_curr / rho_prev;
                }
                else
                {
                    beta = 0.0;
                    if (k == 0)
                        ops::copy(z, p);
                }

                if (k > 0)
                {
                    ops::scal(static_cast<T>(beta), p);
                    ops::axpy(static_cast<T>(1.0), z, p);
                }

                if (u_z == Precision::FP32)
                    ops::quantize_fp32(p);
                else if (u_z == Precision::FP16)
                    ops::quantize_fp16(p);

                // 4. Matrix-Vector Product & Update
                ops::gemv(A, p, q);
                if (u_r == Precision::FP32)
                    ops::quantize_fp32(q);

                double gamma = ops::dot(q, p);
                if (std::abs(gamma) < 1e-50)
                {
                    std::cout << "Breakdown: gamma too small." << std::endl;
                    break;
                }
                double alpha = rho_curr / gamma;

                ops::axpy(static_cast<T>(alpha), p, x);
                ops::axpy(static_cast<T>(-alpha), q, r);
                if (u_r == Precision::FP32)
                    ops::quantize_fp32(r);

                // 5. Update State
                delta_k = ops::norm(r);
                r_norms.push_back(delta_k);
                rho_prev = rho_curr;
                last_u_z = u_z;

                if (k % 50 == 0 || delta_k < params.tol * b_norm)
                {
                    std::cout << "Iter " << std::setw(4) << k
                              << " | ||r||=" << std::scientific << delta_k
                              << " | Prec=" << (int)u_z
                              << std::endl;
                }

                if (delta_k <= params.tol * b_norm)
                {
                    std::cout << "AMP-PCG Converged at iter " << k << std::endl;
                    break;
                }
            }
        }
    } // namespace solver
} // namespace naive

#endif // NAIVE_CG_H