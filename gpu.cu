#include "gpu.cuh"
#include <cuda_runtime.h>

__device__
void compute_eigenvector(float m00, float m01, float m02,
                         float m11, float m12, float m22,
                         float lambda, float* v)
{
    // Solve the homogeneous system:
    // (m00 - lambda)*x1 + m01*x2 + m02*x3 = 0
    // m01*x1 + (m11 - lambda)*x2 + m12*x3 = 0
    // m02*x1 + m12*x2 + (m22 - lambda)*x3 = 0
    
    float a00 = m00 - lambda;
    float a01 = m01;
    float a02 = m02;
    float a10 = m01;
    float a11 = m11 - lambda;
    float a12 = m12;
    float a20 = m02;
    float a21 = m12;
    float a22 = m22 - lambda;

    // Set x3 = 1 and solve the 2x2 system for x1 and x2:
    // a00*x1 + a01*x2 = -a02
    // a10*x1 + a11*x2 = -a12

    float b0 = -a02;
    float b1 = -a12;
    float det = a00 * a11 - a01 * a10;

    if (fabsf(det) > 1e-8f) {
        v[0] = (b0 * a11 - b1 * a01) / det;
        v[1] = (a00 * b1 - a10 * b0) / det;
        v[2] = 1.0f;
    } else {
        // Try x2 = 1, solve for x1 and x3:
        // a00*x1 + a02*x3 = -a01
        // a20*x1 + a22*x3 = -a21
        
        b0 = -a01;
        b1 = -a21;
        det = a00 * a22 - a02 * a20;
        
        if (fabsf(det) > 1e-8f) {
            v[0] = (b0 * a22 - b1 * a02) / det;
            v[1] = 1.0f;
            v[2] = (a00 * b1 - a20 * b0) / det;
        } else {
            // Try x1 = 1, solve for x2 and x3:
            // a01*x2 + a02*x3 = -a00
            // a11*x2 + a12*x3 = -a10
            
            b0 = -a00;
            b1 = -a10;
            det = a01 * a12 - a02 * a11;
            
            if (fabsf(det) > 1e-8f) {
                v[0] = 1.0f;
                v[1] = (b0 * a12 - b1 * a02) / det;
                v[2] = (a01 * b1 - a11 * b0) / det;
            } else {
                v[0] = 1.0f;
                v[1] = 0.0f;
                v[2] = 0.0f;
            }
        }
    }

    // Normalize
    float norm = sqrtf(v[0]*v[0] + v[1]*v[1] + v[2]*v[2]);
    if (norm > 1e-8f) {
        v[0] /= norm;
        v[1] /= norm;
        v[2] /= norm;
    }
}

__global__
void computeA(
    const float* __restrict__ dst,   // reordered: [X-block | Y-block | Z-block]
    float* __restrict__ outA,
    int N_frames,
    int N_atoms,
    int ref_idx
)
{
    int snap = blockIdx.x * blockDim.x + threadIdx.x;
    if (snap >= N_frames || snap == ref_idx)
        return;

    int block = N_atoms * N_frames;   // size of one coord block

    // 3×3 accumulator
    float a00=0, a01=0, a02=0;
    float a10=0, a11=0, a12=0;
    float a20=0, a21=0, a22=0;

    for (int a = 0; a < N_atoms; ++a)
    {
        int xr = 0*block + a*N_frames + ref_idx;
        int yr = 1*block + a*N_frames + ref_idx;
        int zr = 2*block + a*N_frames + ref_idx;

        int xs = 0*block + a*N_frames + snap;
        int ys = 1*block + a*N_frames + snap;
        int zs = 2*block + a*N_frames + snap;

        float Xx = dst[xr];
        float Xy = dst[yr];
        float Xz = dst[zr];

        float Yx = dst[xs];
        float Yy = dst[ys];
        float Yz = dst[zs];

        a00 += Xx * Yx;  a01 += Xx * Yy;  a02 += Xx * Yz;
        a10 += Xy * Yx;  a11 += Xy * Yy;  a12 += Xy * Yz;
        a20 += Xz * Yx;  a21 += Xz * Yy;  a22 += Xz * Yz;
    }

    // Compute M = A^T A (symmetric 3×3)
    float m00 = a00*a00 + a10*a10 + a20*a20;
    float m01 = a00*a01 + a10*a11 + a20*a21;
    float m02 = a00*a02 + a10*a12 + a20*a22;

    float m11 = a01*a01 + a11*a11 + a21*a21;
    float m12 = a01*a02 + a11*a12 + a21*a22;

    float m22 = a02*a02 + a12*a12 + a22*a22;

    // Compute coefficients of characteristic polynomial det(M - λI) = 0
    // For symmetric matrix: -λ³ + c₂λ² + c₁λ + c₀ = 0
    
    float c2 = m00 + m11 + m22;  // trace(M)
    
    float c1 = -(m00*m11 - m01*m01 + m00*m22 - m02*m02 + m11*m22 - m12*m12);
    
    float c0 = m00*(m11*m22 - m12*m12) - m01*(m01*m22 - m12*m02) + m02*(m01*m12 - m11*m02);

    // Solve cubic: λ³ - c₂λ² - c₁λ - c₀ = 0
    // Using substitution λ = t + c₂/3 to get depressed cubic: t³ + pt + q = 0
    
    float p = c1 - c2*c2/3.0f;
    float q = c0 + 2.0f*c2*c2*c2/27.0f - c2*c1/3.0f;
    
    // Cardano's formula
    float discriminant = q*q/4.0f + p*p*p/27.0f;
    
    float lambda0, lambda1, lambda2;
    
    if (discriminant >= 0) {
        // One real root, two complex (shouldn't happen for symmetric positive semi-definite)
        float sqrt_disc = sqrtf(discriminant);
        float u = cbrtf(-q/2.0f + sqrt_disc);
        float v = cbrtf(-q/2.0f - sqrt_disc);
        lambda0 = u + v + c2/3.0f;
        lambda1 = lambda0;
        lambda2 = lambda0;
    } else {
        // Three distinct real roots (typical case)
        float r = sqrtf(-p*p*p/27.0f);
        float theta = acosf(-q/(2.0f*r));
        float rho = cbrtf(r);
        
        lambda0 = 2.0f * rho * cosf(theta/3.0f) + c2/3.0f;
        lambda1 = 2.0f * rho * cosf((theta + 2.0f*3.14159265359f)/3.0f) + c2/3.0f;
        lambda2 = 2.0f * rho * cosf((theta + 4.0f*3.14159265359f)/3.0f) + c2/3.0f;
    }

    if (snap == 1) {
        printf("Debug snap %d:\n", snap);
        printf("  M trace: %.6f\n", c2);
        printf("  M determinant: %.6f\n", c0);
        printf("  Eigenvalues: %.6f, %.6f, %.6f\n", lambda0, lambda1, lambda2);
    }

    // Compute eigenvectors
    float v0[3], v1[3], v2[3];
    
    compute_eigenvector(m00, m01, m02, m11, m12, m22, lambda0, v0);
    compute_eigenvector(m00, m01, m02, m11, m12, m22, lambda1, v1);
    compute_eigenvector(m00, m01, m02, m11, m12, m22, lambda2, v2);

    // Compute singular values (sqrt of eigenvalues, ensuring non-negative)
    float s0 = sqrtf(fmaxf(lambda0, 0.0f));
    float s1 = sqrtf(fmaxf(lambda1, 0.0f));
    float s2 = sqrtf(fmaxf(lambda2, 0.0f));
    
    // Compute A*V
    float u0[3], u1[3], u2[3];
    
    u0[0] = a00*v0[0] + a01*v0[1] + a02*v0[2];
    u0[1] = a10*v0[0] + a11*v0[1] + a12*v0[2];
    u0[2] = a20*v0[0] + a21*v0[1] + a22*v0[2];
    
    u1[0] = a00*v1[0] + a01*v1[1] + a02*v1[2];
    u1[1] = a10*v1[0] + a11*v1[1] + a12*v1[2];
    u1[2] = a20*v1[0] + a21*v1[1] + a22*v1[2];
    
    u2[0] = a00*v2[0] + a01*v2[1] + a02*v2[2];
    u2[1] = a10*v2[0] + a11*v2[1] + a12*v2[2];
    u2[2] = a20*v2[0] + a21*v2[1] + a22*v2[2];
    
    // Normalize by singular values to get U = A*V*Σ^(-1)
    if (s0 > 1e-8f) {
        u0[0] /= s0; u0[1] /= s0; u0[2] /= s0;
    }
    if (s1 > 1e-8f) {
        u1[0] /= s1; u1[1] /= s1; u1[2] /= s1;
    }
    if (s2 > 1e-8f) {
        u2[0] /= s2; u2[1] /= s2; u2[2] /= s2;
    }
    
    // Store rotation matrix U (row-major)
    float* U = outA + snap * 9;
    U[0] = u0[0]; U[1] = u1[0]; U[2] = u2[0];
    U[3] = u0[1]; U[4] = u1[1]; U[5] = u2[1];
    U[6] = u0[2]; U[7] = u1[2]; U[8] = u2[2];

    if (snap == 1) {
        printf("  Singular values: %.6f, %.6f, %.6f\n", s0, s1, s2);
        printf("  Rotation matrix U:\n");
        printf("  [%.6f %.6f %.6f]\n", U[0], U[1], U[2]);
        printf("  [%.6f %.6f %.6f]\n", U[3], U[4], U[5]);
        printf("  [%.6f %.6f %.6f]\n", U[6], U[7], U[8]);
    }
    
    // Calculate RMSD
    float sum_squared_dist = 0.0f;
    
    for (int a = 0; a < N_atoms; ++a)
    {
        int xr = 0*block + a*N_frames + ref_idx;
        int yr = 1*block + a*N_frames + ref_idx;
        int zr = 2*block + a*N_frames + ref_idx;
        
        float Xi_x = dst[xr];
        float Xi_y = dst[yr];
        float Xi_z = dst[zr];
        
        int xs = 0*block + a*N_frames + snap;
        int ys = 1*block + a*N_frames + snap;
        int zs = 2*block + a*N_frames + snap;
        
        float Yi_x = dst[xs];
        float Yi_y = dst[ys];
        float Yi_z = dst[zs];
        
        // Apply rotation: U*Xi
        float UXi_x = U[0]*Xi_x + U[1]*Xi_y + U[2]*Xi_z;
        float UXi_y = U[3]*Xi_x + U[4]*Xi_y + U[5]*Xi_z;
        float UXi_z = U[6]*Xi_x + U[7]*Xi_y + U[8]*Xi_z;
        
        // Compute squared distance
        float diff_x = UXi_x - Yi_x;
        float diff_y = UXi_y - Yi_y;
        float diff_z = UXi_z - Yi_z;
        
        sum_squared_dist += diff_x*diff_x + diff_y*diff_y + diff_z*diff_z;
    }
    
    float rmsd = sqrtf(sum_squared_dist / N_atoms);
    
    if (snap != ref_idx && snap <= 3) {
        printf("  RMSD: %.6f\n", rmsd);
    }
}