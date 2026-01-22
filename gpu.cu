// gpu.cu - Complete corrected version
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

    float b0 = -a02;
    float b1 = -a12;
    float det = a00 * a11 - a01 * a10;

    // z = 1 and solve by inversing the resulting system
    if (fabsf(det) > 1e-8f) {
        v[0] = (b0 * a11 - b1 * a01) / det;
        v[1] = (a00 * b1 - a10 * b0) / det;
        v[2] = 1.0f;
    } else {
        
        b0 = -a01;
        b1 = -a21;
        det = a00 * a22 - a02 * a20;
        
        // y = 1 and solve by inversing the resulting system
        if (fabsf(det) > 1e-8f) {
            v[0] = (b0 * a22 - b1 * a02) / det;
            v[1] = 1.0f;
            v[2] = (a00 * b1 - a20 * b0) / det;
        } else {
            b0 = -a00;
            b1 = -a10;
            det = a01 * a12 - a02 * a11;
            
            // x = 1 and solve by inversing the resulting system
            if (fabsf(det) > 1e-8f) {
                v[0] = 1.0f;
                v[1] = (b0 * a12 - b1 * a02) / det;
                v[2] = (a01 * b1 - a11 * b0) / det;
            } else {
                // Fallback
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

// Robust eigenvalue solver using iterative method
__device__
void compute_eigenvalues_symmetric_3x3(float m00, float m01, float m02,
                                       float m11, float m12, float m22,
                                       float* lambda)
{
    float trace = m00 + m11 + m22;
    float mean = trace / 3.0f;
    
    // Shift matrix
    float sm00 = m00 - mean;
    float sm11 = m11 - mean;
    float sm22 = m22 - mean;
    
    float p = sm00*sm00 + sm11*sm11 + sm22*sm22 + 2.0f*(m01*m01 + m02*m02 + m12*m12);
    p = sqrtf(p / 6.0f);
    
    float invp = (p > 1e-8f) ? (1.0f / p) : 0.0f;
    
    float b00 = sm00 * invp;
    float b01 = m01 * invp;
    float b02 = m02 * invp;
    float b11 = sm11 * invp;
    float b12 = m12 * invp;
    float b22 = sm22 * invp;
    
    float det = b00*(b11*b22 - b12*b12) - b01*(b01*b22 - b12*b02) + b02*(b01*b12 - b11*b02);
    det = det / 2.0f;
    det = fminf(1.0f, fmaxf(-1.0f, det));
    
    float phi = acosf(det) / 3.0f;
    
    lambda[0] = mean + 2.0f * p * cosf(phi);
    lambda[2] = mean + 2.0f * p * cosf(phi + (2.0f * 3.14159265358979323846f / 3.0f));
    lambda[1] = 3.0f * mean - lambda[0] - lambda[2];
    
    // Sort
    if (lambda[0] < lambda[1]) { float tmp = lambda[0]; lambda[0] = lambda[1]; lambda[1] = tmp; }
    if (lambda[1] < lambda[2]) { float tmp = lambda[1]; lambda[1] = lambda[2]; lambda[2] = tmp; }
    if (lambda[0] < lambda[1]) { float tmp = lambda[0]; lambda[0] = lambda[1]; lambda[1] = tmp; }
}

__global__
void RMSD(
    const float* __restrict__ dst,
    int N_frames,
    int N_atoms,
    float* out
)
{
    int snap = blockIdx.x * blockDim.x + threadIdx.x;
    int ref_idx = blockIdx.y * blockDim.y + threadIdx.y;

    // Only compute upper triangle
    if (snap >= N_frames || ref_idx >= N_frames || snap < ref_idx)
        return;

    size_t block = (size_t)N_atoms * N_frames;

    // ----------------- STEP 0: Centroids -----------------
    float cx=0.f, cy=0.f, cz=0.f;
    float sx=0.f, sy=0.f, sz=0.f;

    for (int a=0; a<N_atoms; ++a)
    {
        size_t xr = 0*block + (size_t)a * N_frames + ref_idx;
        size_t yr = 1*block + (size_t)a * N_frames + ref_idx;
        size_t zr = 2*block + (size_t)a * N_frames + ref_idx;

        int xs = 0*block + a*N_frames + snap;
        int ys = 1*block + a*N_frames + snap;
        int zs = 2*block + a*N_frames + snap;

        cx += dst[xr]; cy += dst[yr]; cz += dst[zr];
        sx += dst[xs]; sy += dst[ys]; sz += dst[zs];
    }

    cx/=N_atoms; cy/=N_atoms; cz/=N_atoms;
    sx/=N_atoms; sy/=N_atoms; sz/=N_atoms;

    // ----------------- STEP 1: Correlation matrix A -----------------
    float a00=0,a01=0,a02=0,a10=0,a11=0,a12=0,a20=0,a21=0,a22=0;

    for (int a=0;a<N_atoms;a++)
    {
        int xr = 0*block + a*N_frames + ref_idx;
        int yr = 1*block + a*N_frames + ref_idx;
        int zr = 2*block + a*N_frames + ref_idx;

        int xs = 0*block + a*N_frames + snap;
        int ys = 1*block + a*N_frames + snap;
        int zs = 2*block + a*N_frames + snap;

        float Xx = dst[xr]-cx; float Xy = dst[yr]-cy; float Xz = dst[zr]-cz;
        float Yx = dst[xs]-sx; float Yy = dst[ys]-sy; float Yz = dst[zs]-sz;

        a00 += Xx*Yx; a01 += Xx*Yy; a02 += Xx*Yz;
        a10 += Xy*Yx; a11 += Xy*Yy; a12 += Xy*Yz;
        a20 += Xz*Yx; a21 += Xz*Yy; a22 += Xz*Yz;
    }

    // ----------------- STEP 2: M = A^T * A -----------------
    float m00 = a00*a00 + a10*a10 + a20*a20;
    float m01 = a00*a01 + a10*a11 + a20*a21;
    float m02 = a00*a02 + a10*a12 + a20*a22;
    float m11 = a01*a01 + a11*a11 + a21*a21;
    float m12 = a01*a02 + a11*a12 + a21*a22;
    float m22 = a02*a02 + a12*a12 + a22*a22;

    // ----------------- STEP 3: Eigenvalues -----------------
    float lambda[3];
    compute_eigenvalues_symmetric_3x3(m00,m01,m02,m11,m12,m22,lambda);

    // ----------------- STEP 4: Eigenvectors (V) -----------------
    float vec[3];

    compute_eigenvector(m00,m01,m02,m11,m12,m22,lambda[0], vec);
    float v0x=vec[0], v0y=vec[1], v0z=vec[2];

    compute_eigenvector(m00,m01,m02,m11,m12,m22,lambda[1], vec);
    float v1x=vec[0], v1y=vec[1], v1z=vec[2];

    compute_eigenvector(m00,m01,m02,m11,m12,m22,lambda[2], vec);
    float v2x=vec[0], v2y=vec[1], v2z=vec[2];

    // ----------------- STEP 5: Orthonormalize V -----------------
    // Gram-Schmidt v1
    float dot = v1x*v0x + v1y*v0y + v1z*v0z;
    v1x -= dot*v0x; v1y -= dot*v0y; v1z -= dot*v0z;
    float mag = sqrtf(v1x*v1x + v1y*v1y + v1z*v1z);
    if(mag>1e-8f) { v1x/=mag; v1y/=mag; v1z/=mag; }

    // v2 = v0 × v1
    v2x = v0y*v1z - v0z*v1y;
    v2y = v0z*v1x - v0x*v1z;
    v2z = v0x*v1y - v0y*v1x;

    // ----------------- STEP 6: Compute U = A*V -----------------
    float av0x = a00*v0x + a01*v0y + a02*v0z;
    float av0y = a10*v0x + a11*v0y + a12*v0z;
    float av0z = a20*v0x + a21*v0y + a22*v0z;

    float av1x = a00*v1x + a01*v1y + a02*v1z;
    float av1y = a10*v1x + a11*v1y + a12*v1z;
    float av1z = a20*v1x + a21*v1y + a22*v1z;

    float av2x = a00*v2x + a01*v2y + a02*v2z;
    float av2y = a10*v2x + a11*v2y + a12*v2z;
    float av2z = a20*v2x + a21*v2y + a22*v2z;

    float s0 = (lambda[0]>1e-6f)?sqrtf(lambda[0]):1.f;
    float s1 = (lambda[1]>1e-6f)?sqrtf(lambda[1]):1.f;
    float s2 = (lambda[2]>1e-6f)?sqrtf(lambda[2]):1.f;

    float u0x = av0x/s0, u0y = av0y/s0, u0z = av0z/s0;
    float u1x = av1x/s1, u1y = av1y/s1, u1z = av1z/s1;
    float u2x = av2x/s2, u2y = av2y/s2, u2z = av2z/s2;

    // ----------------- STEP 7: Compute R = U*V^T -----------------
    float R00 = u0x*v0x + u1x*v1x + u2x*v2x;
    float R01 = u0x*v0y + u1x*v1y + u2x*v2y;
    float R02 = u0x*v0z + u1x*v1z + u2x*v2z;

    float R10 = u0y*v0x + u1y*v1x + u2y*v2x;
    float R11 = u0y*v0y + u1y*v1y + u2y*v2y;
    float R12 = u0y*v0z + u1y*v1z + u2y*v2z;

    float R20 = u0z*v0x + u1z*v1x + u2z*v2x;
    float R21 = u0z*v0y + u1z*v1y + u2z*v2y;
    float R22 = u0z*v0z + u1z*v1z + u2z*v2z;

    // ----------------- STEP 8: Compute RMSD -----------------
    float sum2 = 0.f;
    for (int a=0;a<N_atoms;a++)
    {
        int xr = 0*block + a*N_frames + ref_idx;
        int yr = 1*block + a*N_frames + ref_idx;
        int zr = 2*block + a*N_frames + ref_idx;

        int xs = 0*block + a*N_frames + snap;
        int ys = 1*block + a*N_frames + snap;
        int zs = 2*block + a*N_frames + snap;

        float Xi_x = dst[xr]-cx;
        float Xi_y = dst[yr]-cy;
        float Xi_z = dst[zr]-cz;

        float Yi_x = dst[xs]-sx;
        float Yi_y = dst[ys]-sy;
        float Yi_z = dst[zs]-sz;

        float RYi_x = R00*Yi_x + R01*Yi_y + R02*Yi_z;
        float RYi_y = R10*Yi_x + R11*Yi_y + R12*Yi_z;
        float RYi_z = R20*Yi_x + R21*Yi_y + R22*Yi_z;

        float dx = Xi_x - RYi_x;
        float dy = Xi_y - RYi_y;
        float dz = Xi_z - RYi_z;

        sum2 += dx*dx + dy*dy + dz*dz;
    }

    float rmsd = sqrtf(sum2/N_atoms);

    size_t idx1 = (size_t)ref_idx * N_frames + snap;
    size_t idx2 = (size_t)snap * N_frames + ref_idx;

    out[idx1] = rmsd;
    out[idx2] = rmsd;

}


