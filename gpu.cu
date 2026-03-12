#include "gpu.cuh"
#include "utils.cuh"
#include <cuda_runtime.h>
#include <math_constants.h>


// =======================================================
// Eigenvalues solver (symmetric 3x3)
// =======================================================
__device__
void compute_eigenvalues_symmetric_3x3(float m00, float m01, float m02,
                                       float m11, float m12, float m22,
                                       float* lambda)
{
    float trace = m00 + m11 + m22;
    float mean = trace / 3.0f;

    float sm00 = m00 - mean;
    float sm11 = m11 - mean;
    float sm22 = m22 - mean;

    float p = sm00*sm00 + sm11*sm11 + sm22*sm22
            + 2.0f*(m01*m01 + m02*m02 + m12*m12);
    p = sqrtf(p / 6.0f);

    float invp = (p > 1e-8f) ? (1.0f / p) : 0.0f;

    float b00 = sm00 * invp;
    float b01 = m01 * invp;
    float b02 = m02 * invp;
    float b11 = sm11 * invp;
    float b12 = m12 * invp;
    float b22 = sm22 * invp;

    float det = b00*(b11*b22 - b12*b12)
              - b01*(b01*b22 - b12*b02)
              + b02*(b01*b12 - b11*b02);
    det *= 0.5f;
    det = fminf(1.0f, fmaxf(-1.0f, det));

    float phi = acosf(det) / 3.0f;

    lambda[0] = mean + 2.0f * p * cosf(phi);
    lambda[2] = mean + 2.0f * p * cosf(phi + 2.0f * CUDART_PI_F / 3.0f);
    lambda[1] = 3.0f * mean - lambda[0] - lambda[2];

    if (lambda[0] < lambda[1]) { float t=lambda[0]; lambda[0]=lambda[1]; lambda[1]=t; }
    if (lambda[1] < lambda[2]) { float t=lambda[1]; lambda[1]=lambda[2]; lambda[2]=t; }
    if (lambda[0] < lambda[1]) { float t=lambda[0]; lambda[0]=lambda[1]; lambda[1]=t; }
}

__global__
void computeCentroidsG(const float* coords,
                       size_t N_atoms,
                       size_t N_frames,
                       float* cx,
                       float* cy,
                       float* cz,
                       float* G)
{
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    if(idx>=N_frames) return;

    float sx=0,sy=0,sz=0;

    for(int a=0;a<N_atoms;a++){
        size_t off=a*N_frames+idx;
        sx+=coords[0*N_atoms*N_frames+off];
        sy+=coords[1*N_atoms*N_frames+off];
        sz+=coords[2*N_atoms*N_frames+off];
    }

    sx/=N_atoms; sy/=N_atoms; sz/=N_atoms;

    cx[idx]=sx; cy[idx]=sy; cz[idx]=sz;

    float g=0;
    for(int a=0;a<N_atoms;a++){
        size_t off=a*N_frames+idx;
        float rx=coords[0*N_atoms*N_frames+off]-sx;
        float ry=coords[1*N_atoms*N_frames+off]-sy;
        float rz=coords[2*N_atoms*N_frames+off]-sz;
        g+=rx*rx+ry*ry+rz*rz;
    }

    G[idx]=g;
}

__global__
void RMSD(const float* refs,
          const float* tgts,
          size_t N_atoms,
          size_t N_ref,
          size_t N_tgt,
          const float* cx_ref,
          const float* cy_ref,
          const float* cz_ref,
          const float* G_ref,
          const float* cx_tgt,
          const float* cy_tgt,
          const float* cz_tgt,
          const float* G_tgt,
          float* rmsd)
{
    extern __shared__ float smem[];
    const int TILE = blockDim.x;  // number of atoms per tile

    float* s_ref_x = smem;
    float* s_ref_y = s_ref_x + TILE * blockDim.y;
    float* s_ref_z = s_ref_y + TILE * blockDim.y;

    float* s_tgt_x = s_ref_z + TILE * blockDim.y;
    float* s_tgt_y = s_tgt_x + TILE * blockDim.y;
    float* s_tgt_z = s_tgt_y + TILE * blockDim.y;

    int r = blockIdx.y * blockDim.y + threadIdx.y;
    int t = blockIdx.x * blockDim.x + threadIdx.x;

    if(r >= N_ref || t >= N_tgt) return;

    float rcx = cx_ref[r];
    float rcy = cy_ref[r];
    float rcz = cz_ref[r];

    float scx = cx_tgt[t];
    float scy = cy_tgt[t];
    float scz = cz_tgt[t];

    float a00=0,a01=0,a02=0;
    float a10=0,a11=0,a12=0;
    float a20=0,a21=0,a22=0;

    for(int start=0; start<N_atoms; start+=TILE)
    {
        int idx = start + threadIdx.x;

        // load reference atoms into shared memory
        if(idx < N_atoms)
        {
            s_ref_x[threadIdx.x*blockDim.y + threadIdx.y] = refs[0*N_atoms*N_ref + idx*N_ref + r] - rcx;
            s_ref_y[threadIdx.x*blockDim.y + threadIdx.y] = refs[1*N_atoms*N_ref + idx*N_ref + r] - rcy;
            s_ref_z[threadIdx.x*blockDim.y + threadIdx.y] = refs[2*N_atoms*N_ref + idx*N_ref + r] - rcz;

            s_tgt_x[threadIdx.x*blockDim.y + threadIdx.y] = tgts[0*N_atoms*N_tgt + idx*N_tgt + t] - scx;
            s_tgt_y[threadIdx.x*blockDim.y + threadIdx.y] = tgts[1*N_atoms*N_tgt + idx*N_tgt + t] - scy;
            s_tgt_z[threadIdx.x*blockDim.y + threadIdx.y] = tgts[2*N_atoms*N_tgt + idx*N_tgt + t] - scz;
        }

        __syncthreads();

        int tile_end = min(TILE, (int)(N_atoms-start));

        for(int k=0;k<tile_end;k++)
        {
            float rx = s_ref_x[k*blockDim.y + threadIdx.y];
            float ry = s_ref_y[k*blockDim.y + threadIdx.y];
            float rz = s_ref_z[k*blockDim.y + threadIdx.y];

            float sx = s_tgt_x[k*blockDim.y + threadIdx.x];
            float sy = s_tgt_y[k*blockDim.y + threadIdx.x];
            float sz = s_tgt_z[k*blockDim.y + threadIdx.x];

            a00 += rx*sx; a01 += rx*sy; a02 += rx*sz;
            a10 += ry*sx; a11 += ry*sy; a12 += ry*sz;
            a20 += rz*sx; a21 += rz*sy; a22 += rz*sz;
        }

        __syncthreads();
    }

    // covariance matrix M = A^T * A
    float m00=a00*a00+a10*a10+a20*a20;
    float m01=a00*a01+a10*a11+a20*a21;
    float m02=a00*a02+a10*a12+a20*a22;
    float m11=a01*a01+a11*a11+a21*a21;
    float m12=a01*a02+a11*a12+a21*a22;
    float m22=a02*a02+a12*a12+a22*a22;

    float lambda[3];
    compute_eigenvalues_symmetric_3x3(m00,m01,m02,m11,m12,m22,lambda);

    float sigma_sum = sqrtf(fmaxf(lambda[0],0.f)) +
                      sqrtf(fmaxf(lambda[1],0.f)) +
                      sqrtf(fmaxf(lambda[2],0.f));

    float rmsd2 = (G_ref[r] + G_tgt[t] - 2.f*sigma_sum)/N_atoms;
    rmsd[r*N_tgt + t] = sqrtf(fmaxf(rmsd2,0.f));
}