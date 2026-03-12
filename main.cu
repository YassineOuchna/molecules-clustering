#include "FileUtils.hpp"
#include "gpu.cuh"
#include "utils.cuh"

#include <iostream>
#include <fstream>
#include <vector>
#include <iomanip>
#include <chrono>
#include <cmath>
#include <cuda_runtime.h>

static inline double elapsed_s(const std::chrono::high_resolution_clock::time_point& start) {
    return std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - start).count();
}

int main(int argc, char** args)
{
    chrono_type global_start = chrono_time::now();

    if(argc<2){ std::cerr<<"Usage: "<<args[0]<<" <dataset.bin>\n"; return 1; }

    FileUtils file(args[1]);
    size_t N_frames = 90000;
    size_t N_atoms  = file.getN_atoms();
    size_t N_dims   = file.getN_dims();

    std::cout<<"\nDataset info\nFrames: "<<N_frames<<"\nAtoms: "<<N_atoms<<"\n\n";

    std::vector<float> all_data(N_frames*N_atoms*3);
    file.readSnapshotsFastInPlace(0,N_frames-1,all_data);

    const size_t MAX_DATA_CHUNK_SIZE = 12000;
    const size_t NB_FRAMES_PER_CHUNK = get_chunk_frame_nb(MAX_DATA_CHUNK_SIZE,N_atoms,N_dims);
    const size_t NB_ROW_ITERATIONS = (size_t)std::ceil((double)N_frames/NB_FRAMES_PER_CHUNK);

    float* rmsdHostAll = new float[N_frames*N_frames];
    float* rmsdHostChunk = new float[NB_FRAMES_PER_CHUNK*NB_FRAMES_PER_CHUNK];

    std::vector<float> references_coordinates;
    std::vector<float> targets_coordinates;

    float *d_references,*d_targets,*d_rmsd;
    cudaMalloc(&d_references,NB_FRAMES_PER_CHUNK*N_atoms*3*sizeof(float));
    cudaMalloc(&d_targets,NB_FRAMES_PER_CHUNK*N_atoms*3*sizeof(float));
    cudaMalloc(&d_rmsd,NB_FRAMES_PER_CHUNK*NB_FRAMES_PER_CHUNK*sizeof(float));

    float *d_cx_ref,*d_cy_ref,*d_cz_ref,*d_G_ref;
    float *d_cx_tgt,*d_cy_tgt,*d_cz_tgt,*d_G_tgt;
    cudaMalloc(&d_cx_ref,NB_FRAMES_PER_CHUNK*sizeof(float));
    cudaMalloc(&d_cy_ref,NB_FRAMES_PER_CHUNK*sizeof(float));
    cudaMalloc(&d_cz_ref,NB_FRAMES_PER_CHUNK*sizeof(float));
    cudaMalloc(&d_G_ref ,NB_FRAMES_PER_CHUNK*sizeof(float));
    cudaMalloc(&d_cx_tgt,NB_FRAMES_PER_CHUNK*sizeof(float));
    cudaMalloc(&d_cy_tgt,NB_FRAMES_PER_CHUNK*sizeof(float));
    cudaMalloc(&d_cz_tgt,NB_FRAMES_PER_CHUNK*sizeof(float));
    cudaMalloc(&d_G_tgt ,NB_FRAMES_PER_CHUNK*sizeof(float));

    dim3 threads(32,8);

    for(size_t row=0;row<NB_ROW_ITERATIONS;row++)
    {
        size_t start_row = row*NB_FRAMES_PER_CHUNK;
        size_t stop_row  = std::min(start_row+NB_FRAMES_PER_CHUNK,N_frames);
        size_t nb_ref = stop_row-start_row;

        file.extractSnapshotsFastInPlace(start_row,stop_row,all_data,references_coordinates);
        cudaMemcpy(d_references,references_coordinates.data(),references_coordinates.size()*sizeof(float),cudaMemcpyHostToDevice);

        computeCentroidsG<<<(nb_ref+127)/128,128>>>(d_references,N_atoms,nb_ref,d_cx_ref,d_cy_ref,d_cz_ref,d_G_ref);

        for(size_t col=row;col<NB_ROW_ITERATIONS;col++)
        {
            size_t start_col = col*NB_FRAMES_PER_CHUNK;
            size_t stop_col  = std::min(start_col+NB_FRAMES_PER_CHUNK,N_frames);
            size_t nb_tgt = stop_col-start_col;

            file.extractSnapshotsFastInPlace(start_col,stop_col,all_data,targets_coordinates);
            cudaMemcpy(d_targets,targets_coordinates.data(),targets_coordinates.size()*sizeof(float),cudaMemcpyHostToDevice);
            computeCentroidsG<<<(nb_tgt+127)/128,128>>>(d_targets,N_atoms,nb_tgt,d_cx_tgt,d_cy_tgt,d_cz_tgt,d_G_tgt);

            dim3 blocks((nb_tgt+threads.x-1)/threads.x,(nb_ref+threads.y-1)/threads.y);
            int TILE = std::min(128,(int)N_atoms);
            size_t smem_bytes = 6*TILE*threads.y*sizeof(float);

            RMSD<<<blocks,threads,smem_bytes>>>(d_references,d_targets,N_atoms,nb_ref,nb_tgt,d_cx_ref,d_cy_ref,d_cz_ref,d_G_ref,d_cx_tgt,d_cy_tgt,d_cz_tgt,d_G_tgt,d_rmsd);

            cudaMemcpy(rmsdHostChunk,d_rmsd,nb_ref*nb_tgt*sizeof(float),cudaMemcpyDeviceToHost);

            for(size_t i=0;i<nb_ref;i++)
                for(size_t j=0;j<nb_tgt;j++)
                    rmsdHostAll[(start_row+i)*N_frames + start_col+j] = rmsdHostChunk[i*nb_tgt + j];
        }
    }

    // debug upper triangle
    size_t window=5;
    for(size_t i=NB_FRAMES_PER_CHUNK-window;i<NB_FRAMES_PER_CHUNK+window;i++){
        for(size_t j=NB_FRAMES_PER_CHUNK-window;j<NB_FRAMES_PER_CHUNK+window;j++)
            std::cout<<rmsdHostAll[i*N_frames+j]<<" ";
        std::cout<<"\n";
    }

    cudaFree(d_references); cudaFree(d_targets); cudaFree(d_rmsd);
    cudaFree(d_cx_ref); cudaFree(d_cy_ref); cudaFree(d_cz_ref); cudaFree(d_G_ref);
    cudaFree(d_cx_tgt); cudaFree(d_cy_tgt); cudaFree(d_cz_tgt); cudaFree(d_G_tgt);

    delete[] rmsdHostAll;
    delete[] rmsdHostChunk;
}