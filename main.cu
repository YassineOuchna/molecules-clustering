#include "FileUtils.hpp"
#include "gpu.cuh"
#include "utils.cuh"

#include <iostream>
#include <vector>
#include <iomanip>
#include <chrono>
#include <cmath>
#include <cuda_runtime.h>

static inline double elapsed_s(const std::chrono::high_resolution_clock::time_point& start) {
    return std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - start).count();
}

int main(int argc, char** args) {
    auto global_start = std::chrono::high_resolution_clock::now();

    if(argc < 2){
        std::cerr << "Usage: " << args[0] << " <dataset.bin>\n";
        return 1;
    }

    std::string file_name = args[1];
    FileUtils file(file_name);

    size_t N_frames = 90000;
    size_t N_atoms  = file.getN_atoms();
    size_t N_dims   = file.getN_dims();

    std::cout << "\nDataset info\nFrames: " << N_frames
              << "\nAtoms: " << N_atoms << "\n\n";

    // ------------------------------------------------------------
    // Load dataset
    // ------------------------------------------------------------
    std::vector<float> all_data(N_frames * N_atoms * 3);
    auto t_read = std::chrono::high_resolution_clock::now();
    file.readSnapshotsFastInPlace(0, N_frames-1, all_data);
    double read_time = elapsed_s(t_read);

    std::cout << "Read dataset : " << (double)N_frames / read_time
              << " molecules/s (" << read_time << " s)\n";

    // ------------------------------------------------------------
    // Chunk sizing
    // ------------------------------------------------------------
    const size_t MAX_DATA_CHUNK_SIZE = 12000;
    const size_t NB_FRAMES_PER_CHUNK =
        get_chunk_frame_nb(MAX_DATA_CHUNK_SIZE, N_atoms, N_dims);
    const size_t NB_ROW_ITERATIONS =
        (size_t)std::ceil((double)N_frames / NB_FRAMES_PER_CHUNK);

    std::cout << "Frames per chunk : " << NB_FRAMES_PER_CHUNK
              << "\nChunk iterations : " << NB_ROW_ITERATIONS << "\n";

    // ------------------------------------------------------------
    // Allocate RMSD matrices
    // ------------------------------------------------------------
    float* rmsdHostAll   = new float[N_frames * N_frames];
    float* rmsdHostChunk = new float[NB_FRAMES_PER_CHUNK * NB_FRAMES_PER_CHUNK];

    // ------------------------------------------------------------
    // Host buffers
    // ------------------------------------------------------------
    std::vector<float> references_coordinates;
    std::vector<float> targets_coordinates;

    // ------------------------------------------------------------
    // GPU buffers
    // ------------------------------------------------------------
    float *d_references=nullptr, *d_targets=nullptr, *d_rmsd=nullptr;
    cudaMalloc(&d_references, NB_FRAMES_PER_CHUNK*N_atoms*3*sizeof(float));
    cudaMalloc(&d_targets, NB_FRAMES_PER_CHUNK*N_atoms*3*sizeof(float));
    cudaMalloc(&d_rmsd, NB_FRAMES_PER_CHUNK*NB_FRAMES_PER_CHUNK*sizeof(float));

    float *d_cx_ref,*d_cy_ref,*d_cz_ref,*d_G_ref;
    float *d_cx_tgt,*d_cy_tgt,*d_cz_tgt,*d_G_tgt;

    cudaMalloc(&d_cx_ref, NB_FRAMES_PER_CHUNK*sizeof(float));
    cudaMalloc(&d_cy_ref, NB_FRAMES_PER_CHUNK*sizeof(float));
    cudaMalloc(&d_cz_ref, NB_FRAMES_PER_CHUNK*sizeof(float));
    cudaMalloc(&d_G_ref , NB_FRAMES_PER_CHUNK*sizeof(float));
    cudaMalloc(&d_cx_tgt, NB_FRAMES_PER_CHUNK*sizeof(float));
    cudaMalloc(&d_cy_tgt, NB_FRAMES_PER_CHUNK*sizeof(float));
    cudaMalloc(&d_cz_tgt, NB_FRAMES_PER_CHUNK*sizeof(float));
    cudaMalloc(&d_G_tgt , NB_FRAMES_PER_CHUNK*sizeof(float));

    // ------------------------------------------------------------
    // CUDA events
    // ------------------------------------------------------------
    cudaEvent_t start_evt, stop_evt;
    cudaEventCreate(&start_evt);
    cudaEventCreate(&stop_evt);

    float centroid_time_ms = 0.0f;
    float rmsd_time_ms     = 0.0f;

    size_t total_rmsd_pairs = 0;
    size_t total_molecules_processed = 0;

    auto pipeline_start = std::chrono::high_resolution_clock::now();

    // ============================================================
    // RMSD computation
    // ============================================================
    for(size_t row=0; row<NB_ROW_ITERATIONS; row++){
        size_t start_row = row*NB_FRAMES_PER_CHUNK;
        size_t stop_row  = std::min(start_row + NB_FRAMES_PER_CHUNK, N_frames);
        size_t nb_ref = stop_row - start_row;
        total_molecules_processed += nb_ref;

        file.extractSnapshotsFastInPlace(start_row, stop_row, all_data, references_coordinates);
        cudaMemcpy(d_references, references_coordinates.data(),
                   references_coordinates.size()*sizeof(float), cudaMemcpyHostToDevice);

        int threads1D = 128;
        int blocks_ref = (nb_ref + threads1D - 1)/threads1D;
        cudaEventRecord(start_evt);
        computeCentroidsG<<<blocks_ref, threads1D>>>(d_references, N_atoms, nb_ref,
                                                     d_cx_ref, d_cy_ref, d_cz_ref, d_G_ref);
        cudaEventRecord(stop_evt);
        cudaEventSynchronize(stop_evt);
        float ms; cudaEventElapsedTime(&ms, start_evt, stop_evt);
        centroid_time_ms += ms;

        for(size_t col=row; col<NB_ROW_ITERATIONS; col++){
            size_t start_col = col*NB_FRAMES_PER_CHUNK;
            size_t stop_col  = std::min(start_col + NB_FRAMES_PER_CHUNK, N_frames);
            size_t nb_tgt = stop_col - start_col;
            total_molecules_processed += nb_tgt;

            file.extractSnapshotsFastInPlace(start_col, stop_col, all_data, targets_coordinates);
            cudaMemcpy(d_targets, targets_coordinates.data(),
                       targets_coordinates.size()*sizeof(float), cudaMemcpyHostToDevice);

            int blocks_tgt = (nb_tgt + threads1D - 1)/threads1D;
            cudaEventRecord(start_evt);
            computeCentroidsG<<<blocks_tgt, threads1D>>>(d_targets, N_atoms, nb_tgt,
                                                         d_cx_tgt, d_cy_tgt, d_cz_tgt, d_G_tgt);
            cudaEventRecord(stop_evt);
            cudaEventSynchronize(stop_evt);
            cudaEventElapsedTime(&ms, start_evt, stop_evt);
            centroid_time_ms += ms;

            // RMSD kernel launch
            dim3 threads2D(32,8);
            dim3 blocks2D( (nb_tgt+threads2D.x-1)/threads2D.x,
                           (nb_ref+threads2D.y-1)/threads2D.y );

            int TILE = std::min(128, (int)N_atoms);
            size_t smem_bytes = 2*3*TILE*threads2D.y*sizeof(float);

            cudaEventRecord(start_evt);
            RMSD<<<blocks2D, threads2D, smem_bytes>>>(d_references, d_targets,
                                                      N_atoms, nb_ref, nb_tgt,
                                                      d_cx_ref,d_cy_ref,d_cz_ref,d_G_ref,
                                                      d_cx_tgt,d_cy_tgt,d_cz_tgt,d_G_tgt,
                                                      d_rmsd);
            cudaEventRecord(stop_evt);
            cudaEventSynchronize(stop_evt);
            cudaEventElapsedTime(&ms, start_evt, stop_evt);
            rmsd_time_ms += ms;

            total_rmsd_pairs += nb_ref * nb_tgt;

            cudaMemcpy(rmsdHostChunk, d_rmsd, nb_ref*nb_tgt*sizeof(float), cudaMemcpyDeviceToHost);

            // scatter chunk
            for(size_t i=0;i<nb_ref;i++)
                for(size_t j=0;j<nb_tgt;j++)
                    rmsdHostAll[(start_row+i)*N_frames + (start_col+j)] = rmsdHostChunk[i*nb_tgt + j];
        }
    }

    double pipeline_time = elapsed_s(pipeline_start);

    // ------------------------------------------------------------
    // Throughput report
    // ------------------------------------------------------------
    std::cout << "\n===== PERFORMANCE =====\n";
    std::cout << "Centroid compute : "
              << total_molecules_processed / (centroid_time_ms/1000.0) << " molecules/s\n";
    std::cout << "RMSD kernel      : "
              << total_rmsd_pairs / (rmsd_time_ms/1000.0) << " RMSD/s\n";
    std::cout << "Full pipeline    : "
              << total_rmsd_pairs / pipeline_time << " RMSD/s\n";

    // ------------------------------------------------------------
    // Pack upper triangle
    // ------------------------------------------------------------
    size_t upper_triangle_size = N_frames*(N_frames-1)/2;
    float* rmsdUpperTriangle = new float[upper_triangle_size];
    size_t idx = 0;
    for(size_t i=0;i<N_frames;i++)
        for(size_t j=i+1;j<N_frames;j++)
            rmsdUpperTriangle[idx++] = rmsdHostAll[i*N_frames+j];

    delete[] rmsdHostAll;
    delete[] rmsdHostChunk;

    // ------------------------------------------------------------
    // Free GPU memory
    // ------------------------------------------------------------
    cudaFree(d_references); cudaFree(d_targets); cudaFree(d_rmsd);
    cudaFree(d_cx_ref); cudaFree(d_cy_ref); cudaFree(d_cz_ref); cudaFree(d_G_ref);
    cudaFree(d_cx_tgt); cudaFree(d_cy_tgt); cudaFree(d_cz_tgt); cudaFree(d_G_tgt);

    delete[] rmsdUpperTriangle;

    auto global_end = std::chrono::high_resolution_clock::now();
    std::cout << "\nTotal program time: " << elapsed_s(global_start) << " s\n";

    return 0;
}