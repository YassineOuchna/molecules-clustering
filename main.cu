/*
    Compile with:
    nvcc -ccbin /usr/bin/g++-12 -std=c++11 -O3 \
    main.cu FileUtils.cpp gpu.cu \
    -I/usr/local/cuda/include \
    -L/usr/local/cuda/lib64 -lcudart \
    -lchemfiles \
    -o main
*/

#include "FileUtils.h"
#include <iostream>
#include <fstream>
#include <algorithm>
#include <random>
#include <vector>
#include "gpu.cuh"
#include <cuda_runtime.h>
#include <stdio.h>
#include <chrono>
#include <iomanip>
#include <utils.h>


int main(int argc, char** args) {

    // Time measurements
    chrono_type global_start = chrono_time::now();


    int K = 10;
    int MAX_ITER = 50;

    std::string file_name;
    if (argc >= 2) {
        file_name = args[1];
    } else {
        std::cerr<< "Argument for dataset binary file missing, check the Makefile" << std::endl;
        throw std::invalid_argument("Requested frames exceed available frames");
    }
    FileUtils file(file_name); 

    // std::cout << file << std::endl;

    size_t N_frames = 10000;
    // size_t N_frames = file.getN_frames();
    size_t N_atoms  = file.getN_atoms();
    size_t N_dims   = file.getN_dims();

    // Load and reorder into X,Y,Z blocks
    float* frame = file.loadData(N_frames);
    file.reorderByLine(frame, N_frames);

    size_t total_size = N_frames * N_atoms * N_dims * sizeof(float);    

    measure_seconds(global_start, "Loading source data");

    // Copy reordered CPU → GPU
    chrono_type mem_transfer_start = chrono_time::now();
    float* frameGPU;
    CHECK_SUCCESS(cudaMalloc(&frameGPU, total_size), "Allocating frameGPU");
    CHECK_SUCCESS(cudaMemcpy(frameGPU, frame, total_size, cudaMemcpyHostToDevice), "Memcpy frame -> frameGPU");

    // Allocate RMSD matrix
    float* rmsd;
    size_t size_rmsd = N_frames * N_frames * sizeof(float);
    CHECK_SUCCESS(cudaMalloc(&rmsd, size_rmsd), "Allocating rmsd vector on GPU");

    cudaDeviceSynchronize();
    measure_seconds(mem_transfer_start, "CPU to GPU memory transfer");

    dim3 threads(16,16);
    dim3 blocks((N_frames + threads.x - 1) / threads.x, 
                (N_frames + threads.y - 1) / threads.y);

    chrono_type rmsd_kernel_start = chrono_time::now();
    RMSD<<<blocks, threads>>>(
        frameGPU,
        N_frames,
        N_atoms,
        rmsd
    );
    CHECK_SUCCESS(cudaDeviceSynchronize(), "RMSD Kernel");
    measure_seconds(rmsd_kernel_start, "RMSD Kernel");

    float* rmsdHost = new float[N_frames*N_frames];
    CHECK_SUCCESS(cudaMemcpy(rmsdHost, rmsd, size_rmsd, cudaMemcpyDeviceToHost), "Memcpy rmsd -> rmsdHost");

    chrono_type clustering_loop_start = chrono_time::now();
    // Pick first K unique indices
    int* centroids = new int[K];
    int* clusters = new int[N_frames];

    float db_index = runKMedoids(N_frames, K, rmsdHost, MAX_ITER, centroids, clusters);
    // float db_index = k_analysis(rmsdHost, N_frames, MAX_ITER);

    std::cout << "Davies–Bouldin index: " << db_index << std::endl;

    measure_seconds(clustering_loop_start, "Clustering loop");
    measure_seconds(global_start, "Entire program");

    // Print db for random clustering
    float random_db_index = runRandomClustering(N_frames, K, rmsdHost);
    std::cout << "Random Davies–Bouldin index: " << random_db_index << std::endl;

    saveClusters(clusters, N_frames, centroids, K);

    // Cleanup
    delete[] frame;
    delete[] centroids;
    delete[] rmsdHost;
    delete[] clusters;
    cudaFree(frameGPU);
    cudaFree(rmsd);

    return 0;
}
