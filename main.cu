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
#include <algorithm>
#include <random>
#include "gpu.cuh"
#include <cuda_runtime.h>
#include <stdio.h>

// CUDA error checking macro
#define CUDA_CHECK(call) \
do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", \
                __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
} while(0)


void pickRandomCentroids(int N_frames, int K, int* centroids) {
    // Create a vector with all frame indices
    int* indices = new int[N_frames];
    for (int i = 0; i < N_frames; i++) indices[i] = i;

    // Randomly shuffle
    std::random_device rd;
    std::mt19937 g(rd());

    for (size_t i = N_frames - 1; i > 0; --i) {
        std::uniform_int_distribution<size_t> dist(0, i);
        size_t j = dist(g);
        std::swap(indices[i], indices[j]);
    }

    for (int k = 0; k<K; k++) {
        centroids[k] = indices[k];
    }

    delete[] indices;
}

void createClusters(
    int N_frames,
    int K,
    const float* rmsd,
    const int* centroids,
    int* clusters
) {
    for (int i = 0; i < N_frames; i++) {
        float best = 1e30f;
        int best_k = -1;

        for (int k = 0; k < K; k++) {
            float d = rmsd[centroids[k] * N_frames + i];
            if (d < best) {
                best = d;
                best_k = k;
            }
        }
        clusters[i] = best_k;
    }
}



void updateCentroids(
    int N_frames,
    int K,
    const int* clusters,
    const float* rmsdHost,
    int* centroids
) {
    for (int k = 0; k < K; k++) {
        float best_cost = 1e30f;
        int best_idx = -1;

        for (int i = 0; i < N_frames; i++) {
            if (clusters[i] != k) continue;

            float cost = 0.0f;
            for (int j = 0; j < N_frames; j++) {
                if (clusters[j] != k) continue;
                cost += rmsdHost[i * N_frames + j];
            }

            if (cost < best_cost) {
                best_cost = cost;
                best_idx = i;
            }
        }

        if (best_idx != -1)
            centroids[k] = best_idx;
    }
}

float daviesBouldinIndex(
    int N_frames,
    int K,
    const int* clusters,
    const int* centroids,
    const float* rmsd
) {
    std::vector<float> S(K, 0.0f);
    std::vector<int> counts(K, 0);

    // --- Compute S_i (intra-cluster scatter)
    for (int i = 0; i < N_frames; i++) {
        int k = clusters[i];
        S[k] += rmsd[centroids[k] * N_frames + i];
        counts[k]++;
    }

    for (int k = 0; k < K; k++) {
        if (counts[k] > 0)
            S[k] /= counts[k];
    }

    // --- Compute DB
    float db = 0.0f;

    for (int i = 0; i < K; i++) {
        float maxR = 0.0f;

        for (int j = 0; j < K; j++) {
            if (i == j) continue;

            float Mij = rmsd[centroids[i] * N_frames + centroids[j]];
            if (Mij > 0.0f) {
                float Rij = (S[i] + S[j]) / Mij;
                maxR = std::max(maxR, Rij);
            }
        }

        db += maxR;
    }

    return db / K;
}

int main() {

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start, 0));

    int K = 20;
    int MAX_ITER = 50;

    FileUtils file; 

    std::cout << file << std::endl;

    // size_t N_frames = file.getN_frames();
    size_t N_frames = 10000;
    size_t N_atoms  = file.getN_atoms();
    size_t N_dims   = file.getN_dims();

    std::cout << "Processing " << N_frames << " frames with " 
              << N_atoms << " atoms each" << std::endl;

    // Load and reorder into X,Y,Z blocks
    float* frame = file.loadData(N_frames);
    file.reorderByLine(frame, N_frames);

    size_t total_size = N_frames * N_atoms * N_dims * sizeof(float);    

    // Copy reordered CPU → GPU
    float* frameGPU;
    CUDA_CHECK(cudaMalloc(&frameGPU, total_size));
    CUDA_CHECK(cudaMemcpy(frameGPU, frame, total_size, cudaMemcpyHostToDevice));

    std::cout << "Copied " << (total_size / (1024.0*1024.0)) 
              << " MB to GPU" << std::endl;

    // Allocate RMSD matrix
    float* rmsd;
    size_t size_rmsd = N_frames * N_frames * sizeof(float);
    CUDA_CHECK(cudaMalloc(&rmsd, size_rmsd));
    
    std::cout << "Allocated " << (size_rmsd / (1024.0*1024.0)) 
              << " MB for RMSD matrix" << std::endl;

    // FIXED: Better 2D thread configuration
    // Option 1: 2D threading (better for square matrices)
    dim3 threads(16, 16);  // 256 threads per block
    dim3 blocks((N_frames + threads.x - 1) / threads.x, 
                (N_frames + threads.y - 1) / threads.y);
    
    // Option 2: 1D threading with explicit 2D blocks (alternative)
    // dim3 threads(256, 1);
    // dim3 blocks((N_frames + 255) / 256, N_frames);

    std::cout << "Kernel configuration: " << blocks.x << "x" << blocks.y 
              << " blocks, " << threads.x << "x" << threads.y << " threads" << std::endl;
    std::cout << "Total threads: " << (blocks.x * blocks.y * threads.x * threads.y) << std::endl;
    std::cout << "Useful work (upper triangle): " << (N_frames * (N_frames + 1) / 2) << std::endl;

    std::cout << "\nKernel Start" << std::endl;
    RMSD<<<blocks, threads>>>(
        frameGPU,
        N_frames,
        N_atoms,
        rmsd
    );
    
    // Check for kernel launch errors
    CUDA_CHECK(cudaGetLastError());
    
    // Wait for kernel to finish and check for execution errors
    CUDA_CHECK(cudaDeviceSynchronize());
    std::cout << "Kernel Finished" << std::endl;

    // Copy RMSD matrix back to host
    float* rmsdHost = new float[N_frames*N_frames];
    CUDA_CHECK(cudaMemcpy(rmsdHost, rmsd, size_rmsd, cudaMemcpyDeviceToHost));
    
    std::cout << "RMSD matrix copied to host" << std::endl;

    // Verification: Check symmetry
    std::cout << "\nVerifying RMSD matrix symmetry..." << std::endl;
    bool symmetric = true;
    int errors = 0;
    const int max_errors_to_show = 5;
    
    for (int i = 0; i < N_frames && errors < max_errors_to_show; i++) {
        for (int j = i+1; j < N_frames && errors < max_errors_to_show; j++) {
            float val1 = rmsdHost[i * N_frames + j];
            float val2 = rmsdHost[j * N_frames + i];
            if (fabsf(val1 - val2) > 1e-5f) {
                std::cout << "Asymmetry at (" << i << "," << j << "): " 
                          << val1 << " vs " << val2 << std::endl;
                symmetric = false;
                errors++;
            }
        }
    }
    
    if (symmetric) {
        std::cout << "✓ RMSD matrix is symmetric" << std::endl;
    } else {
        std::cout << "✗ WARNING: RMSD matrix has asymmetries!" << std::endl;
    }
    
    // Show some sample values
    std::cout << "\nSample RMSD values:" << std::endl;
    for (int i = 0; i < std::min(5, (int)N_frames); i++) {
        std::cout << "RMSD[0," << i << "] = " << rmsdHost[i] << std::endl;
    }

    // Pick first K unique indices
    int* centroids = new int[K];
    pickRandomCentroids(N_frames, K, centroids);
    int* clusters = new int[N_frames];

    std::cout << "\n=== K-Medoids Clustering ===" << std::endl;
    std::cout << "K = " << K << ", Max iterations = " << MAX_ITER << std::endl;

    // LOOP STARTS HERE
    for (int iter=0; iter<MAX_ITER; iter++) {
        // Assign molecules to nearest centroids
        createClusters(N_frames, K, rmsdHost, centroids, clusters);

        // Update centroids to minimize intra-cluster distance
        int* old_centroids = new int[K];
        memcpy(old_centroids, centroids, K * sizeof(int));
        
        updateCentroids(N_frames, K, clusters, rmsdHost, centroids);
        
        // Check for convergence
        bool converged = true;
        for (int k = 0; k < K; k++) {
            if (centroids[k] != old_centroids[k]) {
                converged = false;
                break;
            }
        }
        
        delete[] old_centroids;
        
        if (converged) {
            std::cout << "Converged at iteration " << (iter + 1) << std::endl;
            break;
        }
    }

    std::cout << "\nFinal centroids: ";
    for (int i=0; i<K; i++) {
        std::cout << centroids[i];
        if (i < K-1) std::cout << ", ";
    }
    std::cout << std::endl;

    // Compute cluster sizes
    std::vector<int> cluster_sizes(K, 0);
    for (int i = 0; i < N_frames; i++) {
        cluster_sizes[clusters[i]]++;
    }
    
    std::cout << "\nCluster sizes: ";
    for (int k = 0; k < K; k++) {
        std::cout << "C" << k << "=" << cluster_sizes[k];
        if (k < K-1) std::cout << ", ";
    }
    std::cout << std::endl;

    float db = daviesBouldinIndex(
        N_frames,
        K,
        clusters,
        centroids,
        rmsdHost
    );

    std::cout << "\nDavies–Bouldin index: " << db << " (lower is better)" << std::endl;

    // Compare with random clustering
    std::cout << "\n=== Random Clustering Baseline ===" << std::endl;
    pickRandomCentroids(N_frames, K, centroids);
    for (int i = 0; i < N_frames; i++) {
        clusters[i] = rand() % K;
    }
    createClusters(N_frames, K, rmsdHost, centroids, clusters);
    db = daviesBouldinIndex(
        N_frames,
        K,
        clusters,
        centroids,
        rmsdHost
    );

    std::cout << "Random Davies–Bouldin index: " << db << std::endl;


    // Cleanup
    delete[] frame;
    delete[] centroids;
    delete[] rmsdHost;
    delete[] clusters;
    CUDA_CHECK(cudaFree(frameGPU));
    CUDA_CHECK(cudaFree(rmsd));

    CUDA_CHECK(cudaEventRecord(stop, 0));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float elapsed_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));

    std::cout << "\n=== Performance ===" << std::endl;
    std::cout << "Total execution time: " << elapsed_ms/1000.0f << " s" << std::endl;
    std::cout << "RMSD computations: " << (unsigned long long)(N_frames * (N_frames + 1) / 2) << " pairs" << std::endl;
    std::cout << "Throughput: " << (N_frames * (N_frames + 1) / 2) / (elapsed_ms/1000.0f) << " pairs/second" << std::endl;

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    return 0;
}