/*
    Compile with:

    nvcc -ccbin /usr/bin/g++-12 -std=c++11 -O3 --use_fast_math -Xcompiler -fopenmp \
    main.cu FileUtils.cpp gpu.cu utils.cu \
    -I/usr/local/cuda/include \
    -L/usr/local/cuda/lib64 -lcudart \
    -lchemfiles \
    -o main
*/

#include "FileUtils.h"
#include <iostream>
#include <fstream>
#include <vector>
#include <iomanip>
#include <algorithm>
#include "gpu.cuh"
#include <cuda_runtime.h>
#include "utils.cuh"

int main(int argc, char** args) {

    // Global timer
    chrono_type global_start = chrono_time::now();

    int K = 10;
    int MAX_ITER = 50;

    // Input file
    if (argc < 2) {
        std::cerr << "Usage: " << args[0] << " <dataset.bin>\n";
        return 1;
    }
    std::string file_name = args[1];
    FileUtils file(file_name);

    size_t N_frames = 20000;
    size_t N_atoms = file.getN_atoms();
    size_t N_dims = file.getN_dims();

    // Chunking parameters
    size_t MAX_DATA_CHUNK_SIZE = 500; // MB
    size_t NB_FRAMES_CHUNK     = get_chunk_frame_nb(MAX_DATA_CHUNK_SIZE, N_atoms, N_dims);
    size_t SQ_TILE_SIZE        = get_optimal_tile_size(MAX_DATA_CHUNK_SIZE, N_atoms, N_dims, N_frames);
    size_t NB_ROW_ITERATIONS   = (size_t) std::ceil((double)N_frames / SQ_TILE_SIZE);
    size_t RMSD_LOOPS_NEEDED   = NB_ROW_ITERATIONS * (NB_ROW_ITERATIONS + 1) / 2;

    std::cout << "\n" << std::string(70,'=') << "\n";
    std::cout << "RMSD COMPUTATION CONFIGURATION\n";
    std::cout << std::string(70,'=') << "\n";
    std::cout << "Max chunk size           : " << MAX_DATA_CHUNK_SIZE << " MB\n";
    std::cout << "Max frames per chunk     : " << NB_FRAMES_CHUNK << "\n";
    std::cout << "Submatrix tile size      : " << SQ_TILE_SIZE << "\n";
    std::cout << "Number of RMSD iterations: " << RMSD_LOOPS_NEEDED << "\n";
    std::cout << "Total frames             : " << N_frames << "\n";
    std::cout << "Number of atoms          : " << N_atoms << "\n";
    std::cout << std::string(70,'=') << "\n\n";

    // Load all frames into CPU memory
    float* frameData = file.loadData(N_frames);

    // Allocate packed upper triangle RMSD matrix
    size_t rmsdSize = N_frames * (N_frames - 1) / 2;
    float* rmsdHost = new float[rmsdSize];

    int row_begin = 0;
    for (size_t tile_iter = 0; tile_iter < RMSD_LOOPS_NEEDED; ++tile_iter) {

        int col_begin = col_index_parcours(tile_iter, NB_ROW_ITERATIONS-1) * SQ_TILE_SIZE;
        int row_end   = std::min(row_begin + (int)SQ_TILE_SIZE, (int)N_frames);
        int col_end   = std::min(col_begin + (int)SQ_TILE_SIZE, (int)N_frames);

        int size_row = row_end - row_begin;
        int size_col = col_end - col_begin;

        // Frames in the subset
        int nb_frames_subset = (col_begin == row_begin) ? size_col : size_row + size_col;

        std::cout << "Iteration " << tile_iter + 1 << "/" << RMSD_LOOPS_NEEDED
                  << " | Tile [" << row_begin << ":" << row_end
                  << ", " << col_begin << ":" << col_end << "] "
                  << "| Frames: " << nb_frames_subset << "\n";

        // Extract and reorder frames subset
        float* frameSubset = file.getFrameSubset(frameData, row_begin, row_end, col_begin, col_end, N_frames);
        file.reorderByLine(frameSubset, nb_frames_subset);

        size_t totalBytes = (size_t)nb_frames_subset * N_atoms * N_dims * sizeof(float);

        // GPU memory for frames and RMSD
        float* frameGPU = nullptr;
        float* rmsdGPU   = nullptr;
        CHECK_SUCCESS(cudaMalloc(&frameGPU, totalBytes), "Allocating frameGPU");
        CHECK_SUCCESS(cudaMemcpy(frameGPU, frameSubset, totalBytes, cudaMemcpyHostToDevice), "Memcpy frame -> GPU");

        size_t rmsdBytes = (size_t)nb_frames_subset * (nb_frames_subset - 1) / 2 * sizeof(float);
        CHECK_SUCCESS(cudaMalloc(&rmsdGPU, rmsdBytes), "Allocating rmsdGPU");

        cudaDeviceSynchronize();

        dim3 threads(16,16);
        dim3 blocks((nb_frames_subset + threads.x - 1)/threads.x, (nb_frames_subset + threads.y - 1)/threads.y);

        RMSD<<<blocks, threads>>>(frameGPU, nb_frames_subset, N_atoms, rmsdGPU);
        CHECK_SUCCESS(cudaDeviceSynchronize(), "RMSD kernel");

        // Copy RMSD back
        float* rmsdSubsetHost = new float[rmsdBytes/sizeof(float)];
        CHECK_SUCCESS(cudaMemcpy(rmsdSubsetHost, rmsdGPU, rmsdBytes, cudaMemcpyDeviceToHost), "Memcpy RMSD -> CPU");

        // Scatter into packed upper triangle
        for (int i = row_begin; i < row_end; ++i) {
            for (int j = col_begin; j < col_end; ++j) {
                if (j <= i) continue; // upper triangle only

                int ii = i - row_begin;
                int jj = (col_begin == row_begin) ? j - col_begin : size_row + (j - col_begin);

                size_t subset_idx = (size_t)std::min(ii,jj) * nb_frames_subset
                                  - ((size_t)std::min(ii,jj)*((size_t)std::min(ii,jj)+1))/2
                                  + (std::max(ii,jj)-std::min(ii,jj)-1);

                size_t global_idx = (size_t)i * N_frames
                                  - ((size_t)i*((size_t)i+1))/2
                                  + (j-i-1);

                rmsdHost[global_idx] = rmsdSubsetHost[subset_idx];
            }
        }

        if (col_end == (int)N_frames) row_begin += SQ_TILE_SIZE;

        delete[] frameSubset;
        delete[] rmsdSubsetHost;
        cudaFree(frameGPU);
        cudaFree(rmsdGPU);
    }

    std::cout << "\n" << std::string(70,'=') << "\n";
    std::cout << "RMSD COMPUTATION COMPLETE\n";
    std::cout << std::string(70,'=') << "\n";
    measure_seconds(global_start, "Total RMSD computation time");
    std::cout << "\n";

    // -------------------------------------------------------------------------
    // K-Medoids clustering
    // -------------------------------------------------------------------------
    std::cout << std::string(70,'=') << "\n";
    std::cout << "K-MEDOIDS CLUSTERING (K=" << K << ", MAX_ITER=" << MAX_ITER << ")\n";
    std::cout << std::string(70,'=') << "\n";

    chrono_type t_cluster = chrono_time::now();
    int* centroids = new int[K];
    int* clusters  = new int[N_frames];

    float db_index = runKMedoids(N_frames, K, rmsdHost, MAX_ITER, centroids, clusters);
    measure_seconds(t_cluster, "K-medoids clustering time");

    // -------------------------------------------------------------------------
    // Display results
    // -------------------------------------------------------------------------
    std::cout << std::string(70,'=') << "\n";
    std::cout << "CLUSTERING RESULTS\n";
    std::cout << std::string(70,'=') << "\n";

    std::cout << std::fixed << std::setprecision(6);
    std::cout << "K-medoids Davies-Bouldin Index: " << db_index << "\n";

    std::vector<int> cluster_sizes(K,0);
    for(int i=0;i<N_frames;i++) cluster_sizes[clusters[i]]++;

    std::cout << "\nCluster centroids and sizes:\n";
    for(int k=0;k<K;k++){
        float percent = 100.0f * cluster_sizes[k] / N_frames;
        std::cout << "  Cluster " << std::setw(2) << k 
                  << " | Centroid: frame " << std::setw(6) << centroids[k] 
                  << " | Size: " << std::setw(6) << cluster_sizes[k] 
                  << " (" << std::setw(5) << std::setprecision(2) << percent << "%)\n";
    }

    std::cout << "\n" << std::string(70,'-') << "\n";
    std::cout << "BASELINE COMPARISON\n";
    std::cout << std::string(70,'-') << "\n";

    float random_db_index = runRandomClustering(N_frames, K, rmsdHost);
    std::cout << std::fixed << std::setprecision(6);
    std::cout << "Random clustering Davies-Bouldin Index: " << random_db_index << "\n";

    float improvement = ((random_db_index - db_index)/random_db_index)*100.0f;
    std::cout << "K-medoids improvement over random: " 
              << std::setprecision(2) << improvement << "% "
              << (improvement>0?"✓ BETTER":"✗ WORSE") << "\n";
    std::cout << std::string(70,'=') << "\n\n";

    saveClusters(clusters, N_frames, centroids, K);
    measure_seconds(global_start, "Total program execution time");

    // Cleanup
    delete[] frameData;
    delete[] rmsdHost;
    delete[] centroids;
    delete[] clusters;

    return 0;
}