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

using chrono_type = std::chrono::high_resolution_clock::time_point;
using chrono_time = std::chrono::high_resolution_clock;

static inline double elapsed_s(const chrono_type& start) {
    return std::chrono::duration<double>(chrono_time::now() - start).count();
}

int main(int argc, char** args)
{
    chrono_type global_start = chrono_time::now();

    if(argc<2){
        std::cerr<<"Usage: "<<args[0]<<" <dataset.bin>\n";
        return 1;
    }

    FileUtils file(args[1]);
    size_t N_frames = 90000;
    size_t N_atoms  = file.getN_atoms();
    size_t N_dims   = file.getN_dims();

    std::cout<<"\n===== DATASET INFO =====\n";
    std::cout<<"Frames : "<<N_frames<<"\n";
    std::cout<<"Atoms  : "<<N_atoms<<"\n\n";

    std::vector<float> all_data(N_frames*N_atoms*3);
    file.readSnapshotsFastInPlace(0,N_frames-1,all_data);

    const size_t MAX_DATA_CHUNK_SIZE = 12000;
    const size_t NB_FRAMES_PER_CHUNK = get_chunk_frame_nb(MAX_DATA_CHUNK_SIZE,N_atoms,N_dims);
    const size_t NB_ROW_ITERATIONS = (size_t)std::ceil((double)N_frames/NB_FRAMES_PER_CHUNK);

    std::cout<<"Chunk size : "<<NB_FRAMES_PER_CHUNK<<" frames\n";
    std::cout<<"Row iterations : "<<NB_ROW_ITERATIONS<<"\n\n";

    float* rmsdHostAll = new float[N_frames*N_frames](); // zero initialize
    float* rmsdHostChunk = new float[NB_FRAMES_PER_CHUNK*NB_FRAMES_PER_CHUNK];

    std::vector<float> references_coordinates;
    std::vector<float> targets_coordinates;

    // GPU buffers
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
    double centroid_time = 0.0;
    double rmsd_time = 0.0;
    size_t total_centroid_frames = 0;
    size_t total_rmsd_pairs = 0;

    // --------------------
    // RMSD computation
    // --------------------
    for(size_t row=0; row<NB_ROW_ITERATIONS; row++)
    {
        size_t start_row = row*NB_FRAMES_PER_CHUNK;
        size_t stop_row  = std::min(start_row+NB_FRAMES_PER_CHUNK,N_frames);
        size_t nb_ref = stop_row-start_row;

        std::cout<<"Processing row chunk "<<row+1<<"/"<<NB_ROW_ITERATIONS
                 <<" ("<<nb_ref<<" frames)\n";

        file.extractSnapshotsFastInPlace(start_row, stop_row, all_data, references_coordinates);
        cudaMemcpy(d_references, references_coordinates.data(), references_coordinates.size()*sizeof(float), cudaMemcpyHostToDevice);

        auto c0 = chrono_time::now();
        computeCentroidsG<<<(nb_ref+127)/128,128>>>(d_references, N_atoms, nb_ref, d_cx_ref, d_cy_ref, d_cz_ref, d_G_ref);
        cudaDeviceSynchronize();
        auto c1 = chrono_time::now();
        centroid_time += elapsed_s(c0);
        total_centroid_frames += nb_ref;

        for(size_t col=row; col<NB_ROW_ITERATIONS; col++)
        {
            size_t start_col = col*NB_FRAMES_PER_CHUNK;
            size_t stop_col  = std::min(start_col+NB_FRAMES_PER_CHUNK,N_frames);
            size_t nb_tgt = stop_col-start_col;

            file.extractSnapshotsFastInPlace(start_col, stop_col, all_data, targets_coordinates);
            cudaMemcpy(d_targets, targets_coordinates.data(), targets_coordinates.size()*sizeof(float), cudaMemcpyHostToDevice);

            auto c2 = chrono_time::now();
            computeCentroidsG<<<(nb_tgt+127)/128,128>>>(d_targets, N_atoms, nb_tgt, d_cx_tgt, d_cy_tgt, d_cz_tgt, d_G_tgt);
            cudaDeviceSynchronize();
            centroid_time += elapsed_s(c2);
            total_centroid_frames += nb_tgt;

            dim3 blocks((nb_tgt+threads.x-1)/threads.x, (nb_ref+threads.y-1)/threads.y);
            int TILE = std::min(128,(int)N_atoms);
            size_t smem_bytes = 6*TILE*threads.y*sizeof(float);

            auto k0 = chrono_time::now();
            RMSD<<<blocks, threads, smem_bytes>>>(d_references, d_targets, N_atoms, nb_ref, nb_tgt,
                                                  d_cx_ref, d_cy_ref, d_cz_ref, d_G_ref,
                                                  d_cx_tgt, d_cy_tgt, d_cz_tgt, d_G_tgt,
                                                  d_rmsd);
            cudaDeviceSynchronize();
            rmsd_time += elapsed_s(k0);
            total_rmsd_pairs += nb_ref*nb_tgt;

            cudaMemcpy(rmsdHostChunk, d_rmsd, nb_ref*nb_tgt*sizeof(float), cudaMemcpyDeviceToHost);

            // copy computed chunk into global RMSD
            for(size_t i=0;i<nb_ref;i++)
                for(size_t j=0;j<nb_tgt;j++)
                    rmsdHostAll[(start_row+i)*N_frames + start_col+j] = rmsdHostChunk[i*nb_tgt + j];
        }
    }

    auto pipeline_time = elapsed_s(global_start);

    // --------------------
    // Throughput
    // --------------------
    std::cout << "\n===== PERFORMANCE =====\n";
    std::cout << "Centroid compute : " << total_centroid_frames/centroid_time << " molecules/s (" << centroid_time << " s)\n";
    std::cout << "RMSD kernel      : " << total_rmsd_pairs/rmsd_time << " RMSD/s (" << rmsd_time << " s)\n";
    std::cout << "Full pipeline    : " << total_rmsd_pairs/pipeline_time << " RMSD/s (" << pipeline_time << " s)\n";

    // --------------------
    // Pack upper triangle for clustering
    // --------------------
    size_t upper_triangle_size = (N_frames*(N_frames-1))/2;
    float* rmsdUpperTriangle = new float[upper_triangle_size];
    size_t idx = 0;
    for(size_t i=0;i<N_frames;i++)
        for(size_t j=i+1;j<N_frames;j++)
            rmsdUpperTriangle[idx++] = rmsdHostAll[i*N_frames+j];

    // --------------------
    // K-Medoids clustering
    // --------------------
    int K = 10;
    int MAX_ITER = 50;
    int* centroids = new int[K];
    int* clusters  = new int[N_frames];

    auto t_clust = chrono_time::now();
    float db_index = runKMedoids(N_frames, K, rmsdUpperTriangle, MAX_ITER, centroids, clusters);
    double clust_time = elapsed_s(t_clust);

    std::cout << "\n===== K-MEDOIDS =====\n";
    std::cout << "Clustering speed : " << N_frames / clust_time << " molecules/s (" << clust_time << " s)\n";
    std::cout << "Davies-Bouldin index : " << db_index << "\n";

    // --------------------
    // Random baseline
    // --------------------
    float random_db = runRandomClustering(N_frames,K,rmsdUpperTriangle);
    std::cout << "Random DB index : " << random_db << "\n";
    std::cout << "Improvement : " << ((random_db-db_index)/random_db)*100.0 << "%\n";

    saveClusters(clusters, N_frames, centroids, K);

    // --------------------
    // Cleanup
    // --------------------
    cudaFree(d_references); cudaFree(d_targets); cudaFree(d_rmsd);
    cudaFree(d_cx_ref); cudaFree(d_cy_ref); cudaFree(d_cz_ref); cudaFree(d_G_ref);
    cudaFree(d_cx_tgt); cudaFree(d_cy_tgt); cudaFree(d_cz_tgt); cudaFree(d_G_tgt);

    delete[] rmsdHostAll;
    delete[] rmsdHostChunk;
    delete[] rmsdUpperTriangle;
    delete[] centroids;
    delete[] clusters;

    measure_seconds(global_start,"Total program time");

    return 0;
}