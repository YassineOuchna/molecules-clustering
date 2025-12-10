/*
    Compile with:
    nvcc -ccbin /usr/bin/g++-12 -std=c++11 -O3 \
    reading_coordinates.cpp FileUtils.cpp gpu.cu \
    -I/usr/local/cuda/include \
    -L/usr/local/cuda/lib64 -lcudart \
    -lchemfiles \
    -o reading_coordinates


    Run:
    ./reading_coordinates
*/

#include "FileUtils.h"
#include <iostream>
#include "gpu.h"

int main() {
    
    FileUtils file; 

    size_t n_subset_frames = 100; // 523mb

    float* frame = file.loadData(n_subset_frames);

    std::cout << frame[0] << "," << frame[1] << "," << frame[2] << std::endl;

    // std::cout << file << std::endl;

    file.reorderByLine(frame, n_subset_frames);

    std::cout << frame[0] << "," << frame[1] << "," << frame[2] << std::endl;

    float* frameGPU;
    allocateOnGPU(&frameGPU, n_subset_frames*file.getN_atoms()*file.getN_dims()*sizeof(float));
    cudaDeviceSynchronize();

    CHECK_SUCCESS(cudaFree(frameGPU), "cudafree");
    delete[] frame;
    return 0;
}