// FileUtils.cpp
#include "FileUtils.h"
#include <iostream>
#include <stdexcept>
#include <vector>

FileUtils::FileUtils() 
    : file("output/snapshots_coords.bin", std::ios::binary) 
{
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open output/snapshots_coords.bin");
    }

    // Read header information
    file.read(reinterpret_cast<char*>(&n_frames), sizeof(size_t));
    file.read(reinterpret_cast<char*>(&n_atoms), sizeof(size_t));
    file.read(reinterpret_cast<char*>(&n_dims), sizeof(size_t));

    if (!file) {
        throw std::runtime_error("Error reading header from snapshots_coords.bin");
    }
}

size_t FileUtils::getN_atoms() const { return n_atoms; }
size_t FileUtils::getN_frames() const { return n_frames; }
size_t FileUtils::getN_dims() const { return n_dims; }

std::string FileUtils::toString() {
    return "UwU";
}

void FileUtils::readFrame(size_t frame_idx) {
    if (frame_idx >= n_frames) {
        throw std::out_of_range("Frame index out of range");
    }

    std::vector<float> frame_data(n_atoms * n_dims);

    file.seekg(3 * sizeof(size_t) + frame_idx * n_atoms * n_dims * sizeof(float), std::ios::beg);
    file.read(reinterpret_cast<char*>(frame_data.data()), n_atoms * n_dims * sizeof(float));

    if (!file) {
        throw std::runtime_error("Error reading frame " + std::to_string(frame_idx));
    }

    for (size_t atom_idx = 0; atom_idx < 10 && atom_idx < n_atoms; atom_idx++) {
        double x = frame_data[atom_idx * n_dims + 0];
        double y = frame_data[atom_idx * n_dims + 1];
        double z = frame_data[atom_idx * n_dims + 2];

        std::cout << "Atom " << atom_idx + 1 << ": (" 
                  << x << ", " << y << ", " << z << ")\n";
    }
}
