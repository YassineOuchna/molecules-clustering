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
std::ifstream& FileUtils::getFile() const { return const_cast<std::ifstream&>(file); }

std::ostream& operator<<(std::ostream& os, const FileUtils& f) {
    size_t n_atoms = f.getN_atoms();
    size_t n_dims = f.getN_dims();
    size_t n_frames = f.getN_frames();
    std::ifstream& file = f.getFile();

    std::vector<float> frame_data(n_atoms * n_dims);
    file.seekg(3 * sizeof(size_t) + 0 * n_atoms * n_dims * sizeof(float), std::ios::beg);
    file.read(reinterpret_cast<char*>(frame_data.data()), n_atoms * n_dims * sizeof(float));
    
    os << "Snapshot 1:" << std::endl
       << "Atom 1: (" << frame_data[0 * n_dims + 0] << ", " << frame_data[0 * n_dims + 1] << ", " << frame_data[0 * n_dims + 2] << ")" << std::endl
       << "Atom 2: (" << frame_data[1 * n_dims + 0] << ", " << frame_data[1 * n_dims + 1] << ", " << frame_data[1 * n_dims + 2] << ")" << std::endl
       << "..." << std::endl
       << "Atom " << n_atoms << ": (" << frame_data[(n_atoms-1) * n_dims + 0] << ", " << frame_data[(n_atoms-1) * n_dims + 1] << ", " << frame_data[(n_atoms-1) * n_dims + 2] << ")" << std::endl << std::endl;
    
    file.seekg(3 * sizeof(size_t) + 1 * n_atoms * n_dims * sizeof(float), std::ios::beg);
    file.read(reinterpret_cast<char*>(frame_data.data()), n_atoms * n_dims * sizeof(float));

    os << "Snapshot 2:" << std::endl
       << "Atom 1: (" << frame_data[0 * n_dims + 0] << ", " << frame_data[0 * n_dims + 1] << ", " << frame_data[0 * n_dims + 2] << ")" << std::endl
       << "Atom 2: (" << frame_data[1 * n_dims + 0] << ", " << frame_data[1 * n_dims + 1] << ", " << frame_data[1 * n_dims + 2] << ")" << std::endl
       << "..." << std::endl
       << "Atom " << n_atoms << ": (" << frame_data[(n_atoms-1) * n_dims + 0] << ", " << frame_data[(n_atoms-1) * n_dims + 1] << ", " << frame_data[(n_atoms-1) * n_dims + 2] << ")" << std::endl << std::endl;
    os << "..." << std::endl << std::endl;

    file.seekg(3 * sizeof(size_t) + n_frames * n_atoms * n_dims * sizeof(float), std::ios::beg);
    file.read(reinterpret_cast<char*>(frame_data.data()), n_atoms * n_dims * sizeof(float));

    os << "Snapshot " << n_frames << ":" << std::endl
       << "Atom 1: (" << frame_data[0 * n_dims + 0] << ", " << frame_data[0 * n_dims + 1] << ", " << frame_data[0 * n_dims + 2] << ")" << std::endl
       << "Atom 2: (" << frame_data[1 * n_dims + 0] << ", " << frame_data[1 * n_dims + 1] << ", " << frame_data[1 * n_dims + 2] << ")" << std::endl
       << "..." << std::endl
       << "Atom " << n_atoms << ": (" << frame_data[(n_atoms-1) * n_dims + 0] << ", " << frame_data[(n_atoms-1) * n_dims + 1] << ", " << frame_data[(n_atoms-1) * n_dims + 2] << ")";
    return os;
}

std::vector<float> FileUtils::readFrame(size_t frame_idx) {

    std::vector<float> frame_data(n_atoms * n_dims);
    file.seekg(3 * sizeof(size_t) + 0 * n_atoms * n_dims * sizeof(float), std::ios::beg);
    file.read(reinterpret_cast<char*>(frame_data.data()), n_atoms * n_dims * sizeof(float));

    return frame_data;
}
