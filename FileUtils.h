// FileUtils.h

#pragma once
#include <fstream>
#include <vector>
#include <iostream>

class FileUtils {
private:
    size_t n_frames;
    size_t n_atoms;
    size_t n_dims;
    std::ifstream file;

public:
    FileUtils();
    size_t getN_atoms() const;
    size_t getN_frames() const;
    size_t getN_dims() const;
    std::string toString();
    void readFrame(size_t frame_idx);
};
