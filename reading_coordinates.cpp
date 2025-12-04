/*
    Compile with:
    g++ -std=c++11 -O3 reading_coordinates.cpp FileUtils.cpp -lchemfiles -o reading_coordinates

    Run:
    ./reading_coordinates
*/

#include "FileUtils.h"

int main() {
    
    FileUtils file; 

    file.readFrame(1);

    return 0;
}