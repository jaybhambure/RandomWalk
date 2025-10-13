#ifndef GRIDRANDOMWALK1D_HPP
#define GRIDRANDOMWALK1D_HPP

#include <random>

class GridRandomWalk1D {
private:
    std::mt19937 generator;
    std::uniform_real_distribution<double> uniform_dist;
    
public:
    GridRandomWalk1D(unsigned int seed = 12345) : generator(seed), uniform_dist(0.0, 1.0) {}
    
    // Generate random step: returns -1 or +1
    int getRandomStep() {
        return (uniform_dist(generator) < 0.5) ? -1 : 1;
    }
};

#endif // GRIDRANDOMWALK1D_HPP