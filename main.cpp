#include "common.h"
#include "mc_Ising.h"

#include <chrono>
#include <exception>
#include <iostream>

int main() {
    try {
        int Lx = 0;
        int Ly = 0;

        GetParaFromInput_int("input.in", "Lx", Lx);
        GetParaFromInput_int("input.in", "Ly", Ly);

        if (Lx < 3 || Ly < 3) {
            throw std::runtime_error("Lx and Ly must be at least 3.");
        }

        std::cout << "============================================================\n";
        std::cout << "Optimized classical Monte Carlo for triangular-lattice SkX\n";
        std::cout << "Lx = " << Lx << ", Ly = " << Ly << '\n';
        std::cout << "============================================================\n";

        const auto start = std::chrono::steady_clock::now();

        Square_Lattice lattice(Lx, Ly);
        MC_Ising simulation(&lattice);

        const auto end = std::chrono::steady_clock::now();
        const double seconds = std::chrono::duration<double>(end - start).count();

        std::cout << "============================================================\n";
        std::cout << "Total time cost: " << seconds << " s\n";
        std::cout << "============================================================\n";

        return 0;
    }
    catch (const std::exception& error) {
        std::cerr << "Fatal error: " << error.what() << '\n';
        return 1;
    }
}