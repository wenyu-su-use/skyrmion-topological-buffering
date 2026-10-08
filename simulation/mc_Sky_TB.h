#pragma once

#include <array>
#include <cstdint>
#include <random>
#include <string>
#include <vector>

class Square_Lattice {
public:
    Square_Lattice(int lx, int ly);

    int get_Lx() const noexcept { return Lx_; }
    int get_Ly() const noexcept { return Ly_; }

    void NN_sites(int site, int* neighbors) const;

private:
    int Lx_ = 0;
    int Ly_ = 0;
};

struct Spin3 {
    double x = 0.0;
    double y = 0.0;
    double z = 1.0;
};

struct MC_Measurements {
    double energy = 0.0;
    double mz = 0.0;
    double skyrmion_number = 0.0;
    double sublattice_sk_order = 0.0;
    double psi6 = 0.0;        // 兼容旧变量名；现在等于 phi6_global
    double chirality = 0.0;

    // Skyrmion-center 版本六重取向序
    double phi6_global = 0.0;     // |Phi6| = |mean_i psi6_i|
    double phi6_local = 0.0;      // mean_i |psi6_i|

    // 用于热涨落型 chi6
    double phi6_global_sq = 0.0;  // |Phi6|^2
    double phi6_local_sq = 0.0;   // (mean_i |psi6_i|)^2
    double chi6_global = 0.0;
    double chi6_local = 0.0;
    double n_sk_center = 0.0;
};

class MC_Ising {
public:
    explicit MC_Ising(Square_Lattice* lattice);

private:
    using NeighborRow = std::array<int, 18>;
    using BondRow = std::array<double, 12>;
    using BondMatrix = std::vector<BondRow>;
    using SpinConfig = std::vector<Spin3>;

    struct DisorderRealization {
        BondMatrix rho;
    };

    struct Settings {
        int n_warmup = 300;
        int n_measure = 1000;
        int n_bins = 4;
        int random_restarts = 2;
        int observable_stride = 10;
        int snapshot_stride = 500;
        int overrelax_sweeps = 2;
        int max_low_t_sweeps = 1200;
        int stable_checks_required = 4;
        std::uint64_t seed = 20260530ULL;

        double target_temperature = 0.005;
        double anneal_start_temperature = 0.9;
        double anneal_rate = 0.84;
        double energy_tolerance_per_site = 1.0e-7;

        double J1 = -1.0 / 3.0;
        double D1 = 1.0;
        double J2 = 0.0;
        double D2 = 0.0;
        double J3 = 0.0;
        double delta_J1 = 0.0;
        double delta_J2 = 0.0;

        bool write_snapshots = false;
        bool calculate_structure_factor = true;
    };

    Square_Lattice* lattice_ = nullptr;
    int Lx_ = 0;
    int Ly_ = 0;
    int N_ = 0;

    Settings settings_;
    std::vector<double> disorder_values_;
    std::vector<double> field_values_;
    std::vector<NeighborRow> neighbors_;

    void ReadSettings();
    void BuildNeighborCache();
    void WriteScanParameters() const;

    void GenerateDisorder(DisorderRealization& disorder, std::mt19937_64& rng) const;
    void BuildInteractions(
        double delta_D,
        const DisorderRealization& disorder,
        BondMatrix& J,
        BondMatrix& D
    ) const;

    void RandomizeSpins(SpinConfig& spins, std::mt19937_64& rng) const;


    double MetropolisSweep(
        double temperature,
        double field,
        SpinConfig& spins,
        const BondMatrix& J,
        const BondMatrix& D,
        std::mt19937_64& rng
    ) const;

    void OverrelaxationSweep(
        double field,
        SpinConfig& spins,
        const BondMatrix& J,
        const BondMatrix& D
    ) const;

    void AnnealOneState(
        double start_temperature,
        double field,
        SpinConfig& spins,
        const BondMatrix& J,
        const BondMatrix& D,
        std::mt19937_64& rng
    ) const;

    void OptimizeState(
        double field,
        SpinConfig& spins,
        const BondMatrix& J,
        const BondMatrix& D,
        std::mt19937_64& rng,
        bool include_existing_state
    ) const;

    MC_Measurements MeasureObservables(
        double field,
        const SpinConfig& spins,
        const BondMatrix& J,
        const BondMatrix& D,
        bool calculate_psi6
    ) const;

    double Hamiltonian(
        double field,
        const SpinConfig& spins,
        const BondMatrix& J,
        const BondMatrix& D
    ) const;

    double MagnetizationZ(const SpinConfig& spins) const;
    double SkyrmionNumber(const SpinConfig& spins) const;
    double SublatticeSkyrmionOrder(const SpinConfig& spins) const;
    double ScalarChirality(const SpinConfig& spins) const;
    double Psi6Order(const SpinConfig& spins) const;

    void MeasureParameterPoint(
        int iD,
        int iB,
        double field,
        std::vector<SpinConfig>& all_spins,
        const std::vector<BondMatrix>& all_J,
        const std::vector<BondMatrix>& all_D,
        std::vector<std::mt19937_64>& dynamics_rng
    ) const;

    void WriteStructureFactor(
        const std::vector<std::vector<Spin3>>& snapshots_per_bin,
        int iD,
        int iB
    ) const;

    void WriteAngleSnapshots(
        const std::vector<std::vector<Spin3>>& snapshots_per_bin,
        int iD,
        int iB
    ) const;
    void WriteFinalConfigurations(
        const std::vector<SpinConfig>& all_spins,
        int iD,
        int iB
    ) const;
};
