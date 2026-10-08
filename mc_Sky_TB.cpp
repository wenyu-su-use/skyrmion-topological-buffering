#define _USE_MATH_DEFINES

#include "mc_Ising.h"
#include "common.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <utility>
#include <string>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace {

    constexpr double PI = 3.141592653589793238462643383279502884;
    constexpr double SQRT3_OVER_2 = 0.866025403784438646763723170752936183;

    const std::array<Spin3, 6> DMI_NN = { {
        { 0.5,  SQRT3_OVER_2, 0.0},
        { 1.0,  0.0,          0.0},
        {-0.5, -SQRT3_OVER_2, 0.0},
        {-1.0,  0.0,          0.0},
        {-0.5,  SQRT3_OVER_2, 0.0},
        { 0.5, -SQRT3_OVER_2, 0.0}
    } };

    const std::array<Spin3, 6> DMI_NNN = { {
        { SQRT3_OVER_2,  0.5, 0.0},
        { 0.0,           1.0, 0.0},
        {-SQRT3_OVER_2,  0.5, 0.0},
        {-SQRT3_OVER_2, -0.5, 0.0},
        { 0.0,          -1.0, 0.0},
        { SQRT3_OVER_2, -0.5, 0.0}
    } };

    Spin3 operator+(const Spin3& a, const Spin3& b) {
        return { a.x + b.x, a.y + b.y, a.z + b.z };
    }

    Spin3 operator-(const Spin3& a, const Spin3& b) {
        return { a.x - b.x, a.y - b.y, a.z - b.z };
    }

    Spin3 operator*(double scalar, const Spin3& spin) {
        return { scalar * spin.x, scalar * spin.y, scalar * spin.z };
    }

    double Dot(const Spin3& a, const Spin3& b) {
        return a.x * b.x + a.y * b.y + a.z * b.z;
    }

    Spin3 Cross(const Spin3& a, const Spin3& b) {
        return {
            a.y * b.z - a.z * b.y,
            a.z * b.x - a.x * b.z,
            a.x * b.y - a.y * b.x
        };
    }

    double NormSquared(const Spin3& spin) {
        return Dot(spin, spin);
    }

    Spin3 Normalize(const Spin3& spin) {
        const double norm_squared = NormSquared(spin);
        if (norm_squared < 1.0e-30) {
            return { 0.0, 0.0, 1.0 };
        }

        const double inv_norm = 1.0 / std::sqrt(norm_squared);
        return inv_norm * spin;
    }

    std::uint64_t SplitMix64(std::uint64_t value) {
        value += 0x9e3779b97f4a7c15ULL;
        value = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9ULL;
        value = (value ^ (value >> 27)) * 0x94d049bb133111ebULL;
        return value ^ (value >> 31);
    }

    std::uint64_t MakeSeed(std::uint64_t base, int iD, int ibin, int stream_id) {
        std::uint64_t value = base;
        value ^= SplitMix64(static_cast<std::uint64_t>(iD + 1009));
        value ^= SplitMix64(static_cast<std::uint64_t>(ibin + 9176));
        value ^= SplitMix64(static_cast<std::uint64_t>(stream_id + 65537));
        return SplitMix64(value);
    }

    Spin3 RandomUnitVector(std::mt19937_64& rng) {
        std::uniform_real_distribution<double> uniform01(0.0, 1.0);

        const double z = 2.0 * uniform01(rng) - 1.0;
        const double phi = 2.0 * PI * uniform01(rng);
        const double radius = std::sqrt(std::max(0.0, 1.0 - z * z));

        return {
            radius * std::cos(phi),
            radius * std::sin(phi),
            z
        };
    }

    Spin3 HeatBathSampleFromField(
        const Spin3& local_field,
        double temperature,
        std::mt19937_64& rng
    ) {
        std::uniform_real_distribution<double> uniform01(0.0, 1.0);

        const double h2 = NormSquared(local_field);
        if (h2 < 1.0e-28 || !(temperature > 0.0)) {
            return RandomUnitVector(rng);
        }

        const double hnorm = std::sqrt(h2);
        const Spin3 hhat = (1.0 / hnorm) * local_field;

        const double beta = 1.0 / std::max(temperature, 1.0e-14);
        const double a = beta * hnorm;

        double r1 = uniform01(rng);
        const double r2 = uniform01(rng);

        // Avoid log(0). uniform_real_distribution may generate exactly 0.
        r1 = std::min(1.0 - 1.0e-16, std::max(1.0e-16, r1));

        double u = 0.0;  // u = cos(theta) relative to local_field direction.

        if (a < 1.0e-8) {
            // Weak-field / high-T limit: nearly uniform on the sphere.
            u = 2.0 * r1 - 1.0;
        }
        else if (a > 50.0) {
            // Low-T / strong-field stable asymptotic form.
            // u = 1 + log(r1)/a, with exponentially small correction omitted.
            u = 1.0 + std::log(r1) / a;
        }
        else {
            // Stable inverse CDF:
            // u = 1 + log(r + (1-r) exp(-2a)) / a
            const double em2a = std::exp(-2.0 * a);
            u = 1.0 + std::log(r1 + (1.0 - r1) * em2a) / a;
        }

        u = std::min(1.0, std::max(-1.0, u));

        const double phi = 2.0 * PI * r2;
        const double sin_theta = std::sqrt(std::max(0.0, 1.0 - u * u));

        Spin3 ref;
        if (std::fabs(hhat.z) < 0.9) {
            ref = { 0.0, 0.0, 1.0 };
        }
        else {
            ref = { 1.0, 0.0, 0.0 };
        }

        Spin3 e1 = Normalize(Cross(ref, hhat));
        Spin3 e2 = Normalize(Cross(hhat, e1));

        Spin3 snew =
            (sin_theta * std::cos(phi)) * e1 +
            (sin_theta * std::sin(phi)) * e2 +
            u * hhat;

        return Normalize(snew);
    }


    double SolidAngle(const Spin3& s0, const Spin3& s1, const Spin3& s2) {
        const double numerator = Dot(s0, Cross(s1, s2));
        const double denominator = 1.0 + Dot(s0, s1) + Dot(s1, s2) + Dot(s2, s0);
        return 2.0 * std::atan2(numerator, denominator);
    }

    double StandardError(const std::vector<double>& values, double mean) {
        if (values.size() <= 1) {
            return 0.0;
        }

        double squared_sum = 0.0;
        for (const double value : values) {
            const double difference = value - mean;
            squared_sum += difference * difference;
        }

        return std::sqrt(
            squared_sum /
            (
                static_cast<double>(values.size()) *
                static_cast<double>(values.size() - 1)
                )
        );
    }

    std::vector<double> DefaultDisorderValues() {
        return {
            0.00, 0.02, 0.04, 0.06, 0.08,
            0.10, 0.12, 0.14, 0.16, 0.18,
            0.20, 0.22, 0.24, 0.26, 0.30,
            0.35, 0.40, 0.45, 0.50, 0.55,
            0.60, 0.65, 0.70, 0.75, 0.80,
            0.85, 0.90, 0.95, 1.00, 1.10
        };
    }

    std::vector<double> DefaultFieldValues() {
        return { 1.20 };
    }

    std::vector<double> ReadListOrDefault(
        const char* fname,
        const char* key,
        const std::vector<double>& fallback
    ) {
        std::ifstream input(fname);
        if (!input) {
            return fallback;
        }

        std::string line;
        while (std::getline(input, line)) {
            const std::size_t comment = line.find('#');
            if (comment != std::string::npos) {
                line.erase(comment);
            }

            const std::size_t equal = line.find('=');
            if (equal == std::string::npos) {
                continue;
            }

            if (Trim(line.substr(0, equal)) != key) {
                continue;
            }

            const std::vector<double> parsed = ParseDoubleList(Trim(line.substr(equal + 1)));
            return parsed.empty() ? fallback : parsed;
        }

        return fallback;
    }


    // ============================================================
    // Skyrmion-center based phi6 helpers
    // Same logic as the Python post-processing:
    //   Sz<0 connected cores -> weighted centers -> PBC Delaunay -> psi6_i
    // ============================================================

    struct Point2 {
        double x = 0.0;
        double y = 0.0;
    };

    struct Phi6FrameResult {
        int nsk = 0;
        double abs_Phi6 = 0.0;        // |mean_i psi6_i|
        double mean_abs_psi6 = 0.0;   // mean_i |psi6_i|
        double std_abs_psi6 = 0.0;
    };

    Point2 LatticeToReal(double u, double v) {
        return { u + 0.5 * v, SQRT3_OVER_2 * v };
    }

    Point2 RealToFrac(const Point2& r, int lx, int ly) {
        const double f2 = r.y / (SQRT3_OVER_2 * static_cast<double>(ly));
        const double f1 =
            (r.x - 0.5 * static_cast<double>(ly) * f2) /
            static_cast<double>(lx);
        return { f1, f2 };
    }

    Point2 FracToReal(const Point2& f, int lx, int ly) {
        return {
            static_cast<double>(lx) * f.x + 0.5 * static_cast<double>(ly) * f.y,
            SQRT3_OVER_2 * static_cast<double>(ly) * f.y
        };
    }

    double Wrap01(double x) {
        x -= std::floor(x);
        if (x >= 1.0) {
            x -= 1.0;
        }
        if (x < 0.0) {
            x += 1.0;
        }
        return x;
    }

    Point2 WrapRealToBox(const Point2& r, int lx, int ly) {
        Point2 f = RealToFrac(r, lx, ly);
        f.x = Wrap01(f.x);
        f.y = Wrap01(f.y);
        return FracToReal(f, lx, ly);
    }

    double PeriodicCenter1D(
        const std::vector<double>& coords,
        const std::vector<double>& weights,
        double period
    ) {
        double re = 0.0;
        double im = 0.0;
        double wsum = 0.0;

        for (std::size_t i = 0; i < coords.size(); ++i) {
            const double w = weights[i];
            const double angle = 2.0 * PI * coords[i] / period;

            re += w * std::cos(angle);
            im += w * std::sin(angle);
            wsum += w;
        }

        if (wsum <= 0.0) {
            if (coords.empty()) {
                return 0.0;
            }

            double mean = 0.0;
            for (const double c : coords) {
                mean += c;
            }
            mean /= static_cast<double>(coords.size());

            double wrapped = std::fmod(mean, period);
            if (wrapped < 0.0) {
                wrapped += period;
            }
            return wrapped;
        }

        double phi = std::atan2(im, re);
        if (phi < 0.0) {
            phi += 2.0 * PI;
        }

        return period * phi / (2.0 * PI);
    }

    Point2 MinImageDeltaReal(
        const Point2& ri,
        const Point2& rj,
        int lx,
        int ly
    ) {
        const double dx0 = rj.x - ri.x;
        const double dy0 = rj.y - ri.y;

        const double df2 = dy0 / (SQRT3_OVER_2 * static_cast<double>(ly));
        const double df1 =
            (dx0 - 0.5 * static_cast<double>(ly) * df2) /
            static_cast<double>(lx);

        double best_r2 = std::numeric_limits<double>::infinity();
        Point2 best{ 0.0, 0.0 };

        for (int m = -1; m <= 1; ++m) {
            for (int n = -1; n <= 1; ++n) {
                const double f1 = df1 - static_cast<double>(m);
                const double f2 = df2 - static_cast<double>(n);

                const double dx =
                    static_cast<double>(lx) * f1 +
                    0.5 * static_cast<double>(ly) * f2;

                const double dy =
                    SQRT3_OVER_2 * static_cast<double>(ly) * f2;

                const double r2 = dx * dx + dy * dy;

                if (r2 < best_r2) {
                    best_r2 = r2;
                    best = { dx, dy };
                }
            }
        }

        return best;
    }

    std::vector<Point2> ExtractSkyrmionCentersBySzCore(
        const std::vector<Spin3>& spins,
        int lx,
        int ly
    ) {
        const int nsite = lx * ly;

        std::vector<unsigned char> mask(nsite, 0);
        std::vector<unsigned char> visited(nsite, 0);

        // Match the Python setting:
        // CORE_SIGN="negative", CORE_THRESHOLD=0, WEIGHT_MODE="sz_core".
        for (int i = 0; i < nsite; ++i) {
            if (spins[i].z < 0.0) {
                mask[i] = 1;
            }
        }

        const auto mod = [](int a, int n) {
            return (a % n + n) % n;
        };

        const auto index = [&](int x, int y) {
            return mod(x, lx) + mod(y, ly) * lx;
        };

        const int dx8[8] = { -1, 0, 1, -1, 1, -1, 0, 1 };
        const int dy8[8] = { -1,-1,-1,  0, 0,  1, 1, 1 };

        constexpr int min_cluster_size = 4;

        std::vector<Point2> centers;
        std::vector<int> stack;
        stack.reserve(1024);

        for (int y0 = 0; y0 < ly; ++y0) {
            for (int x0 = 0; x0 < lx; ++x0) {
                const int start = index(x0, y0);

                if (!mask[start] || visited[start]) {
                    continue;
                }

                stack.clear();
                stack.push_back(start);
                visited[start] = 1;

                std::vector<double> xs;
                std::vector<double> ys;
                std::vector<double> ws;

                while (!stack.empty()) {
                    const int site = stack.back();
                    stack.pop_back();

                    const int x = site % lx;
                    const int y = site / lx;

                    const double w = std::max(-spins[site].z, 0.0);

                    xs.push_back(static_cast<double>(x));
                    ys.push_back(static_cast<double>(y));
                    ws.push_back(w);

                    for (int nn = 0; nn < 8; ++nn) {
                        const int xx = mod(x + dx8[nn], lx);
                        const int yy = mod(y + dy8[nn], ly);
                        const int nb = index(xx, yy);

                        if (mask[nb] && !visited[nb]) {
                            visited[nb] = 1;
                            stack.push_back(nb);
                        }
                    }
                }

                if (static_cast<int>(xs.size()) < min_cluster_size) {
                    continue;
                }

                double wsum = 0.0;
                for (const double w : ws) {
                    wsum += w;
                }

                if (wsum <= 0.0) {
                    continue;
                }

                const double cx_lat =
                    PeriodicCenter1D(xs, ws, static_cast<double>(lx));

                const double cy_lat =
                    PeriodicCenter1D(ys, ws, static_cast<double>(ly));

                Point2 center = LatticeToReal(cx_lat, cy_lat);
                center = WrapRealToBox(center, lx, ly);

                centers.push_back(center);
            }
        }

        return centers;
    }


    
    struct PBCNeighbor {
        int j = -1;
        Point2 dr{0.0, 0.0};  // vector from central i to periodic image of j
    };

    std::vector<std::vector<PBCNeighbor>> BuildPBCDelaunayNeighborList(
        const std::vector<Point2>& centers,
        int lx,
        int ly
    ) {
        // Strict periodic Delaunay graph for the 2D skyrmion centers.
        // Method:
        //   replicate the center set in a 3x3 triangular PBC image cloud,
        //   run Bowyer-Watson Delaunay triangulation on the replicated points,
        //   keep only edges touching the central image, and store the nearest
        //   periodic image vector for each neighbour.
        // This graph is used consistently by phi6, G6(r), and topological defects.
        const int n = static_cast<int>(centers.size());
        std::vector<std::vector<PBCNeighbor>> neigh(n);
        if (n < 3) {
            return neigh;
        }

        struct RepPoint {
            double x = 0.0;
            double y = 0.0;
            int orig = -1;
            int sm = 0;
            int sn = 0;
        };

        struct Triangle {
            int a = -1;
            int b = -1;
            int c = -1;
            double ux = 0.0;
            double uy = 0.0;
            double r2 = -1.0;
            bool bad = false;
        };

        auto shift_real_local = [&](int m, int nn) -> Point2 {
            return {
                static_cast<double>(m) * static_cast<double>(lx)
                    + static_cast<double>(nn) * 0.5 * static_cast<double>(ly),
                static_cast<double>(nn) * SQRT3_OVER_2 * static_cast<double>(ly)
            };
        };

        std::vector<RepPoint> pts;
        pts.reserve(static_cast<std::size_t>(9 * n + 3));

        for (int m = -1; m <= 1; ++m) {
            for (int nn = -1; nn <= 1; ++nn) {
                const Point2 sh = shift_real_local(m, nn);
                for (int i = 0; i < n; ++i) {
                    pts.push_back({
                        centers[i].x + sh.x,
                        centers[i].y + sh.y,
                        i,
                        m,
                        nn
                    });
                }
            }
        }

        const int n_rep = static_cast<int>(pts.size());

        double xmin = pts[0].x;
        double xmax = pts[0].x;
        double ymin = pts[0].y;
        double ymax = pts[0].y;

        for (const auto& p : pts) {
            xmin = std::min(xmin, p.x);
            xmax = std::max(xmax, p.x);
            ymin = std::min(ymin, p.y);
            ymax = std::max(ymax, p.y);
        }

        const double dx_box = xmax - xmin;
        const double dy_box = ymax - ymin;
        const double dmax = std::max(dx_box, dy_box);
        const double xmid = 0.5 * (xmin + xmax);
        const double ymid = 0.5 * (ymin + ymax);

        const int s0 = static_cast<int>(pts.size());
        pts.push_back({ xmid - 20.0 * dmax, ymid - dmax, -1, 0, 0 });
        const int s1 = static_cast<int>(pts.size());
        pts.push_back({ xmid,              ymid + 20.0 * dmax, -1, 0, 0 });
        const int s2 = static_cast<int>(pts.size());
        pts.push_back({ xmid + 20.0 * dmax, ymid - dmax, -1, 0, 0 });

        auto make_triangle = [&](int ia, int ib, int ic) -> Triangle {
            Triangle t;
            t.a = ia;
            t.b = ib;
            t.c = ic;
            t.bad = false;

            const double ax = pts[ia].x;
            const double ay = pts[ia].y;
            const double bx = pts[ib].x;
            const double by = pts[ib].y;
            const double cx = pts[ic].x;
            const double cy = pts[ic].y;

            const double d =
                2.0 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by));

            if (std::fabs(d) < 1.0e-20) {
                t.ux = 0.0;
                t.uy = 0.0;
                t.r2 = -1.0;
                return t;
            }

            const double ax2ay2 = ax * ax + ay * ay;
            const double bx2by2 = bx * bx + by * by;
            const double cx2cy2 = cx * cx + cy * cy;

            t.ux =
                (ax2ay2 * (by - cy)
                    + bx2by2 * (cy - ay)
                    + cx2cy2 * (ay - by)) / d;

            t.uy =
                (ax2ay2 * (cx - bx)
                    + bx2by2 * (ax - cx)
                    + cx2cy2 * (bx - ax)) / d;

            const double ddx = t.ux - ax;
            const double ddy = t.uy - ay;
            t.r2 = ddx * ddx + ddy * ddy;
            return t;
        };

        auto in_circumcircle = [&](const Triangle& t, const RepPoint& p) -> bool {
            if (!(t.r2 > 0.0) || !std::isfinite(t.r2)) {
                return false;
            }
            const double dx = p.x - t.ux;
            const double dy = p.y - t.uy;
            const double r2p = dx * dx + dy * dy;
            return r2p <= t.r2 * (1.0 + 1.0e-12);
        };

        auto same_undirected_edge =
            [](const std::pair<int, int>& e1, const std::pair<int, int>& e2) -> bool {
            return (e1.first == e2.first && e1.second == e2.second) ||
                   (e1.first == e2.second && e1.second == e2.first);
        };

        std::vector<Triangle> tris;
        tris.reserve(static_cast<std::size_t>(2 * n_rep + 16));
        tris.push_back(make_triangle(s0, s1, s2));

        for (int ip = 0; ip < n_rep; ++ip) {
            std::vector<std::pair<int, int>> polygon_edges;
            polygon_edges.reserve(64);

            for (auto& t : tris) {
                t.bad = in_circumcircle(t, pts[ip]);
                if (t.bad) {
                    polygon_edges.push_back({ t.a, t.b });
                    polygon_edges.push_back({ t.b, t.c });
                    polygon_edges.push_back({ t.c, t.a });
                }
            }

            tris.erase(
                std::remove_if(
                    tris.begin(),
                    tris.end(),
                    [](const Triangle& t) { return t.bad; }
                ),
                tris.end()
            );

            std::vector<std::pair<int, int>> boundary;
            boundary.reserve(polygon_edges.size());

            for (std::size_t e = 0; e < polygon_edges.size(); ++e) {
                bool duplicate = false;
                for (std::size_t f = 0; f < polygon_edges.size(); ++f) {
                    if (e == f) {
                        continue;
                    }
                    if (same_undirected_edge(polygon_edges[e], polygon_edges[f])) {
                        duplicate = true;
                        break;
                    }
                }
                if (!duplicate) {
                    boundary.push_back(polygon_edges[e]);
                }
            }

            for (const auto& e : boundary) {
                const Triangle nt = make_triangle(e.first, e.second, ip);
                if (nt.r2 > 0.0 && std::isfinite(nt.r2)) {
                    tris.push_back(nt);
                }
            }
        }

        tris.erase(
            std::remove_if(
                tris.begin(),
                tris.end(),
                [&](const Triangle& t) {
                    return t.a >= n_rep || t.b >= n_rep || t.c >= n_rep;
                }
            ),
            tris.end()
        );

        auto add_neighbor = [&](int ia, int ib, const Point2& dr) {
            if (ia < 0 || ia >= n || ib < 0 || ib >= n || ia == ib) {
                return;
            }

            auto& list = neigh[ia];
            const double r2_new = dr.x * dr.x + dr.y * dr.y;

            for (auto& item : list) {
                if (item.j == ib) {
                    const double r2_old = item.dr.x * item.dr.x + item.dr.y * item.dr.y;
                    if (r2_new < r2_old) {
                        item.dr = dr;
                    }
                    return;
                }
            }
            list.push_back({ ib, dr });
        };

        auto add_edge = [&](int u, int v) {
            const int ia = pts[u].orig;
            const int ib = pts[v].orig;
            if (ia < 0 || ib < 0 || ia == ib) {
                return;
            }

            if (pts[u].sm == 0 && pts[u].sn == 0) {
                Point2 dr{ pts[v].x - centers[ia].x, pts[v].y - centers[ia].y };
                add_neighbor(ia, ib, dr);
            }
            if (pts[v].sm == 0 && pts[v].sn == 0) {
                Point2 dr{ pts[u].x - centers[ib].x, pts[u].y - centers[ib].y };
                add_neighbor(ib, ia, dr);
            }
        };

        for (const auto& t : tris) {
            add_edge(t.a, t.b);
            add_edge(t.b, t.c);
            add_edge(t.c, t.a);
        }

        return neigh;
    }

    std::vector<std::complex<double>> ComputePsi6Knn6PBC(
        const std::vector<Point2>& centers,
        int lx,
        int ly
    ) {
        // Historical name kept for compatibility.  This is a strict PBC
        // Delaunay-neighbour psi6, not KNN=6.
        const int n = static_cast<int>(centers.size());
        std::vector<std::complex<double>> psi6(n, { 0.0, 0.0 });
        if (n < 3) {
            return psi6;
        }

        const auto neigh = BuildPBCDelaunayNeighborList(centers, lx, ly);
        for (int i = 0; i < n; ++i) {
            if (neigh[i].empty()) {
                continue;
            }
            double re = 0.0;
            double im = 0.0;
            for (const auto& item : neigh[i]) {
                const double theta = std::atan2(item.dr.y, item.dr.x);
                re += std::cos(6.0 * theta);
                im += std::sin(6.0 * theta);
            }
            const double inv = 1.0 / static_cast<double>(neigh[i].size());
            psi6[i] = { re * inv, im * inv };
        }
        return psi6;
    }



    Phi6FrameResult ComputePhi6FromSpinConfiguration(
        const std::vector<Spin3>& spins,
        int lx,
        int ly
    ) {
        Phi6FrameResult out;

        const std::vector<Point2> centers =
            ExtractSkyrmionCentersBySzCore(spins, lx, ly);

        out.nsk = static_cast<int>(centers.size());

        if (out.nsk < 3) {
            return out;
        }

        const std::vector<std::complex<double>> psi6 =
            ComputePsi6Knn6PBC(centers, lx, ly);

        std::complex<double> phi_total(0.0, 0.0);
        double abs_sum = 0.0;
        double abs2_sum = 0.0;

        for (const auto& z : psi6) {
            const double az = std::abs(z);

            phi_total += z;
            abs_sum += az;
            abs2_sum += az * az;
        }

        phi_total /= static_cast<double>(out.nsk);

        out.abs_Phi6 = std::abs(phi_total);
        out.mean_abs_psi6 = abs_sum / static_cast<double>(out.nsk);

        const double mean2 = abs2_sum / static_cast<double>(out.nsk);
        const double var =
            std::max(0.0, mean2 - out.mean_abs_psi6 * out.mean_abs_psi6);

        out.std_abs_psi6 = std::sqrt(var);

        return out;
    }


    // ============================================================
    // g(x,0)-1 helpers, strict Python-compatible version
    //
    // Definition:
    //   centers -> ordered pair displacement under triangular PBC
    //   -> rotate by -theta_axis
    //   -> scale by a_sk
    //   -> count pairs with 0 <= dx/a_sk < xmax and |dy/a_sk| < strip_half_width
    //
    // Normalization:
    //   H_ideal_per_bin =
    //       Nsk*(Nsk-1) * [a_sk^2 * dbin * 2*strip_half_width] / Area
    //
    // Average:
    //   for each disorder/bin:
    //       sum histograms over selected measurements first,
    //       sum ideal counts over selected measurements first,
    //       then g_bin = H_bin/I_bin - 1
    //   final:
    //       mean and SEM over disorder bins.
    //
    // Input options:
    //   calculate_gx0=true
    //   gx0_measure_stride=1
    //   gx0_xmax=12.0
    //   gx0_dbin=0.02
    //   gx0_strip_half_width=0.08
    //   gx0_core_sign=negative
    //   gx0_core_threshold=0.0
    //   gx0_min_cluster_size=4
    // ============================================================

    struct Gx0Settings {
        bool enabled = true;
        int measure_stride = 1;              // selected observable measurements
        double xmax = 12.0;
        double dbin = 0.02;
        double strip_half_width = 0.08;
        double nn_rmin_factor = 0.70;
        double nn_rmax_factor = 1.30;
        double core_threshold = 0.0;
        int min_cluster_size = 4;
        std::string core_sign = "negative";  // "negative" or "positive"
    };

    std::string ReadStringOrDefault(
        const char* fname,
        const char* key,
        const std::string& fallback
    ) {
        std::ifstream input(fname);
        if (!input) {
            return fallback;
        }

        std::string line;
        while (std::getline(input, line)) {
            const std::size_t comment = line.find('#');
            if (comment != std::string::npos) {
                line.erase(comment);
            }

            const std::size_t equal = line.find('=');
            if (equal == std::string::npos) {
                continue;
            }

            if (Trim(line.substr(0, equal)) != key) {
                continue;
            }

            std::string value = Trim(line.substr(equal + 1));
            return value.empty() ? fallback : value;
        }

        return fallback;
    }

    Gx0Settings LoadGx0Settings() {
        Gx0Settings s;

        TryGetParaFromInput_bool("input.in", "calculate_gx0", s.enabled);
        TryGetParaFromInput_int("input.in", "gx0_measure_stride", s.measure_stride);
        TryGetParaFromInput_real("input.in", "gx0_xmax", s.xmax);
        TryGetParaFromInput_real("input.in", "gx0_dbin", s.dbin);
        TryGetParaFromInput_real("input.in", "gx0_strip_half_width", s.strip_half_width);
        TryGetParaFromInput_real("input.in", "gx0_nn_rmin_factor", s.nn_rmin_factor);
        TryGetParaFromInput_real("input.in", "gx0_nn_rmax_factor", s.nn_rmax_factor);
        TryGetParaFromInput_real("input.in", "gx0_core_threshold", s.core_threshold);
        TryGetParaFromInput_int("input.in", "gx0_min_cluster_size", s.min_cluster_size);

        s.core_sign = ReadStringOrDefault("input.in", "gx0_core_sign", s.core_sign);

        if (s.measure_stride < 1) {
            s.measure_stride = 1;
        }
        if (!(s.xmax > 0.0)) {
            s.xmax = 12.0;
        }
        if (!(s.dbin > 0.0)) {
            s.dbin = 0.02;
        }
        if (!(s.strip_half_width > 0.0)) {
            s.strip_half_width = 0.08;
        }
        if (s.min_cluster_size < 1) {
            s.min_cluster_size = 1;
        }
        if (s.core_sign != "negative" && s.core_sign != "positive") {
            s.core_sign = "negative";
        }

        return s;
    }

    int Gx0NumberOfBins(const Gx0Settings& s) {
        if (!(s.xmax > 0.0) || !(s.dbin > 0.0) ||
            !std::isfinite(s.xmax) || !std::isfinite(s.dbin)) {
            return 1;
        }

        const double raw = std::ceil(s.xmax / s.dbin);
        if (!std::isfinite(raw) || raw < 1.0) {
            return 1;
        }

        // Prevent accidental enormous allocations from a bad input file.
        const double max_bins = 200000.0;
        if (raw > max_bins) {
            return static_cast<int>(max_bins);
        }

        return static_cast<int>(raw);
    }

    double TriangularBoxArea(int lx, int ly) {
        return static_cast<double>(lx) * static_cast<double>(ly) * SQRT3_OVER_2;
    }

    double AskFromDensity(int nsk, int lx, int ly) {
        if (nsk <= 0) {
            return std::numeric_limits<double>::quiet_NaN();
        }

        return std::sqrt(
            2.0 * TriangularBoxArea(lx, ly) /
            (std::sqrt(3.0) * static_cast<double>(nsk))
        );
    }

    double CanonicalPsi6Axis(double angle) {
        const double period = PI / 3.0;
        double x = std::fmod(angle + 0.5 * period, period);
        if (x < 0.0) {
            x += period;
        }
        return x - 0.5 * period;
    }

    std::vector<Point2> ExtractSkyrmionCentersForGx0(
        const std::vector<Spin3>& spins,
        int lx,
        int ly,
        const Gx0Settings& gx0
    ) {
        const int nsite = lx * ly;

        std::vector<unsigned char> mask(nsite, 0);
        std::vector<unsigned char> visited(nsite, 0);

        const double th = gx0.core_threshold;

        for (int i = 0; i < nsite; ++i) {
            if (gx0.core_sign == "negative") {
                if (spins[i].z < th) {
                    mask[i] = 1;
                }
            }
            else {
                if (spins[i].z > th) {
                    mask[i] = 1;
                }
            }
        }

        const auto mod = [](int a, int n) {
            return (a % n + n) % n;
        };

        const auto index = [&](int x, int y) {
            return mod(x, lx) + mod(y, ly) * lx;
        };

        const int dx8[8] = { -1, 0, 1, -1, 1, -1, 0, 1 };
        const int dy8[8] = { -1,-1,-1,  0, 0,  1, 1, 1 };

        std::vector<Point2> centers;
        std::vector<int> stack;
        stack.reserve(1024);

        for (int y0 = 0; y0 < ly; ++y0) {
            for (int x0 = 0; x0 < lx; ++x0) {
                const int start = index(x0, y0);

                if (!mask[start] || visited[start]) {
                    continue;
                }

                stack.clear();
                stack.push_back(start);
                visited[start] = 1;

                std::vector<double> xs;
                std::vector<double> ys;
                std::vector<double> ws;

                while (!stack.empty()) {
                    const int site = stack.back();
                    stack.pop_back();

                    const int x = site % lx;
                    const int y = site / lx;

                    double w = 0.0;
                    if (gx0.core_sign == "negative") {
                        w = std::max(th - spins[site].z, 0.0);
                    }
                    else {
                        w = std::max(spins[site].z - th, 0.0);
                    }

                    // Fallback for threshold exactly equal to local values.
                    if (w <= 0.0) {
                        w = 1.0e-12;
                    }

                    xs.push_back(static_cast<double>(x));
                    ys.push_back(static_cast<double>(y));
                    ws.push_back(w);

                    for (int nn = 0; nn < 8; ++nn) {
                        const int xx = mod(x + dx8[nn], lx);
                        const int yy = mod(y + dy8[nn], ly);
                        const int nb = index(xx, yy);

                        if (mask[nb] && !visited[nb]) {
                            visited[nb] = 1;
                            stack.push_back(nb);
                        }
                    }
                }

                if (static_cast<int>(xs.size()) < gx0.min_cluster_size) {
                    continue;
                }

                double wsum = 0.0;
                for (const double w : ws) {
                    wsum += w;
                }

                if (wsum <= 0.0) {
                    continue;
                }

                const double cx_lat =
                    PeriodicCenter1D(xs, ws, static_cast<double>(lx));

                const double cy_lat =
                    PeriodicCenter1D(ys, ws, static_cast<double>(ly));

                Point2 center = LatticeToReal(cx_lat, cy_lat);
                center = WrapRealToBox(center, lx, ly);

                centers.push_back(center);
            }
        }

        return centers;
    }

    double ComputePsi6AxisFromCenters(
        const std::vector<Point2>& centers,
        double a_sk,
        int lx,
        int ly,
        const Gx0Settings& gx0
    ) {
        const int n = static_cast<int>(centers.size());

        if (n < 3 || !(a_sk > 0.0) || !std::isfinite(a_sk)) {
            return 0.0;
        }

        const double rmin = gx0.nn_rmin_factor * a_sk;
        const double rmax = gx0.nn_rmax_factor * a_sk;
        const double rmin2 = rmin * rmin;
        const double rmax2 = rmax * rmax;

        double re = 0.0;
        double im = 0.0;
        int count = 0;

        for (int i = 0; i < n - 1; ++i) {
            for (int j = i + 1; j < n; ++j) {
                const Point2 dr = MinImageDeltaReal(centers[i], centers[j], lx, ly);
                const double r2 = dr.x * dr.x + dr.y * dr.y;

                if (r2 > rmin2 && r2 < rmax2) {
                    const double theta = std::atan2(dr.y, dr.x);
                    re += std::cos(6.0 * theta);
                    im += std::sin(6.0 * theta);
                    ++count;
                }
            }
        }

        if (count <= 0) {
            return 0.0;
        }

        return CanonicalPsi6Axis(std::atan2(im, re) / 6.0);
    }

    void AccumulateGx0FrameFromCenters(
        const std::vector<Point2>& centers,
        double a_sk,
        double axis_angle,
        int lx,
        int ly,
        const Gx0Settings& gx0,
        std::vector<double>& hist_x,
        std::vector<double>& ideal_x
    ) {
        const int n = static_cast<int>(centers.size());

        if (n < 2 || !(a_sk > 0.0) || !std::isfinite(a_sk)) {
            return;
        }

        const int nbins = Gx0NumberOfBins(gx0);

        if (static_cast<int>(hist_x.size()) != nbins ||
            static_cast<int>(ideal_x.size()) != nbins) {
            throw std::runtime_error("Gx0 histogram size mismatch.");
        }

        const double c = std::cos(-axis_angle);
        const double s = std::sin(-axis_angle);

        // Ordered pairs: Nsk*(Nsk-1).
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                if (i == j) {
                    continue;
                }

                // MinImageDeltaReal(ri,rj) returns rj-ri.
                // Therefore this gives R_i - R_j, same as Python ordered pair code.
                const Point2 dr = MinImageDeltaReal(centers[j], centers[i], lx, ly);

                const double xr = c * dr.x - s * dr.y;
                const double yr = s * dr.x + c * dr.y;

                const double xs = xr / a_sk;
                const double ys = yr / a_sk;

                if (xs >= 0.0 &&
                    xs < gx0.xmax &&
                    std::fabs(ys) < gx0.strip_half_width &&
                    std::isfinite(xs) && std::isfinite(ys)) {
                    const double raw_bin = std::floor(xs / gx0.dbin);
                    if (std::isfinite(raw_bin)) {
                        const int bin = static_cast<int>(raw_bin);
                        if (bin >= 0 && bin < nbins) {
                            hist_x[bin] += 1.0;
                        }
                    }
                }
            }
        }

        const double strip_area_real =
            a_sk * a_sk * gx0.dbin * (2.0 * gx0.strip_half_width);

        const double ideal_per_bin =
            static_cast<double>(n) *
            static_cast<double>(n - 1) *
            strip_area_real /
            TriangularBoxArea(lx, ly);

        for (int b = 0; b < nbins; ++b) {
            ideal_x[b] += ideal_per_bin;
        }
    }

    std::vector<double> Gx0FromHistogram(
        const std::vector<double>& hist_x,
        const std::vector<double>& ideal_x
    ) {
        const int nbins = static_cast<int>(hist_x.size());
        std::vector<double> gx0(nbins, std::numeric_limits<double>::quiet_NaN());

        for (int b = 0; b < nbins; ++b) {
            if (ideal_x[b] > 0.0) {
                gx0[b] = hist_x[b] / ideal_x[b] - 1.0;
            }
        }

        return gx0;
    }

    void MeanSemStdCurves(
        const std::vector<std::vector<double>>& samples,
        std::vector<double>& mean,
        std::vector<double>& sem,
        std::vector<double>& stddev,
        std::vector<int>& nvalid
    ) {
        const int ns = static_cast<int>(samples.size());
        const int nbins =
            ns > 0 ? static_cast<int>(samples[0].size()) : 0;

        mean.assign(nbins, std::numeric_limits<double>::quiet_NaN());
        sem.assign(nbins, std::numeric_limits<double>::quiet_NaN());
        stddev.assign(nbins, std::numeric_limits<double>::quiet_NaN());
        nvalid.assign(nbins, 0);

        for (int b = 0; b < nbins; ++b) {
            double sum = 0.0;
            int n = 0;

            for (int sidx = 0; sidx < ns; ++sidx) {
                const double v = samples[sidx][b];
                if (std::isfinite(v)) {
                    sum += v;
                    ++n;
                }
            }

            nvalid[b] = n;

            if (n <= 0) {
                continue;
            }

            const double m = sum / static_cast<double>(n);
            mean[b] = m;

            if (n == 1) {
                stddev[b] = 0.0;
                sem[b] = 0.0;
                continue;
            }

            double ss = 0.0;
            for (int sidx = 0; sidx < ns; ++sidx) {
                const double v = samples[sidx][b];
                if (std::isfinite(v)) {
                    const double d = v - m;
                    ss += d * d;
                }
            }

            const double sd = std::sqrt(ss / static_cast<double>(n - 1));
            stddev[b] = sd;
            sem[b] = sd / std::sqrt(static_cast<double>(n));
        }
    }

    void WriteGx0File(
        int iD,
        int iB,
        const Gx0Settings& gx0,
        const std::vector<std::vector<double>>& gx0_samples,
        const std::vector<int>& frames_per_bin,
        const std::vector<double>& avg_nsk_per_bin,
        const std::vector<int>& min_nsk_per_bin,
        const std::vector<int>& max_nsk_per_bin
    ) {
        const int nbins = Gx0NumberOfBins(gx0);

        std::vector<double> mean;
        std::vector<double> sem;
        std::vector<double> stddev;
        std::vector<int> nvalid;

        MeanSemStdCurves(gx0_samples, mean, sem, stddev, nvalid);

        double total_frames = 0.0;
        double weighted_nsk_sum = 0.0;

        for (std::size_t i = 0; i < frames_per_bin.size(); ++i) {
            total_frames += static_cast<double>(frames_per_bin[i]);
            weighted_nsk_sum +=
                static_cast<double>(frames_per_bin[i]) * avg_nsk_per_bin[i];
        }

        const double avg_nsk =
            total_frames > 0.0 ? weighted_nsk_sum / total_frames : 0.0;

        std::ostringstream filename;
        filename << "Gx0_Measures_iD" << iD << "_iB" << iB << ".txt";

        std::ofstream output(filename.str());
        if (!output) {
            throw std::runtime_error("Cannot write Gx0 measurement file.");
        }

        output << "# x_over_ask g_x0_minus1_mean SEM STD N_disorder_bins\n";
        output << "# g(x,0)-1 from Skyrmion centers, strict Python-compatible normalization.\n";
        output << "# ordered pairs, triangular PBC, Psi6-axis alignment, density a_sk.\n";
        output << "# First sum H and I over selected measurements inside each disorder bin; then g=H/I-1; then average over bins.\n";
        output << "# calculate_gx0 " << (gx0.enabled ? "true" : "false") << "\n";
        output << "# gx0_measure_stride " << gx0.measure_stride << "\n";
        output << "# gx0_xmax " << gx0.xmax << "\n";
        output << "# gx0_dbin " << gx0.dbin << "\n";
        output << "# gx0_strip_half_width " << gx0.strip_half_width << "\n";
        output << "# gx0_core_sign " << gx0.core_sign << "\n";
        output << "# gx0_core_threshold " << gx0.core_threshold << "\n";
        output << "# gx0_min_cluster_size " << gx0.min_cluster_size << "\n";
        output << "# disorder_bins " << gx0_samples.size() << "\n";
        output << "# total_gx0_frames " << static_cast<long long>(total_frames) << "\n";
        output << "# average_Nsk " << std::setprecision(16) << avg_nsk << "\n";

        for (int b = 0; b < nbins; ++b) {
            const double xmid = (static_cast<double>(b) + 0.5) * gx0.dbin;

            output << std::setprecision(16)
                << xmid << ' '
                << mean[b] << ' '
                << sem[b] << ' '
                << stddev[b] << ' '
                << nvalid[b] << '\n';
        }

        std::ostringstream bin_filename;
        bin_filename << "Gx0_DisorderSamples_iD" << iD << "_iB" << iB << ".txt";

        std::ofstream bout(bin_filename.str());
        if (!bout) {
            throw std::runtime_error("Cannot write Gx0 disorder-sample file.");
        }

        bout << "# disorder_bin x_over_ask g_x0_minus1\n";

        for (std::size_t ib = 0; ib < gx0_samples.size(); ++ib) {
            for (int b = 0; b < nbins; ++b) {
                const double xmid = (static_cast<double>(b) + 0.5) * gx0.dbin;
                bout << ib << ' '
                    << std::setprecision(16) << xmid << ' '
                    << gx0_samples[ib][b] << '\n';
            }
            bout << '\n';
        }

        std::ostringstream debug_filename;
        debug_filename << "Gx0_Debug_iD" << iD << "_iB" << iB << ".txt";

        std::ofstream dout(debug_filename.str());
        if (!dout) {
            throw std::runtime_error("Cannot write Gx0 debug file.");
        }

        dout << "# disorder_bin selected_gx0_frames avg_Nsk min_Nsk max_Nsk\n";
        for (std::size_t ib = 0; ib < frames_per_bin.size(); ++ib) {
            dout << ib << ' '
                << frames_per_bin[ib] << ' '
                << std::setprecision(16) << avg_nsk_per_bin[ib] << ' '
                << min_nsk_per_bin[ib] << ' '
                << max_nsk_per_bin[ib] << '\n';
        }
    }


    // ============================================================
    // Hierarchical g(x,0)-1 statistics:
    // selected measures -> measure blocks -> disorder samples.
    //
    // Output columns:
    //   x
    //   mean
    //   SEM_disorder
    //   STD_disorder
    //   N_disorder
    //   SEM_block_mean
    //   STD_block_mean
    //   N_blocks_avg
    //   N_frames_avg
    //
    // This keeps the final error as disorder-to-disorder SEM, while
    // also saving the internal block/time fluctuation for diagnostics.
    // ============================================================

    struct Gx0DisorderCurve {
        std::vector<double> mean;
        std::vector<double> sem_block;
        std::vector<double> std_block;
        int n_blocks = 0;
        int selected_frames = 0;
        double avg_nsk = 0.0;
        int min_nsk = 0;
        int max_nsk = 0;
    };

    int LoadGx0BlockSize() {
        int block_size = 20;
        TryGetParaFromInput_int("input.in", "gx0_block_size", block_size);
        if (block_size < 1) {
            block_size = 1;
        }
        return block_size;
    }

    bool LoadGx0WriteBlockSamples() {
        // Keep the historical output behavior by default: write the detailed block file.
        // For very large runs, set write_gx0_block_samples=false in input.in to avoid
        // huge files. Main results and errors are unchanged.
        bool write_blocks = true;
        TryGetParaFromInput_bool("input.in", "write_gx0_block_samples", write_blocks);
        return write_blocks;
    }

    std::vector<double> MeanCurveIgnoringNaN(
        const std::vector<std::vector<double>>& curves
    ) {
        if (curves.empty()) {
            return {};
        }

        const int nbins = static_cast<int>(curves[0].size());
        std::vector<double> mean(nbins, std::numeric_limits<double>::quiet_NaN());

        for (int b = 0; b < nbins; ++b) {
            double sum = 0.0;
            int n = 0;

            for (const auto& c : curves) {
                if (b < static_cast<int>(c.size()) && std::isfinite(c[b])) {
                    sum += c[b];
                    ++n;
                }
            }

            if (n > 0) {
                mean[b] = sum / static_cast<double>(n);
            }
        }

        return mean;
    }

    void MeanSemStdCurvesGeneral(
        const std::vector<std::vector<double>>& samples,
        std::vector<double>& mean,
        std::vector<double>& sem,
        std::vector<double>& stddev,
        std::vector<int>& nvalid
    ) {
        const int ns = static_cast<int>(samples.size());
        const int nbins =
            ns > 0 ? static_cast<int>(samples[0].size()) : 0;

        mean.assign(nbins, std::numeric_limits<double>::quiet_NaN());
        sem.assign(nbins, std::numeric_limits<double>::quiet_NaN());
        stddev.assign(nbins, std::numeric_limits<double>::quiet_NaN());
        nvalid.assign(nbins, 0);

        for (int b = 0; b < nbins; ++b) {
            double sum = 0.0;
            int n = 0;

            for (int sidx = 0; sidx < ns; ++sidx) {
                const double v = samples[sidx][b];
                if (std::isfinite(v)) {
                    sum += v;
                    ++n;
                }
            }

            nvalid[b] = n;

            if (n <= 0) {
                continue;
            }

            const double m = sum / static_cast<double>(n);
            mean[b] = m;

            if (n == 1) {
                stddev[b] = 0.0;
                sem[b] = 0.0;
                continue;
            }

            double ss = 0.0;
            for (int sidx = 0; sidx < ns; ++sidx) {
                const double v = samples[sidx][b];
                if (std::isfinite(v)) {
                    const double d = v - m;
                    ss += d * d;
                }
            }

            const double sd = std::sqrt(ss / static_cast<double>(n - 1));
            stddev[b] = sd;
            sem[b] = sd / std::sqrt(static_cast<double>(n));
        }
    }

    void FinishGx0BlockIfNeeded(
        std::vector<double>& block_hist,
        std::vector<double>& block_ideal,
        int& block_frame_count,
        std::vector<std::vector<double>>& block_curves
    ) {
        if (block_frame_count <= 0) {
            return;
        }

        if (block_hist.empty() || block_hist.size() != block_ideal.size()) {
            std::fill(block_hist.begin(), block_hist.end(), 0.0);
            std::fill(block_ideal.begin(), block_ideal.end(), 0.0);
            block_frame_count = 0;
            return;
        }

        bool has_ideal = false;
        for (const double v : block_ideal) {
            if (v > 0.0 && std::isfinite(v)) {
                has_ideal = true;
                break;
            }
        }

        if (has_ideal) {
            std::vector<double> curve = Gx0FromHistogram(block_hist, block_ideal);
            if (curve.size() == block_hist.size()) {
                block_curves.push_back(std::move(curve));
            }
        }

        std::fill(block_hist.begin(), block_hist.end(), 0.0);
        std::fill(block_ideal.begin(), block_ideal.end(), 0.0);
        block_frame_count = 0;
    }

    Gx0DisorderCurve MakeGx0DisorderCurveFromBlocks(
        const std::vector<std::vector<double>>& block_curves,
        int selected_frames,
        double nsk_sum,
        int min_nsk,
        int max_nsk
    ) {
        Gx0DisorderCurve out;
        out.n_blocks = static_cast<int>(block_curves.size());
        out.selected_frames = selected_frames;
        out.avg_nsk =
            selected_frames > 0 ? nsk_sum / static_cast<double>(selected_frames) : 0.0;
        out.min_nsk = selected_frames > 0 ? min_nsk : 0;
        out.max_nsk = selected_frames > 0 ? max_nsk : 0;

        if (block_curves.empty()) {
            return out;
        }

        std::vector<int> nvalid;
        MeanSemStdCurvesGeneral(
            block_curves,
            out.mean,
            out.sem_block,
            out.std_block,
            nvalid
        );

        return out;
    }

    void WriteGx0HierarchicalFile(
        int iD,
        int iB,
        const Gx0Settings& gx0,
        int gx0_block_size,
        bool write_block_samples,
        const std::vector<Gx0DisorderCurve>& disorder_curves,
        const std::vector<std::vector<std::vector<double>>>& all_block_curves
    ) {
        const int nbins = Gx0NumberOfBins(gx0);

        std::vector<std::vector<double>> disorder_samples;
        disorder_samples.reserve(disorder_curves.size());

        for (const auto& d : disorder_curves) {
            if (static_cast<int>(d.mean.size()) == nbins) {
                disorder_samples.push_back(d.mean);
            }
            else {
                disorder_samples.push_back(
                    std::vector<double>(nbins, std::numeric_limits<double>::quiet_NaN())
                );
            }
        }

        std::vector<double> mean;
        std::vector<double> sem_disorder;
        std::vector<double> std_disorder;
        std::vector<int> nvalid_disorder;

        MeanSemStdCurvesGeneral(
            disorder_samples,
            mean,
            sem_disorder,
            std_disorder,
            nvalid_disorder
        );

        std::vector<double> sem_block_mean(nbins, std::numeric_limits<double>::quiet_NaN());
        std::vector<double> std_block_mean(nbins, std::numeric_limits<double>::quiet_NaN());

        for (int b = 0; b < nbins; ++b) {
            double sum_sem = 0.0;
            double sum_std = 0.0;
            int n_sem = 0;
            int n_std = 0;

            for (const auto& d : disorder_curves) {
                if (b < static_cast<int>(d.sem_block.size()) && std::isfinite(d.sem_block[b])) {
                    sum_sem += d.sem_block[b];
                    ++n_sem;
                }
                if (b < static_cast<int>(d.std_block.size()) && std::isfinite(d.std_block[b])) {
                    sum_std += d.std_block[b];
                    ++n_std;
                }
            }

            if (n_sem > 0) {
                sem_block_mean[b] = sum_sem / static_cast<double>(n_sem);
            }
            if (n_std > 0) {
                std_block_mean[b] = sum_std / static_cast<double>(n_std);
            }
        }

        double total_frames = 0.0;
        double total_blocks = 0.0;
        double weighted_nsk_sum = 0.0;

        for (const auto& d : disorder_curves) {
            total_frames += static_cast<double>(d.selected_frames);
            total_blocks += static_cast<double>(d.n_blocks);
            weighted_nsk_sum +=
                static_cast<double>(d.selected_frames) * d.avg_nsk;
        }

        const double avg_nsk =
            total_frames > 0.0 ? weighted_nsk_sum / total_frames : 0.0;

        const double avg_blocks =
            disorder_curves.empty() ? 0.0 :
            total_blocks / static_cast<double>(disorder_curves.size());

        const double avg_frames =
            disorder_curves.empty() ? 0.0 :
            total_frames / static_cast<double>(disorder_curves.size());

        std::ostringstream filename;
        filename << "Gx0_Measures_iD" << iD << "_iB" << iB << ".txt";

        std::ofstream output(filename.str());
        if (!output) {
            throw std::runtime_error("Cannot write Gx0 measurement file.");
        }

        output << "# x_over_ask g_x0_minus1_mean SEM_disorder STD_disorder "
            << "N_disorder_samples SEM_block_mean STD_block_mean "
            << "N_measure_blocks_avg N_selected_frames_avg\n";
        output << "# Hierarchical average: selected measures -> measure blocks -> disorder samples.\n";
        output << "# Final error used for plotting should usually be SEM_disorder.\n";
        output << "# SEM_block_mean is internal measure/block fluctuation averaged over disorder samples.\n";
        output << "# calculate_gx0 " << (gx0.enabled ? "true" : "false") << "\n";
        output << "# gx0_measure_stride " << gx0.measure_stride << "\n";
        output << "# gx0_block_size " << gx0_block_size << "\n";
        output << "# write_gx0_block_samples " << (write_block_samples ? "true" : "false") << "\n";
        output << "# gx0_xmax " << gx0.xmax << "\n";
        output << "# gx0_dbin " << gx0.dbin << "\n";
        output << "# gx0_strip_half_width " << gx0.strip_half_width << "\n";
        output << "# gx0_core_sign " << gx0.core_sign << "\n";
        output << "# gx0_core_threshold " << gx0.core_threshold << "\n";
        output << "# gx0_min_cluster_size " << gx0.min_cluster_size << "\n";
        output << "# disorder_samples " << disorder_curves.size() << "\n";
        output << "# total_gx0_frames " << static_cast<long long>(total_frames) << "\n";
        output << "# average_Nsk " << std::setprecision(16) << avg_nsk << "\n";
        output << "# average_measure_blocks " << avg_blocks << "\n";
        output << "# average_selected_frames_per_disorder " << avg_frames << "\n";

        for (int b = 0; b < nbins; ++b) {
            const double xmid = (static_cast<double>(b) + 0.5) * gx0.dbin;

            output << std::setprecision(16)
                << xmid << ' '
                << mean[b] << ' '
                << sem_disorder[b] << ' '
                << std_disorder[b] << ' '
                << nvalid_disorder[b] << ' '
                << sem_block_mean[b] << ' '
                << std_block_mean[b] << ' '
                << avg_blocks << ' '
                << avg_frames << '\n';
        }

        std::ostringstream disorder_filename;
        disorder_filename << "Gx0_DisorderSamples_iD" << iD << "_iB" << iB << ".txt";

        std::ofstream dout(disorder_filename.str());
        if (!dout) {
            throw std::runtime_error("Cannot write Gx0 disorder-sample file.");
        }

        dout << "# disorder_bin x_over_ask g_disorder_minus1 "
            << "SEM_block STD_block N_blocks selected_frames avg_Nsk min_Nsk max_Nsk\n";

        for (std::size_t id = 0; id < disorder_curves.size(); ++id) {
            const auto& d = disorder_curves[id];

            for (int b = 0; b < nbins; ++b) {
                const double xmid = (static_cast<double>(b) + 0.5) * gx0.dbin;

                const double gv =
                    b < static_cast<int>(d.mean.size()) ?
                    d.mean[b] :
                    std::numeric_limits<double>::quiet_NaN();

                const double se =
                    b < static_cast<int>(d.sem_block.size()) ?
                    d.sem_block[b] :
                    std::numeric_limits<double>::quiet_NaN();

                const double sd =
                    b < static_cast<int>(d.std_block.size()) ?
                    d.std_block[b] :
                    std::numeric_limits<double>::quiet_NaN();

                dout << id << ' '
                    << std::setprecision(16) << xmid << ' '
                    << gv << ' '
                    << se << ' '
                    << sd << ' '
                    << d.n_blocks << ' '
                    << d.selected_frames << ' '
                    << d.avg_nsk << ' '
                    << d.min_nsk << ' '
                    << d.max_nsk << '\n';
            }

            dout << '\n';
        }

        if (write_block_samples) {
            std::ostringstream block_filename;
            block_filename << "Gx0_BlockSamples_iD" << iD << "_iB" << iB << ".txt";

            std::ofstream bout(block_filename.str());
            if (!bout) {
                throw std::runtime_error("Cannot write Gx0 block-sample file.");
            }

            bout << "# disorder_bin measure_block x_over_ask g_block_minus1\n";

            for (std::size_t id = 0; id < all_block_curves.size(); ++id) {
                for (std::size_t iblock = 0; iblock < all_block_curves[id].size(); ++iblock) {
                    const auto& curve = all_block_curves[id][iblock];

                    for (int b = 0; b < nbins; ++b) {
                        const double xmid = (static_cast<double>(b) + 0.5) * gx0.dbin;
                        const double gv =
                            b < static_cast<int>(curve.size()) ?
                            curve[b] :
                            std::numeric_limits<double>::quiet_NaN();

                        bout << id << ' '
                            << iblock << ' '
                            << std::setprecision(16) << xmid << ' '
                            << gv << '\n';
                    }

                    bout << '\n';
                }
            }
        }

        std::ostringstream debug_filename;
        debug_filename << "Gx0_Debug_iD" << iD << "_iB" << iB << ".txt";

        std::ofstream dbg(debug_filename.str());
        if (!dbg) {
            throw std::runtime_error("Cannot write Gx0 debug file.");
        }

        dbg << "# disorder_bin selected_gx0_frames n_measure_blocks avg_Nsk min_Nsk max_Nsk\n";
        for (std::size_t id = 0; id < disorder_curves.size(); ++id) {
            const auto& d = disorder_curves[id];
            dbg << id << ' '
                << d.selected_frames << ' '
                << d.n_blocks << ' '
                << std::setprecision(16) << d.avg_nsk << ' '
                << d.min_nsk << ' '
                << d.max_nsk << '\n';
        }
    }



    // ============================================================
    // G6(r) and G6(t) helpers
    //
    // G6(r):
    //   centers -> psi6_i by PBC Delaunay neighbours (variable coordination)
    //   unordered pairs i<j
    //   rho = |r_i-r_j| / a_sk
    //   G6(rho) = < Re[ psi6_i * conj(psi6_j) ] >
    //
    // G6(t):
    //   centers -> psi6_i
    //   nearest-neighbour interpolation to a fixed triangular grid
    //   symmetric normalized pixel correlation:
    //       2 Re[psi(t) conj(psi(t+lag))] / (|psi(t)|^2 + |psi(t+lag)|^2)
    //
    // Averaging:
    //   Same hierarchy as g(x,0)-1:
    //       selected measures -> measure blocks -> disorder samples -> disorder mean/SEM
    //   For G6(t), each disorder sample first computes temporal correlations
    //   from its selected fixed-grid fields; then disorder mean/SEM is written.
    // ============================================================

    struct G6rSettings {
        bool enabled = false;
        int measure_stride = 1;
        int block_size = 20;
        double rmax = 12.0;
        double rbin = 0.1;
        bool write_block_samples = true;
    };

    struct G6tSettings {
        bool enabled = false;
        int measure_stride = 1;
        int grid_nx = 0;       // <=0 means use Lx
        int grid_ny = 0;       // <=0 means use Ly
        int max_lag = 0;       // <=0 means all lags
        double dt = 1.0;
        double max_assign_dist_factor = -1.0; // <=0 means no cutoff
        bool write_disorder_samples = true;
    };

    G6rSettings LoadG6rSettings() {
        G6rSettings s;

        TryGetParaFromInput_bool("input.in", "calculate_g6r", s.enabled);
        TryGetParaFromInput_int("input.in", "g6r_measure_stride", s.measure_stride);
        TryGetParaFromInput_int("input.in", "g6r_block_size", s.block_size);
        TryGetParaFromInput_real("input.in", "g6r_rmax", s.rmax);
        TryGetParaFromInput_real("input.in", "g6r_rbin", s.rbin);
        TryGetParaFromInput_bool("input.in", "write_g6r_block_samples", s.write_block_samples);

        if (s.measure_stride < 1) {
            s.measure_stride = 1;
        }
        if (s.block_size < 1) {
            s.block_size = 1;
        }
        if (!(s.rmax > 0.0) || !std::isfinite(s.rmax)) {
            s.rmax = 12.0;
        }
        if (!(s.rbin > 0.0) || !std::isfinite(s.rbin)) {
            s.rbin = 0.1;
        }

        return s;
    }

    G6tSettings LoadG6tSettings(int lx, int ly) {
        G6tSettings s;

        TryGetParaFromInput_bool("input.in", "calculate_g6t", s.enabled);
        TryGetParaFromInput_int("input.in", "g6t_measure_stride", s.measure_stride);
        TryGetParaFromInput_int("input.in", "g6t_grid_nx", s.grid_nx);
        TryGetParaFromInput_int("input.in", "g6t_grid_ny", s.grid_ny);
        TryGetParaFromInput_int("input.in", "g6t_max_lag", s.max_lag);
        TryGetParaFromInput_real("input.in", "g6t_dt", s.dt);
        TryGetParaFromInput_real("input.in", "g6t_max_assign_dist_factor", s.max_assign_dist_factor);
        TryGetParaFromInput_bool("input.in", "write_g6t_disorder_samples", s.write_disorder_samples);

        if (s.measure_stride < 1) {
            s.measure_stride = 1;
        }
        if (s.grid_nx <= 0) {
            s.grid_nx = lx;
        }
        if (s.grid_ny <= 0) {
            s.grid_ny = ly;
        }
        if (s.grid_nx < 1) {
            s.grid_nx = 1;
        }
        if (s.grid_ny < 1) {
            s.grid_ny = 1;
        }
        if (!(s.dt > 0.0) || !std::isfinite(s.dt)) {
            s.dt = 1.0;
        }

        return s;
    }

    int G6rNumberOfBins(const G6rSettings& s) {
        const double nb_real = std::ceil(s.rmax / s.rbin);
        if (!std::isfinite(nb_real) || nb_real < 1.0 || nb_real > 1000000.0) {
            throw std::runtime_error("Invalid G6r bin settings.");
        }
        return static_cast<int>(nb_real);
    }

    struct OrientFrame {
        std::vector<Point2> centers;
        std::vector<std::complex<double>> psi6;
        int nsk = 0;
        double a_sk = std::numeric_limits<double>::quiet_NaN();
        double phi6_global = 0.0;
        double phi6_local = 0.0;
        double phi6_abs_std = 0.0;
    };

    OrientFrame ComputeOrientFrameFromCenters(
        const std::vector<Point2>& centers,
        int lx,
        int ly
    ) {
        OrientFrame out;
        out.centers = centers;
        out.nsk = static_cast<int>(centers.size());
        out.a_sk = AskFromDensity(out.nsk, lx, ly);

        if (out.nsk < 3 || !std::isfinite(out.a_sk) || !(out.a_sk > 0.0)) {
            return out;
        }

        out.psi6 = ComputePsi6Knn6PBC(centers, lx, ly);

        std::complex<double> total(0.0, 0.0);
        double abs_sum = 0.0;
        double abs2_sum = 0.0;
        int n_valid = 0;

        for (const auto& z : out.psi6) {
            if (std::isfinite(z.real()) && std::isfinite(z.imag())) {
                const double az = std::abs(z);
                total += z;
                abs_sum += az;
                abs2_sum += az * az;
                ++n_valid;
            }
        }

        if (n_valid > 0) {
            total /= static_cast<double>(n_valid);
            out.phi6_global = std::abs(total);
            out.phi6_local = abs_sum / static_cast<double>(n_valid);
            const double mean2 = abs2_sum / static_cast<double>(n_valid);
            const double var = std::max(0.0, mean2 - out.phi6_local * out.phi6_local);
            out.phi6_abs_std = std::sqrt(var);
        }

        return out;
    }

    // ============================================================
    // Phase-boundary diagnostics replacing noisy time-fluctuation chi6
    //
    // Average hierarchy:
    //   measurement frames -> one disorder/bin value -> mean/SEM over disorder bins.
    //
    // Quantities:
    //   rho_def = 1 - N6/Nsk from strict 3x3-image PBC Delaunay coordination
    //   N5/N, N6/N, N7/N, Nnon6/N
    //   free/bound dislocations from neighbouring 5-7 Delaunay pairs
    //   G6_tail = average of the disorder-averaged G6(r) curve on [rmin,rmax]
    //   chi_def = <Nsk> Var_dis(rho_def)
    //   chi_g6tail = <Nsk> Var_dis(G6_tail)
    //   Binder cumulants use normalized forms: Phi6 uses O(2) Binder, scalar
    //   magnitudes use 3/2 times the conventional scalar Binder.
    //
    // Note: this file does not depend on Qhull/CGAL.  It implements a local
    // Bowyer-Watson triangulation on a 3x3 periodic image cloud.
    // ============================================================

    struct PhaseBoundarySettings {
        bool enabled = true;
        int defect_measure_stride = 1;

        // Legacy parameters kept for input compatibility.  Defects and phi6 are now
        // computed from strict PBC Delaunay neighbours; these two values are not used
        // by the Delaunay defect counter.
        double defect_nn_rmin_factor = 0.70;
        double defect_nn_rmax_factor = 1.30;

        // A neighbouring 5-fold and 7-fold site is treated as one dislocation core
        // if their separation is smaller than this factor times a_sk.
        double dislocation_pair_rmax_factor = 1.35;

        // A dislocation core is treated as bound if there is another nearby core
        // with approximately opposite polarity within this factor times a_sk.
        // Otherwise it is counted as a free dislocation.
        double dislocation_bind_rmax_factor = 3.0;

        // Opposite-polarity threshold:
        //   dot(p_a,p_b) <= -0.5 means angle >= 120 degrees.
        double dislocation_opposite_cos = -0.5;

        double g6_tail_rmin = 8.0;
        double g6_tail_rmax = 12.0;

        // Translational diagnostics for SkX -> BrG-like onset.
        bool trans_enabled = true;
        bool trans_use_fixed_axis = false;
        double trans_fixed_axis_angle = 0.0;
        double trans_fixed_ask = -1.0;
        double trans_rmax = 12.0;
        double trans_rbin = 0.1;
        double trans_tail_rmin = 8.0;
        double trans_tail_rmax = 12.0;
        double trans_fit_rmin = 3.0;
        double trans_fit_rmax = 12.0;
        double trans_gt_min_for_log = 1.0e-8;

        // Correct reciprocal-grid G_T diagnostics.
        // trans_g_mode:
        //   reference_grid  : use the PBC-compatible reciprocal-grid Bragg vectors
        //                     selected from (trans_reference_iD, trans_reference_iB).
        //                     This is the recommended mode for SkX -> BrG-like.
        //   per_frame_grid  : select reciprocal-grid Bragg vectors independently per frame.
        //                     Useful for clean-crystal sanity tests.
        //   manual          : use one manually supplied vector (trans_manual_gx, trans_manual_gy).
        std::string trans_g_mode = "reference_grid";
        int trans_reference_iD = 0;
        int trans_reference_iB = 0;
        int trans_n_g_vectors = 6;
        // IMPORTANT: the verified Python G_T code uses exactly ONE best reciprocal-grid G
        // unless AVERAGE_SIXFOLD is explicitly enabled.  Keep this false by default
        // so mT = sqrt(Smax/Nsk), matching the Python debug output.
        bool trans_average_multiple_g = false;
        double trans_q0_min_factor = 0.70;
        double trans_q0_max_factor = 1.30;
        double trans_angle_min_separation = 0.25;

        // Additional SkX -> BrG-like static diagnostics.
        // Peak-width diagnostics are computed from the skyrmion-center S(q)
        // in a small reciprocal-grid window around the selected Bragg vector.
        double trans_peak_width_window_factor = 0.35;  // window radius = factor*q0
        double trans_peak_width_background = 1.0;      // random center S(q) background after /N normalization

        double trans_manual_gx = std::numeric_limits<double>::quiet_NaN();
        double trans_manual_gy = std::numeric_limits<double>::quiet_NaN();

        // Output full G_T(r) and G_T(t) curves using the same fixed/reference G vectors.
        int trans_measure_stride = 1;
        int trans_block_size = 20;
        bool write_trans_block_samples = true;
        int trans_t_max_lag = 0;       // <=0 means all lags
        double trans_t_dt = 1.0;
    };

    PhaseBoundarySettings LoadPhaseBoundarySettings() {
        PhaseBoundarySettings s;
        TryGetParaFromInput_bool("input.in", "calculate_phase_diagnostics", s.enabled);
        TryGetParaFromInput_int("input.in", "phase_defect_measure_stride", s.defect_measure_stride);
        TryGetParaFromInput_real("input.in", "phase_defect_nn_rmin_factor", s.defect_nn_rmin_factor);
        TryGetParaFromInput_real("input.in", "phase_defect_nn_rmax_factor", s.defect_nn_rmax_factor);

        TryGetParaFromInput_real("input.in", "dislocation_pair_rmax_factor", s.dislocation_pair_rmax_factor);
        TryGetParaFromInput_real("input.in", "dislocation_bind_rmax_factor", s.dislocation_bind_rmax_factor);
        TryGetParaFromInput_real("input.in", "dislocation_opposite_cos", s.dislocation_opposite_cos);

        TryGetParaFromInput_real("input.in", "g6_tail_rmin", s.g6_tail_rmin);
        TryGetParaFromInput_real("input.in", "g6_tail_rmax", s.g6_tail_rmax);

        TryGetParaFromInput_bool("input.in", "calculate_trans_diagnostics", s.trans_enabled);
        TryGetParaFromInput_bool("input.in", "trans_use_fixed_axis", s.trans_use_fixed_axis);
        TryGetParaFromInput_real("input.in", "trans_fixed_axis_angle", s.trans_fixed_axis_angle);
        TryGetParaFromInput_real("input.in", "trans_fixed_ask", s.trans_fixed_ask);
        TryGetParaFromInput_real("input.in", "trans_rmax", s.trans_rmax);
        TryGetParaFromInput_real("input.in", "trans_rbin", s.trans_rbin);
        TryGetParaFromInput_real("input.in", "trans_tail_rmin", s.trans_tail_rmin);
        TryGetParaFromInput_real("input.in", "trans_tail_rmax", s.trans_tail_rmax);
        TryGetParaFromInput_real("input.in", "trans_fit_rmin", s.trans_fit_rmin);
        TryGetParaFromInput_real("input.in", "trans_fit_rmax", s.trans_fit_rmax);
        TryGetParaFromInput_real("input.in", "trans_gt_min_for_log", s.trans_gt_min_for_log);

        s.trans_g_mode = ReadStringOrDefault("input.in", "trans_g_mode", s.trans_g_mode);
        TryGetParaFromInput_int("input.in", "trans_reference_iD", s.trans_reference_iD);
        TryGetParaFromInput_int("input.in", "trans_reference_iB", s.trans_reference_iB);
        TryGetParaFromInput_int("input.in", "trans_n_g_vectors", s.trans_n_g_vectors);
        TryGetParaFromInput_bool("input.in", "trans_average_multiple_g", s.trans_average_multiple_g);
        TryGetParaFromInput_real("input.in", "trans_q0_min_factor", s.trans_q0_min_factor);
        TryGetParaFromInput_real("input.in", "trans_q0_max_factor", s.trans_q0_max_factor);
        TryGetParaFromInput_real("input.in", "trans_angle_min_separation", s.trans_angle_min_separation);
        TryGetParaFromInput_real("input.in", "trans_peak_width_window_factor", s.trans_peak_width_window_factor);
        TryGetParaFromInput_real("input.in", "trans_peak_width_background", s.trans_peak_width_background);
        TryGetParaFromInput_real("input.in", "trans_manual_gx", s.trans_manual_gx);
        TryGetParaFromInput_real("input.in", "trans_manual_gy", s.trans_manual_gy);
        TryGetParaFromInput_int("input.in", "trans_measure_stride", s.trans_measure_stride);
        TryGetParaFromInput_int("input.in", "trans_block_size", s.trans_block_size);
        TryGetParaFromInput_bool("input.in", "write_trans_block_samples", s.write_trans_block_samples);
        TryGetParaFromInput_int("input.in", "trans_t_max_lag", s.trans_t_max_lag);
        TryGetParaFromInput_real("input.in", "trans_t_dt", s.trans_t_dt);

        if (s.defect_measure_stride < 1) {
            s.defect_measure_stride = 1;
        }
        if (!(s.defect_nn_rmin_factor > 0.0) || !std::isfinite(s.defect_nn_rmin_factor)) {
            s.defect_nn_rmin_factor = 0.70;
        }
        if (!(s.defect_nn_rmax_factor > s.defect_nn_rmin_factor) || !std::isfinite(s.defect_nn_rmax_factor)) {
            s.defect_nn_rmax_factor = 1.30;
        }

        if (!(s.dislocation_pair_rmax_factor > 0.0) || !std::isfinite(s.dislocation_pair_rmax_factor)) {
            s.dislocation_pair_rmax_factor = 1.35;
        }
        if (!(s.dislocation_bind_rmax_factor > 0.0) || !std::isfinite(s.dislocation_bind_rmax_factor)) {
            s.dislocation_bind_rmax_factor = 3.0;
        }
        if (!std::isfinite(s.dislocation_opposite_cos)) {
            s.dislocation_opposite_cos = -0.5;
        }

        if (!std::isfinite(s.g6_tail_rmin)) {
            s.g6_tail_rmin = 8.0;
        }
        if (!(s.g6_tail_rmax > s.g6_tail_rmin) || !std::isfinite(s.g6_tail_rmax)) {
            s.g6_tail_rmax = 12.0;
        }

        if (!std::isfinite(s.trans_fixed_axis_angle)) {
            s.trans_fixed_axis_angle = 0.0;
        }
        if (!(s.trans_fixed_ask > 0.0) || !std::isfinite(s.trans_fixed_ask)) {
            s.trans_fixed_ask = -1.0;
        }
        if (!(s.trans_rmax > 0.0) || !std::isfinite(s.trans_rmax)) {
            s.trans_rmax = 12.0;
        }
        if (!(s.trans_rbin > 0.0) || !std::isfinite(s.trans_rbin)) {
            s.trans_rbin = 0.1;
        }
        if (!std::isfinite(s.trans_tail_rmin)) {
            s.trans_tail_rmin = 8.0;
        }
        if (!(s.trans_tail_rmax > s.trans_tail_rmin) || !std::isfinite(s.trans_tail_rmax)) {
            s.trans_tail_rmax = 12.0;
        }
        if (!std::isfinite(s.trans_fit_rmin)) {
            s.trans_fit_rmin = 3.0;
        }
        if (!(s.trans_fit_rmax > s.trans_fit_rmin) || !std::isfinite(s.trans_fit_rmax)) {
            s.trans_fit_rmax = 12.0;
        }
        if (!(s.trans_gt_min_for_log > 0.0) || !std::isfinite(s.trans_gt_min_for_log)) {
            s.trans_gt_min_for_log = 1.0e-8;
        }

        if (s.trans_g_mode != "reference_grid" &&
            s.trans_g_mode != "per_frame_grid" &&
            s.trans_g_mode != "manual") {
            s.trans_g_mode = "reference_grid";
        }
        if (s.trans_reference_iD < 0) {
            s.trans_reference_iD = 0;
        }
        if (s.trans_reference_iB < 0) {
            s.trans_reference_iB = 0;
        }
        if (s.trans_n_g_vectors < 1) {
            s.trans_n_g_vectors = 1;
        }
        if (!(s.trans_peak_width_window_factor > 0.0) || !std::isfinite(s.trans_peak_width_window_factor)) {
            s.trans_peak_width_window_factor = 0.35;
        }
        if (!std::isfinite(s.trans_peak_width_background)) {
            s.trans_peak_width_background = 1.0;
        }
        if (s.trans_n_g_vectors > 6) {
            s.trans_n_g_vectors = 6;
        }
        if (!(s.trans_q0_min_factor > 0.0) || !std::isfinite(s.trans_q0_min_factor)) {
            s.trans_q0_min_factor = 0.70;
        }
        if (!(s.trans_q0_max_factor > s.trans_q0_min_factor) || !std::isfinite(s.trans_q0_max_factor)) {
            s.trans_q0_max_factor = 1.30;
        }
        if (!(s.trans_angle_min_separation >= 0.0) || !std::isfinite(s.trans_angle_min_separation)) {
            s.trans_angle_min_separation = 0.25;
        }
        if (s.trans_measure_stride < 1) {
            s.trans_measure_stride = 1;
        }
        if (s.trans_block_size < 1) {
            s.trans_block_size = 1;
        }
        if (s.trans_t_max_lag < 0) {
            s.trans_t_max_lag = 0;
        }
        if (!(s.trans_t_dt > 0.0) || !std::isfinite(s.trans_t_dt)) {
            s.trans_t_dt = 1.0;
        }
        return s;
    }

    struct PhaseFrameDiagnostics {
        double rho_def = std::numeric_limits<double>::quiet_NaN();
        double n5_frac = std::numeric_limits<double>::quiet_NaN();
        double n6_frac = std::numeric_limits<double>::quiet_NaN();
        double n7_frac = std::numeric_limits<double>::quiet_NaN();
        double nnon6_frac = std::numeric_limits<double>::quiet_NaN();

        // Dislocation diagnostics.
        // A 5-7 nearest-neighbour pair is counted as one dislocation core.
        // A free dislocation is a core without a nearby opposite-polarity partner.
        double dislocation_frac = std::numeric_limits<double>::quiet_NaN();       // N_D / Nsk
        double free_dislocation_frac = std::numeric_limits<double>::quiet_NaN();  // N_D^free / Nsk
        double bound_dislocation_frac = std::numeric_limits<double>::quiet_NaN(); // N_D^bound / Nsk

        double rho_dislocation = std::numeric_limits<double>::quiet_NaN();        // N_D / Area
        double rho_free_dislocation = std::numeric_limits<double>::quiet_NaN();   // N_D^free / Area

        // xi_D/a_sk = sqrt(Area/N_D^free)/a_sk.
        // If no free dislocation is found, this is a finite-box lower bound sqrt(Area)/a_sk.
        double xiD_free_over_ask = std::numeric_limits<double>::quiet_NaN();
    };

    double Dot2(const Point2& a, const Point2& b) {
        return a.x * b.x + a.y * b.y;
    }

    double Norm2(const Point2& a) {
        return std::sqrt(Dot2(a, a));
    }

    Point2 Normalize2(const Point2& a) {
        const double n = Norm2(a);
        if (!(n > 1.0e-14) || !std::isfinite(n)) {
            return { 1.0, 0.0 };
        }
        return { a.x / n, a.y / n };
    }

    Point2 CoreMidpointPBC(
        const Point2& r5,
        const Point2& r7,
        int lx,
        int ly
    ) {
        const Point2 dr57 = MinImageDeltaReal(r5, r7, lx, ly);
        Point2 mid{ r5.x + 0.5 * dr57.x, r5.y + 0.5 * dr57.y };
        return WrapRealToBox(mid, lx, ly);
    }

    struct DislocationCore {
        int i5 = -1;
        int i7 = -1;
        Point2 r;        // core position, midpoint of the 5-7 pair
        Point2 polarity; // unit vector from 5-fold site to 7-fold site
        bool bound = false;
    };

    struct TranslationalFrameDiagnostics {
        double gt_tail = std::numeric_limits<double>::quiet_NaN();
        double mt = std::numeric_limits<double>::quiet_NaN();
        double eta_T = std::numeric_limits<double>::quiet_NaN();
        double roughness_slope_AB = std::numeric_limits<double>::quiet_NaN();

        // Projected displacement/phase roughness and reciprocal peak widths.
        double phase_roughness_W2 = std::numeric_limits<double>::quiet_NaN();
        double peak_dq_parallel_over_q0 = std::numeric_limits<double>::quiet_NaN();
        double peak_dq_perp_over_q0 = std::numeric_limits<double>::quiet_NaN();
        double xi_parallel_over_ask = std::numeric_limits<double>::quiet_NaN();
        double xi_perp_over_ask = std::numeric_limits<double>::quiet_NaN();
    };

    struct BraggVectorSelection {
        std::vector<Point2> gvecs;
        double q0 = std::numeric_limits<double>::quiet_NaN();
        double a_ref = std::numeric_limits<double>::quiet_NaN();
        double best_S = std::numeric_limits<double>::quiet_NaN();
        int best_n1 = 0;
        int best_n2 = 0;
    };

    struct TransReferenceCache {
        bool valid = false;
        int reference_iD = -1;
        int reference_iB = -1;
        int source_bin = -1;
        BraggVectorSelection selection;
    };

    static TransReferenceCache g_trans_reference_cache;

    Point2 ReciprocalGridQFromIndices(int n1, int n2, int lx, int ly) {
        // Triangular PBC box basis:
        //   L1=(Lx,0), L2=(Ly/2,sqrt(3)Ly/2).
        // Reciprocal vector q satisfies q.L1=2*pi*n1 and q.L2=2*pi*n2.
        const double qx = 2.0 * PI * static_cast<double>(n1) / static_cast<double>(lx);
        const double qy = (2.0 / std::sqrt(3.0)) *
            (2.0 * PI * static_cast<double>(n2) / static_cast<double>(ly) - 0.5 * qx);
        return { qx, qy };
    }

    double StructureFactorAtQFromCenters(
        const std::vector<Point2>& centers,
        const Point2& q
    ) {
        const int n = static_cast<int>(centers.size());
        if (n <= 0) {
            return std::numeric_limits<double>::quiet_NaN();
        }

        double re = 0.0;
        double im = 0.0;
        for (const Point2& r : centers) {
            const double ph = q.x * r.x + q.y * r.y;
            re += std::cos(ph);
            im += std::sin(ph);
        }
        return (re * re + im * im) / static_cast<double>(n);
    }

    double AngleDistanceModulo2Pi(double a, double b) {
        double d = std::fmod(std::fabs(a - b), 2.0 * PI);
        if (d > PI) {
            d = 2.0 * PI - d;
        }
        return d;
    }

    BraggVectorSelection FindBraggVectorsOnReciprocalGrid(
        const std::vector<Point2>& centers,
        int lx,
        int ly,
        double a_ref,
        const PhaseBoundarySettings& phase
    ) {
        BraggVectorSelection out;
        out.a_ref = a_ref;

        if (centers.empty() || !(a_ref > 0.0) || !std::isfinite(a_ref)) {
            return out;
        }

        out.q0 = 4.0 * PI / (std::sqrt(3.0) * a_ref);
        const double qmin = phase.trans_q0_min_factor * out.q0;
        const double qmax = phase.trans_q0_max_factor * out.q0;

        struct Candidate {
            Point2 q;
            double S = -1.0;
            double angle = 0.0;
            int n1 = 0;
            int n2 = 0;
        };

        std::vector<Candidate> candidates;
        candidates.reserve(static_cast<std::size_t>(lx) * static_cast<std::size_t>(ly));

        for (int n1 = -lx / 2; n1 <= lx / 2; ++n1) {
            for (int n2 = -ly / 2; n2 <= ly / 2; ++n2) {
                if (n1 == 0 && n2 == 0) {
                    continue;
                }
                const Point2 q = ReciprocalGridQFromIndices(n1, n2, lx, ly);
                const double qabs = std::sqrt(q.x * q.x + q.y * q.y);
                if (!(qabs >= qmin && qabs <= qmax) || !std::isfinite(qabs)) {
                    continue;
                }

                Candidate c;
                c.q = q;
                c.S = StructureFactorAtQFromCenters(centers, q);
                c.angle = std::atan2(q.y, q.x);
                c.n1 = n1;
                c.n2 = n2;
                if (std::isfinite(c.S)) {
                    candidates.push_back(c);
                }
            }
        }

        if (candidates.empty()) {
            return out;
        }

        std::sort(
            candidates.begin(),
            candidates.end(),
            [](const Candidate& a, const Candidate& b) { return a.S > b.S; }
        );

        // Python-compatible default: select only the strongest reciprocal-grid Bragg vector.
        // This reproduces the verified Python code:
        //     best = find_bragg_reciprocal_grid(...);
        //     gvecs = [[best["qx"], best["qy"]]]  when AVERAGE_SIXFOLD=False.
        // Set trans_average_multiple_g=true only if you intentionally want to average
        // several symmetry-related grid peaks.
        const int n_wanted = phase.trans_average_multiple_g ?
            std::max(1, std::min(6, phase.trans_n_g_vectors)) :
            1;
        const double min_sep = std::max(0.0, phase.trans_angle_min_separation);

        out.best_S = candidates.front().S;
        out.best_n1 = candidates.front().n1;
        out.best_n2 = candidates.front().n2;

        std::vector<double> selected_angles;
        selected_angles.reserve(6);

        for (const Candidate& c : candidates) {
            bool too_close = false;
            for (const double a : selected_angles) {
                if (AngleDistanceModulo2Pi(c.angle, a) < min_sep) {
                    too_close = true;
                    break;
                }
            }
            if (too_close) {
                continue;
            }

            out.gvecs.push_back(c.q);
            selected_angles.push_back(c.angle);

            if (static_cast<int>(out.gvecs.size()) >= n_wanted) {
                break;
            }
        }

        // Fallback: at least use the strongest peak.
        if (out.gvecs.empty()) {
            out.gvecs.push_back(candidates.front().q);
        }

        return out;
    }

    BraggVectorSelection ManualTransBraggVectorSelection(
        const OrientFrame& frame,
        const PhaseBoundarySettings& phase
    ) {
        BraggVectorSelection out;
        out.a_ref =
            (phase.trans_fixed_ask > 0.0 && std::isfinite(phase.trans_fixed_ask)) ?
            phase.trans_fixed_ask :
            frame.a_sk;
        out.q0 = 4.0 * PI / (std::sqrt(3.0) * out.a_ref);
        Point2 q{ phase.trans_manual_gx, phase.trans_manual_gy };
        if (std::isfinite(q.x) && std::isfinite(q.y) && (q.x * q.x + q.y * q.y > 0.0)) {
            out.gvecs.push_back(q);
            out.best_S = StructureFactorAtQFromCenters(frame.centers, q);
        }
        return out;
    }

    BraggVectorSelection GetTransBraggVectorSelection(
        const OrientFrame& frame,
        int lx,
        int ly,
        int iD,
        int iB,
        int ibin,
        const PhaseBoundarySettings& phase
    ) {
        const double a_for_search =
            (phase.trans_fixed_ask > 0.0 && std::isfinite(phase.trans_fixed_ask)) ?
            phase.trans_fixed_ask :
            frame.a_sk;

        if (phase.trans_g_mode == "manual") {
            return ManualTransBraggVectorSelection(frame, phase);
        }

        if (phase.trans_g_mode == "per_frame_grid") {
            return FindBraggVectorsOnReciprocalGrid(frame.centers, lx, ly, a_for_search, phase);
        }

        // Default and recommended for distinguishing SkX from BrG-like:
        // lock all disorder strengths to the first valid reference crystal frame.
        if (phase.trans_g_mode == "reference_grid") {
            if (g_trans_reference_cache.valid &&
                g_trans_reference_cache.reference_iD == phase.trans_reference_iD &&
                g_trans_reference_cache.reference_iB == phase.trans_reference_iB) {
                return g_trans_reference_cache.selection;
            }

            if (iD == phase.trans_reference_iD && iB == phase.trans_reference_iB) {
                const BraggVectorSelection sel =
                    FindBraggVectorsOnReciprocalGrid(frame.centers, lx, ly, a_for_search, phase);

                if (!sel.gvecs.empty()) {
#ifdef _OPENMP
#pragma omp critical(trans_reference_cache_update)
#endif
                    {
                        if (!g_trans_reference_cache.valid ||
                            g_trans_reference_cache.reference_iD != phase.trans_reference_iD ||
                            g_trans_reference_cache.reference_iB != phase.trans_reference_iB) {
                            g_trans_reference_cache.valid = true;
                            g_trans_reference_cache.reference_iD = phase.trans_reference_iD;
                            g_trans_reference_cache.reference_iB = phase.trans_reference_iB;
                            g_trans_reference_cache.source_bin = ibin;
                            g_trans_reference_cache.selection = sel;
                        }
                    }
                }

                if (g_trans_reference_cache.valid) {
                    return g_trans_reference_cache.selection;
                }
                return sel;
            }

            // If the user did not scan the reference point first, fall back to per-frame grid
            // rather than crashing.  The debug/header files make this visible.
            return FindBraggVectorsOnReciprocalGrid(frame.centers, lx, ly, a_for_search, phase);
        }

        // Unknown mode: safe fallback to the correct reciprocal-grid method.
        return FindBraggVectorsOnReciprocalGrid(frame.centers, lx, ly, a_for_search, phase);
    }

    std::vector<std::complex<double>> ComputePsiTForGvecs(
        const std::vector<Point2>& centers,
        const std::vector<Point2>& gvecs
    ) {
        const int n = static_cast<int>(centers.size());
        std::vector<std::complex<double>> psi;
        psi.reserve(gvecs.size());

        if (n <= 0) {
            return psi;
        }

        for (const Point2& g : gvecs) {
            double re = 0.0;
            double im = 0.0;
            for (const Point2& r : centers) {
                const double ph = g.x * r.x + g.y * r.y;
                re += std::cos(ph);
                im += std::sin(ph);
            }
            psi.push_back({ re / static_cast<double>(n), im / static_cast<double>(n) });
        }

        return psi;
    }

    double MTFromPsiT(const std::vector<std::complex<double>>& psiT) {
        if (psiT.empty()) {
            return std::numeric_limits<double>::quiet_NaN();
        }
        double s = 0.0;
        int n = 0;
        for (const auto& z : psiT) {
            if (std::isfinite(z.real()) && std::isfinite(z.imag())) {
                s += std::norm(z);
                ++n;
            }
        }
        if (n <= 0) {
            return std::numeric_limits<double>::quiet_NaN();
        }
        return std::sqrt(std::max(0.0, s / static_cast<double>(n)));
    }

    int TransNumberOfBins(const PhaseBoundarySettings& phase) {
        const double nb_real = std::ceil(phase.trans_rmax / phase.trans_rbin);
        if (!std::isfinite(nb_real) || nb_real < 1.0 || nb_real > 1000000.0) {
            throw std::runtime_error("Invalid translational G_T(r) bin settings.");
        }
        return static_cast<int>(nb_real);
    }

    std::vector<double> ComputeGtrFrameCurveCorrect(
        const OrientFrame& frame,
        int lx,
        int ly,
        const std::vector<Point2>& gvecs,
        double a_ref,
        const PhaseBoundarySettings& phase,
        std::vector<double>* pair_counts = nullptr
    ) {
        const int nbins = TransNumberOfBins(phase);
        std::vector<double> sum(nbins, 0.0);
        std::vector<double> count(nbins, 0.0);
        std::vector<double> curve(nbins, std::numeric_limits<double>::quiet_NaN());

        const int n = frame.nsk;
        if (n < 2 || gvecs.empty() || !(a_ref > 0.0) || !std::isfinite(a_ref)) {
            if (pair_counts != nullptr) {
                pair_counts->assign(nbins, 0.0);
            }
            return curve;
        }

        for (int i = 0; i < n - 1; ++i) {
            for (int j = i + 1; j < n; ++j) {
                const Point2 dr = MinImageDeltaReal(frame.centers[i], frame.centers[j], lx, ly);
                const double dist = std::sqrt(dr.x * dr.x + dr.y * dr.y);
                const double rho = dist / a_ref;
                if (!std::isfinite(rho) || rho < 0.0 || rho >= phase.trans_rmax) {
                    continue;
                }
                const int b = static_cast<int>(std::floor(rho / phase.trans_rbin));
                if (b < 0 || b >= nbins) {
                    continue;
                }

                double v = 0.0;
                for (const Point2& g : gvecs) {
                    v += std::cos(g.x * dr.x + g.y * dr.y);
                }
                v /= static_cast<double>(gvecs.size());

                sum[b] += v;
                count[b] += 1.0;
            }
        }

        for (int b = 0; b < nbins; ++b) {
            if (count[b] > 0.0) {
                curve[b] = sum[b] / count[b];
            }
        }

        if (pair_counts != nullptr) {
            *pair_counts = std::move(count);
        }
        return curve;
    }

    double TailAverageFromCurve(
        const std::vector<double>& curve,
        double rbin,
        double rmin,
        double rmax
    ) {
        double sum = 0.0;
        int n = 0;
        for (int b = 0; b < static_cast<int>(curve.size()); ++b) {
            const double rmid = (static_cast<double>(b) + 0.5) * rbin;
            if (rmid >= rmin && rmid <= rmax) {
                const double v = curve[b];
                if (std::isfinite(v)) {
                    sum += v;
                    ++n;
                }
            }
        }
        return n > 0 ? sum / static_cast<double>(n) : std::numeric_limits<double>::quiet_NaN();
    }

    double MeanGAbsSquared(const std::vector<Point2>& gvecs) {
        if (gvecs.empty()) {
            return std::numeric_limits<double>::quiet_NaN();
        }
        double sum = 0.0;
        for (const Point2& g : gvecs) {
            sum += g.x * g.x + g.y * g.y;
        }
        return sum / static_cast<double>(gvecs.size());
    }


    double WrapAngleMinusPiToPi(double x) {
        while (x <= -PI) {
            x += 2.0 * PI;
        }
        while (x > PI) {
            x -= 2.0 * PI;
        }
        return x;
    }

    // Projected displacement roughness from translational phase fluctuations.
    // For small displacements, phi_j = G.r_j - arg(Psi_G) ~= G.u_j.
    // We report W_G^2 = <phi_j^2>/(|G|^2 a_ref^2), averaged over selected G vectors.
    // This avoids assigning every skyrmion to an ideal lattice site and is therefore robust
    // for finite-temperature/disordered configurations.
    double TranslationalPhaseRoughnessW2(
        const std::vector<Point2>& centers,
        const std::vector<Point2>& gvecs,
        double a_ref
    ) {
        const int n = static_cast<int>(centers.size());
        if (n <= 0 || gvecs.empty() || !(a_ref > 0.0) || !std::isfinite(a_ref)) {
            return std::numeric_limits<double>::quiet_NaN();
        }

        double sum_w2 = 0.0;
        int ngood = 0;

        for (const Point2& g : gvecs) {
            const double g2 = g.x * g.x + g.y * g.y;
            if (!(g2 > 0.0) || !std::isfinite(g2)) {
                continue;
            }

            double re = 0.0;
            double im = 0.0;
            for (const Point2& r : centers) {
                const double ph = g.x * r.x + g.y * r.y;
                re += std::cos(ph);
                im += std::sin(ph);
            }

            const double phase0 = std::atan2(im, re);
            double ss = 0.0;
            int cnt = 0;
            for (const Point2& r : centers) {
                const double ph = WrapAngleMinusPiToPi(g.x * r.x + g.y * r.y - phase0);
                ss += ph * ph;
                ++cnt;
            }

            if (cnt > 0) {
                const double w2 = (ss / static_cast<double>(cnt)) / (g2 * a_ref * a_ref);
                if (std::isfinite(w2)) {
                    sum_w2 += w2;
                    ++ngood;
                }
            }
        }

        return ngood > 0 ? sum_w2 / static_cast<double>(ngood) :
            std::numeric_limits<double>::quiet_NaN();
    }

    struct BraggPeakWidthDiagnostics {
        double dq_parallel_over_q0 = std::numeric_limits<double>::quiet_NaN();
        double dq_perp_over_q0 = std::numeric_limits<double>::quiet_NaN();
        double xi_parallel_over_ask = std::numeric_limits<double>::quiet_NaN();
        double xi_perp_over_ask = std::numeric_limits<double>::quiet_NaN();
    };

    // Estimate radial/tangential Bragg-peak widths from the skyrmion-center S(q)
    // on the PBC-compatible reciprocal grid.  The weight is max[S(q)-S_bg,0],
    // with S_bg~1 the random-point background of |sum exp(iqr)|^2/N.
    BraggPeakWidthDiagnostics ComputeCenterBraggPeakWidths(
        const std::vector<Point2>& centers,
        int lx,
        int ly,
        const std::vector<Point2>& gvecs,
        double a_ref,
        const PhaseBoundarySettings& phase
    ) {
        BraggPeakWidthDiagnostics out;
        if (centers.empty() || gvecs.empty() || !(a_ref > 0.0) || !std::isfinite(a_ref)) {
            return out;
        }

        const double q0 = 4.0 * PI / (std::sqrt(3.0) * a_ref);
        if (!(q0 > 0.0) || !std::isfinite(q0)) {
            return out;
        }

        const double window = phase.trans_peak_width_window_factor * q0;
        const double bg = phase.trans_peak_width_background;

        double wsum = 0.0;
        double par2_sum = 0.0;
        double perp2_sum = 0.0;

        for (const Point2& g : gvecs) {
            const double gabs = std::sqrt(g.x * g.x + g.y * g.y);
            if (!(gabs > 0.0) || !std::isfinite(gabs)) {
                continue;
            }
            const Point2 epar{ g.x / gabs, g.y / gabs };
            const Point2 eperp{ -epar.y, epar.x };

            for (int n1 = -lx / 2; n1 <= lx / 2; ++n1) {
                for (int n2 = -ly / 2; n2 <= ly / 2; ++n2) {
                    const Point2 q = ReciprocalGridQFromIndices(n1, n2, lx, ly);
                    const double dx = q.x - g.x;
                    const double dy = q.y - g.y;
                    const double d2 = dx * dx + dy * dy;
                    if (!(d2 <= window * window) || !std::isfinite(d2)) {
                        continue;
                    }

                    const double S = StructureFactorAtQFromCenters(centers, q);
                    if (!std::isfinite(S)) {
                        continue;
                    }
                    const double w = std::max(0.0, S - bg);
                    if (!(w > 0.0)) {
                        continue;
                    }

                    const double dpar = dx * epar.x + dy * epar.y;
                    const double dperp = dx * eperp.x + dy * eperp.y;
                    wsum += w;
                    par2_sum += w * dpar * dpar;
                    perp2_sum += w * dperp * dperp;
                }
            }
        }

        if (!(wsum > 0.0) || !std::isfinite(wsum)) {
            return out;
        }

        const double dqpar = std::sqrt(std::max(0.0, par2_sum / wsum));
        const double dqperp = std::sqrt(std::max(0.0, perp2_sum / wsum));

        out.dq_parallel_over_q0 = dqpar / q0;
        out.dq_perp_over_q0 = dqperp / q0;
        // Correlation lengths are meaningful only when the width is above the
        // reciprocal-grid resolution.  Otherwise report NaN rather than a huge
        // non-physical number such as 1e14.
        const double dq_min = 1.0e-8;
        if (dqpar > dq_min && std::isfinite(dqpar)) {
            out.xi_parallel_over_ask = 1.0 / (dqpar * a_ref);
        }
        if (dqperp > dq_min && std::isfinite(dqperp)) {
            out.xi_perp_over_ask = 1.0 / (dqperp * a_ref);
        }
        return out;
    }

    double FitEtaFromGtrEnvelope(
        const std::vector<double>& curve,
        const PhaseBoundarySettings& phase
    ) {
        std::vector<double> x;
        std::vector<double> y;

        const int nbins = static_cast<int>(curve.size());
        std::vector<int> peaks;
        for (int b = 1; b + 1 < nbins; ++b) {
            const double v = curve[b];
            if (std::isfinite(v) && v > phase.trans_gt_min_for_log &&
                std::isfinite(curve[b - 1]) && std::isfinite(curve[b + 1]) &&
                v >= curve[b - 1] && v >= curve[b + 1]) {
                peaks.push_back(b);
            }
        }

        const bool use_peaks = static_cast<int>(peaks.size()) >= 3;

        for (int idx = 0; idx < (use_peaks ? static_cast<int>(peaks.size()) : nbins); ++idx) {
            const int b = use_peaks ? peaks[idx] : idx;
            const double rmid = (static_cast<double>(b) + 0.5) * phase.trans_rbin;
            if (rmid < phase.trans_fit_rmin || rmid > phase.trans_fit_rmax) {
                continue;
            }
            const double gt = curve[b];
            if (!std::isfinite(gt) || gt <= phase.trans_gt_min_for_log) {
                continue;
            }
            x.push_back(std::log(rmid));
            y.push_back(std::log(std::min(gt, 1.0)));
        }

        const int n = static_cast<int>(x.size());
        if (n < 2) {
            return std::numeric_limits<double>::quiet_NaN();
        }

        double sx = 0.0;
        double sy = 0.0;
        double sxx = 0.0;
        double sxy = 0.0;
        for (int i = 0; i < n; ++i) {
            sx += x[i];
            sy += y[i];
            sxx += x[i] * x[i];
            sxy += x[i] * y[i];
        }
        const double denom = static_cast<double>(n) * sxx - sx * sx;
        if (std::fabs(denom) < 1.0e-30) {
            return std::numeric_limits<double>::quiet_NaN();
        }

        const double slope = (static_cast<double>(n) * sxy - sx * sy) / denom;
        return -slope;
    }

    TranslationalFrameDiagnostics ComputeTranslationalFrameDiagnostics(
        const OrientFrame& frame,
        int lx,
        int ly,
        const PhaseBoundarySettings& phase,
        int iD = -1,
        int iB = -1,
        int ibin = -1
    ) {
        TranslationalFrameDiagnostics out;

        const int n = frame.nsk;
        if (!phase.trans_enabled || n < 2 || !(frame.a_sk > 0.0) || !std::isfinite(frame.a_sk)) {
            return out;
        }

        const BraggVectorSelection sel =
            GetTransBraggVectorSelection(frame, lx, ly, iD, iB, ibin, phase);
        if (sel.gvecs.empty() || !(sel.a_ref > 0.0) || !std::isfinite(sel.a_ref)) {
            return out;
        }

        const std::vector<std::complex<double>> psiT =
            ComputePsiTForGvecs(frame.centers, sel.gvecs);
        out.mt = MTFromPsiT(psiT);

        const std::vector<double> curve =
            ComputeGtrFrameCurveCorrect(frame, lx, ly, sel.gvecs, sel.a_ref, phase, nullptr);
        out.gt_tail = TailAverageFromCurve(curve, phase.trans_rbin, phase.trans_tail_rmin, phase.trans_tail_rmax);
        out.eta_T = FitEtaFromGtrEnvelope(curve, phase);

        const double g2 = MeanGAbsSquared(sel.gvecs);
        if (std::isfinite(out.eta_T) && std::isfinite(g2) && g2 > 0.0 && sel.a_ref > 0.0) {
            out.roughness_slope_AB = 2.0 * out.eta_T / (g2 * sel.a_ref * sel.a_ref);
        }

        out.phase_roughness_W2 =
            TranslationalPhaseRoughnessW2(frame.centers, sel.gvecs, sel.a_ref);

        const BraggPeakWidthDiagnostics widths =
            ComputeCenterBraggPeakWidths(frame.centers, lx, ly, sel.gvecs, sel.a_ref, phase);
        out.peak_dq_parallel_over_q0 = widths.dq_parallel_over_q0;
        out.peak_dq_perp_over_q0 = widths.dq_perp_over_q0;
        out.xi_parallel_over_ask = widths.xi_parallel_over_ask;
        out.xi_perp_over_ask = widths.xi_perp_over_ask;

        return out;
    }

    
    PhaseFrameDiagnostics ComputeFirstShellCoordinationDiagnostics(
        const OrientFrame& frame,
        int lx,
        int ly,
        const PhaseBoundarySettings& phase
    ) {
        // Historical function name kept for compatibility.  The implementation is
        // now strict PBC Delaunay, not first-shell distance-window coordination.
        PhaseFrameDiagnostics out;
        const int n = frame.nsk;
        if (n <= 0 || !(frame.a_sk > 0.0) || !std::isfinite(frame.a_sk)) {
            return out;
        }

        const auto neigh = BuildPBCDelaunayNeighborList(frame.centers, lx, ly);

        std::vector<int> coord(n, 0);
        int n5 = 0;
        int n6 = 0;
        int n7 = 0;
        int nnon6 = 0;

        for (int i = 0; i < n; ++i) {
            coord[i] = static_cast<int>(neigh[i].size());
            if (coord[i] == 5) {
                ++n5;
            }
            if (coord[i] == 6) {
                ++n6;
            }
            if (coord[i] == 7) {
                ++n7;
            }
            if (coord[i] != 6) {
                ++nnon6;
            }
        }

        // Greedy pairing of adjacent 5-7 Delaunay neighbours into dislocation cores.
        // Each 5-fold and 7-fold site can be used only once.  One 5-7 pair is one
        // dislocation core; therefore N_D/N is not the same normalization as N_non6/N.
        std::vector<unsigned char> used5(n, 0);
        std::vector<unsigned char> used7(n, 0);
        std::vector<DislocationCore> cores;

        const double pair_rmax = phase.dislocation_pair_rmax_factor * frame.a_sk;
        const double pair_rmax2 = pair_rmax * pair_rmax;

        for (int i = 0; i < n; ++i) {
            if (coord[i] != 5 || used5[i]) {
                continue;
            }

            int best7 = -1;
            Point2 best_dr57{0.0, 0.0};
            double best_r2 = std::numeric_limits<double>::infinity();

            for (const auto& nb : neigh[i]) {
                const int j = nb.j;
                if (j < 0 || j >= n || coord[j] != 7 || used7[j]) {
                    continue;
                }
                const double r2 = nb.dr.x * nb.dr.x + nb.dr.y * nb.dr.y;
                // Delaunay adjacency is the strict topological condition.  The extra
                // rmax is only a safety cutoff against pathological triangulation edges.
                if (r2 <= pair_rmax2 && r2 < best_r2) {
                    best_r2 = r2;
                    best7 = j;
                    best_dr57 = nb.dr;
                }
            }

            if (best7 >= 0) {
                used5[i] = 1;
                used7[best7] = 1;

                DislocationCore core;
                core.i5 = i;
                core.i7 = best7;
                core.r = CoreMidpointPBC(frame.centers[i], frame.centers[best7], lx, ly);
                core.polarity = Normalize2(best_dr57);
                core.bound = false;
                cores.push_back(core);
            }
        }

        // Bound/free classification: a core is bound if an opposite-polarity core is nearby.
        const double bind_rmax = phase.dislocation_bind_rmax_factor * frame.a_sk;
        const double bind_rmax2 = bind_rmax * bind_rmax;

        for (int a = 0; a < static_cast<int>(cores.size()); ++a) {
            for (int b = a + 1; b < static_cast<int>(cores.size()); ++b) {
                const Point2 dr = MinImageDeltaReal(cores[a].r, cores[b].r, lx, ly);
                const double r2 = dr.x * dr.x + dr.y * dr.y;
                if (r2 > bind_rmax2) {
                    continue;
                }

                const double cosang = Dot2(cores[a].polarity, cores[b].polarity);
                if (cosang <= phase.dislocation_opposite_cos) {
                    cores[a].bound = true;
                    cores[b].bound = true;
                }
            }
        }

        int n_bound = 0;
        for (const auto& core : cores) {
            if (core.bound) {
                ++n_bound;
            }
        }

        const int n_dislocation = static_cast<int>(cores.size());
        const int n_free = std::max(0, n_dislocation - n_bound);

        const double inv_n = 1.0 / static_cast<double>(n);
        const double area = TriangularBoxArea(lx, ly);

        out.n5_frac = static_cast<double>(n5) * inv_n;
        out.n6_frac = static_cast<double>(n6) * inv_n;
        out.n7_frac = static_cast<double>(n7) * inv_n;
        out.nnon6_frac = static_cast<double>(nnon6) * inv_n;
        out.rho_def = out.nnon6_frac;

        out.dislocation_frac = static_cast<double>(n_dislocation) * inv_n;
        out.free_dislocation_frac = static_cast<double>(n_free) * inv_n;
        out.bound_dislocation_frac = static_cast<double>(n_bound) * inv_n;

        out.rho_dislocation = static_cast<double>(n_dislocation) / area;
        out.rho_free_dislocation = static_cast<double>(n_free) / area;

        if (n_free > 0) {
            out.xiD_free_over_ask =
                std::sqrt(area / static_cast<double>(n_free)) / frame.a_sk;
        }
        else {
            // Lower bound: no free dislocation was observed in this finite box.
            out.xiD_free_over_ask = std::sqrt(area) / frame.a_sk;
        }

        return out;
    }


    double MeanFinite(const std::vector<double>& values) {
        double sum = 0.0;
        int n = 0;
        for (const double v : values) {
            if (std::isfinite(v)) {
                sum += v;
                ++n;
            }
        }
        return n > 0 ? sum / static_cast<double>(n) : std::numeric_limits<double>::quiet_NaN();
    }

    double SemFinite(const std::vector<double>& values, double mean) {
        if (!std::isfinite(mean)) {
            return std::numeric_limits<double>::quiet_NaN();
        }
        int n = 0;
        double ss = 0.0;
        for (const double v : values) {
            if (std::isfinite(v)) {
                const double d = v - mean;
                ss += d * d;
                ++n;
            }
        }
        if (n <= 1) {
            return 0.0;
        }
        return std::sqrt(ss / (static_cast<double>(n) * static_cast<double>(n - 1)));
    }

    double VarianceFinitePopulation(const std::vector<double>& values, double mean) {
        if (!std::isfinite(mean)) {
            return std::numeric_limits<double>::quiet_NaN();
        }
        int n = 0;
        double ss = 0.0;
        for (const double v : values) {
            if (std::isfinite(v)) {
                const double d = v - mean;
                ss += d * d;
                ++n;
            }
        }
        return n > 0 ? ss / static_cast<double>(n) : std::numeric_limits<double>::quiet_NaN();
    }

    double BinderCumulantScalar(const std::vector<double>& values) {
        double m2 = 0.0;
        double m4 = 0.0;
        int n = 0;
        for (const double v : values) {
            if (std::isfinite(v)) {
                const double v2 = v * v;
                m2 += v2;
                m4 += v2 * v2;
                ++n;
            }
        }
        if (n <= 0) {
            return std::numeric_limits<double>::quiet_NaN();
        }
        m2 /= static_cast<double>(n);
        m4 /= static_cast<double>(n);
        if (!(m2 > 0.0) || !std::isfinite(m2)) {
            return std::numeric_limits<double>::quiet_NaN();
        }
        return 1.0 - m4 / (3.0 * m2 * m2);
    }

    double BinderCumulantScalarNormalized(const std::vector<double>& values) {
        // For a scalar non-negative order parameter with the conventional
        // U=1-<m^4>/(3<m^2>^2), ordered constant gives U=2/3.
        // This normalized form maps that ordered limit to 1.
        const double u = BinderCumulantScalar(values);
        return std::isfinite(u) ? 1.5 * u : u;
    }

    double BinderCumulantO2NormalizedFromMagnitude(const std::vector<double>& abs_values) {
        // Correct Binder ratio for a two-component complex order parameter Phi,
        // using stored magnitudes |Phi|:
        //   U_O2_norm = 2 - <|Phi|^4>/<|Phi|^2>^2.
        // Ordered constant -> 1; Gaussian disordered complex vector -> 0.
        double m2 = 0.0;
        double m4 = 0.0;
        int n = 0;
        for (const double v : abs_values) {
            if (std::isfinite(v)) {
                const double v2 = v * v;
                m2 += v2;
                m4 += v2 * v2;
                ++n;
            }
        }
        if (n <= 0) {
            return std::numeric_limits<double>::quiet_NaN();
        }
        m2 /= static_cast<double>(n);
        m4 /= static_cast<double>(n);
        if (!(m2 > 0.0) || !std::isfinite(m2)) {
            return std::numeric_limits<double>::quiet_NaN();
        }
        return 2.0 - m4 / (m2 * m2);
    }


    double SusceptibilityFromDisorderValues(
        const std::vector<double>& values,
        double n_factor
    ) {
        const double mean = MeanFinite(values);
        const double var = VarianceFinitePopulation(values, mean);
        if (!std::isfinite(var) || !std::isfinite(n_factor)) {
            return std::numeric_limits<double>::quiet_NaN();
        }
        return n_factor * var;
    }

    double SusceptibilityJackknifeSem(
        const std::vector<double>& values,
        double n_factor
    ) {
        std::vector<double> x;
        for (const double v : values) {
            if (std::isfinite(v)) {
                x.push_back(v);
            }
        }
        const int n = static_cast<int>(x.size());
        if (n <= 1 || !std::isfinite(n_factor)) {
            return 0.0;
        }

        std::vector<double> jk(n, 0.0);
        for (int leave = 0; leave < n; ++leave) {
            std::vector<double> reduced;
            reduced.reserve(n - 1);
            for (int i = 0; i < n; ++i) {
                if (i != leave) {
                    reduced.push_back(x[i]);
                }
            }
            jk[leave] = SusceptibilityFromDisorderValues(reduced, n_factor);
        }

        const double jk_mean = MeanFinite(jk);
        double ss = 0.0;
        for (const double v : jk) {
            const double d = v - jk_mean;
            ss += d * d;
        }
        return std::sqrt((static_cast<double>(n - 1) / static_cast<double>(n)) * ss);
    }

    double G6TailFromDisorderCurve(
        const Gx0DisorderCurve& curve,
        const G6rSettings& g6r,
        const PhaseBoundarySettings& phase
    ) {
        if (curve.mean.empty()) {
            return std::numeric_limits<double>::quiet_NaN();
        }
        double sum = 0.0;
        int n = 0;
        const int nbins = static_cast<int>(curve.mean.size());
        for (int b = 0; b < nbins; ++b) {
            const double rmid = (static_cast<double>(b) + 0.5) * g6r.rbin;
            if (rmid >= phase.g6_tail_rmin && rmid <= phase.g6_tail_rmax) {
                const double v = curve.mean[b];
                if (std::isfinite(v)) {
                    sum += v;
                    ++n;
                }
            }
        }
        return n > 0 ? sum / static_cast<double>(n) : std::numeric_limits<double>::quiet_NaN();
    }

    void WritePhaseBoundaryDiagnosticsFile(
        int iD,
        int iB,
        const PhaseBoundarySettings& phase,
        double mean_nsk_center,
        const std::vector<double>& phi6_global_values,
        const std::vector<double>& phi6_local_values,
        const std::vector<double>& rho_def_values,
        const std::vector<double>& n5_values,
        const std::vector<double>& n6_values,
        const std::vector<double>& n7_values,
        const std::vector<double>& nnon6_values,
        const std::vector<double>& dislocation_frac_values,
        const std::vector<double>& free_dislocation_frac_values,
        const std::vector<double>& bound_dislocation_frac_values,
        const std::vector<double>& rho_dislocation_values,
        const std::vector<double>& rho_free_dislocation_values,
        const std::vector<double>& xiD_free_over_ask_values,
        const std::vector<double>& gt_tail_values,
        const std::vector<double>& mt_values,
        const std::vector<double>& roughness_slope_AB_values,
        const std::vector<double>& g6tail_values
    ) {
        const double mean_phi6 = MeanFinite(phi6_global_values);
        const double sem_phi6 = SemFinite(phi6_global_values, mean_phi6);
        const double mean_local = MeanFinite(phi6_local_values);
        const double sem_local = SemFinite(phi6_local_values, mean_local);

        const double mean_rho = MeanFinite(rho_def_values);
        const double sem_rho = SemFinite(rho_def_values, mean_rho);
        const double mean_n5 = MeanFinite(n5_values);
        const double mean_n6 = MeanFinite(n6_values);
        const double mean_n7 = MeanFinite(n7_values);
        const double mean_nnon6 = MeanFinite(nnon6_values);

        const double mean_dislocation_frac = MeanFinite(dislocation_frac_values);
        const double mean_free_dislocation_frac = MeanFinite(free_dislocation_frac_values);
        const double mean_bound_dislocation_frac = MeanFinite(bound_dislocation_frac_values);
        const double mean_rho_dislocation = MeanFinite(rho_dislocation_values);
        const double mean_rho_free_dislocation = MeanFinite(rho_free_dislocation_values);
        const double mean_xiD_free_over_ask = MeanFinite(xiD_free_over_ask_values);

        const double sem_dislocation_frac = SemFinite(dislocation_frac_values, mean_dislocation_frac);
        const double sem_free_dislocation_frac = SemFinite(free_dislocation_frac_values, mean_free_dislocation_frac);
        const double sem_bound_dislocation_frac = SemFinite(bound_dislocation_frac_values, mean_bound_dislocation_frac);
        const double sem_rho_dislocation = SemFinite(rho_dislocation_values, mean_rho_dislocation);
        const double sem_rho_free_dislocation = SemFinite(rho_free_dislocation_values, mean_rho_free_dislocation);
        const double sem_xiD_free_over_ask = SemFinite(xiD_free_over_ask_values, mean_xiD_free_over_ask);

        const double mean_g6tail = MeanFinite(g6tail_values);
        const double sem_g6tail = SemFinite(g6tail_values, mean_g6tail);

        const double mean_gt_tail = MeanFinite(gt_tail_values);
        const double sem_gt_tail = SemFinite(gt_tail_values, mean_gt_tail);
        const double mean_mt = MeanFinite(mt_values);
        const double sem_mt = SemFinite(mt_values, mean_mt);
        const double mean_AB = MeanFinite(roughness_slope_AB_values);
        const double sem_AB = SemFinite(roughness_slope_AB_values, mean_AB);

        const double chi_phi6 = SusceptibilityFromDisorderValues(phi6_global_values, mean_nsk_center);
        const double chi_phi6_sem = SusceptibilityJackknifeSem(phi6_global_values, mean_nsk_center);
        const double chi_local = SusceptibilityFromDisorderValues(phi6_local_values, mean_nsk_center);
        const double chi_local_sem = SusceptibilityJackknifeSem(phi6_local_values, mean_nsk_center);
        const double chi_def = SusceptibilityFromDisorderValues(rho_def_values, mean_nsk_center);
        const double chi_def_sem = SusceptibilityJackknifeSem(rho_def_values, mean_nsk_center);
        const double chi_free_dislocation =
            SusceptibilityFromDisorderValues(free_dislocation_frac_values, mean_nsk_center);
        const double chi_free_dislocation_sem =
            SusceptibilityJackknifeSem(free_dislocation_frac_values, mean_nsk_center);
        const double chi_g6tail = SusceptibilityFromDisorderValues(g6tail_values, mean_nsk_center);
        const double chi_g6tail_sem = SusceptibilityJackknifeSem(g6tail_values, mean_nsk_center);
        const double chi_gt_tail = SusceptibilityFromDisorderValues(gt_tail_values, mean_nsk_center);
        const double chi_gt_tail_sem = SusceptibilityJackknifeSem(gt_tail_values, mean_nsk_center);

        const double binder_phi6 = BinderCumulantO2NormalizedFromMagnitude(phi6_global_values);
        const double binder_local = BinderCumulantScalarNormalized(phi6_local_values);
        const double binder_g6tail = BinderCumulantScalarNormalized(g6tail_values);
        const double binder_phi6_scalar_raw = BinderCumulantScalar(phi6_global_values);
        const double binder_local_scalar_raw = BinderCumulantScalar(phi6_local_values);
        const double binder_g6tail_scalar_raw = BinderCumulantScalar(g6tail_values);

        std::ostringstream filename;
        filename << "PhaseBoundary_Measures_iD" << iD << "_iB" << iB << ".txt";
        std::ofstream output(filename.str());
        if (!output) {
            throw std::runtime_error("Cannot write phase-boundary diagnostics file.");
        }

        output << "# One row summary.  Error columns are SEM over disorder bins; chi errors use disorder jackknife.\n";
        output << "# Coordination and dislocations use strict 3x3-image PBC Delaunay neighbours.\n";
        output << "# Dislocation core = neighbouring 5-7 pair. Free dislocation = 5-7 core without nearby opposite-polarity partner.\n";
        output << "# calculate_phase_diagnostics " << (phase.enabled ? "true" : "false") << "\n";
        output << "# phase_defect_measure_stride " << phase.defect_measure_stride << "\n";
        output << "# phase_defect_nn_rmin_factor " << phase.defect_nn_rmin_factor << "\n";
        output << "# phase_defect_nn_rmax_factor " << phase.defect_nn_rmax_factor << "\n";
        output << "# dislocation_pair_rmax_factor " << phase.dislocation_pair_rmax_factor << "\n";
        output << "# dislocation_bind_rmax_factor " << phase.dislocation_bind_rmax_factor << "\n";
        output << "# dislocation_opposite_cos " << phase.dislocation_opposite_cos << "\n";
        output << "# g6_tail_rmin " << phase.g6_tail_rmin << "\n";
        output << "# g6_tail_rmax " << phase.g6_tail_rmax << "\n";
        output << "# calculate_trans_diagnostics " << (phase.trans_enabled ? "true" : "false") << "\n";
        output << "# trans_use_fixed_axis " << (phase.trans_use_fixed_axis ? "true" : "false") << "\n";
        output << "# trans_fixed_axis_angle " << phase.trans_fixed_axis_angle << "\n";
        output << "# trans_fixed_ask " << phase.trans_fixed_ask << "\n";
        output << "# trans_rmax " << phase.trans_rmax << "\n";
        output << "# trans_rbin " << phase.trans_rbin << "\n";
        output << "# trans_tail_rmin " << phase.trans_tail_rmin << "\n";
        output << "# trans_tail_rmax " << phase.trans_tail_rmax << "\n";
        output << "# trans_fit_rmin " << phase.trans_fit_rmin << "\n";
        output << "# trans_fit_rmax " << phase.trans_fit_rmax << "\n";
        output << "# trans_g_mode " << phase.trans_g_mode << "\n";
        output << "# trans_n_g_vectors " << phase.trans_n_g_vectors << "\n";
        output << "# trans_average_multiple_g " << (phase.trans_average_multiple_g ? "true" : "false") << "\n";
        output << "# columns:\n";
        output << "# mean_Nsk Phi6_global Phi6_global_SEM Phi6_local Phi6_local_SEM "
            << "chi_Phi6_dis chi_Phi6_dis_SEM chi_local_dis chi_local_dis_SEM "
            << "rho_def rho_def_SEM chi_def chi_def_SEM "
            << "G6_tail G6_tail_SEM chi_G6tail chi_G6tail_SEM "
            << "GT_tail GT_tail_SEM chi_GTtail chi_GTtail_SEM "
            << "mT mT_SEM AB_roughness AB_roughness_SEM "
            << "Binder_Phi6 Binder_local Binder_G6tail "
            << "Binder_Phi6_scalar_raw Binder_local_scalar_raw Binder_G6tail_scalar_raw "
            << "N5_frac N6_frac N7_frac Nnon6_frac "
            << "Ndisloc_frac Ndisloc_frac_SEM "
            << "Nfree_disloc_frac Nfree_disloc_frac_SEM "
            << "Nbound_disloc_frac Nbound_disloc_frac_SEM "
            << "Rho_disloc Rho_disloc_SEM "
            << "Rho_free_disloc Rho_free_disloc_SEM "
            << "XiD_free_over_ask XiD_free_over_ask_SEM "
            << "Chi_free_disloc Chi_free_disloc_SEM\n";

        output << std::setprecision(16)
            << mean_nsk_center << ' '
            << mean_phi6 << ' ' << sem_phi6 << ' '
            << mean_local << ' ' << sem_local << ' '
            << chi_phi6 << ' ' << chi_phi6_sem << ' '
            << chi_local << ' ' << chi_local_sem << ' '
            << mean_rho << ' ' << sem_rho << ' '
            << chi_def << ' ' << chi_def_sem << ' '
            << mean_g6tail << ' ' << sem_g6tail << ' '
            << chi_g6tail << ' ' << chi_g6tail_sem << ' '
            << mean_gt_tail << ' ' << sem_gt_tail << ' '
            << chi_gt_tail << ' ' << chi_gt_tail_sem << ' '
            << mean_mt << ' ' << sem_mt << ' '
            << mean_AB << ' ' << sem_AB << ' '
            << binder_phi6 << ' ' << binder_local << ' ' << binder_g6tail << ' '
            << binder_phi6_scalar_raw << ' ' << binder_local_scalar_raw << ' ' << binder_g6tail_scalar_raw << ' '
            << mean_n5 << ' ' << mean_n6 << ' ' << mean_n7 << ' ' << mean_nnon6 << ' '
            << mean_dislocation_frac << ' ' << sem_dislocation_frac << ' '
            << mean_free_dislocation_frac << ' ' << sem_free_dislocation_frac << ' '
            << mean_bound_dislocation_frac << ' ' << sem_bound_dislocation_frac << ' '
            << mean_rho_dislocation << ' ' << sem_rho_dislocation << ' '
            << mean_rho_free_dislocation << ' ' << sem_rho_free_dislocation << ' '
            << mean_xiD_free_over_ask << ' ' << sem_xiD_free_over_ask << ' '
            << chi_free_dislocation << ' ' << chi_free_dislocation_sem << '\n';

        std::ostringstream sample_filename;
        sample_filename << "PhaseBoundary_DisorderSamples_iD" << iD << "_iB" << iB << ".txt";
        std::ofstream samples(sample_filename.str());
        if (!samples) {
            throw std::runtime_error("Cannot write phase-boundary disorder samples file.");
        }

        samples << "# disorder_bin Phi6_global Phi6_local rho_def G6_tail GT_tail mT AB_roughness "
            << "N5_frac N6_frac N7_frac Nnon6_frac "
            << "Ndisloc_frac Nfree_disloc_frac Nbound_disloc_frac "
            << "Rho_disloc Rho_free_disloc XiD_free_over_ask\n";

        const std::size_t n = phi6_global_values.size();
        for (std::size_t ib = 0; ib < n; ++ib) {
            const double local = ib < phi6_local_values.size() ? phi6_local_values[ib] : std::numeric_limits<double>::quiet_NaN();
            const double rho = ib < rho_def_values.size() ? rho_def_values[ib] : std::numeric_limits<double>::quiet_NaN();
            const double tail = ib < g6tail_values.size() ? g6tail_values[ib] : std::numeric_limits<double>::quiet_NaN();
            const double gt_tail = ib < gt_tail_values.size() ? gt_tail_values[ib] : std::numeric_limits<double>::quiet_NaN();
            const double mt = ib < mt_values.size() ? mt_values[ib] : std::numeric_limits<double>::quiet_NaN();
            const double ab = ib < roughness_slope_AB_values.size() ? roughness_slope_AB_values[ib] : std::numeric_limits<double>::quiet_NaN();
            const double n5 = ib < n5_values.size() ? n5_values[ib] : std::numeric_limits<double>::quiet_NaN();
            const double n6 = ib < n6_values.size() ? n6_values[ib] : std::numeric_limits<double>::quiet_NaN();
            const double n7 = ib < n7_values.size() ? n7_values[ib] : std::numeric_limits<double>::quiet_NaN();
            const double nnon6 = ib < nnon6_values.size() ? nnon6_values[ib] : std::numeric_limits<double>::quiet_NaN();

            const double ndis = ib < dislocation_frac_values.size() ? dislocation_frac_values[ib] : std::numeric_limits<double>::quiet_NaN();
            const double nfree = ib < free_dislocation_frac_values.size() ? free_dislocation_frac_values[ib] : std::numeric_limits<double>::quiet_NaN();
            const double nbound = ib < bound_dislocation_frac_values.size() ? bound_dislocation_frac_values[ib] : std::numeric_limits<double>::quiet_NaN();
            const double rhod = ib < rho_dislocation_values.size() ? rho_dislocation_values[ib] : std::numeric_limits<double>::quiet_NaN();
            const double rhofree = ib < rho_free_dislocation_values.size() ? rho_free_dislocation_values[ib] : std::numeric_limits<double>::quiet_NaN();
            const double xid = ib < xiD_free_over_ask_values.size() ? xiD_free_over_ask_values[ib] : std::numeric_limits<double>::quiet_NaN();

            samples << ib << ' '
                << std::setprecision(16) << phi6_global_values[ib] << ' '
                << local << ' ' << rho << ' ' << tail << ' '
                << gt_tail << ' ' << mt << ' ' << ab << ' '
                << n5 << ' ' << n6 << ' ' << n7 << ' ' << nnon6 << ' '
                << ndis << ' ' << nfree << ' ' << nbound << ' '
                << rhod << ' ' << rhofree << ' ' << xid << '\n';
        }
    }

    void AccumulateG6rFrame(
        const OrientFrame& frame,
        int lx,
        int ly,
        const G6rSettings& g6r,
        std::vector<double>& sum_re,
        std::vector<double>& count
    ) {
        const int n = frame.nsk;
        if (n < 2 || frame.psi6.size() != frame.centers.size() ||
            !(frame.a_sk > 0.0) || !std::isfinite(frame.a_sk)) {
            return;
        }

        const int nbins = G6rNumberOfBins(g6r);
        if (static_cast<int>(sum_re.size()) != nbins ||
            static_cast<int>(count.size()) != nbins) {
            throw std::runtime_error("G6r histogram size mismatch.");
        }

        for (int i = 0; i < n - 1; ++i) {
            const auto zi = frame.psi6[i];
            if (!std::isfinite(zi.real()) || !std::isfinite(zi.imag())) {
                continue;
            }

            for (int j = i + 1; j < n; ++j) {
                const auto zj = frame.psi6[j];
                if (!std::isfinite(zj.real()) || !std::isfinite(zj.imag())) {
                    continue;
                }

                const Point2 dr = MinImageDeltaReal(frame.centers[i], frame.centers[j], lx, ly);
                const double r2 = dr.x * dr.x + dr.y * dr.y;
                const double rho = std::sqrt(r2) / frame.a_sk;

                if (!(rho >= 0.0) || rho >= g6r.rmax || !std::isfinite(rho)) {
                    continue;
                }

                const int b = static_cast<int>(std::floor(rho / g6r.rbin));
                if (b < 0 || b >= nbins) {
                    continue;
                }

                // Re[psi_i * conj(psi_j)]
                const double prod_re = zi.real() * zj.real() + zi.imag() * zj.imag();

                sum_re[b] += prod_re;
                count[b] += 1.0;
            }
        }
    }

    std::vector<double> G6rFromAccum(
        const std::vector<double>& sum_re,
        const std::vector<double>& count
    ) {
        const int nbins = static_cast<int>(sum_re.size());
        std::vector<double> g(nbins, std::numeric_limits<double>::quiet_NaN());

        for (int b = 0; b < nbins; ++b) {
            if (b < static_cast<int>(count.size()) && count[b] > 0.0) {
                g[b] = sum_re[b] / count[b];
            }
        }

        return g;
    }

    void FinishG6rBlockIfNeeded(
        std::vector<double>& block_sum,
        std::vector<double>& block_count,
        int& block_frame_count,
        std::vector<std::vector<double>>& block_curves
    ) {
        if (block_frame_count <= 0) {
            return;
        }

        bool has_count = false;
        for (const double c : block_count) {
            if (c > 0.0 && std::isfinite(c)) {
                has_count = true;
                break;
            }
        }

        if (has_count) {
            block_curves.push_back(G6rFromAccum(block_sum, block_count));
        }

        std::fill(block_sum.begin(), block_sum.end(), 0.0);
        std::fill(block_count.begin(), block_count.end(), 0.0);
        block_frame_count = 0;
    }

    Gx0DisorderCurve MakeGenericDisorderCurveFromBlocks(
        const std::vector<std::vector<double>>& block_curves,
        int selected_frames,
        double nsk_sum,
        int min_nsk,
        int max_nsk
    ) {
        return MakeGx0DisorderCurveFromBlocks(
            block_curves,
            selected_frames,
            nsk_sum,
            min_nsk,
            max_nsk
        );
    }

    void FinishTransBlockIfNeeded(
        std::vector<std::vector<double>>& block_frame_curves,
        int& block_frame_count,
        std::vector<std::vector<double>>& block_curves
    ) {
        if (block_frame_count <= 0 || block_frame_curves.empty()) {
            block_frame_curves.clear();
            block_frame_count = 0;
            return;
        }

        block_curves.push_back(MeanCurveIgnoringNaN(block_frame_curves));
        block_frame_curves.clear();
        block_frame_count = 0;
    }

    std::vector<double> ComputeTransTOneDisorder(
        const std::vector<std::vector<std::complex<double>>>& psiT_series,
        int max_lag
    ) {
        const int T = static_cast<int>(psiT_series.size());
        if (T <= 0) {
            return {};
        }

        int nlag = T;
        if (max_lag > 0) {
            nlag = std::min(nlag, max_lag + 1);
        }

        std::vector<double> curve(nlag, std::numeric_limits<double>::quiet_NaN());

        for (int lag = 0; lag < nlag; ++lag) {
            double sum_time = 0.0;
            int n_time = 0;

            for (int t0 = 0; t0 + lag < T; ++t0) {
                const auto& a = psiT_series[t0];
                const auto& b = psiT_series[t0 + lag];
                const int ng = std::min(static_cast<int>(a.size()), static_cast<int>(b.size()));
                if (ng <= 0) {
                    continue;
                }

                double sum_g = 0.0;
                int n_g = 0;
                for (int mu = 0; mu < ng; ++mu) {
                    const auto za = a[mu];
                    const auto zb = b[mu];
                    if (!std::isfinite(za.real()) || !std::isfinite(za.imag()) ||
                        !std::isfinite(zb.real()) || !std::isfinite(zb.imag())) {
                        continue;
                    }
                    // Re[Psi_G(t0) conj(Psi_G(t0+lag))]
                    const double val = za.real() * zb.real() + za.imag() * zb.imag();
                    if (std::isfinite(val)) {
                        sum_g += val;
                        ++n_g;
                    }
                }

                if (n_g > 0) {
                    sum_time += sum_g / static_cast<double>(n_g);
                    ++n_time;
                }
            }

            if (n_time > 0) {
                curve[lag] = sum_time / static_cast<double>(n_time);
            }
        }

        return curve;
    }

    void WriteGtrHierarchicalFile(
        int iD,
        int iB,
        const PhaseBoundarySettings& phase,
        const std::vector<Gx0DisorderCurve>& disorder_curves,
        const std::vector<std::vector<std::vector<double>>>& all_block_curves
    ) {
        const int nbins = TransNumberOfBins(phase);

        std::vector<std::vector<double>> disorder_samples;
        disorder_samples.reserve(disorder_curves.size());
        for (const auto& d : disorder_curves) {
            if (static_cast<int>(d.mean.size()) == nbins) {
                disorder_samples.push_back(d.mean);
            }
            else {
                disorder_samples.push_back(
                    std::vector<double>(nbins, std::numeric_limits<double>::quiet_NaN())
                );
            }
        }

        std::vector<double> mean;
        std::vector<double> sem_disorder;
        std::vector<double> std_disorder;
        std::vector<int> nvalid_disorder;
        MeanSemStdCurvesGeneral(disorder_samples, mean, sem_disorder, std_disorder, nvalid_disorder);

        std::vector<double> sem_block_mean(nbins, std::numeric_limits<double>::quiet_NaN());
        std::vector<double> std_block_mean(nbins, std::numeric_limits<double>::quiet_NaN());

        for (int b = 0; b < nbins; ++b) {
            double sum_sem = 0.0;
            double sum_std = 0.0;
            int n_sem = 0;
            int n_std = 0;

            for (const auto& d : disorder_curves) {
                if (b < static_cast<int>(d.sem_block.size()) && std::isfinite(d.sem_block[b])) {
                    sum_sem += d.sem_block[b];
                    ++n_sem;
                }
                if (b < static_cast<int>(d.std_block.size()) && std::isfinite(d.std_block[b])) {
                    sum_std += d.std_block[b];
                    ++n_std;
                }
            }

            if (n_sem > 0) {
                sem_block_mean[b] = sum_sem / static_cast<double>(n_sem);
            }
            if (n_std > 0) {
                std_block_mean[b] = sum_std / static_cast<double>(n_std);
            }
        }

        double total_frames = 0.0;
        double total_blocks = 0.0;
        double weighted_nsk_sum = 0.0;
        for (const auto& d : disorder_curves) {
            total_frames += static_cast<double>(d.selected_frames);
            total_blocks += static_cast<double>(d.n_blocks);
            weighted_nsk_sum += static_cast<double>(d.selected_frames) * d.avg_nsk;
        }

        const double avg_nsk = total_frames > 0.0 ? weighted_nsk_sum / total_frames : 0.0;
        const double avg_blocks = disorder_curves.empty() ? 0.0 : total_blocks / static_cast<double>(disorder_curves.size());
        const double avg_frames = disorder_curves.empty() ? 0.0 : total_frames / static_cast<double>(disorder_curves.size());

        std::ostringstream filename;
        filename << "Gtr_Measures_iD" << iD << "_iB" << iB << ".txt";
        std::ofstream output(filename.str());
        if (!output) {
            throw std::runtime_error("Cannot write corrected Gtr measurement file.");
        }

        output << "# r_over_aref GT_mean SEM_disorder STD_disorder "
            << "N_disorder_samples SEM_block_mean STD_block_mean "
            << "N_measure_blocks_avg N_selected_frames_avg\n";
        output << "# Correct G_T(r) from skyrmion centers using PBC-compatible reciprocal-grid Bragg vector(s).\n";
        output << "# Each selected frame is converted to one G_T(r) curve, then curves are averaged in blocks and disorder bins.\n";
        output << "# trans_g_mode " << phase.trans_g_mode << "\n";
        output << "# trans_reference_iD " << phase.trans_reference_iD << "\n";
        output << "# trans_reference_iB " << phase.trans_reference_iB << "\n";
        output << "# trans_n_g_vectors " << phase.trans_n_g_vectors << "\n";
        output << "# trans_q0_min_factor " << phase.trans_q0_min_factor << "\n";
        output << "# trans_q0_max_factor " << phase.trans_q0_max_factor << "\n";
        output << "# trans_rmax " << phase.trans_rmax << "\n";
        output << "# trans_rbin " << phase.trans_rbin << "\n";
        output << "# disorder_samples " << disorder_curves.size() << "\n";
        output << "# total_trans_frames " << static_cast<long long>(total_frames) << "\n";
        output << "# average_Nsk " << std::setprecision(16) << avg_nsk << "\n";
        output << "# average_measure_blocks " << avg_blocks << "\n";
        output << "# average_selected_frames_per_disorder " << avg_frames << "\n";

        for (int b = 0; b < nbins; ++b) {
            const double rmid = (static_cast<double>(b) + 0.5) * phase.trans_rbin;
            output << std::setprecision(16)
                << rmid << ' '
                << mean[b] << ' '
                << sem_disorder[b] << ' '
                << std_disorder[b] << ' '
                << nvalid_disorder[b] << ' '
                << sem_block_mean[b] << ' '
                << std_block_mean[b] << ' '
                << avg_blocks << ' '
                << avg_frames << '\n';
        }

        std::ostringstream disorder_filename;
        disorder_filename << "Gtr_DisorderSamples_iD" << iD << "_iB" << iB << ".txt";
        std::ofstream dout(disorder_filename.str());
        if (!dout) {
            throw std::runtime_error("Cannot write corrected Gtr disorder-sample file.");
        }
        dout << "# disorder_bin r_over_aref GT_disorder SEM_block STD_block N_blocks selected_frames avg_Nsk min_Nsk max_Nsk\n";

        for (std::size_t id = 0; id < disorder_curves.size(); ++id) {
            const auto& d = disorder_curves[id];
            for (int b = 0; b < nbins; ++b) {
                const double rmid = (static_cast<double>(b) + 0.5) * phase.trans_rbin;
                const double gv = b < static_cast<int>(d.mean.size()) ? d.mean[b] : std::numeric_limits<double>::quiet_NaN();
                const double se = b < static_cast<int>(d.sem_block.size()) ? d.sem_block[b] : std::numeric_limits<double>::quiet_NaN();
                const double sd = b < static_cast<int>(d.std_block.size()) ? d.std_block[b] : std::numeric_limits<double>::quiet_NaN();
                dout << id << ' '
                    << std::setprecision(16) << rmid << ' '
                    << gv << ' ' << se << ' ' << sd << ' '
                    << d.n_blocks << ' ' << d.selected_frames << ' '
                    << d.avg_nsk << ' ' << d.min_nsk << ' ' << d.max_nsk << '\n';
            }
            dout << '\n';
        }

        if (phase.write_trans_block_samples) {
            std::ostringstream block_filename;
            block_filename << "Gtr_BlockSamples_iD" << iD << "_iB" << iB << ".txt";
            std::ofstream bout(block_filename.str());
            if (!bout) {
                throw std::runtime_error("Cannot write corrected Gtr block-sample file.");
            }
            bout << "# disorder_bin measure_block r_over_aref GT_block\n";
            for (std::size_t id = 0; id < all_block_curves.size(); ++id) {
                for (std::size_t iblock = 0; iblock < all_block_curves[id].size(); ++iblock) {
                    const auto& curve = all_block_curves[id][iblock];
                    for (int b = 0; b < nbins; ++b) {
                        const double rmid = (static_cast<double>(b) + 0.5) * phase.trans_rbin;
                        const double gv = b < static_cast<int>(curve.size()) ? curve[b] : std::numeric_limits<double>::quiet_NaN();
                        bout << id << ' ' << iblock << ' '
                            << std::setprecision(16) << rmid << ' ' << gv << '\n';
                    }
                    bout << '\n';
                }
            }
        }
    }

    void WriteTransTFile(
        int iD,
        int iB,
        const PhaseBoundarySettings& phase,
        const std::vector<std::vector<double>>& disorder_curves,
        const std::vector<int>& selected_frames,
        const std::vector<double>& avg_nsk
    ) {
        std::size_t max_curve_size = 0;
        for (const auto& c : disorder_curves) {
            max_curve_size = std::max(max_curve_size, c.size());
        }

        std::vector<std::vector<double>> padded_curves;
        std::vector<std::vector<double>> padded_norm_curves;
        padded_curves.reserve(disorder_curves.size());
        padded_norm_curves.reserve(disorder_curves.size());

        for (const auto& c : disorder_curves) {
            std::vector<double> p(max_curve_size, std::numeric_limits<double>::quiet_NaN());
            std::vector<double> pn(max_curve_size, std::numeric_limits<double>::quiet_NaN());
            const double c0 = (!c.empty() && std::isfinite(c[0]) && std::fabs(c[0]) > 1.0e-30) ? c[0] : std::numeric_limits<double>::quiet_NaN();
            for (std::size_t i = 0; i < c.size(); ++i) {
                p[i] = c[i];
                if (std::isfinite(c[i]) && std::isfinite(c0)) {
                    pn[i] = c[i] / c0;
                }
            }
            padded_curves.push_back(std::move(p));
            padded_norm_curves.push_back(std::move(pn));
        }

        std::vector<double> mean;
        std::vector<double> sem;
        std::vector<double> stddev;
        std::vector<int> nvalid;
        MeanSemStdCurvesGeneral(padded_curves, mean, sem, stddev, nvalid);

        std::vector<double> mean_norm;
        std::vector<double> sem_norm;
        std::vector<double> std_norm;
        std::vector<int> nvalid_norm;
        MeanSemStdCurvesGeneral(padded_norm_curves, mean_norm, sem_norm, std_norm, nvalid_norm);

        std::ostringstream filename;
        filename << "Gtt_Measures_iD" << iD << "_iB" << iB << ".txt";
        std::ofstream output(filename.str());
        if (!output) {
            throw std::runtime_error("Cannot write corrected G_T(t) file.");
        }

        output << "# lag time GTt_mean SEM_disorder STD_disorder N_disorder_samples "
            << "GTt_norm_mean GTt_norm_SEM GTt_norm_STD N_norm_samples\n";
        output << "# G_T(t)=Re < mean_mu Psi_T_mu(t0) conj(Psi_T_mu(t0+lag)) >_t0 using same reciprocal-grid G vectors as G_T(r).\n";
        output << "# trans_t_dt " << phase.trans_t_dt << "\n";
        output << "# trans_t_max_lag " << phase.trans_t_max_lag << "\n";
        output << "# trans_g_mode " << phase.trans_g_mode << "\n";

        for (std::size_t lag = 0; lag < max_curve_size; ++lag) {
            output << std::setprecision(16)
                << lag << ' '
                << static_cast<double>(lag) * phase.trans_t_dt << ' '
                << mean[lag] << ' '
                << sem[lag] << ' '
                << stddev[lag] << ' '
                << nvalid[lag] << ' '
                << mean_norm[lag] << ' '
                << sem_norm[lag] << ' '
                << std_norm[lag] << ' '
                << nvalid_norm[lag] << '\n';
        }

        std::ostringstream sample_filename;
        sample_filename << "Gtt_DisorderSamples_iD" << iD << "_iB" << iB << ".txt";
        std::ofstream samples(sample_filename.str());
        if (!samples) {
            throw std::runtime_error("Cannot write corrected G_T(t) disorder-sample file.");
        }
        samples << "# disorder_bin lag time GTt_disorder GTt_norm selected_frames avg_Nsk\n";
        for (std::size_t ib = 0; ib < disorder_curves.size(); ++ib) {
            const auto& c = disorder_curves[ib];
            const double c0 = (!c.empty() && std::isfinite(c[0]) && std::fabs(c[0]) > 1.0e-30) ? c[0] : std::numeric_limits<double>::quiet_NaN();
            for (std::size_t lag = 0; lag < c.size(); ++lag) {
                const double norm = (std::isfinite(c[lag]) && std::isfinite(c0)) ? c[lag] / c0 : std::numeric_limits<double>::quiet_NaN();
                const int frames = ib < selected_frames.size() ? selected_frames[ib] : 0;
                const double nsk = ib < avg_nsk.size() ? avg_nsk[ib] : std::numeric_limits<double>::quiet_NaN();
                samples << ib << ' '
                    << lag << ' '
                    << std::setprecision(16) << static_cast<double>(lag) * phase.trans_t_dt << ' '
                    << c[lag] << ' '
                    << norm << ' '
                    << frames << ' '
                    << nsk << '\n';
            }
            samples << '\n';
        }
    }

    void WriteTransScalarDiagnosticsFile(
        int iD,
        int iB,
        double mean_nsk_center,
        const std::vector<double>& gt_tail_values,
        const std::vector<double>& mt_values,
        const std::vector<double>& eta_values,
        const std::vector<double>& ab_values,
        const std::vector<double>& w2_values,
        const std::vector<double>& dq_parallel_values,
        const std::vector<double>& dq_perp_values,
        const std::vector<double>& xi_parallel_values,
        const std::vector<double>& xi_perp_values,
        const std::vector<double>& chi_mt_time_values,
        const std::vector<double>& chi_w2_time_values,
        const std::vector<double>& binder_mt_time_values
    ) {
        const double mean_gt_tail = MeanFinite(gt_tail_values);
        const double sem_gt_tail = SemFinite(gt_tail_values, mean_gt_tail);
        const double mean_mt = MeanFinite(mt_values);
        const double sem_mt = SemFinite(mt_values, mean_mt);
        const double mean_eta = MeanFinite(eta_values);
        const double sem_eta = SemFinite(eta_values, mean_eta);
        const double mean_ab = MeanFinite(ab_values);
        const double sem_ab = SemFinite(ab_values, mean_ab);
        const double mean_w2 = MeanFinite(w2_values);
        const double sem_w2 = SemFinite(w2_values, mean_w2);
        const double mean_dqpar = MeanFinite(dq_parallel_values);
        const double sem_dqpar = SemFinite(dq_parallel_values, mean_dqpar);
        const double mean_dqperp = MeanFinite(dq_perp_values);
        const double sem_dqperp = SemFinite(dq_perp_values, mean_dqperp);
        const double mean_xipar = MeanFinite(xi_parallel_values);
        const double sem_xipar = SemFinite(xi_parallel_values, mean_xipar);
        const double mean_xiperp = MeanFinite(xi_perp_values);
        const double sem_xiperp = SemFinite(xi_perp_values, mean_xiperp);
        const double mean_chi_mt_time = MeanFinite(chi_mt_time_values);
        const double sem_chi_mt_time = SemFinite(chi_mt_time_values, mean_chi_mt_time);
        const double mean_chi_w2_time = MeanFinite(chi_w2_time_values);
        const double sem_chi_w2_time = SemFinite(chi_w2_time_values, mean_chi_w2_time);
        const double mean_binder_mt_time = MeanFinite(binder_mt_time_values);
        const double sem_binder_mt_time = SemFinite(binder_mt_time_values, mean_binder_mt_time);

        const double chi_mt_dis = SusceptibilityFromDisorderValues(mt_values, mean_nsk_center);
        const double chi_mt_dis_sem = SusceptibilityJackknifeSem(mt_values, mean_nsk_center);
        const double chi_w2_dis = SusceptibilityFromDisorderValues(w2_values, mean_nsk_center);
        const double chi_w2_dis_sem = SusceptibilityJackknifeSem(w2_values, mean_nsk_center);
        const double chi_ab_dis = SusceptibilityFromDisorderValues(ab_values, mean_nsk_center);
        const double chi_ab_dis_sem = SusceptibilityJackknifeSem(ab_values, mean_nsk_center);
        const double chi_dqpar_dis = SusceptibilityFromDisorderValues(dq_parallel_values, mean_nsk_center);
        const double chi_dqpar_dis_sem = SusceptibilityJackknifeSem(dq_parallel_values, mean_nsk_center);
        const double binder_mt_dis = BinderCumulantScalarNormalized(mt_values);
        const double binder_mt_dis_scalar_raw = BinderCumulantScalar(mt_values);

        std::ostringstream filename;
        filename << "TransScalar_Measures_iD" << iD << "_iB" << iB << ".txt";
        std::ofstream output(filename.str());
        if (!output) {
            throw std::runtime_error("Cannot write translational scalar diagnostics file.");
        }

        output << "# Correct translational diagnostics for SkX -> BrG-like.\n";
        output << "# W2_phase=<wrap(G.r-arg Psi_G)^2>/(G^2 a_sk^2); dQ widths are from center S(q).\n";
        output << "# Columns:\n";
        output << "# mean_Nsk GT_tail GT_tail_SEM mT mT_SEM chi_mT_dis chi_mT_dis_SEM "
            << "chi_mT_time chi_mT_time_SEM Binder_mT_dis Binder_mT_time Binder_mT_time_SEM Binder_mT_dis_scalar_raw "
            << "eta_T eta_T_SEM AB_roughness AB_roughness_SEM "
            << "W2_phase W2_phase_SEM chi_W2_dis chi_W2_dis_SEM chi_W2_time chi_W2_time_SEM "
            << "chi_AB_dis chi_AB_dis_SEM dQpar_over_q0 dQpar_over_q0_SEM "
            << "dQperp_over_q0 dQperp_over_q0_SEM Xi_parallel_over_ask Xi_parallel_over_ask_SEM "
            << "Xi_perp_over_ask Xi_perp_over_ask_SEM chi_dQpar_dis chi_dQpar_dis_SEM\n";

        output << std::setprecision(16)
            << mean_nsk_center << ' '
            << mean_gt_tail << ' ' << sem_gt_tail << ' '
            << mean_mt << ' ' << sem_mt << ' '
            << chi_mt_dis << ' ' << chi_mt_dis_sem << ' '
            << mean_chi_mt_time << ' ' << sem_chi_mt_time << ' '
            << binder_mt_dis << ' '
            << mean_binder_mt_time << ' ' << sem_binder_mt_time << ' '
            << binder_mt_dis_scalar_raw << ' '
            << mean_eta << ' ' << sem_eta << ' '
            << mean_ab << ' ' << sem_ab << ' '
            << mean_w2 << ' ' << sem_w2 << ' '
            << chi_w2_dis << ' ' << chi_w2_dis_sem << ' '
            << mean_chi_w2_time << ' ' << sem_chi_w2_time << ' '
            << chi_ab_dis << ' ' << chi_ab_dis_sem << ' '
            << mean_dqpar << ' ' << sem_dqpar << ' '
            << mean_dqperp << ' ' << sem_dqperp << ' '
            << mean_xipar << ' ' << sem_xipar << ' '
            << mean_xiperp << ' ' << sem_xiperp << ' '
            << chi_dqpar_dis << ' ' << chi_dqpar_dis_sem << '\n';

        std::ostringstream sample_filename;
        sample_filename << "TransScalar_DisorderSamples_iD" << iD << "_iB" << iB << ".txt";
        std::ofstream samples(sample_filename.str());
        if (!samples) {
            throw std::runtime_error("Cannot write translational scalar disorder-sample file.");
        }
        samples << "# disorder_bin GT_tail mT eta_T AB_roughness W2_phase dQpar_over_q0 dQperp_over_q0 "
            << "Xi_parallel_over_ask Xi_perp_over_ask chi_mT_time chi_W2_time Binder_mT_time\n";
        const std::size_t n = mt_values.size();
        for (std::size_t ib = 0; ib < n; ++ib) {
            const double gt = ib < gt_tail_values.size() ? gt_tail_values[ib] : std::numeric_limits<double>::quiet_NaN();
            const double eta = ib < eta_values.size() ? eta_values[ib] : std::numeric_limits<double>::quiet_NaN();
            const double ab = ib < ab_values.size() ? ab_values[ib] : std::numeric_limits<double>::quiet_NaN();
            const double w2 = ib < w2_values.size() ? w2_values[ib] : std::numeric_limits<double>::quiet_NaN();
            const double dqpar = ib < dq_parallel_values.size() ? dq_parallel_values[ib] : std::numeric_limits<double>::quiet_NaN();
            const double dqperp = ib < dq_perp_values.size() ? dq_perp_values[ib] : std::numeric_limits<double>::quiet_NaN();
            const double xipar = ib < xi_parallel_values.size() ? xi_parallel_values[ib] : std::numeric_limits<double>::quiet_NaN();
            const double xiperp = ib < xi_perp_values.size() ? xi_perp_values[ib] : std::numeric_limits<double>::quiet_NaN();
            const double chi_t = ib < chi_mt_time_values.size() ? chi_mt_time_values[ib] : std::numeric_limits<double>::quiet_NaN();
            const double chi_w2_t = ib < chi_w2_time_values.size() ? chi_w2_time_values[ib] : std::numeric_limits<double>::quiet_NaN();
            const double binder_t = ib < binder_mt_time_values.size() ? binder_mt_time_values[ib] : std::numeric_limits<double>::quiet_NaN();
            samples << ib << ' '
                << std::setprecision(16) << gt << ' '
                << mt_values[ib] << ' '
                << eta << ' '
                << ab << ' '
                << w2 << ' '
                << dqpar << ' '
                << dqperp << ' '
                << xipar << ' '
                << xiperp << ' '
                << chi_t << ' '
                << chi_w2_t << ' '
                << binder_t << '\n';
        }
    }

    void WriteG6rHierarchicalFile(
        int iD,
        int iB,
        const G6rSettings& g6r,
        const std::vector<Gx0DisorderCurve>& disorder_curves,
        const std::vector<std::vector<std::vector<double>>>& all_block_curves
    ) {
        const int nbins = G6rNumberOfBins(g6r);

        std::vector<std::vector<double>> disorder_samples;
        disorder_samples.reserve(disorder_curves.size());

        for (const auto& d : disorder_curves) {
            if (static_cast<int>(d.mean.size()) == nbins) {
                disorder_samples.push_back(d.mean);
            }
            else {
                disorder_samples.push_back(
                    std::vector<double>(nbins, std::numeric_limits<double>::quiet_NaN())
                );
            }
        }

        std::vector<double> mean;
        std::vector<double> sem_disorder;
        std::vector<double> std_disorder;
        std::vector<int> nvalid_disorder;

        MeanSemStdCurvesGeneral(
            disorder_samples,
            mean,
            sem_disorder,
            std_disorder,
            nvalid_disorder
        );

        std::vector<double> sem_block_mean(nbins, std::numeric_limits<double>::quiet_NaN());
        std::vector<double> std_block_mean(nbins, std::numeric_limits<double>::quiet_NaN());

        for (int b = 0; b < nbins; ++b) {
            double sum_sem = 0.0;
            double sum_std = 0.0;
            int n_sem = 0;
            int n_std = 0;

            for (const auto& d : disorder_curves) {
                if (b < static_cast<int>(d.sem_block.size()) && std::isfinite(d.sem_block[b])) {
                    sum_sem += d.sem_block[b];
                    ++n_sem;
                }
                if (b < static_cast<int>(d.std_block.size()) && std::isfinite(d.std_block[b])) {
                    sum_std += d.std_block[b];
                    ++n_std;
                }
            }

            if (n_sem > 0) {
                sem_block_mean[b] = sum_sem / static_cast<double>(n_sem);
            }
            if (n_std > 0) {
                std_block_mean[b] = sum_std / static_cast<double>(n_std);
            }
        }

        double total_frames = 0.0;
        double total_blocks = 0.0;
        double weighted_nsk_sum = 0.0;

        for (const auto& d : disorder_curves) {
            total_frames += static_cast<double>(d.selected_frames);
            total_blocks += static_cast<double>(d.n_blocks);
            weighted_nsk_sum += static_cast<double>(d.selected_frames) * d.avg_nsk;
        }

        const double avg_nsk =
            total_frames > 0.0 ? weighted_nsk_sum / total_frames : 0.0;

        const double avg_blocks =
            disorder_curves.empty() ? 0.0 :
            total_blocks / static_cast<double>(disorder_curves.size());

        const double avg_frames =
            disorder_curves.empty() ? 0.0 :
            total_frames / static_cast<double>(disorder_curves.size());

        std::ostringstream filename;
        filename << "G6r_Measures_iD" << iD << "_iB" << iB << ".txt";

        std::ofstream output(filename.str());
        if (!output) {
            throw std::runtime_error("Cannot write G6r measurement file.");
        }

        output << "# r_over_ask ReG6_mean SEM_disorder STD_disorder "
            << "N_disorder_samples SEM_block_mean STD_block_mean "
            << "N_measure_blocks_avg N_selected_frames_avg\n";
        output << "# G6(r)=<Re[psi6_i conj(psi6_j)]> using unordered pairs i<j.\n";
        output << "# Hierarchical average: selected measures -> measure blocks -> disorder samples.\n";
        output << "# calculate_g6r " << (g6r.enabled ? "true" : "false") << "\n";
        output << "# g6r_measure_stride " << g6r.measure_stride << "\n";
        output << "# g6r_block_size " << g6r.block_size << "\n";
        output << "# write_g6r_block_samples " << (g6r.write_block_samples ? "true" : "false") << "\n";
        output << "# g6r_rmax " << g6r.rmax << "\n";
        output << "# g6r_rbin " << g6r.rbin << "\n";
        output << "# disorder_samples " << disorder_curves.size() << "\n";
        output << "# total_g6r_frames " << static_cast<long long>(total_frames) << "\n";
        output << "# average_Nsk " << std::setprecision(16) << avg_nsk << "\n";
        output << "# average_measure_blocks " << avg_blocks << "\n";
        output << "# average_selected_frames_per_disorder " << avg_frames << "\n";

        for (int b = 0; b < nbins; ++b) {
            const double rmid = (static_cast<double>(b) + 0.5) * g6r.rbin;

            output << std::setprecision(16)
                << rmid << ' '
                << mean[b] << ' '
                << sem_disorder[b] << ' '
                << std_disorder[b] << ' '
                << nvalid_disorder[b] << ' '
                << sem_block_mean[b] << ' '
                << std_block_mean[b] << ' '
                << avg_blocks << ' '
                << avg_frames << '\n';
        }

        std::ostringstream disorder_filename;
        disorder_filename << "G6r_DisorderSamples_iD" << iD << "_iB" << iB << ".txt";

        std::ofstream dout(disorder_filename.str());
        if (!dout) {
            throw std::runtime_error("Cannot write G6r disorder-sample file.");
        }

        dout << "# disorder_bin r_over_ask ReG6_disorder SEM_block STD_block "
            << "N_blocks selected_frames avg_Nsk min_Nsk max_Nsk\n";

        for (std::size_t id = 0; id < disorder_curves.size(); ++id) {
            const auto& d = disorder_curves[id];

            for (int b = 0; b < nbins; ++b) {
                const double rmid = (static_cast<double>(b) + 0.5) * g6r.rbin;

                const double gv =
                    b < static_cast<int>(d.mean.size()) ?
                    d.mean[b] :
                    std::numeric_limits<double>::quiet_NaN();

                const double se =
                    b < static_cast<int>(d.sem_block.size()) ?
                    d.sem_block[b] :
                    std::numeric_limits<double>::quiet_NaN();

                const double sd =
                    b < static_cast<int>(d.std_block.size()) ?
                    d.std_block[b] :
                    std::numeric_limits<double>::quiet_NaN();

                dout << id << ' '
                    << std::setprecision(16) << rmid << ' '
                    << gv << ' '
                    << se << ' '
                    << sd << ' '
                    << d.n_blocks << ' '
                    << d.selected_frames << ' '
                    << d.avg_nsk << ' '
                    << d.min_nsk << ' '
                    << d.max_nsk << '\n';
            }

            dout << '\n';
        }

        if (g6r.write_block_samples) {
            std::ostringstream block_filename;
            block_filename << "G6r_BlockSamples_iD" << iD << "_iB" << iB << ".txt";

            std::ofstream bout(block_filename.str());
            if (!bout) {
                throw std::runtime_error("Cannot write G6r block-sample file.");
            }

            bout << "# disorder_bin measure_block r_over_ask ReG6_block\n";

            for (std::size_t id = 0; id < all_block_curves.size(); ++id) {
                for (std::size_t iblock = 0; iblock < all_block_curves[id].size(); ++iblock) {
                    const auto& curve = all_block_curves[id][iblock];

                    for (int b = 0; b < nbins; ++b) {
                        const double rmid = (static_cast<double>(b) + 0.5) * g6r.rbin;
                        const double gv =
                            b < static_cast<int>(curve.size()) ?
                            curve[b] :
                            std::numeric_limits<double>::quiet_NaN();

                        bout << id << ' '
                            << iblock << ' '
                            << std::setprecision(16) << rmid << ' '
                            << gv << '\n';
                    }

                    bout << '\n';
                }
            }
        }

        std::ostringstream debug_filename;
        debug_filename << "G6r_Debug_iD" << iD << "_iB" << iB << ".txt";

        std::ofstream dbg(debug_filename.str());
        if (!dbg) {
            throw std::runtime_error("Cannot write G6r debug file.");
        }

        dbg << "# disorder_bin selected_g6r_frames n_measure_blocks avg_Nsk min_Nsk max_Nsk\n";
        for (std::size_t id = 0; id < disorder_curves.size(); ++id) {
            const auto& d = disorder_curves[id];
            dbg << id << ' '
                << d.selected_frames << ' '
                << d.n_blocks << ' '
                << std::setprecision(16) << d.avg_nsk << ' '
                << d.min_nsk << ' '
                << d.max_nsk << '\n';
        }
    }

    std::vector<Point2> MakeG6tGridPoints(
        int grid_nx,
        int grid_ny,
        int lx,
        int ly
    ) {
        std::vector<Point2> grid;
        grid.reserve(static_cast<std::size_t>(grid_nx) * static_cast<std::size_t>(grid_ny));

        for (int iy = 0; iy < grid_ny; ++iy) {
            for (int ix = 0; ix < grid_nx; ++ix) {
                const double f1 = (static_cast<double>(ix) + 0.5) / static_cast<double>(grid_nx);
                const double f2 = (static_cast<double>(iy) + 0.5) / static_cast<double>(grid_ny);
                grid.push_back(FracToReal({ f1, f2 }, lx, ly));
            }
        }

        return grid;
    }

    std::vector<std::complex<double>> InterpolatePsiToGridNearest(
        const OrientFrame& frame,
        const std::vector<Point2>& grid,
        int lx,
        int ly,
        const G6tSettings& g6t
    ) {
        const double nan = std::numeric_limits<double>::quiet_NaN();
        std::vector<std::complex<double>> out(
            grid.size(),
            std::complex<double>(nan, nan)
        );

        const int n = frame.nsk;
        if (n <= 0 || frame.psi6.size() != frame.centers.size()) {
            return out;
        }

        const double cutoff =
            (g6t.max_assign_dist_factor > 0.0 && frame.a_sk > 0.0 && std::isfinite(frame.a_sk)) ?
            g6t.max_assign_dist_factor * frame.a_sk :
            -1.0;

        const double cutoff2 = cutoff > 0.0 ? cutoff * cutoff : -1.0;

        for (std::size_t ig = 0; ig < grid.size(); ++ig) {
            double best_r2 = std::numeric_limits<double>::infinity();
            int best_id = -1;

            for (int i = 0; i < n; ++i) {
                const auto z = frame.psi6[i];
                if (!std::isfinite(z.real()) || !std::isfinite(z.imag())) {
                    continue;
                }

                const Point2 dr = MinImageDeltaReal(grid[ig], frame.centers[i], lx, ly);
                const double r2 = dr.x * dr.x + dr.y * dr.y;

                if (r2 < best_r2) {
                    best_r2 = r2;
                    best_id = i;
                }
            }

            if (best_id >= 0) {
                if (cutoff2 > 0.0 && best_r2 > cutoff2) {
                    continue;
                }
                out[ig] = frame.psi6[best_id];
            }
        }

        return out;
    }

    std::vector<double> ComputeG6tOneDisorder(
        const std::vector<std::vector<std::complex<double>>>& fields,
        int max_lag
    ) {
        const int T = static_cast<int>(fields.size());
        if (T <= 0) {
            return {};
        }

        int nlag = T;
        if (max_lag > 0) {
            nlag = std::min(nlag, max_lag + 1);
        }

        const int ngrid = static_cast<int>(fields[0].size());
        std::vector<double> curve(nlag, std::numeric_limits<double>::quiet_NaN());

        for (int lag = 0; lag < nlag; ++lag) {
            double sum_time = 0.0;
            int n_time = 0;

            for (int t0 = 0; t0 + lag < T; ++t0) {
                if (static_cast<int>(fields[t0].size()) != ngrid ||
                    static_cast<int>(fields[t0 + lag].size()) != ngrid) {
                    continue;
                }

                double sum_grid = 0.0;
                int n_grid = 0;

                const auto& a = fields[t0];
                const auto& b = fields[t0 + lag];

                for (int ig = 0; ig < ngrid; ++ig) {
                    const auto za = a[ig];
                    const auto zb = b[ig];

                    if (!std::isfinite(za.real()) || !std::isfinite(za.imag()) ||
                        !std::isfinite(zb.real()) || !std::isfinite(zb.imag())) {
                        continue;
                    }

                    const double denom = std::norm(za) + std::norm(zb);
                    if (!(denom > 0.0) || !std::isfinite(denom)) {
                        continue;
                    }

                    const double prod_re = za.real() * zb.real() + za.imag() * zb.imag();
                    const double val = 2.0 * prod_re / denom;

                    if (std::isfinite(val)) {
                        sum_grid += val;
                        ++n_grid;
                    }
                }

                if (n_grid > 0) {
                    sum_time += sum_grid / static_cast<double>(n_grid);
                    ++n_time;
                }
            }

            if (n_time > 0) {
                curve[lag] = sum_time / static_cast<double>(n_time);
            }
        }

        return curve;
    }

    void WriteG6tFile(
        int iD,
        int iB,
        const G6tSettings& g6t,
        const std::vector<std::vector<double>>& disorder_curves,
        const std::vector<int>& selected_frames,
        const std::vector<double>& avg_nsk
    ) {
        std::size_t max_curve_size = 0;
        for (const auto& c : disorder_curves) {
            max_curve_size = std::max(max_curve_size, c.size());
        }

        std::vector<std::vector<double>> padded_curves;
        padded_curves.reserve(disorder_curves.size());
        for (const auto& c : disorder_curves) {
            std::vector<double> p(max_curve_size, std::numeric_limits<double>::quiet_NaN());
            for (std::size_t i = 0; i < c.size(); ++i) {
                p[i] = c[i];
            }
            padded_curves.push_back(std::move(p));
        }

        std::vector<double> mean;
        std::vector<double> sem;
        std::vector<double> stddev;
        std::vector<int> nvalid;

        MeanSemStdCurvesGeneral(padded_curves, mean, sem, stddev, nvalid);

        double frames_sum = 0.0;
        double weighted_nsk_sum = 0.0;
        for (std::size_t i = 0; i < selected_frames.size(); ++i) {
            frames_sum += static_cast<double>(selected_frames[i]);
            weighted_nsk_sum += static_cast<double>(selected_frames[i]) * avg_nsk[i];
        }

        const double avg_frames =
            selected_frames.empty() ? 0.0 :
            frames_sum / static_cast<double>(selected_frames.size());

        const double avg_nsk_all =
            frames_sum > 0.0 ? weighted_nsk_sum / frames_sum : 0.0;

        std::ostringstream filename;
        filename << "G6t_Measures_iD" << iD << "_iB" << iB << ".txt";

        std::ofstream out(filename.str());
        if (!out) {
            throw std::runtime_error("Cannot write G6t measurement file.");
        }

        out << "# t G6t_mean SEM_disorder STD_disorder N_disorder_samples\n";
        out << "# G6(t) from fixed-grid nearest-neighbour interpolation of local psi6.\n";
        out << "# Symmetric normalized correlation: 2 Re[a conj(b)]/(|a|^2+|b|^2).\n";
        out << "# Averaging: temporal average inside each disorder sample, then disorder mean/SEM.\n";
        out << "# calculate_g6t " << (g6t.enabled ? "true" : "false") << "\n";
        out << "# g6t_measure_stride " << g6t.measure_stride << "\n";
        out << "# g6t_grid_nx " << g6t.grid_nx << "\n";
        out << "# g6t_grid_ny " << g6t.grid_ny << "\n";
        out << "# g6t_max_lag " << g6t.max_lag << "\n";
        out << "# g6t_dt " << g6t.dt << "\n";
        out << "# g6t_max_assign_dist_factor " << g6t.max_assign_dist_factor << "\n";
        out << "# disorder_samples " << disorder_curves.size() << "\n";
        out << "# average_selected_frames_per_disorder " << avg_frames << "\n";
        out << "# average_Nsk " << std::setprecision(16) << avg_nsk_all << "\n";

        for (std::size_t lag = 0; lag < mean.size(); ++lag) {
            const double t = static_cast<double>(lag) * g6t.dt;
            out << std::setprecision(16)
                << t << ' '
                << mean[lag] << ' '
                << sem[lag] << ' '
                << stddev[lag] << ' '
                << nvalid[lag] << '\n';
        }

        if (g6t.write_disorder_samples) {
            std::ostringstream dis_filename;
            dis_filename << "G6t_DisorderSamples_iD" << iD << "_iB" << iB << ".txt";

            std::ofstream dout(dis_filename.str());
            if (!dout) {
                throw std::runtime_error("Cannot write G6t disorder-sample file.");
            }

            dout << "# disorder_bin t G6t_disorder selected_frames avg_Nsk\n";

            for (std::size_t id = 0; id < disorder_curves.size(); ++id) {
                const auto& curve = disorder_curves[id];

                for (std::size_t lag = 0; lag < curve.size(); ++lag) {
                    const double t = static_cast<double>(lag) * g6t.dt;
                    dout << id << ' '
                        << std::setprecision(16) << t << ' '
                        << curve[lag] << ' '
                        << selected_frames[id] << ' '
                        << avg_nsk[id] << '\n';
                }
                dout << '\n';
            }
        }

        std::ostringstream debug_filename;
        debug_filename << "G6t_Debug_iD" << iD << "_iB" << iB << ".txt";

        std::ofstream dbg(debug_filename.str());
        if (!dbg) {
            throw std::runtime_error("Cannot write G6t debug file.");
        }

        dbg << "# disorder_bin selected_g6t_frames avg_Nsk\n";
        for (std::size_t id = 0; id < selected_frames.size(); ++id) {
            dbg << id << ' '
                << selected_frames[id] << ' '
                << std::setprecision(16) << avg_nsk[id] << '\n';
        }
    }





    // ============================================================
    // Reciprocal-space entropy / peak-shape diagnostics from S_z(q)
    //
    // These observables are extracted from the first diffraction ring.
    // They are meant to distinguish:
    //   SkX       : sharp six Bragg peaks
    //   BrG-like  : six peaks preserved but broadened/roughened
    //   SkG       : broad spots/ring-like reciprocal pattern
    //
    // Output files written by WriteStructureFactor:
    //   ReciprocalMetrics_Measures_iD*_iB*.txt
    //   ReciprocalMetrics_BinSamples_iD*_iB*.txt
    // ============================================================

    struct ReciprocalRingMetrics {
        double q0 = std::numeric_limits<double>::quiet_NaN();
        double psi6_ring = std::numeric_limits<double>::quiet_NaN();
        double sigma_q = std::numeric_limits<double>::quiet_NaN();
        double entropy_norm = std::numeric_limits<double>::quiet_NaN();
        double cv_theta = std::numeric_limits<double>::quiet_NaN();
        double c6_k = std::numeric_limits<double>::quiet_NaN();
        double c12_k = std::numeric_limits<double>::quiet_NaN();
        double f6_peak = std::numeric_limits<double>::quiet_NaN();
        double ipr_q = std::numeric_limits<double>::quiet_NaN();
        double n_eff_q = std::numeric_limits<double>::quiet_NaN();
        double peak_background_ratio = std::numeric_limits<double>::quiet_NaN();
        double dq_parallel_over_q0 = std::numeric_limits<double>::quiet_NaN();
        double dq_perp_over_q0 = std::numeric_limits<double>::quiet_NaN();
        double peak_spread = std::numeric_limits<double>::quiet_NaN();
        double a6_peak = std::numeric_limits<double>::quiet_NaN();
    };

    struct ReciprocalMetricStats {
        double mean = std::numeric_limits<double>::quiet_NaN();
        double sem = std::numeric_limits<double>::quiet_NaN();
        double stddev = std::numeric_limits<double>::quiet_NaN();
        int n = 0;
    };

    double SafeAngleDiff(double a, double b) {
        double d = std::fmod(a - b, 2.0 * PI);
        if (d > PI) {
            d -= 2.0 * PI;
        }
        if (d < -PI) {
            d += 2.0 * PI;
        }
        return d;
    }

    double PercentileFinite(std::vector<double> values, double percent) {
        values.erase(
            std::remove_if(
                values.begin(),
                values.end(),
                [](double x) { return !std::isfinite(x); }
            ),
            values.end()
        );
        if (values.empty()) {
            return std::numeric_limits<double>::quiet_NaN();
        }
        std::sort(values.begin(), values.end());
        const double p = std::min(100.0, std::max(0.0, percent)) / 100.0;
        const double pos = p * static_cast<double>(values.size() - 1);
        const int i0 = static_cast<int>(std::floor(pos));
        const int i1 = static_cast<int>(std::ceil(pos));
        if (i0 == i1) {
            return values[i0];
        }
        const double w = pos - static_cast<double>(i0);
        return (1.0 - w) * values[i0] + w * values[i1];
    }

    ReciprocalMetricStats StatsForReciprocalMetric(
        const std::vector<ReciprocalRingMetrics>& samples,
        double ReciprocalRingMetrics::* member
    ) {
        ReciprocalMetricStats out;
        std::vector<double> values;
        values.reserve(samples.size());
        for (const auto& s : samples) {
            const double v = s.*member;
            if (std::isfinite(v)) {
                values.push_back(v);
            }
        }
        out.n = static_cast<int>(values.size());
        if (out.n <= 0) {
            return out;
        }
        double sum = 0.0;
        for (const double v : values) {
            sum += v;
        }
        out.mean = sum / static_cast<double>(out.n);
        if (out.n == 1) {
            out.stddev = 0.0;
            out.sem = 0.0;
            return out;
        }
        double ss = 0.0;
        for (const double v : values) {
            const double d = v - out.mean;
            ss += d * d;
        }
        out.stddev = std::sqrt(ss / static_cast<double>(out.n - 1));
        out.sem = out.stddev / std::sqrt(static_cast<double>(out.n));
        return out;
    }

    double WeightedMeanFinite(
        const std::vector<double>& x,
        const std::vector<double>& w,
        const std::vector<int>& ids
    ) {
        double sw = 0.0;
        double sx = 0.0;
        for (const int id : ids) {
            if (id < 0 || id >= static_cast<int>(x.size()) || id >= static_cast<int>(w.size())) {
                continue;
            }
            if (std::isfinite(x[id]) && std::isfinite(w[id]) && w[id] > 0.0) {
                sx += w[id] * x[id];
                sw += w[id];
            }
        }
        return sw > 0.0 ? sx / sw : std::numeric_limits<double>::quiet_NaN();
    }

    double WeightedStdFinite(
        const std::vector<double>& x,
        const std::vector<double>& w,
        const std::vector<int>& ids
    ) {
        const double mu = WeightedMeanFinite(x, w, ids);
        if (!std::isfinite(mu)) {
            return std::numeric_limits<double>::quiet_NaN();
        }
        double sw = 0.0;
        double ss = 0.0;
        for (const int id : ids) {
            if (id < 0 || id >= static_cast<int>(x.size()) || id >= static_cast<int>(w.size())) {
                continue;
            }
            if (std::isfinite(x[id]) && std::isfinite(w[id]) && w[id] > 0.0) {
                const double d = x[id] - mu;
                ss += w[id] * d * d;
                sw += w[id];
            }
        }
        return sw > 0.0 ? std::sqrt(std::max(0.0, ss / sw)) : std::numeric_limits<double>::quiet_NaN();
    }

    ReciprocalRingMetrics ComputeReciprocalRingMetricsFromGrid(
        const std::vector<double>& intensity,
        int qx_count,
        int qy_count,
        int lx,
        int ly
    ) {
        ReciprocalRingMetrics out;
        const int n_q = qx_count * qy_count;
        if (static_cast<int>(intensity.size()) != n_q || n_q <= 0) {
            return out;
        }

        std::vector<double> qx_values(n_q, 0.0);
        std::vector<double> qy_values(n_q, 0.0);
        std::vector<double> r_values(n_q, 0.0);
        std::vector<double> theta_values(n_q, 0.0);
        std::vector<double> intens(n_q, 0.0);

        for (int ky_index = 0; ky_index < qy_count; ++ky_index) {
            const int ky_integer = ky_index - qy_count / 2;
            const double qy = 2.0 * PI * ky_integer / static_cast<double>(ly);
            for (int kx_index = 0; kx_index < qx_count; ++kx_index) {
                const int kx_integer = kx_index - qx_count / 2;
                const double qx = 2.0 * PI * kx_integer / static_cast<double>(lx);
                const int qindex = kx_index + ky_index * qx_count;
                const double r = std::sqrt(qx * qx + qy * qy);
                qx_values[qindex] = qx;
                qy_values[qindex] = qy;
                r_values[qindex] = r;
                theta_values[qindex] = std::atan2(qy, qx);
                const double raw = intensity[qindex];
                intens[qindex] = (std::isfinite(raw) && raw > 0.0) ? raw : 1.0e-12;
            }
        }

        // Locate the first diffraction ring q0 from the radial average.
        const double q_search_min = 0.60;
        const double q_search_max = 5.00;
        const int n_radial_bins = 100;
        std::vector<double> radial_sum(n_radial_bins, 0.0);
        std::vector<int> radial_count(n_radial_bins, 0);
        for (int i = 0; i < n_q; ++i) {
            const double r = r_values[i];
            if (!(r >= q_search_min && r <= q_search_max) || !std::isfinite(r)) {
                continue;
            }
            const double t = (r - q_search_min) / (q_search_max - q_search_min);
            int b = static_cast<int>(std::floor(t * static_cast<double>(n_radial_bins)));
            if (b < 0) {
                b = 0;
            }
            if (b >= n_radial_bins) {
                b = n_radial_bins - 1;
            }
            radial_sum[b] += intens[i];
            radial_count[b] += 1;
        }

        double best_mean = -1.0;
        int best_bin = -1;
        for (int b = 0; b < n_radial_bins; ++b) {
            if (radial_count[b] <= 0) {
                continue;
            }
            const double m = radial_sum[b] / static_cast<double>(radial_count[b]);
            if (m > best_mean) {
                best_mean = m;
                best_bin = b;
            }
        }
        if (best_bin < 0) {
            return out;
        }

        const double radial_bin_width = (q_search_max - q_search_min) / static_cast<double>(n_radial_bins);
        out.q0 = q_search_min + (static_cast<double>(best_bin) + 0.5) * radial_bin_width;
        const double ring_half_width = 0.40;

        std::vector<int> ring_ids;
        ring_ids.reserve(n_q);
        for (int i = 0; i < n_q; ++i) {
            const double r = r_values[i];
            if (r >= out.q0 - ring_half_width && r <= out.q0 + ring_half_width) {
                ring_ids.push_back(i);
            }
        }
        if (ring_ids.size() < 10) {
            return out;
        }

        // Ring radial width using raw positive S_z(q).
        const double q_mean = WeightedMeanFinite(r_values, intens, ring_ids);
        if (std::isfinite(q_mean)) {
            double sw = 0.0;
            double ss = 0.0;
            for (const int id : ring_ids) {
                const double w = intens[id];
                const double d = r_values[id] - q_mean;
                if (std::isfinite(w) && w > 0.0) {
                    ss += w * d * d;
                    sw += w;
                }
            }
            out.sigma_q = sw > 0.0 ? std::sqrt(std::max(0.0, ss / sw)) : std::numeric_limits<double>::quiet_NaN();
        }

        // Angular entropy and CV using binned raw ring intensity.
        const int n_theta_bins = 72;
        std::vector<double> theta_sum(n_theta_bins, 0.0);
        std::vector<int> theta_count(n_theta_bins, 0);
        for (const int id : ring_ids) {
            double t = (theta_values[id] + PI) / (2.0 * PI);
            t -= std::floor(t);
            int b = static_cast<int>(std::floor(t * static_cast<double>(n_theta_bins)));
            if (b < 0) {
                b = 0;
            }
            if (b >= n_theta_bins) {
                b = n_theta_bins - 1;
            }
            theta_sum[b] += intens[id];
            theta_count[b] += 1;
        }

        std::vector<double> theta_mean(n_theta_bins, 0.0);
        double theta_total = 0.0;
        double theta_avg = 0.0;
        for (int b = 0; b < n_theta_bins; ++b) {
            if (theta_count[b] > 0) {
                theta_mean[b] = theta_sum[b] / static_cast<double>(theta_count[b]);
            }
            theta_total += theta_mean[b];
            theta_avg += theta_mean[b];
        }
        theta_avg /= static_cast<double>(n_theta_bins);
        if (theta_total > 0.0) {
            double entropy = 0.0;
            for (const double v : theta_mean) {
                if (v > 0.0) {
                    const double p = v / theta_total;
                    entropy -= p * std::log(p);
                }
            }
            out.entropy_norm = entropy / std::log(static_cast<double>(n_theta_bins));
        }
        if (theta_avg > 0.0) {
            double ss = 0.0;
            for (const double v : theta_mean) {
                const double d = v - theta_avg;
                ss += d * d;
            }
            out.cv_theta = std::sqrt(ss / static_cast<double>(n_theta_bins)) / theta_avg;
        }

        // Background-subtracted weights on the ring.
        std::vector<double> ring_intens;
        ring_intens.reserve(ring_ids.size());
        for (const int id : ring_ids) {
            ring_intens.push_back(intens[id]);
        }
        const double bg = PercentileFinite(ring_intens, 5.0);
        std::vector<double> w_ring(n_q, 0.0);
        double wsum = 0.0;
        for (const int id : ring_ids) {
            double w = std::isfinite(bg) ? std::max(0.0, intens[id] - bg) : intens[id];
            if (!(w > 0.0)) {
                w = 0.0;
            }
            w_ring[id] = w;
            wsum += w;
        }
        if (!(wsum > 1.0e-12)) {
            for (const int id : ring_ids) {
                w_ring[id] = std::max(0.0, intens[id]);
                wsum += w_ring[id];
            }
        }
        if (!(wsum > 1.0e-12)) {
            return out;
        }

        double re6 = 0.0;
        double im6 = 0.0;
        double re12 = 0.0;
        double im12 = 0.0;
        for (const int id : ring_ids) {
            const double w = w_ring[id];
            const double th = theta_values[id];
            re6 += w * std::cos(6.0 * th);
            im6 += w * std::sin(6.0 * th);
            re12 += w * std::cos(12.0 * th);
            im12 += w * std::sin(12.0 * th);
        }
        out.c6_k = std::sqrt(re6 * re6 + im6 * im6) / wsum;
        out.c12_k = std::sqrt(re12 * re12 + im12 * im12) / wsum;
        out.psi6_ring = out.c6_k;

        double ipr = 0.0;
        for (const int id : ring_ids) {
            const double p = w_ring[id] / wsum;
            ipr += p * p;
        }
        out.ipr_q = ipr;
        out.n_eff_q = ipr > 0.0 ? 1.0 / ipr : std::numeric_limits<double>::quiet_NaN();

        const double phi0 = std::atan2(im6, re6) / 6.0;
        std::array<double, 6> centers{};
        for (int n = 0; n < 6; ++n) {
            centers[n] = phi0 + static_cast<double>(n) * PI / 3.0;
        }
        const double dtheta_peak = PI / 12.0;

        double peak_weight = 0.0;
        double peak_sum = 0.0;
        double bg_sum = 0.0;
        int peak_count = 0;
        int bg_count = 0;

        std::array<double, 6> peak_integrals{};
        std::array<int, 6> peak_point_counts{};
        std::array<double, 6> peak_spreads{};
        std::array<double, 6> peak_dqpar{};
        std::array<double, 6> peak_dqperp{};
        std::array<unsigned char, 6> peak_valid{};

        for (const int id : ring_ids) {
            double nearest = std::numeric_limits<double>::infinity();
            for (const double c : centers) {
                nearest = std::min(nearest, std::fabs(SafeAngleDiff(theta_values[id], c)));
            }
            if (nearest < dtheta_peak) {
                peak_weight += w_ring[id];
                peak_sum += w_ring[id];
                ++peak_count;
            }
            else {
                bg_sum += w_ring[id];
                ++bg_count;
            }
        }
        out.f6_peak = peak_weight / wsum;
        if (peak_count > 0 && bg_count > 0) {
            const double peak_mean = peak_sum / static_cast<double>(peak_count);
            const double bg_mean = bg_sum / static_cast<double>(bg_count);
            out.peak_background_ratio = bg_mean > 1.0e-12 ? peak_mean / bg_mean : std::numeric_limits<double>::quiet_NaN();
        }

        for (int ip = 0; ip < 6; ++ip) {
            std::vector<int> ids;
            ids.reserve(ring_ids.size());
            for (const int id : ring_ids) {
                if (std::fabs(SafeAngleDiff(theta_values[id], centers[ip])) < dtheta_peak) {
                    ids.push_back(id);
                }
            }
            if (ids.size() < 5) {
                continue;
            }

            double sw = 0.0;
            double maxw = 0.0;
            for (const int id : ids) {
                sw += w_ring[id];
                maxw = std::max(maxw, w_ring[id]);
            }
            if (!(sw > 1.0e-12) || !(maxw > 1.0e-12)) {
                continue;
            }

            const double cth = std::cos(centers[ip]);
            const double sth = std::sin(centers[ip]);
            std::vector<double> qpar(n_q, 0.0);
            std::vector<double> qperp(n_q, 0.0);
            for (const int id : ids) {
                qpar[id] = qx_values[id] * cth + qy_values[id] * sth - out.q0;
                qperp[id] = -qx_values[id] * sth + qy_values[id] * cth;
            }

            const double dpar = WeightedStdFinite(qpar, w_ring, ids);
            const double dperp = WeightedStdFinite(qperp, w_ring, ids);
            if (std::isfinite(dpar) && std::isfinite(dperp) && out.q0 > 1.0e-12) {
                peak_dqpar[ip] = dpar / out.q0;
                peak_dqperp[ip] = dperp / out.q0;
                peak_spreads[ip] = sw / maxw;
                peak_integrals[ip] = sw;
                peak_point_counts[ip] = static_cast<int>(ids.size());
                peak_valid[ip] = 1;
            }
        }

        double sum_dpar = 0.0;
        double sum_dperp = 0.0;
        double sum_spread = 0.0;
        int n_peak_valid = 0;
        std::vector<double> valid_integrals;
        for (int ip = 0; ip < 6; ++ip) {
            if (!peak_valid[ip]) {
                continue;
            }
            sum_dpar += peak_dqpar[ip];
            sum_dperp += peak_dqperp[ip];
            sum_spread += peak_spreads[ip];
            valid_integrals.push_back(peak_integrals[ip]);
            ++n_peak_valid;
        }
        if (n_peak_valid > 0) {
            out.dq_parallel_over_q0 = sum_dpar / static_cast<double>(n_peak_valid);
            out.dq_perp_over_q0 = sum_dperp / static_cast<double>(n_peak_valid);
            out.peak_spread = sum_spread / static_cast<double>(n_peak_valid);
        }
        if (valid_integrals.size() > 1) {
            double mean_int = 0.0;
            for (const double v : valid_integrals) {
                mean_int += v;
            }
            mean_int /= static_cast<double>(valid_integrals.size());
            if (mean_int > 1.0e-12) {
                double ss = 0.0;
                for (const double v : valid_integrals) {
                    const double d = v - mean_int;
                    ss += d * d;
                }
                out.a6_peak = std::sqrt(ss / static_cast<double>(valid_integrals.size() - 1)) / mean_int;
            }
        }

        return out;
    }

    void WriteReciprocalMetricsFiles(
        int iD,
        int iB,
        const std::vector<ReciprocalRingMetrics>& bin_metrics
    ) {
        auto stat = [&](double ReciprocalRingMetrics::* member) {
            return StatsForReciprocalMetric(bin_metrics, member);
        };

        const ReciprocalMetricStats q0 = stat(&ReciprocalRingMetrics::q0);
        const ReciprocalMetricStats psi6 = stat(&ReciprocalRingMetrics::psi6_ring);
        const ReciprocalMetricStats sigq = stat(&ReciprocalRingMetrics::sigma_q);
        const ReciprocalMetricStats sent = stat(&ReciprocalRingMetrics::entropy_norm);
        const ReciprocalMetricStats cvt = stat(&ReciprocalRingMetrics::cv_theta);
        const ReciprocalMetricStats c6 = stat(&ReciprocalRingMetrics::c6_k);
        const ReciprocalMetricStats c12 = stat(&ReciprocalRingMetrics::c12_k);
        const ReciprocalMetricStats f6 = stat(&ReciprocalRingMetrics::f6_peak);
        const ReciprocalMetricStats ipr = stat(&ReciprocalRingMetrics::ipr_q);
        const ReciprocalMetricStats neff = stat(&ReciprocalRingMetrics::n_eff_q);
        const ReciprocalMetricStats pbr = stat(&ReciprocalRingMetrics::peak_background_ratio);
        const ReciprocalMetricStats dqpar = stat(&ReciprocalRingMetrics::dq_parallel_over_q0);
        const ReciprocalMetricStats dqperp = stat(&ReciprocalRingMetrics::dq_perp_over_q0);
        const ReciprocalMetricStats spread = stat(&ReciprocalRingMetrics::peak_spread);
        const ReciprocalMetricStats a6 = stat(&ReciprocalRingMetrics::a6_peak);

        std::ostringstream filename;
        filename << "ReciprocalMetrics_Measures_iD" << iD << "_iB" << iB << ".txt";
        std::ofstream output(filename.str());
        if (!output) {
            throw std::runtime_error("Cannot write reciprocal metrics file.");
        }

        output << "# Reciprocal-space diagnostics extracted from S_z(q) first diffraction ring.\n";
        output << "# SkX->BrG-like: dQpar/q0, dQperp/q0, PeakSpread, N_eff_q should be most sensitive.\n";
        output << "# BrG-like->SkG: S_norm, C6_k, f6_peak, PBR, A6_peak should be most sensitive.\n";
        output << "# Columns:\n";
        output << "# "
            << "q0 q0_SEM q0_STD q0_N "
            << "Psi6_ring Psi6_ring_SEM Psi6_ring_STD Psi6_ring_N "
            << "SigmaQ SigmaQ_SEM SigmaQ_STD SigmaQ_N "
            << "S_norm S_norm_SEM S_norm_STD S_norm_N "
            << "CV_theta CV_theta_SEM CV_theta_STD CV_theta_N "
            << "C6_k C6_k_SEM C6_k_STD C6_k_N "
            << "C12_k C12_k_SEM C12_k_STD C12_k_N "
            << "f6_peak f6_peak_SEM f6_peak_STD f6_peak_N "
            << "IPR_q IPR_q_SEM IPR_q_STD IPR_q_N "
            << "N_eff_q N_eff_q_SEM N_eff_q_STD N_eff_q_N "
            << "PBR PBR_SEM PBR_STD PBR_N "
            << "dQpar_over_q0 dQpar_over_q0_SEM dQpar_over_q0_STD dQpar_over_q0_N "
            << "dQperp_over_q0 dQperp_over_q0_SEM dQperp_over_q0_STD dQperp_over_q0_N "
            << "PeakSpread PeakSpread_SEM PeakSpread_STD PeakSpread_N "
            << "A6_peak A6_peak_SEM A6_peak_STD A6_peak_N\n";

        auto write_stat = [&](const ReciprocalMetricStats& s) {
            output << s.mean << ' ' << s.sem << ' ' << s.stddev << ' ' << s.n << ' ';
        };

        output << std::setprecision(16);
        write_stat(q0);
        write_stat(psi6);
        write_stat(sigq);
        write_stat(sent);
        write_stat(cvt);
        write_stat(c6);
        write_stat(c12);
        write_stat(f6);
        write_stat(ipr);
        write_stat(neff);
        write_stat(pbr);
        write_stat(dqpar);
        write_stat(dqperp);
        write_stat(spread);
        output << a6.mean << ' ' << a6.sem << ' ' << a6.stddev << ' ' << a6.n << '\n';

        std::ostringstream sample_filename;
        sample_filename << "ReciprocalMetrics_BinSamples_iD" << iD << "_iB" << iB << ".txt";
        std::ofstream samples(sample_filename.str());
        if (!samples) {
            throw std::runtime_error("Cannot write reciprocal metrics bin-sample file.");
        }
        samples << "# bin q0 Psi6_ring SigmaQ S_norm CV_theta C6_k C12_k f6_peak IPR_q N_eff_q PBR "
            << "dQpar_over_q0 dQperp_over_q0 PeakSpread A6_peak\n";
        for (std::size_t ib = 0; ib < bin_metrics.size(); ++ib) {
            const auto& m = bin_metrics[ib];
            samples << std::setprecision(16)
                << ib << ' '
                << m.q0 << ' '
                << m.psi6_ring << ' '
                << m.sigma_q << ' '
                << m.entropy_norm << ' '
                << m.cv_theta << ' '
                << m.c6_k << ' '
                << m.c12_k << ' '
                << m.f6_peak << ' '
                << m.ipr_q << ' '
                << m.n_eff_q << ' '
                << m.peak_background_ratio << ' '
                << m.dq_parallel_over_q0 << ' '
                << m.dq_perp_over_q0 << ' '
                << m.peak_spread << ' '
                << m.a6_peak << '\n';
        }
    }

} // namespace

Square_Lattice::Square_Lattice(int lx, int ly)
    : Lx_(lx), Ly_(ly) {
    if (Lx_ <= 0 || Ly_ <= 0) {
        throw std::runtime_error("Lattice dimensions must be positive.");
    }
}

void Square_Lattice::NN_sites(int site, int* neighbors) const {
    const int x = site % Lx_;
    const int y = site / Lx_;

    const auto mod = [](int value, int size) {
        return (value % size + size) % size;
    };

    const auto index = [&](int xx, int yy) {
        return mod(xx, Lx_) + mod(yy, Ly_) * Lx_;
    };

    neighbors[0] = index(x, y + 1);
    neighbors[1] = index(x + 1, y);
    neighbors[2] = index(x, y - 1);
    neighbors[3] = index(x - 1, y);
    neighbors[4] = index(x - 1, y + 1);
    neighbors[5] = index(x + 1, y - 1);

    neighbors[6] = index(x - 1, y - 1);
    neighbors[7] = index(x - 2, y + 1);
    neighbors[8] = index(x - 1, y + 2);
    neighbors[9] = index(x + 1, y + 1);
    neighbors[10] = index(x + 2, y - 1);
    neighbors[11] = index(x + 1, y - 2);

    neighbors[12] = index(x, y + 2);
    neighbors[13] = index(x + 2, y);
    neighbors[14] = index(x, y - 2);
    neighbors[15] = index(x - 2, y);
    neighbors[16] = index(x - 2, y + 2);
    neighbors[17] = index(x + 2, y - 2);
}

MC_Ising::MC_Ising(Square_Lattice* lattice)
    : lattice_(lattice),
    Lx_(lattice->get_Lx()),
    Ly_(lattice->get_Ly()),
    N_(Lx_* Ly_) {
    ReadSettings();
    BuildNeighborCache();
    WriteScanParameters();

#ifdef _OPENMP
    omp_set_dynamic(0);

    const char* omp_env = std::getenv("OMP_NUM_THREADS");
    if (omp_env != nullptr) {
        const int requested_threads = std::atoi(omp_env);
        if (requested_threads > 0) {
            omp_set_num_threads(requested_threads);
        }
    }

    std::cout << "OpenMP enabled. max threads = "
              << omp_get_max_threads() << '\n';

    if (settings_.n_bins < omp_get_max_threads()) {
        std::cout << "Warning: n_bins = " << settings_.n_bins
                  << " is smaller than OpenMP threads = "
                  << omp_get_max_threads()
                  << ". Some threads may be idle. For full use, set n_bins >= OMP_NUM_THREADS.\n";
    }
#else
    std::cout << "OpenMP is NOT enabled. Recompile with -fopenmp or /openmp.\n";
#endif

    std::cout << "Sites = " << N_ << ", bins = " << settings_.n_bins << '\n';
    std::cout << "Random restarts per parameter point = "
        << settings_.random_restarts << '\n';
    std::cout << "MC update = heat-bath single-spin sampler + overrelaxation (MetropolisSweep name kept for compatibility).\n";

    std::vector<DisorderRealization> base_disorder(settings_.n_bins);
    for (int ibin = 0; ibin < settings_.n_bins; ++ibin) {
        std::mt19937_64 disorder_rng(MakeSeed(settings_.seed, 0, ibin, 101));
        GenerateDisorder(base_disorder[ibin], disorder_rng);
    }

    std::vector<BondMatrix> all_J(settings_.n_bins, BondMatrix(N_));
    std::vector<BondMatrix> all_D(settings_.n_bins, BondMatrix(N_));
    std::vector<SpinConfig> all_spins(settings_.n_bins, SpinConfig(N_));

    std::vector<std::mt19937_64> dynamics_rng;
    dynamics_rng.reserve(settings_.n_bins);

    for (int ibin = 0; ibin < settings_.n_bins; ++ibin) {
        dynamics_rng.emplace_back(MakeSeed(settings_.seed, 0, ibin, 202));
    }

    bool have_previous_parameter_state = false;

    for (std::size_t iD = 0; iD < disorder_values_.size(); ++iD) {
        const double delta_D = disorder_values_[iD];

#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1)
#endif
        for (int ibin = 0; ibin < settings_.n_bins; ++ibin) {
            BuildInteractions(delta_D, base_disorder[ibin], all_J[ibin], all_D[ibin]);
        }

        for (std::size_t iB = 0; iB < field_values_.size(); ++iB) {
            const double field = field_values_[iB];

            std::cout << "\n>>> iD=" << iD << "/" << disorder_values_.size() - 1
                << ", iB=" << iB << "/" << field_values_.size() - 1
                << ", delta_D=" << delta_D
                << ", B=" << field << '\n';

            const auto warmup_start = std::chrono::steady_clock::now();

#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1)
#endif
            for (int ibin = 0; ibin < settings_.n_bins; ++ibin) {
                const bool include_existing_state =
                    have_previous_parameter_state || (iB > 0);

                OptimizeState(
                    field,
                    all_spins[ibin],
                    all_J[ibin],
                    all_D[ibin],
                    dynamics_rng[ibin],
                    include_existing_state
                );
            }

            const auto warmup_end = std::chrono::steady_clock::now();

            MeasureParameterPoint(
                static_cast<int>(iD),
                static_cast<int>(iB),
                field,
                all_spins,
                all_J,
                all_D,
                dynamics_rng
            );

            const auto measure_end = std::chrono::steady_clock::now();

            const double warmup_seconds =
                std::chrono::duration<double>(warmup_end - warmup_start).count();

            const double measure_seconds =
                std::chrono::duration<double>(measure_end - warmup_end).count();

            std::cout << "Warmup: " << warmup_seconds
                << " s, measure: " << measure_seconds << " s\n";

            have_previous_parameter_state = true;
        }
    }
}

void MC_Ising::ReadSettings() {
    GetParaFromInput_int("input.in", "n_warmup", settings_.n_warmup);
    GetParaFromInput_int("input.in", "n_measure", settings_.n_measure);
    GetParaFromInput_int("input.in", "n_bins", settings_.n_bins);

    TryGetParaFromInput_int("input.in", "random_restarts", settings_.random_restarts);
    TryGetParaFromInput_int("input.in", "observable_stride", settings_.observable_stride);
    TryGetParaFromInput_int("input.in", "snapshot_stride", settings_.snapshot_stride);
    TryGetParaFromInput_int("input.in", "overrelax_sweeps", settings_.overrelax_sweeps);
    TryGetParaFromInput_int("input.in", "max_low_t_sweeps", settings_.max_low_t_sweeps);
    TryGetParaFromInput_int("input.in", "stable_checks_required", settings_.stable_checks_required);

    int seed_as_int = static_cast<int>(settings_.seed);
    if (TryGetParaFromInput_int("input.in", "seed", seed_as_int)) {
        settings_.seed = static_cast<std::uint64_t>(seed_as_int);
    }

    TryGetParaFromInput_real("input.in", "T_target", settings_.target_temperature);
    TryGetParaFromInput_real("input.in", "T_anneal_start", settings_.anneal_start_temperature);
    TryGetParaFromInput_real("input.in", "anneal_rate", settings_.anneal_rate);
    TryGetParaFromInput_real("input.in", "energy_tolerance_per_site", settings_.energy_tolerance_per_site);

    TryGetParaFromInput_real("input.in", "J1", settings_.J1);
    TryGetParaFromInput_real("input.in", "D1", settings_.D1);
    TryGetParaFromInput_real("input.in", "J2", settings_.J2);
    TryGetParaFromInput_real("input.in", "D2", settings_.D2);
    TryGetParaFromInput_real("input.in", "J3", settings_.J3);
    TryGetParaFromInput_real("input.in", "delta_J1", settings_.delta_J1);
    TryGetParaFromInput_real("input.in", "delta_J2", settings_.delta_J2);

    TryGetParaFromInput_bool("input.in", "write_snapshots", settings_.write_snapshots);
    TryGetParaFromInput_bool("input.in", "calculate_structure_factor", settings_.calculate_structure_factor);

    disorder_values_ = ReadListOrDefault("input.in", "D_values", DefaultDisorderValues());
    field_values_ = ReadListOrDefault("input.in", "B_values", DefaultFieldValues());

    if (settings_.n_warmup <= 0 || settings_.n_measure <= 0 || settings_.n_bins <= 0) {
        throw std::runtime_error("n_warmup, n_measure and n_bins must be positive.");
    }

    if (settings_.observable_stride <= 0 || settings_.snapshot_stride <= 0) {
        throw std::runtime_error("observable_stride and snapshot_stride must be positive.");
    }

    if (!(settings_.anneal_rate > 0.0 && settings_.anneal_rate < 1.0)) {
        throw std::runtime_error("anneal_rate must satisfy 0 < anneal_rate < 1.");
    }

    if (settings_.target_temperature <= 0.0 ||
        settings_.anneal_start_temperature <= 0.0) {
        throw std::runtime_error("Temperatures must be positive.");
    }
}

void MC_Ising::BuildNeighborCache() {
    neighbors_.resize(N_);

    for (int site = 0; site < N_; ++site) {
        lattice_->NN_sites(site, neighbors_[site].data());
    }
}

void MC_Ising::WriteScanParameters() const {
    std::ofstream output("Scan_Parameters.txt");
    if (!output) {
        throw std::runtime_error("Cannot write Scan_Parameters.txt.");
    }

    output << "Index_D Value_D\n";
    for (std::size_t i = 0; i < disorder_values_.size(); ++i) {
        output << i << ' ' << disorder_values_[i] << '\n';
    }

    output << "\nIndex_B Value_B\n";
    for (std::size_t i = 0; i < field_values_.size(); ++i) {
        output << i << ' ' << field_values_[i] << '\n';
    }
}

void MC_Ising::GenerateDisorder(
    DisorderRealization& disorder,
    std::mt19937_64& rng
) const {
    disorder.rho.assign(N_, BondRow{});

    std::uniform_real_distribution<double> uniform_minus1_plus1(-1.0, 1.0);

    for (int i = 0; i < N_; ++i) {
        for (int n = 0; n < 12; ++n) {
            const int j = neighbors_[i][n];

            if (j <= i) {
                continue;
            }

            const double rho = uniform_minus1_plus1(rng);
            disorder.rho[i][n] = rho;

            bool reverse_found = false;
            for (int reverse_n = 0; reverse_n < 12; ++reverse_n) {
                if (neighbors_[j][reverse_n] == i) {
                    disorder.rho[j][reverse_n] = rho;
                    reverse_found = true;
                    break;
                }
            }

            if (!reverse_found) {
                throw std::runtime_error("Reverse bond is missing in neighbor table.");
            }
        }
    }
}

void MC_Ising::BuildInteractions(
    double delta_D,
    const DisorderRealization& disorder,
    BondMatrix& J,
    BondMatrix& D
) const {
    J.assign(N_, BondRow{});
    D.assign(N_, BondRow{});

    for (int i = 0; i < N_; ++i) {
        for (int n = 0; n < 6; ++n) {
            const double rho = disorder.rho[i][n];
            J[i][n] = settings_.J1 * (1.0 + settings_.delta_J1 * rho);
            D[i][n] = settings_.D1 * (1.0 + delta_D * rho);
        }

        for (int n = 6; n < 12; ++n) {
            const double rho = disorder.rho[i][n];
            J[i][n] = settings_.J2 * (1.0 + settings_.delta_J2 * rho);
            D[i][n] = settings_.D2 * (1.0 + delta_D * rho);
        }
    }
}

void MC_Ising::RandomizeSpins(
    SpinConfig& spins,
    std::mt19937_64& rng
) const {
    spins.resize(N_);

    for (Spin3& spin : spins) {
        spin = RandomUnitVector(rng);
    }
}

double MC_Ising::MetropolisSweep(
    double temperature,
    double field,
    SpinConfig& spins,
    const BondMatrix& J,
    const BondMatrix& D,
    std::mt19937_64& rng
) const {
    // High-efficiency heat-bath update.
    //
    // The function name is kept as MetropolisSweep to remain compatible
    // with the existing mc_Ising.h declarations and all call sites.
    //
    // For fixed neighbours, the single-spin energy is
    //     E_i = - S_i . H_i + const.
    // Therefore the correct conditional distribution is
    //     P(S_i) propto exp(beta S_i . H_i).
    //
    // Local field convention is consistent with OverrelaxationSweep:
    //   J term       H += -J_ij S_j       because E contains +J S_i.S_j
    //   DMI term     H +=  D_ij e_ij x S_j because E contains D e_ij.(S_i x S_j)
    //   field term   H += (0,0,B)
    //   J3 term      H += -J3 S_j
    (void)rng; // keeps static analyzers quiet if the compiler inlines the helper unexpectedly.

    std::uniform_int_distribution<int> random_start(0, N_ - 1);
    const int first_site = random_start(rng);

    for (int step = 0; step < N_; ++step) {
        const int i = (first_site + step) % N_;

        Spin3 local_field{ 0.0, 0.0, field };

        for (int n = 0; n < 6; ++n) {
            const int j = neighbors_[i][n];
            local_field = local_field + (-J[i][n]) * spins[j];
            local_field = local_field + D[i][n] * Cross(DMI_NN[n], spins[j]);
        }

        for (int n = 6; n < 12; ++n) {
            const int j = neighbors_[i][n];
            local_field = local_field + (-J[i][n]) * spins[j];
            local_field = local_field + D[i][n] * Cross(DMI_NNN[n - 6], spins[j]);
        }

        for (int n = 12; n < 18; ++n) {
            const int j = neighbors_[i][n];
            local_field = local_field + (-settings_.J3) * spins[j];
        }

        spins[i] = HeatBathSampleFromField(local_field, temperature, rng);
    }

    // Heat-bath has no reject step. Return 1.0 as an effective update ratio.
    return 1.0;
}

void MC_Ising::OverrelaxationSweep(
    double field,
    SpinConfig& spins,
    const BondMatrix& J,
    const BondMatrix& D
) const {
    for (int i = 0; i < N_; ++i) {
        Spin3 local_field{ 0.0, 0.0, field };

        for (int n = 0; n < 6; ++n) {
            const int j = neighbors_[i][n];

            local_field = local_field + (-J[i][n]) * spins[j];
            local_field = local_field + D[i][n] * Cross(DMI_NN[n], spins[j]);
        }

        for (int n = 6; n < 12; ++n) {
            const int j = neighbors_[i][n];

            local_field = local_field + (-J[i][n]) * spins[j];
            local_field = local_field + D[i][n] * Cross(DMI_NNN[n - 6], spins[j]);
        }

        for (int n = 12; n < 18; ++n) {
            const int j = neighbors_[i][n];

            local_field = local_field + (-settings_.J3) * spins[j];
        }

        const double h2 = NormSquared(local_field);
        if (h2 < 1.0e-24) {
            continue;
        }

        const double projection = Dot(spins[i], local_field);
        spins[i] = Normalize((2.0 * projection / h2) * local_field - spins[i]);
    }
}

void MC_Ising::AnnealOneState(
    double start_temperature,
    double field,
    SpinConfig& spins,
    const BondMatrix& J,
    const BondMatrix& D,
    std::mt19937_64& rng
) const {
    double temperature = start_temperature;

    while (temperature > settings_.target_temperature * 1.001) {
        int sweeps_this_temperature = 40;

        if (temperature < 0.1) {
            sweeps_this_temperature = std::max(80, settings_.n_warmup / 2);
        }

        for (int sweep = 0; sweep < sweeps_this_temperature; ++sweep) {
            MetropolisSweep(temperature, field, spins, J, D, rng);

            for (int k = 0; k < settings_.overrelax_sweeps; ++k) {
                OverrelaxationSweep(field, spins, J, D);
            }
        }

        temperature *= settings_.anneal_rate;
    }

    double previous_energy = Hamiltonian(field, spins, J, D);
    int stable_counter = 0;

    for (int sweep = 0; sweep < settings_.max_low_t_sweeps; ++sweep) {
        MetropolisSweep(settings_.target_temperature, field, spins, J, D, rng);

        for (int k = 0; k < settings_.overrelax_sweeps; ++k) {
            OverrelaxationSweep(field, spins, J, D);
        }

        if ((sweep + 1) % 20 == 0) {
            const double current_energy = Hamiltonian(field, spins, J, D);
            const double difference = std::fabs(current_energy - previous_energy);

            if (difference < settings_.energy_tolerance_per_site) {
                ++stable_counter;
            }
            else {
                stable_counter = 0;
            }

            previous_energy = current_energy;

            if (stable_counter >= settings_.stable_checks_required) {
                break;
            }
        }
    }
}

void MC_Ising::OptimizeState(
    double field,
    SpinConfig& spins,
    const BondMatrix& J,
    const BondMatrix& D,
    std::mt19937_64& rng,
    bool include_existing_state
) const {
    SpinConfig best_spins;
    double best_energy = std::numeric_limits<double>::infinity();

    const int total_candidates =
        std::max(1, settings_.random_restarts) + (include_existing_state ? 1 : 0);

    for (int candidate = 0; candidate < total_candidates; ++candidate) {
        SpinConfig trial_spins;

        if (candidate == 0 && include_existing_state && !spins.empty()) {
            trial_spins = spins;
        }
        else {
            RandomizeSpins(trial_spins, rng);
        }

        AnnealOneState(
            settings_.anneal_start_temperature,
            field,
            trial_spins,
            J,
            D,
            rng
        );

        const double energy = Hamiltonian(field, trial_spins, J, D);

        if (energy < best_energy) {
            best_energy = energy;
            best_spins = std::move(trial_spins);
        }
    }

    spins = std::move(best_spins);
}

double MC_Ising::Hamiltonian(
    double field,
    const SpinConfig& spins,
    const BondMatrix& J,
    const BondMatrix& D
) const {
    double energy = 0.0;

    for (const Spin3& spin : spins) {
        energy -= field * spin.z;
    }

    for (int i = 0; i < N_; ++i) {
        for (int n = 0; n < 6; ++n) {
            const int j = neighbors_[i][n];

            if (j > i) {
                energy += J[i][n] * Dot(spins[i], spins[j]);
                energy += D[i][n] * Dot(DMI_NN[n], Cross(spins[i], spins[j]));
            }
        }

        for (int n = 6; n < 12; ++n) {
            const int j = neighbors_[i][n];

            if (j > i) {
                energy += J[i][n] * Dot(spins[i], spins[j]);
                energy += D[i][n] * Dot(DMI_NNN[n - 6], Cross(spins[i], spins[j]));
            }
        }

        for (int n = 12; n < 18; ++n) {
            const int j = neighbors_[i][n];

            if (j > i) {
                energy += settings_.J3 * Dot(spins[i], spins[j]);
            }
        }
    }

    return energy / static_cast<double>(N_);
}

double MC_Ising::MagnetizationZ(const SpinConfig& spins) const {
    double mz = 0.0;

    for (const Spin3& spin : spins) {
        mz += spin.z;
    }

    return mz / static_cast<double>(N_);
}

double MC_Ising::SkyrmionNumber(const SpinConfig& spins) const {
    double total_solid_angle = 0.0;

    for (int i = 0; i < N_; ++i) {
        const int right = neighbors_[i][1];
        const int up = neighbors_[i][0];
        const int left = neighbors_[i][3];
        const int down = neighbors_[i][2];

        total_solid_angle += SolidAngle(spins[i], spins[right], spins[up]);
        total_solid_angle += SolidAngle(spins[i], spins[left], spins[down]);
    }

    return total_solid_angle / (4.0 * PI);
}

double MC_Ising::SublatticeSkyrmionOrder(const SpinConfig& spins) const {
    double q0 = 0.0;
    double q1 = 0.0;
    double q2 = 0.0;

    for (int y = 0; y < Ly_; ++y) {
        for (int x = 0; x < Lx_; ++x) {
            const int i = x + y * Lx_;
            const int sub = ((x - y) % 3 + 3) % 3;

            const int right = neighbors_[i][1];
            const int up = neighbors_[i][0];

            const double omega = SolidAngle(spins[i], spins[right], spins[up]);

            if (sub == 0) {
                q0 += omega;
            }
            else if (sub == 1) {
                q1 += omega;
            }
            else {
                q2 += omega;
            }
        }
    }

    q0 /= 4.0 * PI;
    q1 /= 4.0 * PI;
    q2 /= 4.0 * PI;

    return std::sqrt(q0 * q0 + q1 * q1 + q2 * q2);
}

double MC_Ising::ScalarChirality(const SpinConfig& spins) const {
    double chirality = 0.0;

    for (int i = 0; i < N_; ++i) {
        const int right = neighbors_[i][1];
        const int up = neighbors_[i][0];
        const int left = neighbors_[i][3];
        const int down = neighbors_[i][2];

        chirality += Dot(spins[i], Cross(spins[right], spins[up]));
        chirality += Dot(spins[i], Cross(spins[left], spins[down]));
    }

    return chirality / static_cast<double>(2 * N_);
}

double MC_Ising::Psi6Order(const SpinConfig& spins) const {
    const Phi6FrameResult phi =
        ComputePhi6FromSpinConfiguration(spins, Lx_, Ly_);

    return phi.abs_Phi6;
}


MC_Measurements MC_Ising::MeasureObservables(
    double field,
    const SpinConfig& spins,
    const BondMatrix& J,
    const BondMatrix& D,
    bool calculate_psi6
) const {
    MC_Measurements result;

    result.energy = Hamiltonian(field, spins, J, D);
    result.mz = MagnetizationZ(spins);
    result.skyrmion_number = SkyrmionNumber(spins);
    result.sublattice_sk_order = SublatticeSkyrmionOrder(spins);
    result.chirality = ScalarChirality(spins);
    result.psi6 = calculate_psi6 ? Psi6Order(spins) : 0.0;

    return result;
}


void MC_Ising::MeasureParameterPoint(
    int iD,
    int iB,
    double field,
    std::vector<SpinConfig>& all_spins,
    const std::vector<BondMatrix>& all_J,
    const std::vector<BondMatrix>& all_D,
    std::vector<std::mt19937_64>& dynamics_rng
) const {
    const Gx0Settings gx0 = LoadGx0Settings();
    const int gx0_block_size = LoadGx0BlockSize();
    const bool write_gx0_block_samples = LoadGx0WriteBlockSamples();

    const G6rSettings g6r = LoadG6rSettings();
    const G6tSettings g6t = LoadG6tSettings(Lx_, Ly_);
    const PhaseBoundarySettings phase = LoadPhaseBoundarySettings();
    const std::vector<Point2> g6t_grid =
        g6t.enabled ? MakeG6tGridPoints(g6t.grid_nx, g6t.grid_ny, Lx_, Ly_) :
        std::vector<Point2>();

    std::vector<MC_Measurements> bin_measurements(settings_.n_bins);
    std::vector<std::vector<Spin3>> snapshots_per_bin(settings_.n_bins);

    std::vector<double> bin_phi6_global(settings_.n_bins, std::numeric_limits<double>::quiet_NaN());
    std::vector<double> bin_phi6_local(settings_.n_bins, std::numeric_limits<double>::quiet_NaN());
    std::vector<double> bin_chi_phi6_global_time(settings_.n_bins, std::numeric_limits<double>::quiet_NaN());
    std::vector<double> bin_chi_phi6_local_time(settings_.n_bins, std::numeric_limits<double>::quiet_NaN());
    std::vector<double> bin_nsk_center(settings_.n_bins, std::numeric_limits<double>::quiet_NaN());

    // Measurement-averaged per-disorder phase diagnostics.
    std::vector<double> bin_rho_def(settings_.n_bins, std::numeric_limits<double>::quiet_NaN());
    std::vector<double> bin_n5_frac(settings_.n_bins, std::numeric_limits<double>::quiet_NaN());
    std::vector<double> bin_n6_frac(settings_.n_bins, std::numeric_limits<double>::quiet_NaN());
    std::vector<double> bin_n7_frac(settings_.n_bins, std::numeric_limits<double>::quiet_NaN());
    std::vector<double> bin_nnon6_frac(settings_.n_bins, std::numeric_limits<double>::quiet_NaN());

    std::vector<double> bin_dislocation_frac(settings_.n_bins, std::numeric_limits<double>::quiet_NaN());
    std::vector<double> bin_free_dislocation_frac(settings_.n_bins, std::numeric_limits<double>::quiet_NaN());
    std::vector<double> bin_bound_dislocation_frac(settings_.n_bins, std::numeric_limits<double>::quiet_NaN());
    std::vector<double> bin_rho_dislocation(settings_.n_bins, std::numeric_limits<double>::quiet_NaN());
    std::vector<double> bin_rho_free_dislocation(settings_.n_bins, std::numeric_limits<double>::quiet_NaN());
    std::vector<double> bin_xiD_free_over_ask(settings_.n_bins, std::numeric_limits<double>::quiet_NaN());

    std::vector<double> bin_gt_tail(settings_.n_bins, std::numeric_limits<double>::quiet_NaN());
    std::vector<double> bin_mt(settings_.n_bins, std::numeric_limits<double>::quiet_NaN());
    std::vector<double> bin_eta_T(settings_.n_bins, std::numeric_limits<double>::quiet_NaN());
    std::vector<double> bin_AB_roughness(settings_.n_bins, std::numeric_limits<double>::quiet_NaN());
    std::vector<double> bin_chi_mt_time(settings_.n_bins, std::numeric_limits<double>::quiet_NaN());
    std::vector<double> bin_binder_mt_time(settings_.n_bins, std::numeric_limits<double>::quiet_NaN());

    // Additional elastic-roughening and Bragg-peak-width diagnostics.
    std::vector<double> bin_phase_W2(settings_.n_bins, std::numeric_limits<double>::quiet_NaN());
    std::vector<double> bin_chi_W2_time(settings_.n_bins, std::numeric_limits<double>::quiet_NaN());
    std::vector<double> bin_dq_parallel(settings_.n_bins, std::numeric_limits<double>::quiet_NaN());
    std::vector<double> bin_dq_perp(settings_.n_bins, std::numeric_limits<double>::quiet_NaN());
    std::vector<double> bin_xi_parallel(settings_.n_bins, std::numeric_limits<double>::quiet_NaN());
    std::vector<double> bin_xi_perp(settings_.n_bins, std::numeric_limits<double>::quiet_NaN());

    const int gx0_nbins = Gx0NumberOfBins(gx0);
    std::vector<Gx0DisorderCurve> gx0_disorder_curves(settings_.n_bins);
    std::vector<std::vector<std::vector<double>>> gx0_all_block_curves(settings_.n_bins);

    const int g6r_nbins = G6rNumberOfBins(g6r);
    std::vector<Gx0DisorderCurve> g6r_disorder_curves(settings_.n_bins);
    std::vector<std::vector<std::vector<double>>> g6r_all_block_curves(settings_.n_bins);

    std::vector<std::vector<double>> g6t_disorder_curves(settings_.n_bins);
    std::vector<int> g6t_selected_frames(settings_.n_bins, 0);
    std::vector<double> g6t_avg_nsk(settings_.n_bins, 0.0);

    const int trans_nbins = TransNumberOfBins(phase);
    std::vector<Gx0DisorderCurve> trans_r_disorder_curves(settings_.n_bins);
    std::vector<std::vector<std::vector<double>>> trans_r_all_block_curves(settings_.n_bins);
    std::vector<std::vector<double>> trans_t_disorder_curves(settings_.n_bins);
    std::vector<int> trans_t_selected_frames(settings_.n_bins, 0);
    std::vector<double> trans_t_avg_nsk(settings_.n_bins, 0.0);

#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1)
#endif
    for (int ibin = 0; ibin < settings_.n_bins; ++ibin) {
        MC_Measurements accumulated;
        int measurement_count = 0;

        double sum_phi6_global = 0.0;
        double sum_phi6_local = 0.0;
        double sum_phi6_global_sq = 0.0;
        double sum_phi6_local_sq = 0.0;
        double sum_nsk_center = 0.0;

        // g(x,0)-1 block statistics
        std::vector<double> gx0_block_hist(gx0_nbins, 0.0);
        std::vector<double> gx0_block_ideal(gx0_nbins, 0.0);
        int gx0_block_frame_count = 0;
        std::vector<std::vector<double>> gx0_block_curves;
        int gx0_selected_frames = 0;
        double gx0_nsk_sum = 0.0;
        int gx0_min_nsk = std::numeric_limits<int>::max();
        int gx0_max_nsk = 0;

        // G6(r) block statistics
        std::vector<double> g6r_block_sum(g6r_nbins, 0.0);
        std::vector<double> g6r_block_count(g6r_nbins, 0.0);
        int g6r_block_frame_count = 0;
        std::vector<std::vector<double>> g6r_block_curves;
        int g6r_selected_frames = 0;
        double g6r_nsk_sum = 0.0;
        int g6r_min_nsk = std::numeric_limits<int>::max();
        int g6r_max_nsk = 0;

        // G6(t) fields for this disorder sample
        std::vector<std::vector<std::complex<double>>> g6t_fields;
        int g6t_frame_count = 0;
        double g6t_nsk_sum = 0.0;

        // Correct translational G_T(r) and G_T(t) diagnostics.
        std::vector<std::vector<double>> trans_block_frame_curves;
        int trans_block_frame_count = 0;
        std::vector<std::vector<double>> trans_block_curves;
        std::vector<std::vector<std::complex<double>>> trans_psiT_series;
        std::vector<double> trans_mT_values;
        std::vector<double> trans_eta_values;
        std::vector<double> trans_AB_values;
        std::vector<double> trans_W2_values;
        std::vector<double> trans_dq_parallel_values;
        std::vector<double> trans_dq_perp_values;
        std::vector<double> trans_xi_parallel_values;
        std::vector<double> trans_xi_perp_values;
        std::vector<double> trans_GT_tail_values;
        int trans_selected_frames = 0;
        double trans_nsk_sum = 0.0;
        int trans_min_nsk = std::numeric_limits<int>::max();
        int trans_max_nsk = 0;

        // Phase-boundary diagnostics: measurement average inside this disorder sample.
        int phase_diag_frame_count = 0;
        double sum_rho_def = 0.0;
        double sum_n5_frac = 0.0;
        double sum_n6_frac = 0.0;
        double sum_n7_frac = 0.0;
        double sum_nnon6_frac = 0.0;

        double sum_dislocation_frac = 0.0;
        double sum_free_dislocation_frac = 0.0;
        double sum_bound_dislocation_frac = 0.0;
        double sum_rho_dislocation = 0.0;
        double sum_rho_free_dislocation = 0.0;
        double sum_xiD_free_over_ask = 0.0;

        double sum_gt_tail = 0.0;
        double sum_mt = 0.0;
        double sum_AB_roughness = 0.0;
        int count_gt_tail = 0;
        int count_mt = 0;
        int count_AB_roughness = 0;

        for (int sweep = 1; sweep <= settings_.n_measure; ++sweep) {
            MetropolisSweep(
                settings_.target_temperature,
                field,
                all_spins[ibin],
                all_J[ibin],
                all_D[ibin],
                dynamics_rng[ibin]
            );

            for (int k = 0; k < settings_.overrelax_sweeps; ++k) {
                OverrelaxationSweep(
                    field,
                    all_spins[ibin],
                    all_J[ibin],
                    all_D[ibin]
                );
            }

            if (sweep % settings_.observable_stride == 0) {
                const MC_Measurements current =
                    MeasureObservables(
                        field,
                        all_spins[ibin],
                        all_J[ibin],
                        all_D[ibin],
                        false
                    );

                const std::vector<Point2> centers =
                    ExtractSkyrmionCentersForGx0(all_spins[ibin], Lx_, Ly_, gx0);

                const OrientFrame orient =
                    ComputeOrientFrameFromCenters(centers, Lx_, Ly_);

                const double phi6_global = orient.phi6_global;
                const double phi6_local = orient.phi6_local;

                accumulated.energy += current.energy;
                accumulated.mz += current.mz;
                accumulated.skyrmion_number += current.skyrmion_number;
                accumulated.sublattice_sk_order += current.sublattice_sk_order;
                accumulated.psi6 += phi6_global;
                accumulated.chirality += current.chirality;

                sum_phi6_global += phi6_global;
                sum_phi6_local += phi6_local;
                sum_phi6_global_sq += phi6_global * phi6_global;
                sum_phi6_local_sq += phi6_local * phi6_local;
                sum_nsk_center += static_cast<double>(orient.nsk);

                // Phase-boundary diagnostics: selected measures -> disorder mean.
                // These replace the noisy time-fluctuation chi6 as transition indicators.
                if (
                    phase.enabled &&
                    (measurement_count % phase.defect_measure_stride == 0) &&
                    orient.nsk >= 3 &&
                    std::isfinite(orient.a_sk) &&
                    orient.a_sk > 0.0
                    ) {
                    const PhaseFrameDiagnostics diag =
                        ComputeFirstShellCoordinationDiagnostics(
                            orient,
                            Lx_,
                            Ly_,
                            phase
                        );

                    if (std::isfinite(diag.rho_def)) {
                        sum_rho_def += diag.rho_def;
                        sum_n5_frac += diag.n5_frac;
                        sum_n6_frac += diag.n6_frac;
                        sum_n7_frac += diag.n7_frac;
                        sum_nnon6_frac += diag.nnon6_frac;

                        sum_dislocation_frac += diag.dislocation_frac;
                        sum_free_dislocation_frac += diag.free_dislocation_frac;
                        sum_bound_dislocation_frac += diag.bound_dislocation_frac;
                        sum_rho_dislocation += diag.rho_dislocation;
                        sum_rho_free_dislocation += diag.rho_free_dislocation;
                        sum_xiD_free_over_ask += diag.xiD_free_over_ask;

                        ++phase_diag_frame_count;
                    }
                }

                // Correct G_T(r), G_T(t), m_T, chi_T and Binder diagnostics.
                // This is independent of the old first-shell axis estimate: the Bragg vector(s)
                // are selected on the PBC-compatible reciprocal grid, or fixed from the reference SkX.
                if (
                    phase.trans_enabled &&
                    (measurement_count % phase.trans_measure_stride == 0) &&
                    orient.nsk >= 2 &&
                    std::isfinite(orient.a_sk) &&
                    orient.a_sk > 0.0
                    ) {
                    const BraggVectorSelection trans_sel =
                        GetTransBraggVectorSelection(orient, Lx_, Ly_, iD, iB, ibin, phase);

                    if (!trans_sel.gvecs.empty() &&
                        trans_sel.a_ref > 0.0 &&
                        std::isfinite(trans_sel.a_ref)) {
                        std::vector<double> pair_counts;
                        const std::vector<double> gt_curve =
                            ComputeGtrFrameCurveCorrect(
                                orient,
                                Lx_,
                                Ly_,
                                trans_sel.gvecs,
                                trans_sel.a_ref,
                                phase,
                                &pair_counts
                            );

                        trans_block_frame_curves.push_back(gt_curve);
                        ++trans_block_frame_count;
                        ++trans_selected_frames;
                        trans_nsk_sum += static_cast<double>(orient.nsk);
                        trans_min_nsk = std::min(trans_min_nsk, orient.nsk);
                        trans_max_nsk = std::max(trans_max_nsk, orient.nsk);

                        const std::vector<std::complex<double>> psiT =
                            ComputePsiTForGvecs(orient.centers, trans_sel.gvecs);
                        const double mt_now = MTFromPsiT(psiT);
                        if (std::isfinite(mt_now)) {
                            trans_mT_values.push_back(mt_now);
                            sum_mt += mt_now;
                            ++count_mt;
                        }
                        if (!psiT.empty()) {
                            trans_psiT_series.push_back(psiT);
                        }

                        const double gt_tail_now =
                            TailAverageFromCurve(
                                gt_curve,
                                phase.trans_rbin,
                                phase.trans_tail_rmin,
                                phase.trans_tail_rmax
                            );
                        if (std::isfinite(gt_tail_now)) {
                            trans_GT_tail_values.push_back(gt_tail_now);
                            sum_gt_tail += gt_tail_now;
                            ++count_gt_tail;
                        }

                        const double eta_now = FitEtaFromGtrEnvelope(gt_curve, phase);
                        if (std::isfinite(eta_now)) {
                            trans_eta_values.push_back(eta_now);
                        }

                        const double g2 = MeanGAbsSquared(trans_sel.gvecs);
                        if (std::isfinite(eta_now) && std::isfinite(g2) &&
                            g2 > 0.0 && trans_sel.a_ref > 0.0) {
                            const double ab_now =
                                2.0 * eta_now / (g2 * trans_sel.a_ref * trans_sel.a_ref);
                            if (std::isfinite(ab_now)) {
                                trans_AB_values.push_back(ab_now);
                                sum_AB_roughness += ab_now;
                                ++count_AB_roughness;
                            }
                        }

                        const double w2_now =
                            TranslationalPhaseRoughnessW2(orient.centers, trans_sel.gvecs, trans_sel.a_ref);
                        if (std::isfinite(w2_now)) {
                            trans_W2_values.push_back(w2_now);
                        }

                        const BraggPeakWidthDiagnostics width_now =
                            ComputeCenterBraggPeakWidths(
                                orient.centers,
                                Lx_,
                                Ly_,
                                trans_sel.gvecs,
                                trans_sel.a_ref,
                                phase
                            );
                        if (std::isfinite(width_now.dq_parallel_over_q0)) {
                            trans_dq_parallel_values.push_back(width_now.dq_parallel_over_q0);
                        }
                        if (std::isfinite(width_now.dq_perp_over_q0)) {
                            trans_dq_perp_values.push_back(width_now.dq_perp_over_q0);
                        }
                        if (std::isfinite(width_now.xi_parallel_over_ask)) {
                            trans_xi_parallel_values.push_back(width_now.xi_parallel_over_ask);
                        }
                        if (std::isfinite(width_now.xi_perp_over_ask)) {
                            trans_xi_perp_values.push_back(width_now.xi_perp_over_ask);
                        }

                        if (trans_block_frame_count >= phase.trans_block_size) {
                            FinishTransBlockIfNeeded(
                                trans_block_frame_curves,
                                trans_block_frame_count,
                                trans_block_curves
                            );
                        }
                    }
                }

                // g(x,0)-1: selected measures -> blocks -> disorder curve
                if (
                    gx0.enabled &&
                    (measurement_count % gx0.measure_stride == 0) &&
                    orient.nsk >= 2 &&
                    std::isfinite(orient.a_sk) &&
                    orient.a_sk > 0.0
                    ) {
                    const double axis_angle =
                        ComputePsi6AxisFromCenters(orient.centers, orient.a_sk, Lx_, Ly_, gx0);

                    AccumulateGx0FrameFromCenters(
                        orient.centers,
                        orient.a_sk,
                        axis_angle,
                        Lx_,
                        Ly_,
                        gx0,
                        gx0_block_hist,
                        gx0_block_ideal
                    );

                    ++gx0_block_frame_count;
                    ++gx0_selected_frames;
                    gx0_nsk_sum += static_cast<double>(orient.nsk);
                    gx0_min_nsk = std::min(gx0_min_nsk, orient.nsk);
                    gx0_max_nsk = std::max(gx0_max_nsk, orient.nsk);

                    if (gx0_block_frame_count >= gx0_block_size) {
                        FinishGx0BlockIfNeeded(
                            gx0_block_hist,
                            gx0_block_ideal,
                            gx0_block_frame_count,
                            gx0_block_curves
                        );
                    }
                }

                // G6(r): same hierarchical average as g(x,0)-1
                if (
                    g6r.enabled &&
                    (measurement_count % g6r.measure_stride == 0) &&
                    orient.nsk >= 3 &&
                    orient.psi6.size() == orient.centers.size() &&
                    std::isfinite(orient.a_sk) &&
                    orient.a_sk > 0.0
                    ) {
                    AccumulateG6rFrame(
                        orient,
                        Lx_,
                        Ly_,
                        g6r,
                        g6r_block_sum,
                        g6r_block_count
                    );

                    ++g6r_block_frame_count;
                    ++g6r_selected_frames;
                    g6r_nsk_sum += static_cast<double>(orient.nsk);
                    g6r_min_nsk = std::min(g6r_min_nsk, orient.nsk);
                    g6r_max_nsk = std::max(g6r_max_nsk, orient.nsk);

                    if (g6r_block_frame_count >= g6r.block_size) {
                        FinishG6rBlockIfNeeded(
                            g6r_block_sum,
                            g6r_block_count,
                            g6r_block_frame_count,
                            g6r_block_curves
                        );
                    }
                }

                // G6(t): store fixed-grid field for this selected measurement
                if (
                    g6t.enabled &&
                    (measurement_count % g6t.measure_stride == 0) &&
                    orient.nsk >= 3 &&
                    orient.psi6.size() == orient.centers.size() &&
                    std::isfinite(orient.a_sk) &&
                    orient.a_sk > 0.0
                    ) {
                    g6t_fields.push_back(
                        InterpolatePsiToGridNearest(
                            orient,
                            g6t_grid,
                            Lx_,
                            Ly_,
                            g6t
                        )
                    );

                    ++g6t_frame_count;
                    g6t_nsk_sum += static_cast<double>(orient.nsk);
                }

                ++measurement_count;
            }

            if (
                settings_.write_snapshots &&
                sweep % settings_.snapshot_stride == 0
                ) {
#ifdef _OPENMP
#pragma omp critical
#endif
                    {
                        snapshots_per_bin[ibin].insert(
                            snapshots_per_bin[ibin].end(),
                            all_spins[ibin].begin(),
                            all_spins[ibin].end()
                        );
                    }
            }
        }

        // Finish incomplete correlation blocks.
        FinishGx0BlockIfNeeded(
            gx0_block_hist,
            gx0_block_ideal,
            gx0_block_frame_count,
            gx0_block_curves
        );

        FinishG6rBlockIfNeeded(
            g6r_block_sum,
            g6r_block_count,
            g6r_block_frame_count,
            g6r_block_curves
        );

        FinishTransBlockIfNeeded(
            trans_block_frame_curves,
            trans_block_frame_count,
            trans_block_curves
        );

        if (measurement_count > 0) {
            const double inv_count =
                1.0 / static_cast<double>(measurement_count);

            accumulated.energy *= inv_count;
            accumulated.mz *= inv_count;
            accumulated.skyrmion_number *= inv_count;
            accumulated.sublattice_sk_order *= inv_count;
            accumulated.psi6 *= inv_count;
            accumulated.chirality *= inv_count;

            const double mean_phi6_global = sum_phi6_global * inv_count;
            const double mean_phi6_local = sum_phi6_local * inv_count;
            const double mean_nsk_center = sum_nsk_center * inv_count;

            bin_phi6_global[ibin] = mean_phi6_global;
            bin_phi6_local[ibin] = mean_phi6_local;
            bin_nsk_center[ibin] = mean_nsk_center;

            const double var_phi6_global_time =
                std::max(0.0, sum_phi6_global_sq * inv_count - mean_phi6_global * mean_phi6_global);
            const double var_phi6_local_time =
                std::max(0.0, sum_phi6_local_sq * inv_count - mean_phi6_local * mean_phi6_local);
            if (std::isfinite(mean_nsk_center)) {
                bin_chi_phi6_global_time[ibin] = mean_nsk_center * var_phi6_global_time;
                bin_chi_phi6_local_time[ibin] = mean_nsk_center * var_phi6_local_time;
            }

            if (phase_diag_frame_count > 0) {
                const double inv_phase = 1.0 / static_cast<double>(phase_diag_frame_count);
                bin_rho_def[ibin] = sum_rho_def * inv_phase;
                bin_n5_frac[ibin] = sum_n5_frac * inv_phase;
                bin_n6_frac[ibin] = sum_n6_frac * inv_phase;
                bin_n7_frac[ibin] = sum_n7_frac * inv_phase;
                bin_nnon6_frac[ibin] = sum_nnon6_frac * inv_phase;

                bin_dislocation_frac[ibin] = sum_dislocation_frac * inv_phase;
                bin_free_dislocation_frac[ibin] = sum_free_dislocation_frac * inv_phase;
                bin_bound_dislocation_frac[ibin] = sum_bound_dislocation_frac * inv_phase;
                bin_rho_dislocation[ibin] = sum_rho_dislocation * inv_phase;
                bin_rho_free_dislocation[ibin] = sum_rho_free_dislocation * inv_phase;
                bin_xiD_free_over_ask[ibin] = sum_xiD_free_over_ask * inv_phase;
            }

            if (trans_selected_frames > 0) {
                bin_gt_tail[ibin] = MeanFinite(trans_GT_tail_values);
                bin_mt[ibin] = MeanFinite(trans_mT_values);
                bin_eta_T[ibin] = MeanFinite(trans_eta_values);
                bin_AB_roughness[ibin] = MeanFinite(trans_AB_values);
                bin_phase_W2[ibin] = MeanFinite(trans_W2_values);
                bin_dq_parallel[ibin] = MeanFinite(trans_dq_parallel_values);
                bin_dq_perp[ibin] = MeanFinite(trans_dq_perp_values);
                bin_xi_parallel[ibin] = MeanFinite(trans_xi_parallel_values);
                bin_xi_perp[ibin] = MeanFinite(trans_xi_perp_values);

                const double w2_mean_bin = MeanFinite(trans_W2_values);
                const double w2_var_bin = VarianceFinitePopulation(trans_W2_values, w2_mean_bin);
                const double mt_mean_bin = MeanFinite(trans_mT_values);
                const double mt_var_bin = VarianceFinitePopulation(trans_mT_values, mt_mean_bin);
                const double n_factor_bin = trans_nsk_sum / static_cast<double>(trans_selected_frames);
                if (std::isfinite(mt_var_bin) && std::isfinite(n_factor_bin)) {
                    bin_chi_mt_time[ibin] = n_factor_bin * mt_var_bin;
                }
                if (std::isfinite(w2_var_bin) && std::isfinite(n_factor_bin)) {
                    bin_chi_W2_time[ibin] = n_factor_bin * w2_var_bin;
                }
                bin_binder_mt_time[ibin] = BinderCumulantScalar(trans_mT_values);
            }
        }
        else {
            accumulated = MeasureObservables(
                field,
                all_spins[ibin],
                all_J[ibin],
                all_D[ibin],
                false
            );

            const Phi6FrameResult phi =
                ComputePhi6FromSpinConfiguration(all_spins[ibin], Lx_, Ly_);

            accumulated.psi6 = phi.abs_Phi6;

            bin_phi6_global[ibin] = phi.abs_Phi6;
            bin_phi6_local[ibin] = phi.mean_abs_psi6;
            bin_nsk_center[ibin] = static_cast<double>(phi.nsk);

            const std::vector<Point2> centers =
                ExtractSkyrmionCentersForGx0(all_spins[ibin], Lx_, Ly_, gx0);
            const OrientFrame orient = ComputeOrientFrameFromCenters(centers, Lx_, Ly_);
            const PhaseFrameDiagnostics diag =
                ComputeFirstShellCoordinationDiagnostics(orient, Lx_, Ly_, phase);
            bin_rho_def[ibin] = diag.rho_def;
            bin_n5_frac[ibin] = diag.n5_frac;
            bin_n6_frac[ibin] = diag.n6_frac;
            bin_n7_frac[ibin] = diag.n7_frac;
            bin_nnon6_frac[ibin] = diag.nnon6_frac;

            bin_dislocation_frac[ibin] = diag.dislocation_frac;
            bin_free_dislocation_frac[ibin] = diag.free_dislocation_frac;
            bin_bound_dislocation_frac[ibin] = diag.bound_dislocation_frac;
            bin_rho_dislocation[ibin] = diag.rho_dislocation;
            bin_rho_free_dislocation[ibin] = diag.rho_free_dislocation;
            bin_xiD_free_over_ask[ibin] = diag.xiD_free_over_ask;

            const TranslationalFrameDiagnostics tdiag =
                ComputeTranslationalFrameDiagnostics(orient, Lx_, Ly_, phase, iD, iB, ibin);
            bin_gt_tail[ibin] = tdiag.gt_tail;
            bin_mt[ibin] = tdiag.mt;
            bin_eta_T[ibin] = tdiag.eta_T;
            bin_AB_roughness[ibin] = tdiag.roughness_slope_AB;
            bin_phase_W2[ibin] = tdiag.phase_roughness_W2;
            bin_dq_parallel[ibin] = tdiag.peak_dq_parallel_over_q0;
            bin_dq_perp[ibin] = tdiag.peak_dq_perp_over_q0;
            bin_xi_parallel[ibin] = tdiag.xi_parallel_over_ask;
            bin_xi_perp[ibin] = tdiag.xi_perp_over_ask;
        }

        bin_measurements[ibin] = accumulated;

        gx0_disorder_curves[ibin] =
            MakeGx0DisorderCurveFromBlocks(
                gx0_block_curves,
                gx0_selected_frames,
                gx0_nsk_sum,
                gx0_min_nsk,
                gx0_max_nsk
            );

        gx0_all_block_curves[ibin] = std::move(gx0_block_curves);

        g6r_disorder_curves[ibin] =
            MakeGenericDisorderCurveFromBlocks(
                g6r_block_curves,
                g6r_selected_frames,
                g6r_nsk_sum,
                g6r_min_nsk,
                g6r_max_nsk
            );

        g6r_all_block_curves[ibin] = std::move(g6r_block_curves);

        trans_r_disorder_curves[ibin] =
            MakeGenericDisorderCurveFromBlocks(
                trans_block_curves,
                trans_selected_frames,
                trans_nsk_sum,
                trans_min_nsk,
                trans_max_nsk
            );

        trans_r_all_block_curves[ibin] = std::move(trans_block_curves);

        if (phase.trans_enabled) {
            trans_t_disorder_curves[ibin] =
                ComputeTransTOneDisorder(
                    trans_psiT_series,
                    phase.trans_t_max_lag
                );

            trans_t_selected_frames[ibin] = trans_selected_frames;
            trans_t_avg_nsk[ibin] =
                trans_selected_frames > 0 ?
                trans_nsk_sum / static_cast<double>(trans_selected_frames) :
                0.0;
        }

        if (g6t.enabled) {
            g6t_disorder_curves[ibin] =
                ComputeG6tOneDisorder(
                    g6t_fields,
                    g6t.max_lag
                );

            g6t_selected_frames[ibin] = g6t_frame_count;
            g6t_avg_nsk[ibin] =
                g6t_frame_count > 0 ?
                g6t_nsk_sum / static_cast<double>(g6t_frame_count) :
                0.0;
        }
    }

    MC_Measurements mean;

    std::vector<double> energy_values;
    std::vector<double> mz_values;
    std::vector<double> sk_values;
    std::vector<double> sub_sk_values;
    std::vector<double> psi6_values;
    std::vector<double> chirality_values;

    energy_values.reserve(settings_.n_bins);
    mz_values.reserve(settings_.n_bins);
    sk_values.reserve(settings_.n_bins);
    sub_sk_values.reserve(settings_.n_bins);
    psi6_values.reserve(settings_.n_bins);
    chirality_values.reserve(settings_.n_bins);

    for (const MC_Measurements& measurement : bin_measurements) {
        mean.energy += measurement.energy;
        mean.mz += measurement.mz;
        mean.skyrmion_number += measurement.skyrmion_number;
        mean.sublattice_sk_order += measurement.sublattice_sk_order;
        mean.psi6 += measurement.psi6;
        mean.chirality += measurement.chirality;

        energy_values.push_back(measurement.energy);
        mz_values.push_back(measurement.mz);
        sk_values.push_back(measurement.skyrmion_number);
        sub_sk_values.push_back(measurement.sublattice_sk_order);
        psi6_values.push_back(measurement.psi6);
        chirality_values.push_back(measurement.chirality);
    }

    const double inv_bins = 1.0 / static_cast<double>(settings_.n_bins);

    mean.energy *= inv_bins;
    mean.mz *= inv_bins;
    mean.skyrmion_number *= inv_bins;
    mean.sublattice_sk_order *= inv_bins;
    mean.psi6 *= inv_bins;
    mean.chirality *= inv_bins;

    const double mean_phi6_global =
        std::accumulate(bin_phi6_global.begin(), bin_phi6_global.end(), 0.0) * inv_bins;

    const double mean_phi6_local =
        std::accumulate(bin_phi6_local.begin(), bin_phi6_local.end(), 0.0) * inv_bins;

    const double mean_nsk_center = MeanFinite(bin_nsk_center);

    std::vector<double> bin_g6_tail(settings_.n_bins, std::numeric_limits<double>::quiet_NaN());
    for (int ib = 0; ib < settings_.n_bins; ++ib) {
        bin_g6_tail[ib] = G6TailFromDisorderCurve(g6r_disorder_curves[ib], g6r, phase);
    }

    const double mean_rho_def = MeanFinite(bin_rho_def);
    const double mean_n5_frac = MeanFinite(bin_n5_frac);
    const double mean_n6_frac = MeanFinite(bin_n6_frac);
    const double mean_n7_frac = MeanFinite(bin_n7_frac);
    const double mean_nnon6_frac = MeanFinite(bin_nnon6_frac);

    const double mean_dislocation_frac = MeanFinite(bin_dislocation_frac);
    const double mean_free_dislocation_frac = MeanFinite(bin_free_dislocation_frac);
    const double mean_bound_dislocation_frac = MeanFinite(bin_bound_dislocation_frac);
    const double mean_rho_dislocation = MeanFinite(bin_rho_dislocation);
    const double mean_rho_free_dislocation = MeanFinite(bin_rho_free_dislocation);
    const double mean_xiD_free_over_ask = MeanFinite(bin_xiD_free_over_ask);

    const double mean_g6_tail = MeanFinite(bin_g6_tail);
    const double mean_gt_tail = MeanFinite(bin_gt_tail);
    const double mean_mt = MeanFinite(bin_mt);
    const double mean_eta_T = MeanFinite(bin_eta_T);
    const double mean_AB_roughness = MeanFinite(bin_AB_roughness);
    const double mean_phase_W2 = MeanFinite(bin_phase_W2);
    const double mean_chi_W2_time = MeanFinite(bin_chi_W2_time);
    const double mean_dq_parallel = MeanFinite(bin_dq_parallel);
    const double mean_dq_perp = MeanFinite(bin_dq_perp);
    const double mean_xi_parallel = MeanFinite(bin_xi_parallel);
    const double mean_xi_perp = MeanFinite(bin_xi_perp);
    const double mean_chi_phi6_global_time = MeanFinite(bin_chi_phi6_global_time);
    const double mean_chi_phi6_local_time = MeanFinite(bin_chi_phi6_local_time);
    const double mean_chi_mt_time = MeanFinite(bin_chi_mt_time);
    const double mean_binder_mt_time = MeanFinite(bin_binder_mt_time);

    const double chi_phi6_global_dis =
        SusceptibilityFromDisorderValues(bin_phi6_global, mean_nsk_center);
    const double chi_phi6_local_dis =
        SusceptibilityFromDisorderValues(bin_phi6_local, mean_nsk_center);
    const double chi_def =
        SusceptibilityFromDisorderValues(bin_rho_def, mean_nsk_center);
    const double chi_free_dislocation =
        SusceptibilityFromDisorderValues(bin_free_dislocation_frac, mean_nsk_center);
    const double chi_g6_tail =
        SusceptibilityFromDisorderValues(bin_g6_tail, mean_nsk_center);
    const double chi_gt_tail =
        SusceptibilityFromDisorderValues(bin_gt_tail, mean_nsk_center);
    const double chi_mt_dis =
        SusceptibilityFromDisorderValues(bin_mt, mean_nsk_center);
    const double chi_W2_dis =
        SusceptibilityFromDisorderValues(bin_phase_W2, mean_nsk_center);
    const double chi_AB_dis =
        SusceptibilityFromDisorderValues(bin_AB_roughness, mean_nsk_center);
    const double chi_dq_parallel_dis =
        SusceptibilityFromDisorderValues(bin_dq_parallel, mean_nsk_center);

    const double binder_phi6 = BinderCumulantO2NormalizedFromMagnitude(bin_phi6_global);
    const double binder_local = BinderCumulantScalarNormalized(bin_phi6_local);
    const double binder_g6_tail = BinderCumulantScalarNormalized(bin_g6_tail);
    const double binder_mt_dis = BinderCumulantScalarNormalized(bin_mt);
    const double binder_phi6_scalar_raw = BinderCumulantScalar(bin_phi6_global);
    const double binder_local_scalar_raw = BinderCumulantScalar(bin_phi6_local);
    const double binder_g6_tail_scalar_raw = BinderCumulantScalar(bin_g6_tail);
    const double binder_mt_dis_scalar_raw = BinderCumulantScalar(bin_mt);

    MC_Measurements sem;

    sem.energy = StandardError(energy_values, mean.energy);
    sem.mz = StandardError(mz_values, mean.mz);
    sem.skyrmion_number = StandardError(sk_values, mean.skyrmion_number);
    sem.sublattice_sk_order = StandardError(sub_sk_values, mean.sublattice_sk_order);
    sem.psi6 = StandardError(psi6_values, mean.psi6);
    sem.chirality = StandardError(chirality_values, mean.chirality);

    const double sem_phi6_global =
        StandardError(bin_phi6_global, mean_phi6_global);

    const double sem_phi6_local =
        StandardError(bin_phi6_local, mean_phi6_local);

    const double sem_nsk_center =
        SemFinite(bin_nsk_center, mean_nsk_center);

    const double sem_rho_def = SemFinite(bin_rho_def, mean_rho_def);
    const double sem_dislocation_frac = SemFinite(bin_dislocation_frac, mean_dislocation_frac);
    const double sem_free_dislocation_frac = SemFinite(bin_free_dislocation_frac, mean_free_dislocation_frac);
    const double sem_bound_dislocation_frac = SemFinite(bin_bound_dislocation_frac, mean_bound_dislocation_frac);
    const double sem_rho_dislocation = SemFinite(bin_rho_dislocation, mean_rho_dislocation);
    const double sem_rho_free_dislocation = SemFinite(bin_rho_free_dislocation, mean_rho_free_dislocation);
    const double sem_xiD_free_over_ask = SemFinite(bin_xiD_free_over_ask, mean_xiD_free_over_ask);

    const double sem_g6_tail = SemFinite(bin_g6_tail, mean_g6_tail);
    const double sem_gt_tail = SemFinite(bin_gt_tail, mean_gt_tail);
    const double sem_mt = SemFinite(bin_mt, mean_mt);
    const double sem_eta_T = SemFinite(bin_eta_T, mean_eta_T);
    const double sem_AB_roughness = SemFinite(bin_AB_roughness, mean_AB_roughness);
    const double sem_phase_W2 = SemFinite(bin_phase_W2, mean_phase_W2);
    const double sem_chi_W2_time = SemFinite(bin_chi_W2_time, mean_chi_W2_time);
    const double sem_dq_parallel = SemFinite(bin_dq_parallel, mean_dq_parallel);
    const double sem_dq_perp = SemFinite(bin_dq_perp, mean_dq_perp);
    const double sem_xi_parallel = SemFinite(bin_xi_parallel, mean_xi_parallel);
    const double sem_xi_perp = SemFinite(bin_xi_perp, mean_xi_perp);
    const double sem_chi_phi6_global_time = SemFinite(bin_chi_phi6_global_time, mean_chi_phi6_global_time);
    const double sem_chi_phi6_local_time = SemFinite(bin_chi_phi6_local_time, mean_chi_phi6_local_time);
    const double sem_chi_mt_time = SemFinite(bin_chi_mt_time, mean_chi_mt_time);
    const double sem_binder_mt_time = SemFinite(bin_binder_mt_time, mean_binder_mt_time);
    const double sem_chi_phi6_global_dis =
        SusceptibilityJackknifeSem(bin_phi6_global, mean_nsk_center);
    const double sem_chi_phi6_local_dis =
        SusceptibilityJackknifeSem(bin_phi6_local, mean_nsk_center);
    const double sem_chi_def =
        SusceptibilityJackknifeSem(bin_rho_def, mean_nsk_center);
    const double sem_chi_free_dislocation =
        SusceptibilityJackknifeSem(bin_free_dislocation_frac, mean_nsk_center);
    const double sem_chi_g6_tail =
        SusceptibilityJackknifeSem(bin_g6_tail, mean_nsk_center);
    const double sem_chi_gt_tail =
        SusceptibilityJackknifeSem(bin_gt_tail, mean_nsk_center);
    const double sem_chi_mt_dis =
        SusceptibilityJackknifeSem(bin_mt, mean_nsk_center);
    const double sem_chi_W2_dis =
        SusceptibilityJackknifeSem(bin_phase_W2, mean_nsk_center);
    const double sem_chi_AB_dis =
        SusceptibilityJackknifeSem(bin_AB_roughness, mean_nsk_center);
    const double sem_chi_dq_parallel_dis =
        SusceptibilityJackknifeSem(bin_dq_parallel, mean_nsk_center);

    std::ostringstream filename;
    filename << "Averaged_Measures_iD" << iD << "_iB" << iB << ".txt";

    std::ofstream output(filename.str());
    if (!output) {
        throw std::runtime_error("Cannot write averaged measurement file.");
    }

    output << "# Row 1: mean, Row 2: SEM over disorder bins\n";
    output << "# Energy_per_site Mz Skyrmion_number Sublattice_Sk_order "
        << "Psi6 Chirality "
        << "Phi6_global Phi6_local ChiPhi6_dis ChiLocal_dis ChiPhi6_time ChiLocal_time "
        << "Rho_def Chi_def G6_tail Chi_G6tail "
        << "GT_tail Chi_GTtail mT Chi_mT_dis Chi_mT_time Binder_mT_dis Binder_mT_time eta_T AB_roughness "
        << "W2_phase Chi_W2_dis Chi_W2_time Chi_AB_dis dQpar_over_q0 dQperp_over_q0 "
        << "Xi_parallel_over_ask Xi_perp_over_ask Chi_dQpar_dis "
        << "Binder_Phi6 Binder_local Binder_G6tail "
        << "Binder_Phi6_scalar_raw Binder_local_scalar_raw Binder_G6tail_scalar_raw Binder_mT_dis_scalar_raw "
        << "Nsk_center N5_frac N6_frac N7_frac Nnon6_frac "
        << "Ndisloc_frac Nfree_disloc_frac Nbound_disloc_frac "
        << "Rho_disloc Rho_free_disloc XiD_free_over_ask Chi_free_disloc\n";

    output << std::setprecision(16)
        << mean.energy << ' '
        << mean.mz << ' '
        << mean.skyrmion_number << ' '
        << mean.sublattice_sk_order << ' '
        << mean.psi6 << ' '
        << mean.chirality << ' '
        << mean_phi6_global << ' '
        << mean_phi6_local << ' '
        << chi_phi6_global_dis << ' '
        << chi_phi6_local_dis << ' '
        << mean_chi_phi6_global_time << ' '
        << mean_chi_phi6_local_time << ' '
        << mean_rho_def << ' '
        << chi_def << ' '
        << mean_g6_tail << ' '
        << chi_g6_tail << ' '
        << mean_gt_tail << ' '
        << chi_gt_tail << ' '
        << mean_mt << ' '
        << chi_mt_dis << ' '
        << mean_chi_mt_time << ' '
        << binder_mt_dis << ' '
        << mean_binder_mt_time << ' '
        << mean_eta_T << ' '
        << mean_AB_roughness << ' '
        << mean_phase_W2 << ' '
        << chi_W2_dis << ' '
        << mean_chi_W2_time << ' '
        << chi_AB_dis << ' '
        << mean_dq_parallel << ' '
        << mean_dq_perp << ' '
        << mean_xi_parallel << ' '
        << mean_xi_perp << ' '
        << chi_dq_parallel_dis << ' '
        << binder_phi6 << ' '
        << binder_local << ' '
        << binder_g6_tail << ' '
        << binder_phi6_scalar_raw << ' '
        << binder_local_scalar_raw << ' '
        << binder_g6_tail_scalar_raw << ' '
        << binder_mt_dis_scalar_raw << ' '
        << mean_nsk_center << ' '
        << mean_n5_frac << ' '
        << mean_n6_frac << ' '
        << mean_n7_frac << ' '
        << mean_nnon6_frac << ' '
        << mean_dislocation_frac << ' '
        << mean_free_dislocation_frac << ' '
        << mean_bound_dislocation_frac << ' '
        << mean_rho_dislocation << ' '
        << mean_rho_free_dislocation << ' '
        << mean_xiD_free_over_ask << ' '
        << chi_free_dislocation << '\n';

    output << std::setprecision(16)
        << sem.energy << ' '
        << sem.mz << ' '
        << sem.skyrmion_number << ' '
        << sem.sublattice_sk_order << ' '
        << sem.psi6 << ' '
        << sem.chirality << ' '
        << sem_phi6_global << ' '
        << sem_phi6_local << ' '
        << sem_chi_phi6_global_dis << ' '
        << sem_chi_phi6_local_dis << ' '
        << sem_chi_phi6_global_time << ' '
        << sem_chi_phi6_local_time << ' '
        << sem_rho_def << ' '
        << sem_chi_def << ' '
        << sem_g6_tail << ' '
        << sem_chi_g6_tail << ' '
        << sem_gt_tail << ' '
        << sem_chi_gt_tail << ' '
        << sem_mt << ' '
        << sem_chi_mt_dis << ' '
        << sem_chi_mt_time << ' '
        << 0.0 << ' '
        << sem_binder_mt_time << ' '
        << sem_eta_T << ' '
        << sem_AB_roughness << ' '
        << sem_phase_W2 << ' '
        << sem_chi_W2_dis << ' '
        << sem_chi_W2_time << ' '
        << sem_chi_AB_dis << ' '
        << sem_dq_parallel << ' '
        << sem_dq_perp << ' '
        << sem_xi_parallel << ' '
        << sem_xi_perp << ' '
        << sem_chi_dq_parallel_dis << ' '
        << 0.0 << ' '
        << 0.0 << ' '
        << 0.0 << ' '
        << 0.0 << ' '
        << sem_nsk_center << ' '
        << SemFinite(bin_n5_frac, mean_n5_frac) << ' '
        << SemFinite(bin_n6_frac, mean_n6_frac) << ' '
        << SemFinite(bin_n7_frac, mean_n7_frac) << ' '
        << SemFinite(bin_nnon6_frac, mean_nnon6_frac) << ' '
        << sem_dislocation_frac << ' '
        << sem_free_dislocation_frac << ' '
        << sem_bound_dislocation_frac << ' '
        << sem_rho_dislocation << ' '
        << sem_rho_free_dislocation << ' '
        << sem_xiD_free_over_ask << ' '
        << sem_chi_free_dislocation << '\n';

    WritePhaseBoundaryDiagnosticsFile(
        iD,
        iB,
        phase,
        mean_nsk_center,
        bin_phi6_global,
        bin_phi6_local,
        bin_rho_def,
        bin_n5_frac,
        bin_n6_frac,
        bin_n7_frac,
        bin_nnon6_frac,
        bin_dislocation_frac,
        bin_free_dislocation_frac,
        bin_bound_dislocation_frac,
        bin_rho_dislocation,
        bin_rho_free_dislocation,
        bin_xiD_free_over_ask,
        bin_gt_tail,
        bin_mt,
        bin_AB_roughness,
        bin_g6_tail
    );

    if (phase.trans_enabled) {
        WriteGtrHierarchicalFile(
            iD,
            iB,
            phase,
            trans_r_disorder_curves,
            trans_r_all_block_curves
        );

        WriteTransTFile(
            iD,
            iB,
            phase,
            trans_t_disorder_curves,
            trans_t_selected_frames,
            trans_t_avg_nsk
        );

        WriteTransScalarDiagnosticsFile(
            iD,
            iB,
            mean_nsk_center,
            bin_gt_tail,
            bin_mt,
            bin_eta_T,
            bin_AB_roughness,
            bin_phase_W2,
            bin_dq_parallel,
            bin_dq_perp,
            bin_xi_parallel,
            bin_xi_perp,
            bin_chi_mt_time,
            bin_chi_W2_time,
            bin_binder_mt_time
        );
    }

    WriteGx0HierarchicalFile(
        iD,
        iB,
        gx0,
        gx0_block_size,
        write_gx0_block_samples,
        gx0_disorder_curves,
        gx0_all_block_curves
    );

    if (g6r.enabled) {
        WriteG6rHierarchicalFile(
            iD,
            iB,
            g6r,
            g6r_disorder_curves,
            g6r_all_block_curves
        );
    }

    if (g6t.enabled) {
        WriteG6tFile(
            iD,
            iB,
            g6t,
            g6t_disorder_curves,
            g6t_selected_frames,
            g6t_avg_nsk
        );
    }

    if (settings_.calculate_structure_factor) {
        WriteStructureFactor(all_spins, iD, iB);
    }

    if (settings_.write_snapshots) {
        WriteAngleSnapshots(snapshots_per_bin, iD, iB);
    }
}


void MC_Ising::WriteStructureFactor(
    const std::vector<std::vector<Spin3>>& snapshots_per_bin,
    int iD,
    int iB
) const {
    const int number_of_bins = static_cast<int>(snapshots_per_bin.size());

    const int qx_count = Lx_;
    const int qy_count = Ly_;
    const int n_q = qx_count * qy_count;

    std::vector<double> mean_perp(n_q, 0.0);
    std::vector<double> mean_z(n_q, 0.0);
    std::vector<double> square_perp(n_q, 0.0);
    std::vector<double> square_z(n_q, 0.0);

    std::vector<ReciprocalRingMetrics> reciprocal_bin_metrics;
    reciprocal_bin_metrics.reserve(static_cast<std::size_t>(std::max(0, number_of_bins)));

    for (int ibin = 0; ibin < number_of_bins; ++ibin) {
        const SpinConfig& spins = snapshots_per_bin[ibin];
        std::vector<double> sz_frame(n_q, 0.0);

        for (int ky_index = 0; ky_index < qy_count; ++ky_index) {
            const int ky_integer = ky_index - qy_count / 2;
            const double qy = 2.0 * PI * ky_integer / static_cast<double>(Ly_);

            for (int kx_index = 0; kx_index < qx_count; ++kx_index) {
                const int kx_integer = kx_index - qx_count / 2;
                const double qx = 2.0 * PI * kx_integer / static_cast<double>(Lx_);

                std::complex<double> sx(0.0, 0.0);
                std::complex<double> sy(0.0, 0.0);
                std::complex<double> sz(0.0, 0.0);

                for (int y = 0; y < Ly_; ++y) {
                    for (int x = 0; x < Lx_; ++x) {
                        const int site = x + y * Lx_;

                        const double rx = static_cast<double>(x) + 0.5 * static_cast<double>(y);
                        const double ry = SQRT3_OVER_2 * static_cast<double>(y);

                        const double phase = -(qx * rx + qy * ry);
                        const std::complex<double> factor(std::cos(phase), std::sin(phase));

                        sx += spins[site].x * factor;
                        sy += spins[site].y * factor;
                        sz += spins[site].z * factor;
                    }
                }

                const double norm = static_cast<double>(N_);
                const double s_perp =
                    (std::norm(sx) + std::norm(sy)) / norm;

                const double s_z =
                    std::norm(sz) / norm;

                const int qindex = kx_index + ky_index * qx_count;

                mean_perp[qindex] += s_perp;
                mean_z[qindex] += s_z;
                square_perp[qindex] += s_perp * s_perp;
                square_z[qindex] += s_z * s_z;
                sz_frame[qindex] = s_z;
            }
        }

        // Per-bin reciprocal-space entropy / peak-shape diagnostics from S_z(q).
        reciprocal_bin_metrics.push_back(
            ComputeReciprocalRingMetricsFromGrid(
                sz_frame,
                qx_count,
                qy_count,
                Lx_,
                Ly_
            )
        );
    }

    if (!reciprocal_bin_metrics.empty()) {
        WriteReciprocalMetricsFiles(iD, iB, reciprocal_bin_metrics);
    }

    std::ostringstream filename;
    filename << "SF_Averaged_iD" << iD << "_iB" << iB << ".txt";

    std::ofstream output(filename.str());
    if (!output) {
        throw std::runtime_error("Cannot write structure factor file.");
    }

    output << "# qx qy S_perp_mean S_perp_sem S_z_mean S_z_sem\n";

    for (int ky_index = 0; ky_index < qy_count; ++ky_index) {
        const int ky_integer = ky_index - qy_count / 2;
        const double qy = 2.0 * PI * ky_integer / static_cast<double>(Ly_);

        for (int kx_index = 0; kx_index < qx_count; ++kx_index) {
            const int kx_integer = kx_index - qx_count / 2;
            const double qx = 2.0 * PI * kx_integer / static_cast<double>(Lx_);

            const int qindex = kx_index + ky_index * qx_count;

            const double inv_bins = number_of_bins > 0 ?
                1.0 / static_cast<double>(number_of_bins) :
                0.0;

            const double avg_perp = mean_perp[qindex] * inv_bins;
            const double avg_z = mean_z[qindex] * inv_bins;

            double sem_perp = 0.0;
            double sem_z = 0.0;

            if (number_of_bins > 1) {
                const double variance_perp =
                    square_perp[qindex] * inv_bins - avg_perp * avg_perp;

                const double variance_z =
                    square_z[qindex] * inv_bins - avg_z * avg_z;

                sem_perp = std::sqrt(
                    std::max(0.0, variance_perp) /
                    static_cast<double>(number_of_bins)
                );

                sem_z = std::sqrt(
                    std::max(0.0, variance_z) /
                    static_cast<double>(number_of_bins)
                );
            }

            output << std::setprecision(12)
                << qx << ' '
                << qy << ' '
                << avg_perp << ' '
                << sem_perp << ' '
                << avg_z << ' '
                << sem_z << '\n';
        }
    }
}

void MC_Ising::WriteAngleSnapshots(
    const std::vector<std::vector<Spin3>>& snapshots_per_bin,
    int iD,
    int iB
) const {
    for (std::size_t ibin = 0; ibin < snapshots_per_bin.size(); ++ibin) {
        std::ostringstream filename;
        filename << "Snapshot_iD" << iD << "_iB" << iB << "_bin" << ibin << ".bin";

        std::vector<double> angle_data;
        angle_data.reserve(snapshots_per_bin[ibin].size() * 2);

        for (const Spin3& spin : snapshots_per_bin[ibin]) {
            const double theta = std::acos(std::max(-1.0, std::min(1.0, spin.z)));
            const double phi = std::atan2(spin.y, spin.x);

            angle_data.push_back(theta);
            angle_data.push_back(phi);
        }

        if (!angle_data.empty()) {
            Vec_fwrite_double(
                filename.str().c_str(),
                angle_data.data(),
                angle_data.size()
            );
        }
    }
}