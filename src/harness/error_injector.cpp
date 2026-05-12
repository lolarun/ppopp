#include "error_injector.h"
#include <algorithm>
#include <vector>
#include <numeric>

template<typename W>
Certificate<W> inject_errors(
    const CSR<W>&          g,
    const Certificate<W>&  original,
    ErrorKind              kind,
    int                    n_errors,
    int                    seed)
{
    Certificate<W> c = original;  // copy
    std::mt19937 rng(seed);

    // Collect reachable vertices (d[v] != INF, v != source assumed at v=0 for simplicity)
    std::vector<vid_t> reachable;
    for (vid_t v = 0; v < g.n_vertices; ++v)
        if (c.d[v] != Sentinel<W>::inf && v != 0)
            reachable.push_back(v);

    if (reachable.empty()) return c;

    std::shuffle(reachable.begin(), reachable.end(), rng);
    int n = std::min((int)reachable.size(), n_errors);

    std::uniform_real_distribution<W> delta_dist(W{0.1}, W{10.0});
    std::uniform_int_distribution<vid_t> vid_dist(0, g.n_vertices - 1);

    switch (kind) {
        case ErrorKind::DISTANCE_PERTURB:
            for (int i = 0; i < n; ++i)
                c.d[reachable[i]] += delta_dist(rng);
            break;

        case ErrorKind::PREDECESSOR_RANDOM:
            for (int i = 0; i < n; ++i) {
                vid_t v = reachable[i];
                vid_t new_pi;
                do { new_pi = vid_dist(rng); } while (new_pi == c.pi[v]);
                c.pi[v] = new_pi;
            }
            break;

        case ErrorKind::INCONSISTENT:
            // Change d[v] but leave pi[v] pointing to old predecessor
            for (int i = 0; i < n; ++i)
                c.d[reachable[i]] += delta_dist(rng);
            // pi not touched → d[v] != d[pi[v]] + w(pi[v],v)
            break;

        case ErrorKind::MISSED_UNREACHABLE:
            // Set reachable vertex as INF (pretend it's unreachable)
            for (int i = 0; i < n; ++i) {
                c.d[reachable[i]]  = Sentinel<W>::inf;
                c.pi[reachable[i]] = INVALID_VID;
            }
            break;

        case ErrorKind::CYCLE:
            // Pair up vertices and create pi[a]=b, pi[b]=a
            for (int i = 0; i + 1 < n; i += 2) {
                vid_t a = reachable[i], b = reachable[i + 1];
                c.pi[a] = b;
                c.pi[b] = a;
            }
            break;
    }

    return c;
}

template Certificate<float>  inject_errors<float> (const CSR<float>&,  const Certificate<float>&,  ErrorKind, int, int);
template Certificate<double> inject_errors<double>(const CSR<double>&, const Certificate<double>&, ErrorKind, int, int);
