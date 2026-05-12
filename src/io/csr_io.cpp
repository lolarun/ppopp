// csr_io.cpp — Binary CSR serialisation / deserialisation.
//
// File format (.csr) — see csr_io.h for specification.
// CRC64: Jones polynomial (0xad93d23594c935a9), no pre/post-conditioning.

#include "csr_io.h"
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>

// ── CRC64 (Jones polynomial) ──────────────────────────────────────────────────

static const uint64_t CRC64_POLY = 0xad93d23594c935a9ULL;

static uint64_t crc64_update(uint64_t crc, const void* data, size_t len) {
    const uint8_t* p = static_cast<const uint8_t*>(data);
    for (size_t i = 0; i < len; ++i) {
        crc ^= (uint64_t)p[i] << 56;
        for (int b = 0; b < 8; ++b) {
            if (crc & (1ULL << 63))
                crc = (crc << 1) ^ CRC64_POLY;
            else
                crc <<= 1;
        }
    }
    return crc;
}

// ── Magic / precision tags ────────────────────────────────────────────────────

static const uint64_t MAGIC   = 0x4353520000000001ULL;  // CSR\0 + version 1
static const uint32_t TAG_F32 = 0x46503332u;  // "FP32"
static const uint32_t TAG_F64 = 0x46503634u;  // "FP64"

template<typename W> struct PrecTag;
template<> struct PrecTag<float>  { static constexpr uint32_t value = 0x46503332u; };
template<> struct PrecTag<double> { static constexpr uint32_t value = 0x46503634u; };

// ── Write ─────────────────────────────────────────────────────────────────────

template<typename W>
void write_csr(const CSR<W>& g, const char* path) {
    FILE* fp = fopen(path, "wb");
    if (!fp) throw std::runtime_error(std::string("write_csr: cannot open ") + path);

    uint64_t crc = 0;

    auto write_and_crc = [&](const void* buf, size_t sz) {
        fwrite(buf, 1, sz, fp);
        crc = crc64_update(crc, buf, sz);
    };

    uint64_t magic = MAGIC;
    write_and_crc(&magic, 8);

    uint32_t tag = PrecTag<W>::value;
    write_and_crc(&tag, 4);

    uint64_t nv = g.n_vertices, ne = g.n_edges;
    write_and_crc(&nv, 8);
    write_and_crc(&ne, 8);

    // row_offsets: eid_t = uint64_t
    write_and_crc(g.row_offsets.data(), (nv + 1) * sizeof(eid_t));
    // col_indices: vid_t = uint32_t
    write_and_crc(g.col_indices.data(), ne * sizeof(vid_t));
    // weights: W
    write_and_crc(g.weights.data(), ne * sizeof(W));

    fwrite(&crc, 8, 1, fp);
    fclose(fp);
}

// ── Read ──────────────────────────────────────────────────────────────────────

template<typename W>
CSR<W> read_csr(const char* path) {
    FILE* fp = fopen(path, "rb");
    if (!fp) throw std::runtime_error(std::string("read_csr: cannot open ") + path);

    uint64_t crc = 0;
    auto read_and_crc = [&](void* buf, size_t sz) -> bool {
        if (fread(buf, 1, sz, fp) != sz) return false;
        crc = crc64_update(crc, buf, sz);
        return true;
    };

    uint64_t magic;
    if (!read_and_crc(&magic, 8) || magic != MAGIC) {
        fclose(fp);
        throw std::runtime_error("read_csr: bad magic");
    }

    uint32_t tag;
    if (!read_and_crc(&tag, 4)) {
        fclose(fp); throw std::runtime_error("read_csr: truncated");
    }
    uint32_t expected_tag = PrecTag<W>::value;
    if (tag != expected_tag) {
        fclose(fp);
        throw std::runtime_error("read_csr: precision tag mismatch");
    }

    uint64_t nv, ne;
    read_and_crc(&nv, 8);
    read_and_crc(&ne, 8);

    CSR<W> g;
    g.n_vertices = (vid_t)nv;
    g.n_edges    = (eid_t)ne;
    g.row_offsets.resize(nv + 1);
    g.col_indices.resize(ne);
    g.weights.resize(ne);

    read_and_crc(g.row_offsets.data(), (nv + 1) * sizeof(eid_t));
    read_and_crc(g.col_indices.data(), ne * sizeof(vid_t));
    read_and_crc(g.weights.data(),     ne * sizeof(W));

    uint64_t stored_crc;
    if (fread(&stored_crc, 8, 1, fp) != 1) {
        fclose(fp); throw std::runtime_error("read_csr: missing CRC");
    }
    fclose(fp);

    if (stored_crc != crc) throw std::runtime_error("read_csr: CRC mismatch");
    return g;
}

// ── Explicit instantiations ───────────────────────────────────────────────────

template void    write_csr<float> (const CSR<float>&,  const char*);
template void    write_csr<double>(const CSR<double>&, const char*);
template CSR<float>  read_csr<float> (const char*);
template CSR<double> read_csr<double>(const char*);
