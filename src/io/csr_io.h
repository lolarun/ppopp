#pragma once
// csr_io.h — Binary CSR serialisation / deserialisation.
//
// File format (.csr):
//   [8 bytes]  magic = 0x43535200'00000001ULL  (CSR\0 + version 1)
//   [4 bytes]  precision tag: 0x46503332 = "FP32", 0x46503634 = "FP64"
//   [8 bytes]  n_vertices (uint64_t)
//   [8 bytes]  n_edges    (uint64_t)
//   [n_vertices+1 × 8 bytes]  row_offsets (uint64_t[])
//   [n_edges      × 4 bytes]  col_indices (uint32_t[])
//   [n_edges      × sizeof(W)] weights    (W[])
//   [8 bytes]  CRC64 of all preceding bytes
//
// CSR files are the primary on-disk format; all experiments read from .csr.
// Conversion: DIMACS/SNAP/GAP → CSR via scripts/preprocess.sh.

#include "../core/csr_sssp.h"
#include <cstdint>

// Write CSR to binary file.  Overwrites if exists.
template<typename W>
void write_csr(const CSR<W>& g, const char* path);

// Read CSR from binary file.  Throws std::runtime_error on format/CRC error.
template<typename W>
CSR<W> read_csr(const char* path);

extern template void    write_csr<float> (const CSR<float>&,  const char*);
extern template void    write_csr<double>(const CSR<double>&, const char*);
extern template CSR<float>  read_csr<float> (const char*);
extern template CSR<double> read_csr<double>(const char*);
