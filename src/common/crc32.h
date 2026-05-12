#pragma once

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <vector>

static inline uint32_t crc32_zlib(const void* data, size_t n) {
    static uint32_t table[256];
    static bool init = false;
    if (!init) {
        for (uint32_t i = 0; i < 256; ++i) {
            uint32_t c = i;
            for (int k = 0; k < 8; ++k) c = (c & 1) ? (0xEDB88320u ^ (c >> 1)) : (c >> 1);
            table[i] = c;
        }
        init = true;
    }
    uint32_t c = 0xFFFFFFFFu;
    const uint8_t* p = (const uint8_t*)data;
    for (size_t i = 0; i < n; ++i) c = table[(c ^ p[i]) & 0xFFu] ^ (c >> 8);
    return c ^ 0xFFFFFFFFu;
}

template<typename T>
static inline void crc32_hex(const std::vector<T>& data, char buf[9]) {
    uint32_t crc = crc32_zlib(data.data(), data.size() * sizeof(T));
    std::snprintf(buf, 9, "%08x", crc);
}
