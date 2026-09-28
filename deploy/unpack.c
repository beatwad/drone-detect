/* Packed accelerator output -> float32, for finn/util/data_packing.py's fast
 * path. NumPy's uint64 shift-and-mask takes ~7 ms per frame on the A53
 * (build_notes §11.14); this is the same arithmetic in C.
 *
 * Each of `words` packed words is `nbytes` little-endian bytes (<= 8) holding
 * `n` fields of `bits` bits, element 0 in the lowest bits -- FINN's layout with
 * reverse_inner and reverse_endian both set. Fields are sign-extended if `sgn`,
 * mapped 0/1 -> -1/+1 if `bipolar`, and written as float32, as FINN returns.
 * No libc, so the .so has no dependency on the board's glibc.
 *
 * Build (host, Vitis 2022.2 cross compiler):
 *   ~/Xilinx/Vitis/2022.2/gnu/aarch64/lin/aarch64-linux/bin/aarch64-linux-gnu-gcc \
 *       -O2 -shared -fPIC -nostdlib -o deploy/libunpack.so deploy/unpack.c
 */
#include <stdint.h>

void unpack_le_fields(const uint8_t *src, long words, int nbytes, int n, int bits,
                      int sgn, int bipolar, float *dst)
{
    const uint64_t mask = (1ULL << bits) - 1;
    const int64_t sign = 1LL << (bits - 1);
    for (long w = 0; w < words; w++, src += nbytes) {
        uint64_t acc = 0;
        for (int b = 0; b < nbytes; b++)
            acc |= (uint64_t)src[b] << (8 * b);
        for (int j = 0; j < n; j++) {
            int64_t v = (int64_t)((acc >> (j * bits)) & mask);
            if (bipolar)
                v = 2 * v - 1;
            else if (sgn)
                v = (v ^ sign) - sign;
            *dst++ = (float)v;
        }
    }
}
