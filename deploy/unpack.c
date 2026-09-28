/* Packed accelerator output -> float32, for finn/util/data_packing.py's fast
 * path. NumPy's uint64 shift-and-mask takes ~7 ms per frame on the A53
 * (build_notes §11.14); this is the same arithmetic in C.
 *
 * Each of `words` packed words is `nbytes` little-endian bytes (<= 8) holding
 * `n` fields of `bits` bits, element 0 in the lowest bits -- FINN's layout with
 * reverse_inner and reverse_endian both set. Fields are sign-extended if `sgn`,
 * mapped 0/1 -> -1/+1 if `bipolar`, and written as float32, as FINN returns.
 * No libc, so the .so has no dependency on the board's glibc. scan_cells, at
 * the end, serves postprocess.decode_packed.
 *
 * Build (host, Vitis 2022.2 cross compiler):
 *   ~/Xilinx/Vitis/2022.2/gnu/aarch64/lin/aarch64-linux/bin/aarch64-linux-gnu-gcc \
 *       -O2 -ffp-contract=off -shared -fPIC -nostdlib -o deploy/libunpack.so deploy/unpack.c
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

static inline int64_t read_signed(const uint8_t *p, int nbytes, int bits)
{
    uint64_t acc = 0;
    for (int b = 0; b < nbytes; b++)
        acc |= (uint64_t)p[b] << (8 * b);
    const int64_t sign = 1LL << (bits - 1);
    return ((int64_t)(acc & ((1ULL << bits) - 1)) ^ sign) - sign;
}

/* For postprocess.decode_packed: scan the packed output of `cells` cells x `nch`
 * signed `bits`-bit elements, one element per `nbytes`-byte word, straight out
 * of the DMA buffer. A cell passes if its channel `key` dequantizes to a logit
 * >= logit_lo; each passing cell's index goes to idx and all its nch elements,
 * as float32, to vals. Returns the number of passing cells (at most max_n).
 * The caller picks logit_lo a little below the threshold and re-applies the
 * exact test in NumPy, so float rounding here can only let extra cells in. */
long scan_cells(const uint8_t *src, long cells, int nch, int nbytes, int bits, int key,
                float scale_k, float bias_k, float logit_lo, int32_t *idx, float *vals,
                long max_n)
{
    long n = 0;
    for (long c = 0; c < cells && n < max_n; c++) {
        const uint8_t *cell = src + c * nch * nbytes;
        float logit = (float)read_signed(cell + key * nbytes, nbytes, bits) * scale_k + bias_k;
        if (!(logit >= logit_lo))
            continue;
        idx[n] = (int32_t)c;
        for (int ch = 0; ch < nch; ch++)
            vals[n * nch + ch] = (float)read_signed(cell + ch * nbytes, nbytes, bits);
        n++;
    }
    return n;
}
