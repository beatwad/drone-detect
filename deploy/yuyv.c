/* YUYV window -> RGB, for capture.py. NumPy takes 13 ms per 320x192 window on
 * the A53; this is the same arithmetic in C, 1.6 ms (build_notes §12.8).
 *
 * Bit-exact with OpenCV's COLOR_YUV2RGB_YUYV: ITU-R BT.601 limited range, 20-bit
 * fixed point. No libc, so the .so has no dependency on the board's glibc.
 *
 * Build (host, Vitis 2022.2 cross compiler):
 *   ~/Xilinx/Vitis/2022.2/gnu/aarch64/lin/aarch64-linux/bin/aarch64-linux-gnu-gcc \
 *       -O2 -shared -fPIC -nostdlib -o deploy/libyuyv.so deploy/yuyv.c
 */
#include <stdint.h>

#define SHIFT 20
#define CY 1220542
#define CUB 2116026
#define CUG (-409993)
#define CVG (-852492)
#define CVR 1673527

static inline uint8_t clip(int32_t x)
{
    return x < 0 ? 0 : x > 255 ? 255 : (uint8_t)x;
}

/* src: YUYV frame, `stride` bytes per row. Converts the w x h window whose
 * top-left pixel is (x0, y0) -- x0 even -- into dst, packed RGB, w*3 per row. */
void yuyv_window_to_rgb(const uint8_t *src, int stride, int x0, int y0,
                        int w, int h, uint8_t *dst)
{
    for (int r = 0; r < h; r++) {
        const uint8_t *s = src + (y0 + r) * stride + 2 * x0;
        uint8_t *d = dst + r * w * 3;
        for (int c = 0; c < w; c += 2, s += 4, d += 6) {
            int32_t u = s[1] - 128, v = s[3] - 128;
            int32_t half = 1 << (SHIFT - 1);
            int32_t ruv = half + CVR * v;
            int32_t guv = half + CVG * v + CUG * u;
            int32_t buv = half + CUB * u;
            int32_t y0v = (s[0] > 16 ? s[0] - 16 : 0) * CY;
            int32_t y1v = (s[2] > 16 ? s[2] - 16 : 0) * CY;
            d[0] = clip((y0v + ruv) >> SHIFT);
            d[1] = clip((y0v + guv) >> SHIFT);
            d[2] = clip((y0v + buv) >> SHIFT);
            d[3] = clip((y1v + ruv) >> SHIFT);
            d[4] = clip((y1v + guv) >> SHIFT);
            d[5] = clip((y1v + buv) >> SHIFT);
        }
    }
}
