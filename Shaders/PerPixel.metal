#include <metal_stdlib>
using namespace metal;

// 예시: per-pixel confidence threshold
kernel void validate_confidence(
    device const float *confidence [[ buffer(0) ]],
    device uchar       *accept     [[ buffer(1) ]],
    constant float     &threshold  [[ buffer(2) ]],
    uint                gid        [[ thread_position_in_grid ]]
) {
    accept[gid] = confidence[gid] >= threshold ? 1 : 0;
}

// 예시: dirty region dilate
kernel void dilate_mask(
    device const uchar *src [[ buffer(0) ]],
    device uchar       *dst [[ buffer(1) ]],
    constant uint      &w   [[ buffer(2) ]],
    constant uint      &h   [[ buffer(3) ]],
    uint2               gid [[ thread_position_in_grid ]]
) {
    if (gid.x >= w || gid.y >= h) return;
    uchar v = 0;
    for (int dy = -1; dy <= 1; ++dy) {
        for (int dx = -1; dx <= 1; ++dx) {
            int nx = int(gid.x) + dx;
            int ny = int(gid.y) + dy;
            if (nx < 0 || ny < 0 || nx >= int(w) || ny >= int(h)) continue;
            v |= src[ny * int(w) + nx];
        }
    }
    dst[gid.y * w + gid.x] = v;
}
