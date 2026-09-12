import Foundation
import Metal

public final class MetalRuntime: @unchecked Sendable {
    public static let shared: MetalRuntime? = MetalRuntime()

    public let device: MTLDevice
    public let queue: MTLCommandQueue
    public let library: MTLLibrary

    private init?() {
        guard let d = MTLCreateSystemDefaultDevice(),
              let q = d.makeCommandQueue() else { return nil }
        self.device = d
        self.queue = q
        do {
            self.library = try d.makeLibrary(source: MetalRuntime.shaderSource, options: nil)
        } catch {
            FileHandle.standardError.write("metal library compile failed: \(error)\n".data(using: .utf8)!)
            return nil
        }
    }

    public static let shaderSource = """
    #include <metal_stdlib>
    using namespace metal;

    static inline float luma_of(device const float* rgba, uint i) {
        return 0.299f * rgba[i] + 0.587f * rgba[i + 1] + 0.114f * rgba[i + 2];
    }

    kernel void sobel_classify(
        device const float* rgba     [[ buffer(0) ]],
        device uchar*       classID  [[ buffer(1) ]],
        device float*       conf     [[ buffer(2) ]],
        constant uint&      w        [[ buffer(3) ]],
        constant uint&      h        [[ buffer(4) ]],
        constant float&     edgeTh   [[ buffer(5) ]],
        constant float&     texTh    [[ buffer(6) ]],
        uint2 gid [[ thread_position_in_grid ]]
    ) {
        if (gid.x >= w || gid.y >= h) return;
        uint x = gid.x, y = gid.y;
        uint idx = y * w + x;

        float g = 0.0f;
        bool interior = (x > 0) && (x + 1 < w) && (y > 0) && (y + 1 < h);
        if (interior) {
            uint tl = ((y - 1) * w + (x - 1)) * 4;
            uint tc = ((y - 1) * w +  x     ) * 4;
            uint tr = ((y - 1) * w + (x + 1)) * 4;
            uint ml = ( y      * w + (x - 1)) * 4;
            uint mr = ( y      * w + (x + 1)) * 4;
            uint bl = ((y + 1) * w + (x - 1)) * 4;
            uint bc = ((y + 1) * w +  x     ) * 4;
            uint br = ((y + 1) * w + (x + 1)) * 4;

            float a = luma_of(rgba, tl);
            float b = luma_of(rgba, tc);
            float c = luma_of(rgba, tr);
            float d = luma_of(rgba, ml);
            float e = luma_of(rgba, mr);
            float f = luma_of(rgba, bl);
            float gg = luma_of(rgba, bc);
            float hh = luma_of(rgba, br);

            float gx = (c + 2.0f * e + hh) - (a + 2.0f * d + f);
            float gy = (f + 2.0f * gg + hh) - (a + 2.0f * b + c);
            g = sqrt(gx * gx + gy * gy);
        }

        uchar cid;
        float cf;
        if (g >= edgeTh) {
            cid = 1;
            cf = min(1.0f, 0.5f + (g - edgeTh) / edgeTh);
        } else if (g >= texTh) {
            cid = 2;
            cf = 0.5f + (g - texTh) / (edgeTh - texTh) * 0.4f;
        } else {
            cid = 0;
            cf = min(1.0f, 0.6f + (texTh - g) / texTh * 0.4f);
        }
        classID[idx] = cid;
        conf[idx] = cf;
    }

    kernel void validate_pixels(
        device const uchar* classID   [[ buffer(0) ]],
        device const float* confidence[[ buffer(1) ]],
        device uchar* accept          [[ buffer(2) ]],
        device uchar* error           [[ buffer(3) ]],
        constant uint&  w             [[ buffer(4) ]],
        constant uint&  h             [[ buffer(5) ]],
        constant float& confThreshold [[ buffer(6) ]],
        constant float& neighborLimit [[ buffer(7) ]],
        uint2 gid                     [[ thread_position_in_grid ]]
    ) {
        if (gid.x >= w || gid.y >= h) return;
        uint idx = gid.y * w + gid.x;
        uchar c = classID[idx];
        float cf = confidence[idx];

        if (cf < confThreshold) {
            accept[idx] = 0; error[idx] = 1; return;
        }

        int disagree = 0;
        int total = 0;
        for (int dy = -1; dy <= 1; ++dy) {
            for (int dx = -1; dx <= 1; ++dx) {
                if (dx == 0 && dy == 0) continue;
                int nx = int(gid.x) + dx;
                int ny = int(gid.y) + dy;
                if (nx < 0 || ny < 0 || nx >= int(w) || ny >= int(h)) continue;
                uchar nc = classID[ny * int(w) + nx];
                int diff = int(nc) - int(c);
                if (diff < 0) diff = -diff;
                if (diff >= 2) disagree += 1;
                total += 1;
            }
        }

        float ratio = total > 0 ? float(disagree) / float(total) : 0.0f;
        if (ratio > neighborLimit) {
            accept[idx] = 0; error[idx] = 1;
        } else {
            accept[idx] = 1; error[idx] = 0;
        }
    }

    kernel void dilate3x3(
        device const uchar* src [[ buffer(0) ]],
        device uchar* dst       [[ buffer(1) ]],
        constant uint& w        [[ buffer(2) ]],
        constant uint& h        [[ buffer(3) ]],
        uint2 gid               [[ thread_position_in_grid ]]
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

    kernel void reprocess_vote(
        device const uchar* classID  [[ buffer(0) ]],
        device const float* confIn   [[ buffer(1) ]],
        device const uchar* errorMask[[ buffer(2) ]],
        device uchar*       outClass [[ buffer(3) ]],
        device float*       outConf  [[ buffer(4) ]],
        constant uint&      w        [[ buffer(5) ]],
        constant uint&      h        [[ buffer(6) ]],
        constant int&       radius   [[ buffer(7) ]],
        uint2 gid [[ thread_position_in_grid ]]
    ) {
        if (gid.x >= w || gid.y >= h) return;
        uint idx = gid.y * w + gid.x;
        outClass[idx] = classID[idx];
        outConf[idx]  = confIn[idx];

        if (errorMask[idx] == 0) return;

        int votes[3] = {0, 0, 0};
        int total = 0;
        for (int dy = -radius; dy <= radius; ++dy) {
            for (int dx = -radius; dx <= radius; ++dx) {
                if (dx == 0 && dy == 0) continue;
                int nx = int(gid.x) + dx;
                int ny = int(gid.y) + dy;
                if (nx < 0 || ny < 0 || nx >= int(w) || ny >= int(h)) continue;
                uint nidx = uint(ny) * w + uint(nx);
                if (errorMask[nidx] == 1) continue;
                votes[classID[nidx]] += 1;
                total += 1;
            }
        }
        if (total == 0) return;
        int best = 0;
        for (int i = 1; i < 3; ++i) if (votes[i] > votes[best]) best = i;
        outClass[idx] = uchar(best);
        outConf[idx]  = 0.80f;
    }
    """
}
