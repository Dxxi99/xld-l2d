let rigShaderSource = """

#include <metal_stdlib>
using namespace metal;

struct VertexIn {
    float2 position   [[attribute(0)]];
    float2 basis      [[attribute(1)]];
    float2 uv         [[attribute(2)]];
    float  z          [[attribute(3)]];
    float  pad        [[attribute(4)]];
};

struct VertexOut {
    float4 position [[position]];
    float2 uv;
};

struct Uniforms {
    float2 viewport;
    float2 image_size;
    float2 offset;
    float  scale;
    float  pad;
    float  head_weight_base;
    float  head_z_lo;
    float  head_z_hi;
    float  lower_y_start;
    float  lower_y_end;
    float  lower_fade;
    float  yaw_rad;
    float  pitch_rad;
    float  face_pivot_y;
    float  face_rot_scale;
    float  face_persp;
    float  reserved3;
};

vertex VertexOut rig_vertex(uint vid [[vertex_id]],
                            const device VertexIn* verts [[buffer(0)]],
                            constant Uniforms& u [[buffer(1)]]) {
    VertexIn v = verts[vid];
    float z = clamp(v.z, 0.0, 1.0);
    float t = clamp((z - u.head_z_lo) / max(u.head_z_hi - u.head_z_lo, 1e-6), 0.0, 1.0);
    float ss = t * t * (3.0 - 2.0 * t);

    // ---- 얼굴: 가로 압축만 (회전/shear 없음) ----
    float2 facePivot = float2(u.image_size.x * 0.5, u.image_size.y * u.face_pivot_y);
    float2 dF = v.position - facePivot;
    float scaleX = 1.0 - abs(u.yaw_rad) * u.face_persp;
    float scaleY = 1.0 - abs(u.pitch_rad) * u.face_persp * 0.6;
    float2 facePos = facePivot + float2(dF.x * scaleX, dF.y * scaleY);
    float2 pos = mix(v.position, facePos, ss);

    // ---- 전역 변형 ----
    float w = u.head_weight_base + (1.0 - u.head_weight_base) * ss;
    float yn = clamp(pos.y / u.image_size.y, 0.0, 1.0);
    float lf = clamp((yn - u.lower_y_start) / max(u.lower_y_end - u.lower_y_start, 1e-6), 0.0, 1.0);
    float lfs = lf * lf * (3.0 - 2.0 * lf);
    w *= (1.0 - lfs * u.lower_fade);
    float2 p = pos + v.basis * u.offset * w;
    float2 centered = p - u.image_size * 0.5;
    float2 screen = centered * u.scale;
    float2 ndc = float2(screen.x / (u.viewport.x * 0.5),
                        -screen.y / (u.viewport.y * 0.5));
    VertexOut o;
    o.position = float4(ndc, 0, 1);
    o.uv = v.uv;
    return o;
}

fragment float4 rig_fragment(VertexOut in [[stage_in]],
                             texture2d<float> tex [[texture(0)]],
                             sampler s [[sampler(0)]]) {
    float4 c = tex.sample(s, in.uv);
    if (c.a < 0.5) discard_fragment();
    return float4(c.rgb * c.a, c.a);
}

"""
