let rigShaderSource = """

#include <metal_stdlib>
using namespace metal;

struct VertexIn {
    float2 position   [[attribute(0)]];
    float2 basis      [[attribute(1)]];
    float2 uv         [[attribute(2)]];
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
};

vertex VertexOut rig_vertex(uint vid [[vertex_id]],
                            const device VertexIn* verts [[buffer(0)]],
                            constant Uniforms& u [[buffer(1)]]) {
    VertexIn v = verts[vid];
    float2 p = v.position + v.basis * u.offset;
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
    // discard transparent
    if (c.a < 0.5) discard_fragment();
    // premultiply for blending
    return float4(c.rgb * c.a, c.a);
}

"""
