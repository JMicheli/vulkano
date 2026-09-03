struct VSInput {
    float3 position : POSITION;
    float3 normal : NORMAL;
};

struct VSOutput {
    float4 position : SV_Position;
    float3 v_normal : NORMAL;
};

struct Data {
    float4x4 world;
    float4x4 view;
    float4x4 proj;
    float4x4 norm;
};

ConstantBuffer<Data> uniforms;

[shader("vertex")]
VSOutput main(VSInput input) {
    VSOutput output;

    float4x4 worldview = mul(uniforms.view, uniforms.world);
    output.v_normal = mul((float3x3)uniforms.norm, input.normal);
    output.position = mul(uniforms.proj, mul(worldview, float4(input.position, 1.0)));

    return output;
}
