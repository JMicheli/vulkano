struct FSInput {
    float3 v_normal;
};

static const float3 LIGHT = float3(0.0, 0.0, 1.0);

[shader("fragment")]
float4 main(FSInput input) {
    float brightness = dot(normalize(input.v_normal), normalize(LIGHT));
    float3 dark_color = float3(0.6, 0.0, 0.0);
    float3 regular_color = float3(1.0, 0.0, 0.0);

    return float4(lerp(dark_color, regular_color, brightness), 1.0);
}
