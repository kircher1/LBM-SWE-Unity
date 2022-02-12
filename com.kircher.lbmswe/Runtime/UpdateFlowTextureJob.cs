using Unity.Burst;
using Unity.Collections;
using Unity.Jobs;
using Unity.Mathematics;

[BurstCompile]
struct UpdateFlowTextureJob : IJobParallelFor
{
    private static readonly float2 Half = new float2(0.5f, 0.5f);

    [ReadOnly]
    private int _textureWidth;
    [ReadOnly]
    private float _maxSpeed;
    [ReadOnly]
    private float _textureScale;
    [ReadOnly]
    private NativeArray<bool> _solid;
    [ReadOnly]
    private NativeArray<float2> _velocity;

    [NativeDisableParallelForRestriction]
    private NativeArray<byte> _destination;

    public UpdateFlowTextureJob(int textureWidth, float maxSpeed, float textureScale, NativeArray<bool> solid, NativeArray<float2> velocity, NativeArray<byte> destination)
    {
        _textureWidth = textureWidth;
        _maxSpeed = maxSpeed;
        _textureScale = textureScale;
        _solid = solid;
        _velocity = velocity;
        _destination = destination;
    }

    public void Execute(int rowIdx)
    {
        var startIdx = rowIdx * _textureWidth;
        for (var nodeIdx = startIdx; nodeIdx < startIdx + _textureWidth; ++nodeIdx)
        {
            if (_solid[nodeIdx])
            {
                _destination[2 * nodeIdx + 0] = 127;
                _destination[2 * nodeIdx + 1] = 127;
            }
            else
            {
                var velocity = _velocity[nodeIdx];
                var rescaledValue = 0.5f * math.sign(velocity) * math.pow(math.abs(velocity) / _maxSpeed, _textureScale) + Half;
                var bytes = math.round(255.0f * math.saturate(rescaledValue));
                _destination[2 * nodeIdx + 0] = (byte)bytes.x;
                _destination[2 * nodeIdx + 1] = (byte)bytes.y;
            }
        }
    }
}
