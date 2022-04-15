using Unity.Burst;
using Unity.Collections;
using Unity.Jobs;
using Unity.Mathematics;

namespace LatticeBoltzmannMethods
{
    [BurstCompile(FloatPrecision.Standard, FloatMode.Fast)]
    struct UpdateFlowTextureJob : IJobParallelFor
    {
        private static readonly float2 Half = new float2(0.5f, 0.5f);

        [ReadOnly]
        private int _textureWidth;
        [ReadOnly]
        private float _oneHalfOverMaxSpeed;
        [ReadOnly]
        private NativeArray<byte> _solid;
        [ReadOnly]
        private NativeArray<float2> _velocity;

        [WriteOnly]
        [NativeDisableParallelForRestriction]
        private NativeArray<byte> _destination;

        public UpdateFlowTextureJob(int textureWidth, float maxSpeed, NativeArray<byte> solid, NativeArray<float2> velocity, NativeArray<byte> destination)
        {
            _textureWidth = textureWidth;
            _oneHalfOverMaxSpeed = 0.5f / maxSpeed;
            _solid = solid;
            _velocity = velocity;
            _destination = destination;
        }

        public void Execute(int rowIdx)
        {
            var startIdx = rowIdx * _textureWidth;
            for (var nodeIdx = startIdx; nodeIdx < startIdx + _textureWidth; ++nodeIdx)
            {
                var velocity = _velocity[nodeIdx];
                var rescaledValue = _solid[nodeIdx] * _oneHalfOverMaxSpeed * velocity + Half;
                var bytes = math.round(255.0f * math.saturate(rescaledValue));
                _destination[2 * nodeIdx + 0] = (byte)bytes.x;
                _destination[2 * nodeIdx + 1] = (byte)bytes.y;
            }
        }
    }
}
