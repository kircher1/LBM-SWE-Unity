using Unity.Burst;
using Unity.Collections;
using Unity.Jobs;
using Unity.Mathematics;

namespace LatticeBoltzmannMethods
{
    /// <summary>
    /// Assumes destination is two bytes per pixel.
    /// </summary>
    [BurstCompile]
    struct UpdateForceTextureJob : IJobParallelFor
    {
        [ReadOnly]
        private int _textureWidth;
        [ReadOnly]
        private NativeArray<float2> _minMaxForce;
        [ReadOnly]
        private NativeArray<float2> _force;

        [NativeDisableParallelForRestriction]
        private NativeArray<byte> _destination;

        public UpdateForceTextureJob(int textureWidth, NativeArray<float2> minMaxForce, NativeArray<float2> force, NativeArray<byte> destination)
        {
            _textureWidth = textureWidth;
            _minMaxForce = minMaxForce;
            _force = force;
            _destination = destination;
        }

        public void Execute(int rowIdx)
        {
            var minForce = _minMaxForce[0];
            var maxForce = _minMaxForce[1];
            var forceRange = maxForce - minForce;

            var startIdx = rowIdx * _textureWidth;
            for (var idx = startIdx; idx < startIdx + _textureWidth; ++idx)
            {
                var force = _force[idx];
                var rescaled = math.round(255.0f * math.saturate((force - minForce) / forceRange));
                _destination[2 * idx + 0] = (byte)rescaled.x;
                _destination[2 * idx + 1] = (byte)rescaled.y;
            }
        }
    }
}
