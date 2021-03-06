using Unity.Burst;
using Unity.Collections;
using Unity.Jobs;
using Unity.Mathematics;

namespace LatticeBoltzmannMethods
{
    [BurstCompile]
    struct UpdateHeightTextureJob : IJobParallelFor
    {
        [ReadOnly]
        private int _textureWidth;
        [ReadOnly]
        private float _maxHeight;
        [ReadOnly]
        private NativeArray<float> _waterHeight;

        [WriteOnly]
        [NativeDisableParallelForRestriction]
        private NativeArray<half> _destination;

        public UpdateHeightTextureJob(int textureWidth, float maxHeight, NativeArray<float> waterHeight, NativeArray<half> destination)
        {
            _textureWidth = textureWidth;
            _maxHeight = maxHeight;
            _waterHeight = waterHeight;
            _destination = destination;
        }

        public void Execute(int rowIdx)
        {
            var startIdx = rowIdx * _textureWidth;
            for (var idx = startIdx; idx < startIdx + _textureWidth; ++idx)
            {
                var height = _waterHeight[idx];
                _destination[idx] = (half)math.saturate(height / _maxHeight);
            }
        }
    }
}
