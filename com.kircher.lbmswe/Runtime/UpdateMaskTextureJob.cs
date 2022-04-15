using Unity.Burst;
using Unity.Collections;
using Unity.Jobs;

namespace LatticeBoltzmannMethods
{
    [BurstCompile]
    struct UpdateMaskTextureJob : IJobParallelFor
    {
        [ReadOnly]
        private int _textureWidth;
        [ReadOnly]
        private NativeArray<byte> _solid;

        [WriteOnly]
        [NativeDisableParallelForRestriction]
        private NativeArray<byte> _destination;

        public UpdateMaskTextureJob(int textureWidth, NativeArray<byte> solid, NativeArray<byte> destination)
        {
            _textureWidth = textureWidth;
            _solid = solid;
            _destination = destination;
        }

        public void Execute(int rowIdx)
        {
            var startIdx = rowIdx * _textureWidth;
            for (var idx = startIdx; idx < startIdx + _textureWidth; ++idx)
            {
                _destination[idx] = (byte)((1 - _solid[idx]) * 255);
            }
        }
    }
}
