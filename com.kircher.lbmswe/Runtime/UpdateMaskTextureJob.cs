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
        private NativeArray<bool> _solid;
        [NativeDisableParallelForRestriction]
        private NativeArray<byte> _destination;

        public UpdateMaskTextureJob(int textureWidth, NativeArray<bool> solid, NativeArray<byte> destination)
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
                _destination[idx] = _solid[idx] ? (byte)255 : (byte)0;
            }
        }
    }
}
