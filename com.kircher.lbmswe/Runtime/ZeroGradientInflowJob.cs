using Unity.Burst;
using Unity.Collections;
using Unity.Jobs;

namespace LatticeBoltzmannMethods
{
    /// <summary>
    /// Computes missing inlet link distributions by copying from neighboring column.
    /// TODO: Handle corners.
    /// </summary>
    [BurstCompile]
    public struct ZeroGradientInflowJob : IJob
    {
        [ReadOnly]
        private int _latticeWidth;
        [ReadOnly]
        private int _latticeHeight;
        [ReadOnly]
        private NativeArray<bool> _solid;

        private NativeArray<float> _distribution;

        public ZeroGradientInflowJob(
            int latticeWidth,
            int latticeHeight,
            NativeArray<bool> solid,
            NativeArray<float> distribution)
        {
            _latticeWidth = latticeWidth;
            _latticeHeight = latticeHeight;
            _solid = solid;
            _distribution = distribution;
        }

        public void Execute()
        {
            for (var rowIdx = 0; rowIdx < _latticeHeight; rowIdx++)
            {
                var nodeIdx = rowIdx * _latticeWidth;
                if (!_solid[nodeIdx] && !_solid[nodeIdx + 1])
                {
                    var neighborNodeIdx = nodeIdx + 1;
                    _distribution[9 * nodeIdx + 1] = _distribution[9 * neighborNodeIdx + 1];
                    _distribution[9 * nodeIdx + 2] = _distribution[9 * neighborNodeIdx + 2];
                    _distribution[9 * nodeIdx + 8] = _distribution[9 * neighborNodeIdx + 8];
                }
            }
        }
    }
}
