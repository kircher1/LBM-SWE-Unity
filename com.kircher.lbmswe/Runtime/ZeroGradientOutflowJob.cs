using Unity.Burst;
using Unity.Collections;
using Unity.Jobs;
using Unity.Mathematics;

namespace LatticeBoltzmannMethods
{
    /// <summary>
    /// Computes missing outlet link distributions by copying from neighboring column.
    /// </summary>
    [BurstCompile]
    public struct ZeroGradientOutflowJob : IJob
    {
        [ReadOnly]
        private int _latticeWidth;
        [ReadOnly]
        private int _latticeHeight;
        [ReadOnly]
        private NativeArray<bool> _solid;

        private NativeArray<float> _distribution;
        private NativeArray<float> _height;
        private NativeArray<float2> _velocity;

        public ZeroGradientOutflowJob(
            int latticeWidth,
            int latticeHeight,
            NativeArray<bool> solid,
            NativeArray<float> distribution,
            NativeArray<float> height,
            NativeArray<float2> velocity)
        {
            _latticeWidth = latticeWidth;
            _latticeHeight = latticeHeight;
            _solid = solid;
            _distribution = distribution;
            _height = height;
            _velocity = velocity;
        }

        public void Execute()
        {
            for (var rowIdx = 0; rowIdx < _latticeHeight; rowIdx++)
            {
                var nodeIdx = rowIdx * _latticeWidth + _latticeWidth - 1;
                if (_solid[nodeIdx] || _solid[nodeIdx - 1])
                {
                    continue;
                }

                if (rowIdx == 0) // bottom-right corner
                {
                    // 2 and 3 also undertmined. Take from upper-left neighbor.
                    var neighborNodeIdx = nodeIdx + _latticeWidth - 1;
                    _distribution[9 * nodeIdx + 2] = _distribution[9 * neighborNodeIdx + 2];
                    _distribution[9 * nodeIdx + 3] = _distribution[9 * neighborNodeIdx + 3];
                    _distribution[9 * nodeIdx + 4] = _distribution[9 * neighborNodeIdx + 4];
                    _distribution[9 * nodeIdx + 5] = _distribution[9 * neighborNodeIdx + 5];
                    _distribution[9 * nodeIdx + 6] = _distribution[9 * neighborNodeIdx + 6];
                    _velocity[nodeIdx] = _velocity[neighborNodeIdx];
                    _height[nodeIdx] = _height[neighborNodeIdx];
                }
                else if (rowIdx == _latticeHeight - 1) // top-right corner
                {
                    // 7 and 8 also undertmined. Take from lower-left neighbor.
                    var neighborNodeIdx = nodeIdx - _latticeWidth - 1;
                    _distribution[9 * nodeIdx + 4] = _distribution[9 * neighborNodeIdx + 4];
                    _distribution[9 * nodeIdx + 5] = _distribution[9 * neighborNodeIdx + 5];
                    _distribution[9 * nodeIdx + 6] = _distribution[9 * neighborNodeIdx + 6];
                    _distribution[9 * nodeIdx + 7] = _distribution[9 * neighborNodeIdx + 7];
                    _distribution[9 * nodeIdx + 8] = _distribution[9 * neighborNodeIdx + 8];
                    _velocity[nodeIdx] = _velocity[neighborNodeIdx];
                    _height[nodeIdx] = _height[neighborNodeIdx];
                }
                else
                {
                    var neighborNodeIdx = nodeIdx - 1;
                    _distribution[9 * nodeIdx + 4] = _distribution[9 * neighborNodeIdx + 4];
                    _distribution[9 * nodeIdx + 5] = _distribution[9 * neighborNodeIdx + 5];
                    _distribution[9 * nodeIdx + 6] = _distribution[9 * neighborNodeIdx + 6];
                    _velocity[nodeIdx] = _velocity[neighborNodeIdx];
                    _height[nodeIdx] = _height[neighborNodeIdx];
                }
            }
        }
    }
}
