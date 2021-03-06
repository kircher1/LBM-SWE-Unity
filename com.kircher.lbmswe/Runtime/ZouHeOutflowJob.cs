using Unity.Burst;
using Unity.Collections;
using Unity.Jobs;
using Unity.Mathematics;

namespace LatticeBoltzmannMethods
{
    /// <summary>
    /// Computes missing outlet link distributions by applying method by Zou and He
    /// and updates height and  velocity on outlet nodes.
    /// </summary>
    [BurstCompile]
    public struct ZouHeOutflowJob : IJob
    {
        [ReadOnly]
        private int _latticeWidth;
        [ReadOnly]
        private int _latticeHeight;
        [ReadOnly]
        private float _inverseE;
        [ReadOnly]
        private NativeArray<byte> _solid;

        private NativeArray<float> _restDistribution;
        private NativeArray<float> _distribution;
        private NativeArray<float> _height;
        private NativeArray<float2> _velocity;

        public ZouHeOutflowJob(
            int latticeWidth,
            int latticeHeight,
            float inverseE,
            NativeArray<byte> solid,
            NativeArray<float> restDistribution,
            NativeArray<float> distribution,
            NativeArray<float> height,
            NativeArray<float2> velocity)
        {
            _latticeWidth = latticeWidth;
            _latticeHeight = latticeHeight;
            _inverseE = inverseE;
            _solid = solid;
            _restDistribution = restDistribution;
            _distribution = distribution;
            _height = height;
            _velocity = velocity;
        }

        public void Execute()
        {
            for (var rowIdx = 0; rowIdx < _latticeHeight; rowIdx++)
            {
                // Really, this assumes no solid nodes on inlet column, but skip if encountered node is solid.
                var nodeIdx = rowIdx * _latticeWidth + _latticeWidth - 1;
                if (_solid[nodeIdx] == 0)
                {
                    continue;
                }

                if (rowIdx == 0) // Bottom-right
                {
                    var neighborHeight = _height[nodeIdx - 1 + _latticeWidth];
                    _distribution[8 * nodeIdx + 2] = _distribution[8 * nodeIdx + 6];
                    _distribution[8 * nodeIdx + 3] = _distribution[8 * nodeIdx + 7];
                    _distribution[8 * nodeIdx + 4] = _distribution[8 * nodeIdx + 0];
                    var distributionSumFor2and6 =
                        _restDistribution[nodeIdx] +
                        _distribution[8 * nodeIdx + 0] +
                        _distribution[8 * nodeIdx + 2] +
                        _distribution[8 * nodeIdx + 3] +
                        _distribution[8 * nodeIdx + 4] +
                        _distribution[8 * nodeIdx + 6] +
                        _distribution[8 * nodeIdx + 7];
                    _distribution[8 * nodeIdx + 1] = _distribution[8 * nodeIdx + 5] = 0.5f * (neighborHeight - distributionSumFor2and6);
                    _velocity[nodeIdx] = float2.zero;
                    _height[nodeIdx] = neighborHeight;
                }
                else if (rowIdx == _latticeHeight - 1) // Top-right
                {
                    var neighborHeight = _height[nodeIdx - 1 - _latticeWidth];
                    _distribution[8 * nodeIdx + 4] = _distribution[8 * nodeIdx + 2];
                    _distribution[8 * nodeIdx + 5] = _distribution[8 * nodeIdx + 1];
                    _distribution[8 * nodeIdx + 6] = _distribution[8 * nodeIdx + 0];
                    var distributionSumFor4and8 =
                        _restDistribution[nodeIdx] +
                        _distribution[8 * nodeIdx + 0] +
                        _distribution[8 * nodeIdx + 1] +
                        _distribution[8 * nodeIdx + 2] +
                        _distribution[8 * nodeIdx + 4] +
                        _distribution[8 * nodeIdx + 5] +
                        _distribution[8 * nodeIdx + 6];
                    _distribution[8 * nodeIdx + 3] = _distribution[8 * nodeIdx + 7] = 0.5f * (neighborHeight - distributionSumFor4and8);
                    _velocity[nodeIdx] = float2.zero;
                    _height[nodeIdx] = neighborHeight;
                }
                else
                {
                    var neighborHeight = _height[nodeIdx - 1];
                    var neighborVelocity = _velocity[nodeIdx - 1];
                    var u = neighborVelocity.x;
                    _distribution[8 * nodeIdx + 4] = _distribution[8 * nodeIdx + 0] - (2.0f / 3.0f) * _inverseE * neighborHeight * u;
                    _distribution[8 * nodeIdx + 3] = -(1.0f / 6.0f) * _inverseE * neighborHeight * u + _distribution[8 * nodeIdx + 7] + 0.5f * (_distribution[8 * nodeIdx + 6] - _distribution[8 * nodeIdx + 2]);
                    _distribution[8 * nodeIdx + 5] = -(1.0f / 6.0f) * _inverseE * neighborHeight * u + _distribution[8 * nodeIdx + 1] + 0.5f * (_distribution[8 * nodeIdx + 2] - _distribution[8 * nodeIdx + 6]);
                    _velocity[nodeIdx] = new float2(u, 0.0f);
                    _height[nodeIdx] = neighborHeight;
                }
            }
        }
    }
}
