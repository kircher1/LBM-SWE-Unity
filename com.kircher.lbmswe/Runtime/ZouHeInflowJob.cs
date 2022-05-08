using System;
using Unity.Burst;
using Unity.Collections;
using Unity.Jobs;
using Unity.Mathematics;

namespace LatticeBoltzmannMethods
{
    /// <summary>
    /// Computes missing inlet link distributions by applying method by Zou and He
    /// and updates height and  velocity on inlet nodes.
    /// </summary>
    [BurstCompile]
    public struct ZouHeInflowJob : IJob
    {
        [ReadOnly]
        private int _latticeWidth;
        [ReadOnly]
        private int _latticeHeight;
        [ReadOnly]
        private float _inverseE;
        [ReadOnly]
        private NativeArray<byte> _solid;
        [ReadOnly]
        private float _inletWaterHeight;
        [ReadOnly]
        private float2 _inletVelocity;

        private NativeArray<float> _restDistribution;
        private NativeArray<float> _distribution;
        private NativeArray<float> _height;
        private NativeArray<float2> _velocity;

        public ZouHeInflowJob(
            int latticeWidth,
            int latticeHeight,
            float inverseE,
            NativeArray<byte> solid,
            float inletWaterHeight,
            float2 inletVelocity,
            NativeArray<float> restDistribution,
            NativeArray<float> distribution,
            NativeArray<float> height,
            NativeArray<float2> velocity)
        {
            if (inletVelocity.y != 0)
            {
                throw new ArgumentException("Inlet y-velocity must be zero.");
            }

            _latticeWidth = latticeWidth;
            _latticeHeight = latticeHeight;
            _inverseE = inverseE;
            _solid = solid;
            _inletWaterHeight = inletWaterHeight;
            _inletVelocity = inletVelocity;
            _restDistribution = restDistribution;
            _distribution = distribution;
            _height = height;
            _velocity = velocity;
        }

        public void Execute()
        {
            var u = _inletVelocity.x;
            for (var rowIdx = 0; rowIdx < _latticeHeight; rowIdx++)
            {
                var nodeIdx = rowIdx * _latticeWidth;

                // Really, this assumes no solid nodes on inlet column, but skip if encountered node is solid.
                if (_solid[nodeIdx] == 0)
                {
                    continue;
                }

                if (rowIdx == 0) // Bottom-left corner. (only works if diagonally adjacent flow node is not solid.)
                {
                    var neighborHeight = _height[nodeIdx + 1 + _latticeWidth];
                    _distribution[8 * nodeIdx + 0] = _distribution[8 * nodeIdx + 4];
                    _distribution[8 * nodeIdx + 1] = _distribution[8 * nodeIdx + 5];
                    _distribution[8 * nodeIdx + 2] = _distribution[8 * nodeIdx + 6];
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
                }
                else if (rowIdx == _latticeHeight - 1) // Top-left corner. (only works if diagonally adjacent flow node is not solid.)
                {
                    var neighborHeight = _height[nodeIdx + 1 - _latticeWidth];
                    _distribution[8 * nodeIdx + 0] = _distribution[8 * nodeIdx + 4];
                    _distribution[8 * nodeIdx + 6] = _distribution[8 * nodeIdx + 2];
                    _distribution[8 * nodeIdx + 7] = _distribution[8 * nodeIdx + 3];
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
                }
                else // General case, not a corner.
                {
                    _distribution[8 * nodeIdx + 0] = _distribution[8 * nodeIdx + 4] + (2.0f / 3.0f) * _inverseE * _inletWaterHeight * u;
                    _distribution[8 * nodeIdx + 1] = (1.0f / 6.0f) * _inverseE * _inletWaterHeight * u + _distribution[8 * nodeIdx + 5] + 0.5f * (_distribution[8 * nodeIdx + 6] - _distribution[8 * nodeIdx + 2]);
                    _distribution[8 * nodeIdx + 7] = (1.0f / 6.0f) * _inverseE * _inletWaterHeight * u + _distribution[8 * nodeIdx + 3] + 0.5f * (_distribution[8 * nodeIdx + 2] - _distribution[8 * nodeIdx + 6]);
                    _velocity[nodeIdx] = _inletVelocity;
                }

                _height[nodeIdx] = _inletWaterHeight;
            }
        }
    }
}
