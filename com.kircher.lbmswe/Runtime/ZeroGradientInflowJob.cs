﻿using Unity.Burst;
using Unity.Collections;
using Unity.Jobs;
using Unity.Mathematics;

namespace LatticeBoltzmannMethods
{
    /// <summary>
    /// Computes missing inlet link distributions by copying from neighboring column.
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
        [ReadOnly]
        private float _inletWaterHeight;
        [ReadOnly]
        private float2 _inletVelocity;

        private NativeArray<float> _distribution;
        private NativeArray<float> _height;
        private NativeArray<float2> _velocity;

        public ZeroGradientInflowJob(
            int latticeWidth,
            int latticeHeight,
            NativeArray<bool> solid,
            float inletWaterHeight,
            float2 inletVelocity,
            NativeArray<float> distribution,
            NativeArray<float> height,
            NativeArray<float2> velocity)
        {
            _latticeWidth = latticeWidth;
            _latticeHeight = latticeHeight;
            _solid = solid;
            _inletWaterHeight = inletWaterHeight;
            _inletVelocity = inletVelocity;
            _distribution = distribution;
            _height = height;
            _velocity = velocity;
        }

        public void Execute()
        {
            for (var rowIdx = 0; rowIdx < _latticeHeight; rowIdx++)
            {
                var nodeIdx = rowIdx * _latticeWidth;
                if (_solid[nodeIdx] || _solid[nodeIdx + 1])
                {
                    continue;
                }

                if (rowIdx == 0) // bottom-left corner
                {
                    // 3 and 4 also undertmined. Take from upper-right neighbor.
                    var neighborNodeIdx = nodeIdx + _latticeWidth + 1;
                    _distribution[9 * nodeIdx + 1] = _distribution[9 * neighborNodeIdx + 1];
                    _distribution[9 * nodeIdx + 2] = _distribution[9 * neighborNodeIdx + 2];
                    _distribution[9 * nodeIdx + 8] = _distribution[9 * neighborNodeIdx + 8];
                    _distribution[9 * nodeIdx + 3] = _distribution[9 * neighborNodeIdx + 3];
                    _distribution[9 * nodeIdx + 4] = _distribution[9 * neighborNodeIdx + 4];
                    _velocity[nodeIdx] = float2.zero;
                    _height[nodeIdx] = _height[neighborNodeIdx];
                }
                else if (rowIdx == _latticeHeight - 1) // top-left corner
                {
                    // 6 and 7 also undertmined. Take from lower-right neighbor.
                    var neighborNodeIdx = nodeIdx - _latticeWidth + 1;
                    _distribution[9 * nodeIdx + 1] = _distribution[9 * neighborNodeIdx + 1];
                    _distribution[9 * nodeIdx + 2] = _distribution[9 * neighborNodeIdx + 2];
                    _distribution[9 * nodeIdx + 8] = _distribution[9 * neighborNodeIdx + 8];
                    _distribution[9 * nodeIdx + 6] = _distribution[9 * neighborNodeIdx + 6];
                    _distribution[9 * nodeIdx + 7] = _distribution[9 * neighborNodeIdx + 7];
                    _velocity[nodeIdx] = float2.zero;
                    _height[nodeIdx] = _height[neighborNodeIdx];
                }
                else
                {
                    var neighborNodeIdx = nodeIdx + 1;
                    _distribution[9 * nodeIdx + 1] = _distribution[9 * neighborNodeIdx + 1];
                    _distribution[9 * nodeIdx + 2] = _distribution[9 * neighborNodeIdx + 2];
                    _distribution[9 * nodeIdx + 8] = _distribution[9 * neighborNodeIdx + 8];
                }
                _velocity[nodeIdx] = _inletVelocity;
                _height[nodeIdx] = _inletWaterHeight;
            }
        }
    }
}
