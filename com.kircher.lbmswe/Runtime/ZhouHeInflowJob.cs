using System;
using Unity.Burst;
using Unity.Collections;
using Unity.Jobs;
using Unity.Mathematics;

namespace LatticeBoltzmannMethods
{
    /// <summary>
    /// Computes missing inlet link distributions by applying method by Zhou and He,
    /// and updates height and  velocity on inlet nodes.
    /// TODO: Handle corners.
    /// </summary>
    [BurstCompile]
    public struct ZhouHeInflowJob : IJob
    {
        [ReadOnly]
        private int _latticeWidth;
        [ReadOnly]
        private int _latticeHeight;
        [ReadOnly]
        private float _inverseE;
        [ReadOnly]
        private NativeArray<bool> _solid;
        [ReadOnly]
        private float _inletWaterHeight;
        [ReadOnly]
        private float2 _inletVelocity;

        private NativeArray<float> _distribution;
        private NativeArray<float> _height;
        private NativeArray<float2> _velocity;

        public ZhouHeInflowJob(
            int latticeWidth,
            int latticeHeight,
            float inverseE,
            NativeArray<bool> solid,
            float inletWaterHeight,
            float2 inletVelocity,
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
                if (!_solid[nodeIdx])
                {
                    _distribution[9 * nodeIdx + 1] = _distribution[9 * nodeIdx + 5] + (2.0f / 3.0f) * _inverseE * _inletWaterHeight * u;
                    _distribution[9 * nodeIdx + 2] = (1.0f / 6.0f) * _inverseE * _inletWaterHeight * u + _distribution[9 * nodeIdx + 6] + 0.5f * (_distribution[9 * nodeIdx + 7] - _distribution[9 * nodeIdx + 3]);
                    _distribution[9 * nodeIdx + 8] = (1.0f / 6.0f) * _inverseE * _inletWaterHeight * u + _distribution[9 * nodeIdx + 4] + 0.5f * (_distribution[9 * nodeIdx + 3] - _distribution[9 * nodeIdx + 7]);

                    _velocity[nodeIdx] = _inletVelocity;
                    _height[nodeIdx] = _inletWaterHeight;
                }
            }
        }
    }
}
