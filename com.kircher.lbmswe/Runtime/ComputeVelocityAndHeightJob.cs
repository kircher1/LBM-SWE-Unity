using Unity.Burst;
using Unity.Collections;
using Unity.Jobs;
using Unity.Mathematics;

namespace LatticeBoltzmannMethods
{
    [BurstCompile]
    public struct ComputeVelocityAndHeightJob : IJobParallelFor
    {
        [ReadOnly]
        private int _latticeWidth;
        [ReadOnly]
        private float _e;
        [ReadOnly]
        private float _maxHeight;
        [ReadOnly]
        private float _gravitationalForce;
        [ReadOnly]
        private NativeArray<float2> _linkDirection;
        [ReadOnly]
        private NativeArray<bool> _solid;

        // Inout
        [NativeDisableParallelForRestriction]
        private NativeArray<float> _distribution;

        // Out
        [NativeDisableParallelForRestriction]
        private NativeArray<float> _height;
        [NativeDisableParallelForRestriction]
        private NativeArray<float2> _velocity;

        public ComputeVelocityAndHeightJob(
            int latticeWidth,
            float e,
            float maxHeight,
            float gravitationalForce,
            NativeArray<float2> linkDirection,
            NativeArray<bool> solid,
            NativeArray<float> distributuion,
            NativeArray<float> height,
            NativeArray<float2> velocity)
        {
            _latticeWidth = latticeWidth;
            _e = e;
            _maxHeight = maxHeight;
            _gravitationalForce = gravitationalForce;
            _linkDirection = linkDirection;
            _solid = solid;
            _distribution = distributuion;
            _height = height;
            _velocity = velocity;
        }

        public void Execute(int rowIdx)
        {
            // Note, inflow and outflow jobs are responsible for updating height/velocity for inlet/outlet nodes.
            var rowStartIdx = rowIdx * _latticeWidth;
            for (var colIdx = 1; colIdx < _latticeWidth - 1; colIdx++)
            {
                var nodeIdx = rowStartIdx + colIdx;
                var height = 0.0f;
                var velocity = float2.zero;
                if (!_solid[nodeIdx])
                {
                    // Handle center link.
                    {
                        height += _distribution[9 * nodeIdx];
                    }

                    // Handle directional links.
                    for (var linkIdx = 0; linkIdx < 8; linkIdx++)
                    {
                        var linkDistribution = _distribution[9 * nodeIdx + linkIdx + 1];
                        height += linkDistribution;
                        velocity += linkDistribution * _linkDirection[linkIdx];
                    }
                }

                const float minHeight = 0.001f;
                if (height < minHeight)
                {
                    height = minHeight;
                    velocity = float2.zero;
                }
                else
                {
                    if (height > _maxHeight)
                    {
                        height = _maxHeight;
                        var rescale = _maxHeight / height;
                        for (var linkIdx = 0; linkIdx < 9; linkIdx++)
                        {
                            _distribution[9 * nodeIdx + linkIdx] *= rescale;
                        }
                    }

                    velocity *= _e / height;

                    const float FroudeNumberLimit = 0.75f;
                    var froudeNumber = math.length(velocity) / math.sqrt(_gravitationalForce * height);
                    if (froudeNumber >= FroudeNumberLimit)
                    {
                        var scale = FroudeNumberLimit / froudeNumber;
                        velocity *= scale;
                    }
                }

                _height[nodeIdx] = height;
                _velocity[nodeIdx] = velocity;
            }
        }
    }
}
