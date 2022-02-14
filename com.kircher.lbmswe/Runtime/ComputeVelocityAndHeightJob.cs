using Unity.Burst;
using Unity.Collections;
using Unity.Jobs;
using Unity.Mathematics;

namespace LatticeBoltzmannMethods
{
    [BurstCompile]
    public struct ComputeVelocityAndHeightJob : IJob
    {
        [ReadOnly]
        private int _latticeWidth;
        [ReadOnly]
        private int _latticeHeight;
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
        private NativeArray<float> _distribution;

        // Out
        private NativeArray<float> _height;
        private NativeArray<float2> _velocity;

        public ComputeVelocityAndHeightJob(
            int latticeWidth,
            int latticeHeight,
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
            _latticeHeight = latticeHeight;
            _e = e;
            _maxHeight = maxHeight;
            _gravitationalForce = gravitationalForce;
            _linkDirection = linkDirection;
            _solid = solid;
            _distribution = distributuion;
            _height = height;
            _velocity = velocity;
        }

        public void Execute()
        {
            for (var nodeIdx = 0; nodeIdx < _latticeWidth * _latticeHeight; nodeIdx++)
            {
                var height = 0.0f;
                var velocity = float2.zero;
                if (!_solid[nodeIdx])
                {
                    for (var linkIdx = 0; linkIdx < 9; linkIdx++)
                    {
                        var linkDistribution = _distribution[9 * nodeIdx + linkIdx];
                        height += linkDistribution;
                        velocity += _e * linkDistribution * _linkDirection[linkIdx];
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

                    velocity /= height;

                    const float FroudeNumberLimit = 0.75f;
                    var froudeNumber = math.length(velocity) / math.sqrt(_gravitationalForce * height);
                    if (froudeNumber >= FroudeNumberLimit)
                    {
                        var scale = FroudeNumberLimit / froudeNumber;
                        velocity *= scale;
                    }

                    // Rescale velocity so we don't blow up.
                    //var speed = velocity.magnitude;
                    //if (speed > maxSpeed)
                    //{
                    //    velocity *= maxSpeed / speed;
                    //}
                }

                _height[nodeIdx] = height;
                _velocity[nodeIdx] = velocity;
            }
        }
    }
}
