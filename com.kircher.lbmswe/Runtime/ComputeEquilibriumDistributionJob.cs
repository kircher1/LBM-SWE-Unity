using Unity.Burst;
using Unity.Collections;
using Unity.Jobs;
using Unity.Mathematics;

namespace LatticeBoltzmannMethods
{
    // TODO: ParallelJob-ify
    [BurstCompile]
    public struct ComputeEquilibriumDistributionJob : IJob
    {
        [ReadOnly]
        private int _latticeWidth;
        [ReadOnly]
        private int _latticeHeight;
        [ReadOnly]
        private float _e;
        [ReadOnly]
        private float _gravitationalForce;
        [ReadOnly]
        private NativeArray<float2> _linkDirection;
        [ReadOnly]
        private NativeArray<bool> _solid;
        [ReadOnly]
        private NativeArray<float> _height;
        [ReadOnly]
        private NativeArray<float2> _velocity;

        private NativeArray<float> _equilibriumDistribution; // output

        public ComputeEquilibriumDistributionJob(
            int latticeWidth,
            int latticeHeight,
            float e,
            float gravitationalForce,
            NativeArray<float2> linkDirection,
            NativeArray<bool> solid,
            NativeArray<float> height,
            NativeArray<float2> velocity,
            NativeArray<float> equilibriumDistribution)
        {
            _latticeWidth = latticeWidth;
            _latticeHeight = latticeHeight;
            _e = e;
            _gravitationalForce = gravitationalForce;
            _linkDirection = linkDirection;
            _solid = solid;
            _height = height;
            _velocity = velocity;
            _equilibriumDistribution = equilibriumDistribution;
        }

        public void Execute()
        {
            var inverseESq = 1.0f / (_e * _e);
            var inverseEQd = inverseESq * inverseESq;

            for (var nodeIdx = 0; nodeIdx < _latticeWidth * _latticeHeight; nodeIdx++)
            {
                if (_solid[nodeIdx])
                {
                    continue;
                }

                var height = _height[nodeIdx];
                var heightSq = height * height;
                var gravityTimesHeightSq = _gravitationalForce * heightSq;

                var velocity = _velocity[nodeIdx];
                var velocitySq = velocity * velocity;
                var sqMagnitudeOfVelocity = velocitySq.x + velocitySq.y;

                // Link 0
                _equilibriumDistribution[9 * nodeIdx] =
                        height
                    - (5.0f / 6.0f) * inverseESq * gravityTimesHeightSq
                    - (2.0f / 3.0f) * inverseESq * height * sqMagnitudeOfVelocity;

                for (var linkIdx = 1; linkIdx < 9; linkIdx++)
                {
                    var linkDirection = _e * _linkDirection[linkIdx];
                    var linkDirectionDotVelocity = math.dot(linkDirection, velocity);
                    var bigVelocityTerm =
                        linkDirection.x * linkDirection.x * velocitySq.x +
                        linkDirection.y * linkDirection.y * velocitySq.y +
                        2.0f * linkDirection.x * velocity.x * linkDirection.y * velocity.y;

                    if (linkIdx % 2 == 1)
                    {
                        _equilibriumDistribution[9 * nodeIdx + linkIdx] =
                              (1.0f / 6.0f) * inverseESq * gravityTimesHeightSq
                            + (1.0f / 3.0f) * inverseESq * height * linkDirectionDotVelocity
                            + (1.0f / 2.0f) * inverseEQd * height * bigVelocityTerm
                            - (1.0f / 6.0f) * inverseESq * height * sqMagnitudeOfVelocity;
                    }
                    else // if (linkIdx % 2 == 0)
                    {
                        _equilibriumDistribution[9 * nodeIdx + linkIdx] =
                              (1.0f / 24.0f) * inverseESq * gravityTimesHeightSq
                            + (1.0f / 12.0f) * inverseESq * height * linkDirectionDotVelocity
                            + (1.0f / 8.00f) * inverseEQd * height * bigVelocityTerm
                            - (1.0f / 24.0f) * inverseESq * height * sqMagnitudeOfVelocity;
                    }
                }
            }
        }
    }
}
