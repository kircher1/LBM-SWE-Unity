using Unity.Burst;
using Unity.Collections;
using Unity.Jobs;
using Unity.Mathematics;

namespace LatticeBoltzmannMethods
{
    [BurstCompile]
    public struct ComputeEquilibriumDistributionJob : IJobParallelFor
    {
        [ReadOnly]
        private int _latticeWidth;
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

        [WriteOnly]
        [NativeDisableParallelForRestriction]
        private NativeArray<float> _equilibriumDistribution; // output

        public ComputeEquilibriumDistributionJob(
            int latticeWidth,
            float e,
            float gravitationalForce,
            NativeArray<float2> linkDirection,
            NativeArray<bool> solid,
            NativeArray<float> height,
            NativeArray<float2> velocity,
            NativeArray<float> equilibriumDistribution)
        {
            _latticeWidth = latticeWidth;
            _e = e;
            _gravitationalForce = gravitationalForce;
            _linkDirection = linkDirection;
            _solid = solid;
            _height = height;
            _velocity = velocity;
            _equilibriumDistribution = equilibriumDistribution;
        }

        public void Execute(int rowIdx)
        {
            var inverseESq = 1.0f / (_e * _e);
            var inverseEQd = inverseESq * inverseESq;
            var rowStartIdx = rowIdx * _latticeWidth;
            for (var colIdx = 0; colIdx < _latticeWidth; colIdx++)
            {
                var nodeIdx = rowStartIdx + colIdx;
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

                for (var linkIdx = 0; linkIdx < 8; linkIdx++)
                {
                    var linkDirection = _e * _linkDirection[linkIdx];
                    var linkDirectionDotVelocity = math.dot(linkDirection, velocity);
                    var bigVelocityTerm =
                        linkDirection.x * linkDirection.x * velocitySq.x +
                        linkDirection.y * linkDirection.y * velocitySq.y +
                        2.0f * linkDirection.x * velocity.x * linkDirection.y * velocity.y;

                    if (linkIdx % 2 == 0)
                    {
                        _equilibriumDistribution[9 * nodeIdx + linkIdx + 1] =
                              (1.0f / 6.0f) * inverseESq * gravityTimesHeightSq
                            + (1.0f / 3.0f) * inverseESq * height * linkDirectionDotVelocity
                            + (1.0f / 2.0f) * inverseEQd * height * bigVelocityTerm
                            - (1.0f / 6.0f) * inverseESq * height * sqMagnitudeOfVelocity;
                    }
                    else // if (linkIdx % 2 == 1)
                    {
                        _equilibriumDistribution[9 * nodeIdx + linkIdx + 1] =
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
