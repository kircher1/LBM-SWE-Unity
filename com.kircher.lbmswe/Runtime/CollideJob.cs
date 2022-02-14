using Unity.Burst;
using Unity.Collections;
using Unity.Jobs;
using Unity.Mathematics;

namespace LatticeBoltzmannMethods
{
    [BurstCompile]
    public struct CollideJob : IJob
    {
        [ReadOnly]
        private float _deltaT;
        [ReadOnly]
        private int _latticeWidth;
        [ReadOnly]
        private int _latticeHeight;
        [ReadOnly]
        private float _e;
        [ReadOnly]
        private float _inverseESq;
        [ReadOnly]
        private bool _applyEddyRelaxationTime;
        [ReadOnly]
        private float _relaxationTime;
        [ReadOnly]
        private float _relaxationTimeSq;
        [ReadOnly]
        private float _smagorinskyConstantSq;
        [ReadOnly]
        private NativeArray<float2> _linkDirection;
        [ReadOnly]
        private NativeArray<bool> _solid;
        [ReadOnly]
        private NativeArray<float> _equilibriumDistribution;
        [ReadOnly]
        private NativeArray<float2> _force;
        [ReadOnly]
        private NativeArray<float> _waterHeight;

        private NativeArray<float> _distribution;

        public CollideJob(
            float deltaT,
            int latticeWidth,
            int latticeHeight,
            float e,
            float inverseESq,
            bool applyEddyRelaxationTime,
            float relaxationTime,
            float relaxationTimeSq,
            float smagorinskyConstantSq,
            NativeArray<float2> linkDirection,
            NativeArray<bool> solid,
            NativeArray<float> equilibriumDistribution,
            NativeArray<float2> force,
            NativeArray<float> waterHeight,
            NativeArray<float> distribution)
        {
            _deltaT = deltaT;
            _latticeWidth = latticeWidth;
            _latticeHeight = latticeHeight;
            _e = e;
            _inverseESq = inverseESq;
            _applyEddyRelaxationTime = applyEddyRelaxationTime;
            _relaxationTime = relaxationTime;
            _relaxationTimeSq = relaxationTimeSq;
            _smagorinskyConstantSq = smagorinskyConstantSq;
            _linkDirection = linkDirection;
            _solid = solid;
            _equilibriumDistribution = equilibriumDistribution;
            _force = force;
            _waterHeight = waterHeight;
            _distribution = distribution;
        }

        public void Execute()
        {
            var inverseRelaxationTime = 1.0f / _relaxationTime;
            for (var nodeIdx = 0; nodeIdx < _latticeWidth * _latticeHeight; nodeIdx++)
            {
                if (_solid[nodeIdx])
                {
                    // bounce back
                    float temp;
                    temp = _distribution[9 * nodeIdx + 1]; _distribution[9 * nodeIdx + 1] = _distribution[9 * nodeIdx + 5]; _distribution[9 * nodeIdx + 5] = temp;
                    temp = _distribution[9 * nodeIdx + 2]; _distribution[9 * nodeIdx + 2] = _distribution[9 * nodeIdx + 6]; _distribution[9 * nodeIdx + 6] = temp;
                    temp = _distribution[9 * nodeIdx + 3]; _distribution[9 * nodeIdx + 3] = _distribution[9 * nodeIdx + 7]; _distribution[9 * nodeIdx + 7] = temp;
                    temp = _distribution[9 * nodeIdx + 4]; _distribution[9 * nodeIdx + 4] = _distribution[9 * nodeIdx + 8]; _distribution[9 * nodeIdx + 8] = temp;
                }
                else
                {
                    if (_applyEddyRelaxationTime)
                    {
                        var momentumFluxTensor = 0.0f;
                        for (var linkIdx = 0; linkIdx < 9; linkIdx++)
                        {
                            var equilibriumDistribution = _equilibriumDistribution[9 * nodeIdx + linkIdx];
                            var currentDistribution = _distribution[9 * nodeIdx + linkIdx];
                            var distributionDelta = currentDistribution - equilibriumDistribution;
                            var linkDirection = _e * _linkDirection[linkIdx];
                            momentumFluxTensor +=
                                distributionDelta *
                                (linkDirection.x * linkDirection.x
                                    + 2.0f * linkDirection.x * linkDirection.y
                                    + linkDirection.y * linkDirection.y);
                        }
                        momentumFluxTensor = math.abs(momentumFluxTensor);
                        var totalRelaxationTime =
                            0.5f * (
                                _relaxationTime +
                                math.sqrt(
                                    _relaxationTimeSq + 18.0f * _smagorinskyConstantSq * momentumFluxTensor * _inverseESq / _waterHeight[nodeIdx]));

                        inverseRelaxationTime = 1.0f / totalRelaxationTime;
                    }

                    var force = _force[nodeIdx];
                    for (var linkIdx = 0; linkIdx < 9; linkIdx++)
                    {
                        var equilibriumDistribution = _equilibriumDistribution[9 * nodeIdx + linkIdx];
                        var currentDistribution = _distribution[9 * nodeIdx + linkIdx];
                        var relaxationTerm = inverseRelaxationTime * (currentDistribution - equilibriumDistribution);
                        var forceTerm = linkIdx == 0 ? 0.0f : (1.0f / 6.0f) * _inverseESq * _deltaT * math.dot(force, _e * _linkDirection[linkIdx]);
                        _distribution[9 * nodeIdx + linkIdx] = currentDistribution - relaxationTerm + forceTerm;
                    }
                }
            }
        }
    }
}
