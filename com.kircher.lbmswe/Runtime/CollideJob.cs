using Unity.Burst;
using Unity.Collections;
using Unity.Jobs;
using Unity.Mathematics;

namespace LatticeBoltzmannMethods
{
    [BurstCompile]
    public struct CollideJob : IJobParallelFor
    {
        // Found a table of manning's coefficients here: https://www.engineeringtoolbox.com/mannings-roughness-d_799.html
        private const float WallManningsCoefficient = 0.025f;
        private const float BottomManningsCoefficient = 0.025f;

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
        private float _gravitationalForce;
        [ReadOnly]
        private float2 _bedSlope;
        [ReadOnly]
        private bool _applyShearForces;
        [ReadOnly]
        private NativeArray<float2> _linkDirection;
        [ReadOnly]
        private NativeArray<sbyte> _linkOffsetX;
        [ReadOnly]
        private NativeArray<sbyte> _linkOffsetY;
        [ReadOnly]
        private NativeArray<bool> _solid;
        [ReadOnly]
        private NativeArray<float> _equilibriumDistribution;
        [ReadOnly]
        private NativeArray<float> _height;
        [ReadOnly]
        private NativeArray<float2> _velocity;

        [NativeDisableParallelForRestriction]
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
            float gravitationalForce,
            float2 bedSlope,
            bool applyShearForces,
            NativeArray<float2> linkDirection,
            NativeArray<sbyte> linkOffsetX,
            NativeArray<sbyte> linkOffsetY,
            NativeArray<bool> solid,
            NativeArray<float> equilibriumDistribution,
            NativeArray<float> height,
            NativeArray<float2> velocity,
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
            _gravitationalForce = gravitationalForce;
            _bedSlope = bedSlope;
            _applyShearForces = applyShearForces;
            _linkDirection = linkDirection;
            _linkOffsetX = linkOffsetX;
            _linkOffsetY = linkOffsetY;
            _solid = solid;
            _equilibriumDistribution = equilibriumDistribution;
            _height = height;
            _velocity = velocity;
            _distribution = distribution;
        }

        public void Execute(int rowIdx)
        {
            var inverseRelaxationTime = 1.0f / _relaxationTime;
            var rowStartIdx = rowIdx * _latticeWidth;
            for (var colIdx = 0; colIdx < _latticeWidth; colIdx++)
            {
                var nodeIdx = rowStartIdx + colIdx;
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
                    var currentHeight = _height[nodeIdx];
                    var currentVelocity = _velocity[nodeIdx];

                    if (_applyEddyRelaxationTime)
                    {
                        // Skipping link 0 since result for that link is always 0.
                        var momentumFluxTensor = 0.0f;
                        for (var linkIdx = 0; linkIdx < 8; linkIdx++)
                        {
                            var equilibriumDistribution = _equilibriumDistribution[9 * nodeIdx + linkIdx + 1];
                            var currentDistribution = _distribution[9 * nodeIdx + linkIdx + 1];
                            var distributionDelta = currentDistribution - equilibriumDistribution;
                            var linkDirection = _linkDirection[linkIdx];
                            momentumFluxTensor +=
                                distributionDelta *
                                (linkDirection.x * linkDirection.x
                                    + 2.0f * linkDirection.x * linkDirection.y
                                    + linkDirection.y * linkDirection.y);
                        }
                        momentumFluxTensor = _e * math.abs(momentumFluxTensor);

                        var totalRelaxationTime =
                            0.5f * (
                                _relaxationTime +
                                math.sqrt(
                                    _relaxationTimeSq + 18.0f * _smagorinskyConstantSq * momentumFluxTensor * _inverseESq / currentHeight));

                        inverseRelaxationTime = 1.0f / totalRelaxationTime;
                    }

                    // Link 0.
                    {
                        var equilibriumDistribution = _equilibriumDistribution[9 * nodeIdx];
                        var currentDistribution = _distribution[9 * nodeIdx];
                        var relaxationTerm = inverseRelaxationTime * (currentDistribution - equilibriumDistribution);
                        _distribution[9 * nodeIdx] = currentDistribution - relaxationTerm;
                    }

                    // Other links.
                    var forceTermCoefficient = (1.0f / 6.0f) * _e * _inverseESq * _deltaT;
                    for (var linkIdx = 0; linkIdx < 8; linkIdx++)
                    {
                        var equilibriumDistribution = _equilibriumDistribution[9 * nodeIdx + linkIdx + 1];
                        var currentDistribution = _distribution[9 * nodeIdx + linkIdx + 1];
                        var relaxationTerm = inverseRelaxationTime * (currentDistribution - equilibriumDistribution);
                        var forceTerm = ComputeForceTerm(rowIdx, colIdx, linkIdx, currentHeight, currentVelocity);
                        _distribution[9 * nodeIdx + linkIdx + 1] = currentDistribution - relaxationTerm + forceTermCoefficient * forceTerm;
                    }
                }
            }
        }

        /// <summary>
        /// Computes the force term between the current node (specified by rowIdx and colIdx) and it's neighboring node specified by the linkIdx.
        /// </summary>
        private float ComputeForceTerm(int rowIdx, int colIdx, int linkIdx, float currentHeight, float2 currentVelocity)
        {
            var linkDirection = _linkDirection[linkIdx];
            var linkOffsetX = _linkOffsetX[linkIdx];
            var linkOffsetY = _linkOffsetY[linkIdx];

            var neighborRowIdx = math.clamp(rowIdx + linkOffsetY, 0, _latticeHeight - 1);
            var neighborColIdx = math.clamp(colIdx + linkOffsetX, 0, _latticeWidth - 1);
            var neighborIdx = neighborRowIdx * _latticeWidth + neighborColIdx;

            var isWallNeighbor = _solid[neighborIdx];
            var neighborHeight = isWallNeighbor ? currentHeight : _height[neighborIdx];
            var centeredHeight = 0.5f * (currentHeight + neighborHeight);

            var bedShearStress = float2.zero;
            var wallShearStress = float2.zero;

            if (_applyShearForces)
            {
                // Also trying the centered scheme with velocity too... Not sure it's valid.
                //var neighborVelocity = isWallNeighbor ? float2.zero : _velocity[neighborIdx];
                var centeredVelocity = currentVelocity; // 0.5f * (currentVelocity + neighborVelocity);
                var velocitySq = centeredVelocity * math.length(centeredVelocity);

                var chezyCoefficient = math.pow(centeredHeight, 1.0f / 6.0f) / BottomManningsCoefficient;
                var bedFrictionCoefficient = _gravitationalForce / (chezyCoefficient * chezyCoefficient);
                bedShearStress = velocitySq * bedFrictionCoefficient;

                // Collision is using a bounce back scheme so disable the additional wall friction for now.
                //if (isWallNeighbor)
                //{
                //    var wallFrictionCoefficient = -_gravitationalForce * WallManningsCoefficient * WallManningsCoefficient / math.pow(centeredHeight, 1.0f / 3.0f);
                //    wallShearStress = velocitySq * wallFrictionCoefficient;
                //}
            }

            var gravitationalForce = _gravitationalForce * centeredHeight * _bedSlope;

            return math.dot(linkDirection, -gravitationalForce - bedShearStress + wallShearStress);
        }
    }
}
