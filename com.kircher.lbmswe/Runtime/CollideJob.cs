using System.Runtime.CompilerServices;
using Unity.Burst;
using Unity.Burst.CompilerServices;
using Unity.Collections;
using Unity.Jobs;
using Unity.Mathematics;

namespace LatticeBoltzmannMethods
{
    [BurstCompile(FloatPrecision.Standard, FloatMode.Fast)]
    public struct CollideJob : IJobParallelFor
    {
        #if LBM_APPLY_SHEAR_FORCES
        // Found a table of manning's coefficients here: https://www.engineeringtoolbox.com/mannings-roughness-d_799.html
        // private const float WallManningsCoefficient = 0.025f;
        private const float BottomManningsCoefficient = 0.025f;
        #endif

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
        private float _relaxationTime;
        [ReadOnly]
        private float _gravitationalForce;
        [ReadOnly]
        private float2 _bedSlope;
        [ReadOnly]
        private NativeArray<float2> _linkDirection;
        [ReadOnly]
        private NativeArray<sbyte> _linkOffsetX;
        [ReadOnly]
        private NativeArray<sbyte> _linkOffsetY;
        [ReadOnly]
        private NativeArray<byte> _solid;
        [ReadOnly]
        private NativeArray<float> _equilibriumDistribution;
        [ReadOnly]
        private NativeArray<float> _height;
        [ReadOnly]
        private NativeArray<float2> _velocity;
        [ReadOnly]
        private NativeArray<float> _inverseEddyRelaxationTime;

        [NativeDisableParallelForRestriction]
        private NativeArray<float> _distribution;

        [ReadOnly]
        private float _forceTermCoefficient;

        public CollideJob(
            float deltaT,
            int latticeWidth,
            int latticeHeight,
            float e,
            float inverseESq,
            float relaxationTime,
            float gravitationalForce,
            float2 bedSlope,
            NativeArray<float2> linkDirection,
            NativeArray<sbyte> linkOffsetX,
            NativeArray<sbyte> linkOffsetY,
            NativeArray<byte> solid,
            NativeArray<float> equilibriumDistribution,
            NativeArray<float> height,
            NativeArray<float2> velocity,
            NativeArray<float> distribution,
            NativeArray<float> inverseEddyRelaxationTime)
        {
            _deltaT = deltaT;
            _latticeWidth = latticeWidth;
            _latticeHeight = latticeHeight;
            _e = e;
            _inverseESq = inverseESq;
            _relaxationTime = relaxationTime;
            _gravitationalForce = gravitationalForce;
            _bedSlope = bedSlope;
            _linkDirection = linkDirection;
            _linkOffsetX = linkOffsetX;
            _linkOffsetY = linkOffsetY;
            _solid = solid;
            _equilibriumDistribution = equilibriumDistribution;
            _height = height;
            _velocity = velocity;
            _distribution = distribution;
            _inverseEddyRelaxationTime = inverseEddyRelaxationTime;

            _forceTermCoefficient = (1.0f / 6.0f) * _e * _inverseESq * _deltaT;
        }

        public void Execute(int rowIdx)
        {
            var rowStartIdx = rowIdx * _latticeWidth;

            // Apply bounce back to solidn nodes.
            for (var colIdx = 0; colIdx < _latticeWidth; colIdx++)
            {
                var nodeIdx = rowStartIdx + colIdx;
                if (_solid[nodeIdx] == 0)
                {
                    var nodeOffset = 9 * nodeIdx;
                    float temp;
                    // swap 1 <---> 5
                    temp = _distribution[nodeOffset + 1];
                    _distribution[nodeOffset + 1] = _distribution[nodeOffset + 5];
                    _distribution[nodeOffset + 5] = temp;
                    // swap 2 <---> 6
                    temp = _distribution[nodeOffset + 2];
                    _distribution[nodeOffset + 2] = _distribution[nodeOffset + 6];
                    _distribution[nodeOffset + 6] = temp;
                    // swap 3 <---> 7
                    temp = _distribution[nodeOffset + 3];
                    _distribution[nodeOffset + 3] = _distribution[nodeOffset + 7];
                    _distribution[nodeOffset + 7] = temp;
                    // swap 4 <---> 8
                    temp = _distribution[nodeOffset + 4];
                    _distribution[nodeOffset + 4] = _distribution[nodeOffset + 8];
                    _distribution[nodeOffset + 8] = temp;
                }
            }

            // Update distributions for liquid link 0.
            for (var colIdx = 0; colIdx < _latticeWidth; colIdx++)
            {
                var nodeIdx = rowStartIdx + colIdx;
                if (_solid[nodeIdx] != 0)
                {
                    var inverseRelaxationTime = _inverseEddyRelaxationTime[nodeIdx];
                    var nodeOffset = 9 * nodeIdx;
                    var equilibriumDistribution = _equilibriumDistribution[nodeOffset];
                    var currentDistribution = _distribution[nodeOffset];
                    var relaxationTerm = inverseRelaxationTime * (currentDistribution - equilibriumDistribution);
                    _distribution[nodeOffset] = currentDistribution - relaxationTerm;
                }
            }

            // Update distributions, applying force term, for remaining links.
            for (var colIdx = 0; colIdx < _latticeWidth; colIdx++)
            {
                var nodeIdx = rowStartIdx + colIdx;
                if (_solid[nodeIdx] != 0)
                {
                    var inverseRelaxationTime = _inverseEddyRelaxationTime[nodeIdx];
                    var currentHeight = _height[nodeIdx];
                    var currentVelocity = _velocity[nodeIdx];

                    var nodeOffset = 9 * nodeIdx + 1;
                    for (var linkIdx = 0; linkIdx < 8; linkIdx++)
                    {
                        //Loop.ExpectVectorized();
                        var equilibriumDistribution = _equilibriumDistribution[nodeOffset + linkIdx];
                        var currentDistribution = _distribution[nodeOffset + linkIdx];
                        var relaxationTerm = inverseRelaxationTime * (currentDistribution - equilibriumDistribution);
                        var forceTerm = ComputeForceTerm(rowIdx, colIdx, linkIdx, currentHeight, currentVelocity);
                        _distribution[nodeOffset + linkIdx] = currentDistribution - relaxationTerm + _forceTermCoefficient * forceTerm;
                    }
                }
            }
        }

        /// <summary>
        /// Computes the force term between the current node (specified by rowIdx and colIdx) and it's neighboring node specified by the linkIdx.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private float ComputeForceTerm(int rowIdx, int colIdx, [AssumeRange(0, 8)] int linkIdx, float currentHeight, float2 currentVelocity)
        {
            var linkDirection = _linkDirection[linkIdx];
            var linkOffsetX = _linkOffsetX[linkIdx];
            var linkOffsetY = _linkOffsetY[linkIdx];

            // TODO: Handle periodic boundary.
            var neighborRowIdx = math.clamp(rowIdx + linkOffsetY, 0, _latticeHeight - 1);
            var neighborColIdx = math.clamp(colIdx + linkOffsetX, 0, _latticeWidth - 1);
            var neighborIdx = neighborRowIdx * _latticeWidth + neighborColIdx;

            var neighborHeight = math.lerp(currentHeight, _height[neighborIdx], _solid[neighborIdx]);
            var centeredHeight = 0.5f * (currentHeight + neighborHeight);

            var gravitationalForce = _gravitationalForce * centeredHeight * _bedSlope;

            #if LBM_APPLY_SHEAR_FORCES
            {
                // Also trying the centered scheme with velocity too... Not sure it's valid.
                //var neighborVelocity = isWallNeighbor ? float2.zero : _velocity[neighborIdx];
                var centeredVelocity = currentVelocity; // 0.5f * (currentVelocity + neighborVelocity);
                var velocitySq = centeredVelocity * math.length(centeredVelocity);

                var chezyCoefficient = math.pow(centeredHeight, 1.0f / 6.0f) / BottomManningsCoefficient;
                var bedFrictionCoefficient = _gravitationalForce / (chezyCoefficient * chezyCoefficient);
                var bedShearStress = velocitySq * bedFrictionCoefficient;

                // Collision is using a bounce back scheme so disable the additional wall friction for now.
                //if (isWallNeighbor)
                //{
                //    var wallFrictionCoefficient = -_gravitationalForce * WallManningsCoefficient * WallManningsCoefficient / math.pow(centeredHeight, 1.0f / 3.0f);
                //    wallShearStress = (1.0f - _solid[neighborIdx]) * velocitySq * wallFrictionCoefficient;
                //}

                return math.dot(linkDirection, -gravitationalForce - bedShearStress/* + wallShearStress*/);
            }
            #else
            {
                return math.dot(linkDirection, -gravitationalForce);
            }
            #endif
        }
    }
}
