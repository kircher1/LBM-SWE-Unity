using Unity.Burst;
using Unity.Collections;
using Unity.Jobs;
using Unity.Mathematics;

namespace LatticeBoltzmannMethods
{
    [BurstCompile]
    public struct UpdateForcesJob : IJobParallelFor
    {
        // Found a table of manning's coefficients here: https://www.engineeringtoolbox.com/mannings-roughness-d_799.html
        private const float WallManningsCoefficient = 0.025f;
        private const float BottomManningsCoefficient = 0.025f;

        [ReadOnly]
        private int _latticeWidth;
        [ReadOnly]
        private int _latticeHeight;
        [ReadOnly]
        private float _gravitationalForce;
        [ReadOnly]
        private float2 _bedSlope;
        [ReadOnly]
        private bool _applyShearForces;
        [ReadOnly]
        private NativeArray<float2> _linkDirection;
        [ReadOnly]
        private NativeArray<int> _linkOffsetX;
        [ReadOnly]
        private NativeArray<int> _linkOffsetY;
        [ReadOnly]
        private NativeArray<bool> _solid;
        [ReadOnly]
        private NativeArray<float> _height;
        [ReadOnly]
        private NativeArray<float2> _velocity;

        [NativeDisableParallelForRestriction]
        private NativeArray<float2> _force;

        public UpdateForcesJob(
            int latticeWidth,
            int latticeHeight,
            float gravitationalForce,
            float2 bedSlope,
            bool applyShearForces,
            NativeArray<float2> linkDirection,
            NativeArray<int> linkOffsetX,
            NativeArray<int> linkOffsetY,
            NativeArray<bool> solid,
            NativeArray<float> waterHeight,
            NativeArray<float2> velocity,
            NativeArray<float2> force)
        {
            _latticeWidth = latticeWidth;
            _latticeHeight = latticeHeight;
            _gravitationalForce = gravitationalForce;
            _bedSlope = bedSlope;
            _applyShearForces = applyShearForces;
            _linkDirection = linkDirection;
            _linkOffsetX = linkOffsetX;
            _linkOffsetY = linkOffsetY;
            _solid = solid;
            _height = waterHeight;
            _velocity = velocity;
            _force = force;
        }

        public void Execute(int rowIdx)
        {
            var rowStartIdx = rowIdx * _latticeWidth;
            for (var colIdx = 0; colIdx < _latticeWidth; colIdx++)
            {
                var nodeIdx = rowStartIdx + colIdx;
                if (_solid[nodeIdx])
                {
                    _force[nodeIdx] = float2.zero;
                    continue;
                }

                var gravitationalForce = float2.zero;
                var bedShearStress = float2.zero;
                var wallShearStress = float2.zero;

                // For each link, add the force term.
                for (var linkIdx = 1; linkIdx < 9; linkIdx++)
                {
                    var linkOffsetX = _linkOffsetX[linkIdx];
                    var linkOffsetY = _linkOffsetY[linkIdx];
                    var neighborRowIdx = math.clamp(rowIdx + linkOffsetY, 0, _latticeHeight - 1);
                    var neighborColIdx = math.clamp(colIdx + linkOffsetX, 0, _latticeWidth - 1);
                    var neighborIdx = neighborRowIdx * _latticeWidth + neighborColIdx;

                    var isWallNeighbor = _solid[neighborIdx];
                    var currentHeight = _height[nodeIdx];
                    var neighborWaterHeight = isWallNeighbor ? currentHeight : _height[neighborIdx];
                    var height = 0.5f * (currentHeight + neighborWaterHeight);

                    var velocity = _velocity[nodeIdx];

                    var linkDirection = math.normalize(_linkDirection[linkIdx]);

                    if (_applyShearForces)
                    {
                        var velocityDotDirection = math.max(0.0f, math.dot(velocity, linkDirection));
                        var projectedVelocity = velocityDotDirection * velocity;
                        var speed = math.length(projectedVelocity);
                        var scaledVelocity = projectedVelocity * speed;

                        var chezyCoefficient = math.pow(height, 1.0f / 6.0f) / BottomManningsCoefficient;
                        var bedFrictionCoefficient = _gravitationalForce / (chezyCoefficient * chezyCoefficient);
                        bedShearStress += scaledVelocity * bedFrictionCoefficient;

                        if (isWallNeighbor)
                        {
                            var wallFrictionCoefficient = -_gravitationalForce * WallManningsCoefficient * WallManningsCoefficient / math.pow(height, 1.0f / 3.0f);
                            wallShearStress += scaledVelocity * wallFrictionCoefficient;
                        }
                    }

                    var bedSlopeDotDirection = math.dot(_bedSlope, linkDirection);
                    gravitationalForce += _gravitationalForce * height * bedSlopeDotDirection * _bedSlope;

                    // ?
                    //gravitationalForce /= 9.0f;
                    //bedShearStress /= 9.0f;
                    //wallShearStress /= 9.0f;
                }

                _force[nodeIdx] = -gravitationalForce - bedShearStress + wallShearStress;
            }
        }
    }
}
