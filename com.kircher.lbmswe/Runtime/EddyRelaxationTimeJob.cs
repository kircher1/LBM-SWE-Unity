using Unity.Burst;
using Unity.Collections;
using Unity.Jobs;
using Unity.Mathematics;

namespace LatticeBoltzmannMethods
{
    /// <summary>
    /// Computes a per-node eddy relaxation time using Smagorinsky SGS model.
    /// </summary>
    [BurstCompile(FloatPrecision.Standard, FloatMode.Fast)]
    public struct EddyRelaxationTimeJob : IJobParallelFor
    {
        [ReadOnly]
        private int _latticeWidth;
        [ReadOnly]
        private float _e;
        [ReadOnly]
        private float _inverseESq;
        [ReadOnly]
        private float _relaxationTime;
        [ReadOnly]
        private float _relaxationTimeSq;
        [ReadOnly]
        private float _smagorinskyConstantSq;
        [ReadOnly]
        private NativeArray<float2> _linkDirection;
        [ReadOnly]
        private NativeArray<byte> _solid;
        [ReadOnly]
        private NativeArray<float> _equilibriumDistribution;
        [ReadOnly]
        private NativeArray<float> _height;
        [ReadOnly]
        private NativeArray<float> _distribution;

        [NativeDisableParallelForRestriction]
        private NativeArray<float> _inverseEddyRelaxationTime;

        [ReadOnly]
        private float _momentumFluxTensorTerm;

        public EddyRelaxationTimeJob(
            int latticeWidth,
            float e,
            float inverseESq,
            float relaxationTime,
            float relaxationTimeSq,
            float smagorinskyConstantSq,
            NativeArray<float2> linkDirection,
            NativeArray<byte> solid,
            NativeArray<float> equilibriumDistribution,
            NativeArray<float> height,
            NativeArray<float> distribution,
            NativeArray<float> inverseEddyRelaxationTime)
        {
            _latticeWidth = latticeWidth;
            _e = e;
            _inverseESq = inverseESq;
            _relaxationTime = relaxationTime;
            _relaxationTimeSq = relaxationTimeSq;
            _smagorinskyConstantSq = smagorinskyConstantSq;
            _linkDirection = linkDirection;
            _solid = solid;
            _equilibriumDistribution = equilibriumDistribution;
            _height = height;
            _distribution = distribution;
            _inverseEddyRelaxationTime = inverseEddyRelaxationTime;

            _momentumFluxTensorTerm = 18.0f * _smagorinskyConstantSq * _inverseESq;
        }

        public void Execute(int rowIdx)
        {
            var rowStartIdx = rowIdx * _latticeWidth;
            for (var colIdx = 0; colIdx < _latticeWidth; colIdx++)
            {
                var nodeIdx = rowStartIdx + colIdx;
                var currentHeight = _height[nodeIdx];

                // Skipping actual link 0 since result for that link is always 0.
                // TODO: Convert to float3?
                var momentumFluxTensorXX = 0.0f; var momentumFluxTensorXY = 0.0f; var momentumFluxTensorYY = 0.0f;
                for (var linkIdx = 0; linkIdx < 8; linkIdx++)
                {
                    //Loop.ExpectVectorized();
                    var equilibriumDistribution = _equilibriumDistribution[9 * nodeIdx + linkIdx + 1];
                    var currentDistribution = _distribution[9 * nodeIdx + linkIdx + 1];
                    var nonEquilibriumDistribution = currentDistribution - equilibriumDistribution;
                    var linkDirection = _e * _linkDirection[linkIdx]; // TODO: Precompute (and linkDirection xx/xy/yy).
                    momentumFluxTensorXX += nonEquilibriumDistribution * linkDirection.x * linkDirection.x;
                    momentumFluxTensorXY += nonEquilibriumDistribution * linkDirection.x * linkDirection.y;
                    momentumFluxTensorYY += nonEquilibriumDistribution * linkDirection.y * linkDirection.y;
                }

                var doubleDotProduct = momentumFluxTensorXX * momentumFluxTensorXX + 2.0f * momentumFluxTensorXY * momentumFluxTensorXY + momentumFluxTensorYY * momentumFluxTensorYY;
                var momentumFluxTensorMagnitude = math.sqrt(doubleDotProduct);
                var turubulentTime = _momentumFluxTensorTerm * momentumFluxTensorMagnitude / currentHeight;
                _inverseEddyRelaxationTime[nodeIdx] = _solid[nodeIdx] / (0.5f * (_relaxationTime + math.sqrt(_relaxationTimeSq + turubulentTime)));
            }
        }
    }
}
