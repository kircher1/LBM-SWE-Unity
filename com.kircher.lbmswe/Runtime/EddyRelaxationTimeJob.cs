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

        [ReadOnly]
        [DeallocateOnJobCompletion]
        private NativeArray<float4> _strainRateTensorValues;

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

            // Precompute some intermediate values used to evaluate the magnitude of the momentum flux tensor.
            _strainRateTensorValues = new NativeArray<float4>(8, Allocator.TempJob);
            for (var linkIdx = 0; linkIdx < 8; linkIdx++)
            {
                var eLinkDirection = _e * _linkDirection[linkIdx];
                _strainRateTensorValues[linkIdx] =
                    new float4(
                        eLinkDirection.x * eLinkDirection.x,
                        eLinkDirection.x * eLinkDirection.y,
                        eLinkDirection.y * eLinkDirection.x,
                        eLinkDirection.y * eLinkDirection.y);
            }
        }

        public void Execute(int rowIdx)
        {
            var rowStartIdx = rowIdx * _latticeWidth;
            for (var colIdx = 0; colIdx < _latticeWidth; colIdx++)
            {
                var nodeIdx = rowStartIdx + colIdx;
                var currentHeight = _height[nodeIdx];

                // Skipping actual link 0 since result for that link is always 0.
                var nodeOffset = 9 * nodeIdx + 1;
                var momentumFluxTensor = float4.zero; // really, a 2x2 matrix.
                for (var linkIdx = 0; linkIdx < 8; linkIdx++)
                {
                    //Loop.ExpectVectorized();
                    var equilibriumDistribution = _equilibriumDistribution[nodeOffset + linkIdx];
                    var currentDistribution = _distribution[nodeOffset + linkIdx];
                    var nonEquilibriumDistribution = currentDistribution - equilibriumDistribution;
                    momentumFluxTensor += nonEquilibriumDistribution * _strainRateTensorValues[linkIdx];
                }

                var doubleDotProduct = math.dot(momentumFluxTensor, momentumFluxTensor);
                var momentumFluxTensorMagnitude = math.sqrt(doubleDotProduct);
                var turubulentTime = _momentumFluxTensorTerm * momentumFluxTensorMagnitude / currentHeight;
                _inverseEddyRelaxationTime[nodeIdx] = _solid[nodeIdx] / (0.5f * (_relaxationTime + math.sqrt(_relaxationTimeSq + turubulentTime)));
            }
        }
    }
}
