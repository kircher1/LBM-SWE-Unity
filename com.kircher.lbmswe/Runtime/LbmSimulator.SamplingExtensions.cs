using System.Runtime.CompilerServices;
using Unity.Mathematics;

namespace LatticeBoltzmannMethods
{
    public static class SamplingExtensions
    {
        /// <summary>
        /// Sample the veloicty at the specified UV coordinate.
        /// UV-space is normalized between the lattice edges.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float2 SampleVelocity(this LbmSimulator lbmSimulator, float2 uv)
        {
            SamplingMath.GenerateLinearSampleUvCoords(
                uv,
                lbmSimulator.LatticeDimension,
                lbmSimulator.HalfLatticeSpacingInUVSpace,
                out var upperLeftIdx,
                out var lowerLeftIdx,
                out var upperRightIdx,
                out var lowerRightIdx,
                out float2 weights);
            var upperLeft = lbmSimulator.Velocity[upperLeftIdx];
            var lowerLeft = lbmSimulator.Velocity[lowerLeftIdx];
            var upperRight = lbmSimulator.Velocity[upperRightIdx];
            var lowerRight = lbmSimulator.Velocity[lowerRightIdx];
            return SamplingMath.LinearBlend(upperLeft, lowerLeft, upperRight, lowerRight, weights);
        }

        /// <summary>
        /// Sample the solid mask at the specified UV coordinate. 0.0 is liquid, 1.0 is solid.
        /// UV-space is normalized between the lattice edges.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float SampleSolid(this LbmSimulator lbmSimulator, float2 uv)
        {
            SamplingMath.GenerateLinearSampleUvCoords(
                uv,
                lbmSimulator.LatticeDimension,
                lbmSimulator.HalfLatticeSpacingInUVSpace,
                out var upperLeftIdx,
                out var lowerLeftIdx,
                out var upperRightIdx,
                out var lowerRightIdx,
                out float2 weights);
            var upperLeft = lbmSimulator.Solid[upperLeftIdx];
            var lowerLeft = lbmSimulator.Solid[lowerLeftIdx];
            var upperRight = lbmSimulator.Solid[upperRightIdx];
            var lowerRight = lbmSimulator.Solid[lowerRightIdx];
            return 1.0f - SamplingMath.LinearBlend(upperLeft, lowerLeft, upperRight, lowerRight, weights);
        }

        /// <summary>
        /// Sample the height at the specified UV coordinate.
        /// UV-space is normalized between the lattice edges.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float SampleHeight(this LbmSimulator lbmSimulator, float2 uv)
        {
            SamplingMath.GenerateLinearSampleUvCoords(
                uv,
                lbmSimulator.LatticeDimension,
                lbmSimulator.HalfLatticeSpacingInUVSpace,
                out var upperLeftIdx,
                out var lowerLeftIdx,
                out var upperRightIdx,
                out var lowerRightIdx,
                out float2 weights);
            var upperLeft = lbmSimulator.Height[upperLeftIdx];
            var lowerLeft = lbmSimulator.Height[lowerLeftIdx];
            var upperRight = lbmSimulator.Height[upperRightIdx];
            var lowerRight = lbmSimulator.Height[lowerRightIdx];
            return SamplingMath.LinearBlend(upperLeft, lowerLeft, upperRight, lowerRight, weights);
        }
    }
}
