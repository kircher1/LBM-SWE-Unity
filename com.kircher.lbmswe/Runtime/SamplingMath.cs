using System.Runtime.CompilerServices;
using Unity.Mathematics;

namespace LatticeBoltzmannMethods
{
    public static class SamplingMath
    {
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void GenerateLinearSampleUvCoords(
            float2 uv,
            int2 dimensionInPixels,
            float2 halfPixelSizeInUVSpace,
            out int upperLeftIdx,
            out int lowerLeftIdx,
            out int upperRightIdx,
            out int lowerRightIdx,
            out float2 weights)
        {
            var upperLeft = uv - halfPixelSizeInUVSpace;
            var upperLeftRowCol = (int2)math.round((dimensionInPixels - 1) * math.saturate(upperLeft));
            var texelCoord = uv - upperLeftRowCol / (float2)dimensionInPixels - halfPixelSizeInUVSpace;
            weights = math.saturate(texelCoord * dimensionInPixels);

            // And compute the indices for looking up the corners.
            var lowerRightRowCol = math.min((dimensionInPixels - 1), upperLeftRowCol + 1);
            upperLeftIdx = upperLeftRowCol.y * dimensionInPixels.x + upperLeftRowCol.x;
            upperRightIdx = upperLeftRowCol.y * dimensionInPixels.x + lowerRightRowCol.x;
            lowerLeftIdx = lowerRightRowCol.y * dimensionInPixels.x + upperLeftRowCol.x;
            lowerRightIdx = lowerRightRowCol.y * dimensionInPixels.x + lowerRightRowCol.x;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float LinearBlend(float upperLeft, float lowerLeft, float upperRight, float lowerRight, float2 weights)
        {
            return
                math.lerp(
                    math.lerp(upperLeft, lowerLeft, weights.y),
                    math.lerp(upperRight, lowerRight, weights.y),
                    weights.x);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float2 LinearBlend(float2 upperLeft, float2 lowerLeft, float2 upperRight, float2 lowerRight, float2 weights)
        {
            return
                math.lerp(
                    math.lerp(upperLeft, lowerLeft, weights.y),
                    math.lerp(upperRight, lowerRight, weights.y),
                    weights.x);
        }
    }
}
