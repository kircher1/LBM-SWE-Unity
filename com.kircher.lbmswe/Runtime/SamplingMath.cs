using Unity.Mathematics;

namespace LatticeBoltzmannMethods
{
    public static class SamplingMath
    {
        public static void GenerateLinearSampleUvCoords(
            float2 uv,
            int textureWidth,
            int textureHeight,
            out int upperLeftIdx,
            out int lowerLeftIdx,
            out int upperRightIdx,
            out int lowerRightIdx,
            out float2 weights)
        {
            var pixelSize = new float2(textureWidth - 1, textureHeight - 1);
            var halfTexelSize = new float2(-0.5f, -0.5f) / pixelSize;
            var upperLeft = math.saturate(uv + halfTexelSize);
            var lowerRight = math.saturate(uv - halfTexelSize);

            var upperLeftColRowIdx = (int2)math.round(pixelSize * upperLeft);
            var lowerRightColRowIdx = (int2)math.round(pixelSize * lowerRight);

            upperLeftIdx = upperLeftColRowIdx.y * textureWidth + upperLeftColRowIdx.x;
            lowerLeftIdx = lowerRightColRowIdx.y * textureWidth + upperLeftColRowIdx.x;
            upperRightIdx = lowerRightColRowIdx.y * textureWidth + lowerRightColRowIdx.x;
            lowerRightIdx = lowerRightColRowIdx.y * textureWidth + lowerRightColRowIdx.x;

            weights = math.saturate((uv - lowerRight) / (2.0f * halfTexelSize));
        }

        public static float LinearBlend(float upperLeft, float lowerLeft, float upperRight, float lowerRight, float2 weights)
        {
            return
                math.lerp(
                    math.lerp(upperLeft, lowerLeft, weights.x),
                    math.lerp(upperRight, lowerRight, weights.x),
                    weights.y);
        }

        public static float2 LinearBlend(float2 upperLeft, float2 lowerLeft, float2 upperRight, float2 lowerRight, float2 weights)
        {
            return
                math.lerp(
                    math.lerp(upperLeft, lowerLeft, weights.x),
                    math.lerp(upperRight, lowerRight, weights.x),
                    weights.y);
        }
    }
}
