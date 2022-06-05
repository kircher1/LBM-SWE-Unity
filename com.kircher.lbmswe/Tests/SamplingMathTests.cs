using LatticeBoltzmannMethods;
using NUnit.Framework;
using Unity.Collections;
using Unity.Mathematics;

public class SamplingMathTests
{
    [Test]
    public void CenterSampleTest()
    {
        var data = new NativeArray<float>(4, Allocator.Temp);
        data[0] = 0;
        data[1] = 0;
        data[2] = 1;
        data[3] = 1;

        SamplingMath.GenerateLinearSampleUvCoords(
            new float2(0.5f, 0.5f),
            new int2(2, 2),
            0.5f / new float2(2, 2),
            out var upperLeftIdx,
            out var lowerLeftIdx,
            out var upperRightIdx,
            out var lowerRightIdx,
            out var weights);
        Assert.AreEqual(0.5, weights.x, 0.0001);
        Assert.AreEqual(0.5, weights.y, 0.0001);
        Assert.AreEqual(0, upperLeftIdx);
        Assert.AreEqual(1, upperRightIdx);
        Assert.AreEqual(2, lowerLeftIdx);
        Assert.AreEqual(3, lowerRightIdx);

        var blended = SamplingMath.LinearBlend(data[upperLeftIdx], data[lowerLeftIdx], data[upperRightIdx], data[lowerRightIdx], weights);
        Assert.AreEqual(0.5, blended, 0.0001);
    }

    [Test]
    public void LeftCenterSampleTest()
    {
        var data = new NativeArray<float>(4, Allocator.Temp);
        data[0] = 0;
        data[1] = 10;
        data[2] = 1;
        data[3] = 10;

        SamplingMath.GenerateLinearSampleUvCoords(
            new float2(0.0f, 0.5f),
            new int2(2, 2),
            0.5f / new float2(2, 2),
            out var upperLeftIdx,
            out var lowerLeftIdx,
            out var upperRightIdx,
            out var lowerRightIdx,
            out var weights);
        Assert.AreEqual(0.0, weights.x, 0.0001);
        Assert.AreEqual(0.5, weights.y, 0.0001);

        var blended = SamplingMath.LinearBlend(data[upperLeftIdx], data[lowerLeftIdx], data[upperRightIdx], data[lowerRightIdx], weights);
        Assert.AreEqual(0.5, blended, 0.0001);
    }

    [Test]
    public void RightCenterSampleTest()
    {
        var data = new NativeArray<float>(4, Allocator.Temp);
        data[0] = 0;
        data[1] = 10;
        data[2] = 1;
        data[3] = 20;

        SamplingMath.GenerateLinearSampleUvCoords(
            new float2(1.0f, 0.5f),
            new int2(2, 2),
            0.5f / new float2(2, 2),
            out var upperLeftIdx,
            out var lowerLeftIdx,
            out var upperRightIdx,
            out var lowerRightIdx,
            out var weights);
        Assert.AreEqual(0.5, weights.x, 0.0001);
        Assert.AreEqual(0.5, weights.y, 0.0001);

        var blended = SamplingMath.LinearBlend(data[upperLeftIdx], data[lowerLeftIdx], data[upperRightIdx], data[lowerRightIdx], weights);
        Assert.AreEqual(15.0, blended, 0.0001);
    }

    [Test]
    public void SampleOffCenterTest()
    {
        var data = new NativeArray<float>(4, Allocator.Temp);
        data[0] = 0;
        data[1] = 0;
        data[2] = 1;
        data[3] = 1;

        SamplingMath.GenerateLinearSampleUvCoords(
            new float2(0.6f, 0.6f),
            new int2(2, 2),
            0.5f / new float2(2, 2),
            out var upperLeftIdx,
            out var lowerLeftIdx,
            out var upperRightIdx,
            out var lowerRightIdx,
            out var weights);
        Assert.AreEqual(0.7, weights.x, 0.0001);
        Assert.AreEqual(0.7, weights.y, 0.0001);

        var blended = SamplingMath.LinearBlend(data[upperLeftIdx], data[lowerLeftIdx], data[upperRightIdx], data[lowerRightIdx], weights);
        Assert.AreEqual(0.7, blended, 0.0001);
    }
}
