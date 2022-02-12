using System;
using Unity.Burst;
using Unity.Collections;
using Unity.Jobs;
using Unity.Mathematics;

/// <summary>
/// Result is a NativeArray of size 2 where the 0th index is the min and the next index is the max.
/// </summary>
[BurstCompile]
struct ComputeMinMaxFloat2Job : IJob
{
    [ReadOnly]
    private NativeArray<float2> _input;
    [ReadOnly]
    private NativeArray<float2> _result;

    public ComputeMinMaxFloat2Job(NativeArray<float2> input, NativeArray<float2> result)
    {
        if (result.Length < 2)
        {
            throw new ArgumentException(nameof(result));
        }

        _input = input;
        _result = result;
    }

    public void Execute()
    {
        var min = new float2(float.MaxValue, float.MaxValue);
        var max = new float2(float.MinValue, float.MinValue);
        for (var idx = 0; idx < _input.Length; idx++)
        {
            min = math.min(min, _input[idx]);
            max = math.max(max, _input[idx]);
        }
        _result[0] = min;
        _result[1] = max;
    }
}
