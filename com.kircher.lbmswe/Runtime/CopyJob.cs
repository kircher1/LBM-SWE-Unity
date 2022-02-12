using System;
using Unity.Burst;
using Unity.Collections;
using Unity.Jobs;

[BurstCompile]
public struct CopyJob : IJob
{
    [ReadOnly]
    private NativeArray<float> _src;
    private NativeArray<float> _dst;

    public CopyJob(NativeArray<float> src, NativeArray<float> dst)
    {
        if (dst.Length < src.Length)
        {
            throw new ArgumentException(nameof(src));
        }

        _src = src;
        _dst = dst;
    }

    public void Execute()
    {
        NativeArray<float>.Copy(_src, _dst, _src.Length);
    }
}
