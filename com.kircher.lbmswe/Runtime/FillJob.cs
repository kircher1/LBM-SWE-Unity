using Unity.Burst;
using Unity.Collections;
using Unity.Jobs;

namespace LatticeBoltzmannMethods
{
    [BurstCompile]
    public struct FillJob : IJob
    {
        private NativeArray<float> _array;

        public FillJob(NativeArray<float> array)
        {
            _array = array;
        }

        public void Execute()
        {
            for (var idx = 0; idx < _array.Length; idx++)
            {
                _array[idx] = 0;
            }
        }
    }
}
