using Unity.Burst;
using Unity.Collections;
using Unity.Jobs;

namespace LatticeBoltzmannMethods
{
    /// <summary>
    /// Runs an iteration of flood fill to set the height in a solid node to the average height in non-solid neighboring nodes.
    /// This won't affect the actual sim, but can be useful for visualization.
    /// </summary>
    [BurstCompile]
    public struct FloodSolidHeightsJob : IJobParallelFor
    {
        [ReadOnly]
        private int _latticeWidth;
        [ReadOnly]
        private NativeArray<bool> _solid;

        [NativeDisableParallelForRestriction]
        private NativeArray<float> _height;

        public FloodSolidHeightsJob(int latticeWidth, NativeArray<bool> solid, NativeArray<float> height)
        {
            _latticeWidth = latticeWidth;
            _solid = solid;
            _height = height;
        }

        public void Execute(int rowIdx)
        {
            if (rowIdx == 0)
            {
                return;
            }

            var rowStartIdx = rowIdx * _latticeWidth;
            for (var nodeIdx = rowStartIdx + 1; nodeIdx < rowStartIdx + _latticeWidth - 1; ++nodeIdx)
            {
                if (!_solid[nodeIdx])
                {
                    continue;
                }

                var count = 0;
                var totalHeight = 0.0f;
                if (!_solid[nodeIdx - _latticeWidth - 1])
                {
                    totalHeight += _height[nodeIdx - _latticeWidth - 1];
                    count++;
                }
                if (!_solid[nodeIdx - _latticeWidth])
                {
                    totalHeight += _height[nodeIdx - _latticeWidth];
                    count++;
                }
                if (!_solid[nodeIdx - _latticeWidth + 1])
                {
                    totalHeight += _height[nodeIdx - _latticeWidth + 1];
                    count++;
                }
                if (!_solid[nodeIdx - 1])
                {
                    totalHeight += _height[nodeIdx - 1];
                    count++;
                }
                if (!_solid[nodeIdx + 1])
                {
                    totalHeight += _height[nodeIdx + 1];
                    count++;
                }
                if (!_solid[nodeIdx + _latticeWidth - 1])
                {
                    totalHeight += _height[nodeIdx + _latticeWidth - 1];
                    count++;
                }
                if (!_solid[nodeIdx + _latticeWidth])
                {
                    totalHeight += _height[nodeIdx + _latticeWidth];
                    count++;
                }
                if (!_solid[nodeIdx + _latticeWidth + 1])
                {
                    totalHeight += _height[nodeIdx + _latticeWidth + 1];
                    count++;
                }

                if (count > 0)
                {
                    _height[nodeIdx] = totalHeight / count;
                }
                else
                {
                    _height[nodeIdx] = -5.0f;
                }
            }
        }
    }
}
