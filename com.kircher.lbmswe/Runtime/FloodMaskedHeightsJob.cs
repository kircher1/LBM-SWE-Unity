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
    public struct FloodSolidHeightsJob : IJob
    {
        [ReadOnly]
        private int _latticeWidth;
        [ReadOnly]
        private int _latticeHeight;
        [ReadOnly]
        private NativeArray<bool> _solid;

        private NativeArray<float> _height;

        public FloodSolidHeightsJob(int latticeWidth, int latticeHeight, NativeArray<bool> solid, NativeArray<float> height)
        {
            _latticeWidth = latticeWidth;
            _latticeHeight = latticeHeight;
            _solid = solid;
            _height = height;
        }

        public void Execute()
        {
            for (var rowIdx = 1; rowIdx < _latticeHeight - 1; rowIdx++)
            {
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
}
