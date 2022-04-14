using Unity.Burst;
using Unity.Collections;
using Unity.Jobs;

namespace LatticeBoltzmannMethods
{
    [BurstCompile]
    public struct StreamJob : IJobParallelFor
    {
        [ReadOnly]
        private bool _usePeriodicBoundary;
        [ReadOnly]
        private int _latticeWidth;
        [ReadOnly]
        private int _latticeHeight;
        [ReadOnly]
        private NativeArray<sbyte> _linkOffsetX;
        [ReadOnly]
        private NativeArray<sbyte> _linkOffsetY;
        [ReadOnly]
        private NativeArray<float> _lastDistribution;

        [WriteOnly]
        [NativeDisableParallelForRestriction]
        private NativeArray<float> _distribution;

        public StreamJob(
            bool usePreiodicBoundary,
            int latticeWidth,
            int latticeHeight,
            NativeArray<sbyte> linkOffsetX,
            NativeArray<sbyte> linkOffsetY,
            NativeArray<float> lastDistribution,
            NativeArray<float> distribution)
        {
            _usePeriodicBoundary = usePreiodicBoundary;
            _latticeWidth = latticeWidth;
            _latticeHeight = latticeHeight;
            _linkOffsetX = linkOffsetX;
            _linkOffsetY = linkOffsetY;
            _lastDistribution = lastDistribution;
            _distribution = distribution;
        }

        public void Execute(int rowIdx)
        {
            var rowStartIdx = rowIdx * _latticeWidth;
            for (var colIdx = 0; colIdx < _latticeWidth; colIdx++)
            {
                var nodeIdx = rowStartIdx + colIdx;

                // Link 0
                {
                    _distribution[9 * nodeIdx] = _lastDistribution[9 * nodeIdx];
                }

                // Remaining links.
                for (var linkIdx = 1; linkIdx < 9; linkIdx++)
                {
                    var propagatedColIdx = colIdx + _linkOffsetX[linkIdx - 1];
                    var propagatedRowIdx = rowIdx + _linkOffsetY[linkIdx - 1];

                    if (_usePeriodicBoundary)
                    {
                        if (propagatedColIdx >= _latticeWidth) propagatedColIdx -= _latticeWidth;
                        if (propagatedColIdx < 0) propagatedColIdx += _latticeWidth;
                        if (propagatedRowIdx >= _latticeHeight) propagatedRowIdx -= _latticeHeight;
                        if (propagatedRowIdx < 0) propagatedRowIdx += _latticeHeight;
                    }

                    if (propagatedColIdx >= 0 && propagatedColIdx < _latticeWidth && propagatedRowIdx >= 0 && propagatedRowIdx < _latticeHeight)
                    {
                        var distributionToStream = _lastDistribution[9 * nodeIdx + linkIdx];
                        var propagatedIdx = propagatedRowIdx * _latticeWidth + propagatedColIdx;
                        _distribution[9 * propagatedIdx + linkIdx] = distributionToStream;
                    }
                }
            }
        }
    }
}
