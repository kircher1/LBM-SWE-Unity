using Unity.Burst;
using Unity.Collections;
using Unity.Jobs;

namespace LatticeBoltzmannMethods
{
    // TODO: Separate into one job for edge nodes and another job for the rest.
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
        private NativeArray<float> _lastRestDistribution;
        [ReadOnly]
        private NativeArray<float> _lastDistribution;

        [WriteOnly]
        [NativeDisableParallelForRestriction]
        private NativeArray<float> _restDistribution;
        [WriteOnly]
        [NativeDisableParallelForRestriction]
        private NativeArray<float> _distribution;

        public StreamJob(
            bool usePreiodicBoundary,
            int latticeWidth,
            int latticeHeight,
            NativeArray<sbyte> linkOffsetX,
            NativeArray<sbyte> linkOffsetY,
            NativeArray<float> lastRestDistribution,
            NativeArray<float> lastDistribution,
            NativeArray<float> restDistribution,
            NativeArray<float> distribution)
        {
            _usePeriodicBoundary = usePreiodicBoundary;
            _latticeWidth = latticeWidth;
            _latticeHeight = latticeHeight;
            _linkOffsetX = linkOffsetX;
            _linkOffsetY = linkOffsetY;
            _lastRestDistribution = lastRestDistribution;
            _lastDistribution = lastDistribution;
            _restDistribution = restDistribution;
            _distribution = distribution;
        }

        public void Execute(int rowIdx)
        {
            var rowStartIdx = rowIdx * _latticeWidth;
            for (var colIdx = 0; colIdx < _latticeWidth; colIdx++)
            {
                var nodeIdx = rowStartIdx + colIdx;

                // Rest link.
                {
                    _restDistribution[nodeIdx] = _lastRestDistribution[nodeIdx];
                }

                // Remaining links.
                for (var linkIdx = 0; linkIdx < 8; linkIdx++)
                {
                    var propagatedColIdx = colIdx + _linkOffsetX[linkIdx];
                    var propagatedRowIdx = rowIdx + _linkOffsetY[linkIdx];

                    if (_usePeriodicBoundary)
                    {
                        if (propagatedColIdx >= _latticeWidth) propagatedColIdx -= _latticeWidth;
                        if (propagatedColIdx < 0) propagatedColIdx += _latticeWidth;
                        if (propagatedRowIdx >= _latticeHeight) propagatedRowIdx -= _latticeHeight;
                        if (propagatedRowIdx < 0) propagatedRowIdx += _latticeHeight;
                    }

                    if (propagatedColIdx >= 0 && propagatedColIdx < _latticeWidth && propagatedRowIdx >= 0 && propagatedRowIdx < _latticeHeight)
                    {
                        var distributionToStream = _lastDistribution[8 * nodeIdx + linkIdx];
                        var propagatedIdx = propagatedRowIdx * _latticeWidth + propagatedColIdx;
                        _distribution[8 * propagatedIdx + linkIdx] = distributionToStream;
                    }
                }
            }
        }
    }
}
