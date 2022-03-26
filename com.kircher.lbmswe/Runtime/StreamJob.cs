using Unity.Burst;
using Unity.Collections;
using Unity.Jobs;

namespace LatticeBoltzmannMethods
{
    [BurstCompile]
    public struct StreamJob : IJob
    {
        [ReadOnly]
        private bool _usePeriodicBoundary;
        [ReadOnly]
        private int _latticeWidth;
        [ReadOnly]
        private int _latticeHeight;
        [ReadOnly]
        private NativeArray<int> _linkOffsetX;
        [ReadOnly]
        private NativeArray<int> _linkOffsetY;
        [ReadOnly]
        private NativeArray<bool> _solid;
        [ReadOnly]
        private NativeArray<float> _lastDistribution;

        private NativeArray<float> _distribution;

        public StreamJob(
            bool usePreiodicBoundary,
            int latticeWidth,
            int latticeHeight,
            NativeArray<int> linkOffsetX,
            NativeArray<int> linkOffsetY,
            NativeArray<bool> solid,
            NativeArray<float> lastDistribution,
            NativeArray<float> distribution)
        {
            _usePeriodicBoundary = usePreiodicBoundary;
            _latticeWidth = latticeWidth;
            _latticeHeight = latticeHeight;
            _linkOffsetX = linkOffsetX;
            _linkOffsetY = linkOffsetY;
            _solid = solid;
            _lastDistribution = lastDistribution;
            _distribution = distribution;
        }

        public void Execute()
        {
            for (var rowIdx = 0; rowIdx < _latticeHeight; rowIdx++)
            {
                var rowStartIdx = rowIdx * _latticeWidth;
                for (var colIdx = 0; colIdx < _latticeWidth; colIdx++)
                {
                    var nodeIdx = rowStartIdx + colIdx;
                    for (var linkIdx = 0; linkIdx < 9; linkIdx++)
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
                            var distributionToStream = _lastDistribution[9 * nodeIdx + linkIdx];
                            var propagatedIdx = propagatedRowIdx * _latticeWidth + propagatedColIdx;
                            _distribution[9 * propagatedIdx + linkIdx] += distributionToStream;
                        }
                    }
                }
            }

            // Handle missing distributions at outlet.
            // TODO: Separate job.
            if (!_usePeriodicBoundary)
            {
                // var inverseE = 1.0f / e;

                for (var rowIdx = 0; rowIdx < _latticeHeight; rowIdx++)
                {
                    var nodeIdx = rowIdx * _latticeWidth + _latticeWidth - 1;
                    if (!_solid[nodeIdx])
                    {
                        // Strat 1: Copy from neighbor.
                        _distribution[9 * nodeIdx + 4] = _distribution[9 * (nodeIdx - 1) + 4];
                        _distribution[9 * nodeIdx + 5] = _distribution[9 * (nodeIdx - 1) + 5];
                        _distribution[9 * nodeIdx + 6] = _distribution[9 * (nodeIdx - 1) + 6];

                        // TODO: This does not work as expected.
                        // Start 2: Zou and He.
                        //var waterHeight = _height[nodeIdx];
                        //var u = _velocity[nodeIdx].x;
                        //_newDistribution[9 * nodeIdx + 5] = _newDistribution[9 * nodeIdx + 1] - (2.0f / 3.0f) * inverseE * waterHeight * u;
                        //_newDistribution[9 * nodeIdx + 4] = -(1.0f / 6.0f) * inverseE * waterHeight * u + _newDistribution[9 * nodeIdx + 8] + 0.5f * (_newDistribution[9 * nodeIdx + 7] - _newDistribution[9 * nodeIdx + 3]);
                        //_newDistribution[9 * nodeIdx + 6] = -(1.0f / 6.0f) * inverseE * waterHeight * u + _newDistribution[9 * nodeIdx + 2] + 0.5f * (_newDistribution[9 * nodeIdx + 3] - _newDistribution[9 * nodeIdx + 7]);
                    }
                }
            }
        }
    }
}
