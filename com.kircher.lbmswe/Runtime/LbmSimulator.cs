using System;
using System.Collections.Generic;
using Unity.Collections;
using Unity.Jobs;
using Unity.Mathematics;
using UnityEngine;

namespace LatticeBoltzmannMethods
{
    // TODO: Optimization ideas
    // - Split link data into separate arrays. Then run equilibrium distribution jobs in parallel.

    /// <summary>
    /// Simulates shallow water flows on a 2D lattice. Origin node is at the "bottom-left" corner of lattice.
    /// </summary>
    [ExecuteInEditMode]
    public class LbmSimulator : MonoBehaviour
    {
        private const float GravitationalForce = 9.8f;
        private const float PiOverFour = 0.78539816339f;

        private static readonly ValueTuple<JobHandle?, Texture2D, TextureEventHandler> NoTextureProcessedResult = (null, null, null);

        [SerializeField]
        [Range(0.001f, 1.0f)]
        private float _simulationStepTime = 0.016f;

        [SerializeField]
        private float _latticeSpacingInMeters = 0.05f;

        [SerializeField]
        private int _latticeWidth = 65;

        public int LatticeWidth => _latticeWidth;

        [SerializeField]
        private int _latticeHeight = 193;

        public int LatticeHeight => _latticeHeight;

        [SerializeField]
        private float _startingHeight = 0.1f;

        [SerializeField]
        [Range(0.501f, 2.0f)]
        private float _relaxationTime = 0.51f;

        [SerializeField]
        private float2 _bedSlope = new float2(-0.005f, 0.0f);

        [SerializeField]
        [Range(0.06f, 0.5f)]
        private float _smagorinskyConstant = 0.18f;

        [SerializeField]
        private InletOutletBoundaryCondition _inletOutletBoundaryCondition = InletOutletBoundaryCondition.ZouHe;

        /// <summary>
        /// Flood fills solid nodes with the average of non-solid neighboring heights.
        /// Does not affect the simulation.
        /// </summary>
        [SerializeField]
        private bool _fixupSolidHeights = false;

        /// <summary>
        /// Flood fills solid nodes with the average of non-solid neighboring heights.
        /// Does not affect the simulation.
        /// </summary>
        public bool FixupSolidHeights { get => _fixupSolidHeights; set => _fixupSolidHeights = value; }

        [SerializeField]
        private bool _showMarkers = true;

        [SerializeField]
        private GameObject _markerPrefab;

        [SerializeField]
        [Range(0.01f, 20.0f)]
        private float _markerFrequency = 5.0f;

        [SerializeField]
        private bool _dumpStats = false;

        public delegate void TextureEventHandler(object sender, Texture2D e);

        public event EventHandler SimulationStepCompleted;
        public event TextureEventHandler FlowTextureUpdated;
        public event TextureEventHandler HeightTextureUpdated;
        public event TextureEventHandler MaskTextureUpdated;

        public float MaxSpeed { get; private set; } = 1.0f;

        private static readonly float2 Half = new float2(0.5f, 0.5f);

        private NativeArray<float2> _linkDirection;
        private NativeArray<sbyte> _linkOffsetX;
        private NativeArray<sbyte> _linkOffsetY;

        /// <summary>
        /// Note, does not include origin link, which has direction (0, 0).
        /// </summary>
        public NativeArray<float2> LinkDirection => _linkDirection;

        /// <summary>
        /// Note, does not include origin link, which has X-offset of 0.
        /// </summary>
        public NativeArray<sbyte> LinkOffsetX => _linkOffsetX;

        /// <summary>
        /// Note, does not include origin link, which has Y-offset of 0.
        /// </summary>
        public NativeArray<sbyte> LinkOffsetY => _linkOffsetY;

        // Simulation data. These are actively being updated frame to frame. Don't read from outside of jobs.
        private NativeArray<byte> _solid;
        private NativeArray<float2> _velocity;
        private NativeArray<float> _height;
        private NativeArray<float> _lastDistribution;
        private NativeArray<float> _newDistribution;
        private NativeArray<float> _equilibriumDistribution;
        public NativeArray<float> _inverseEddyRelaxationTime;
        private ValueTuple<JobHandle, JobHandle>? _lastJobHandles = null;

        // Sim result data. This is what can be queried by clients.
        private NativeArray<byte> _solidResult;
        /// <summary>
        /// Name will probably change to 'liquid', as currently the value is either 0 => solid, or 1 => liquid.
        /// </summary>
        public NativeArray<byte> Solid => _solidResult;
        private NativeArray<float2> _velocityResult;
        public NativeArray<float2> Velocity => _velocityResult;
        private NativeArray<float> _heightResult;
        public NativeArray<float> Height => _heightResult;

        private readonly List<GameObject> _markers = new List<GameObject>();
        private float _markerTimer = 1.0f;

        // These are results of sim, which can be optionally generated.
        private Texture2D _flowTexture;
        private Texture2D _heightTexture;
        private Texture2D _maskTexture;

        private float _e;
        private float _inverseE;
        private float _maxHeight;
        private float _initialHeight;
        private float2 _initialVelocity;

        private static void InitializeLinkData(
            out NativeArray<float2> linkDirection,
            out NativeArray<sbyte> linkOffsetX,
            out NativeArray<sbyte> linkOffsetY)
        {
            linkDirection = new NativeArray<float2>(8, Allocator.Persistent);
            linkOffsetX = new NativeArray<sbyte>(8, Allocator.Persistent);
            linkOffsetY = new NativeArray<sbyte>(8, Allocator.Persistent);

            var _linkOffsetX = new sbyte[] { 1,  1,  0, -1, -1, -1,  0,  1 };
            var _linkOffsetY = new sbyte[] { 0,  1,  1,  1,  0, -1, -1, -1 };

            for (var linkIdx = 0; linkIdx < 8; linkIdx++)
            {
                var angle = PiOverFour * linkIdx;
                linkDirection[linkIdx] = math.normalize(new float2(math.cos(angle), math.sin(angle)));
                if (linkIdx % 2 == 1)
                {
                    linkDirection[linkIdx] *= math.SQRT2;
                }

                linkOffsetX[linkIdx] = _linkOffsetX[linkIdx];
                linkOffsetY[linkIdx] = _linkOffsetY[linkIdx];
            }
        }

        private void OnEnable()
        {
            InitializeLinkData(out _linkDirection, out _linkOffsetX, out _linkOffsetY);

            _markerTimer = _markerFrequency;

            InitializeSimData();
        }

        private void InitializeNativeArrays()
        {
            if (_solid.IsCreated)
            {
                return;
            }

            _solid = new NativeArray<byte>(_latticeWidth * _latticeHeight, Allocator.Persistent);
            for (var idx = 0; idx < _solid.Length; idx++)
            {
                _solid[idx] = 1;
            }
            _lastDistribution = new NativeArray<float>(_latticeWidth * _latticeHeight * 9, Allocator.Persistent);
            _newDistribution = new NativeArray<float>(_latticeWidth * _latticeHeight * 9, Allocator.Persistent);
            _equilibriumDistribution = new NativeArray<float>(_latticeWidth * _latticeHeight * 9, Allocator.Persistent);
            _inverseEddyRelaxationTime = new NativeArray<float>(_latticeWidth * _latticeHeight, Allocator.Persistent);
            _height = new NativeArray<float>(_latticeWidth * _latticeHeight, Allocator.Persistent);
            _velocity = new NativeArray<float2>(_latticeWidth * _latticeHeight, Allocator.Persistent);

            _heightResult = new NativeArray<float>(_latticeWidth * _latticeHeight, Allocator.Persistent);
            _velocityResult = new NativeArray<float2>(_latticeWidth * _latticeHeight, Allocator.Persistent);
        }

        private void InitializeSimData()
        {
            InitializeNativeArrays();

            var startupLog = "LbmSimulator ---\r\n";

            // Fill in solid rails unless along top and bottom edge unless some solid nodes have already been set by someone else.
            if (!_solidResult.IsCreated)
            {
                startupLog += "Added solid rails.\r\n";
                for (var idx = 0; idx < _solid.Length; idx++)
                {
                    _solid[idx] = 0;
                }
                for (var rowIdx = 1; rowIdx < _latticeHeight - 1; rowIdx++)
                {
                    var rowStartIdx = rowIdx * _latticeWidth;
                    for (var colIdx = 0; colIdx < _latticeWidth; colIdx++)
                    {
                        _solid[rowStartIdx + colIdx] = 1;
                    }
                }

                // Ensure corners are liquid.
                //_solid[0] = 1;
                //_solid[_latticeWidth - 1] = 1;
                //_solid[(_latticeHeight - 1) * _latticeWidth] = 1;
                //_solid[(_latticeHeight - 1) * _latticeWidth + _latticeWidth - 1] = 1;

                _solidResult = new NativeArray<byte>(_latticeWidth* _latticeHeight, Allocator.Persistent);
                NativeArray<byte>.Copy(_solid, _solidResult);
            }
            else
            {
                startupLog += "Solid nodes previously set.\r\n";
                NativeArray<byte>.Copy(_solidResult, _solid);
            }

            // Compute per-simulation terms.
            _e = _latticeSpacingInMeters / _simulationStepTime;
            _inverseE = 1.0f / _e;
            MaxSpeed = math.abs(_e) - 0.001f;
            _maxHeight = _e * _e / GravitationalForce - 0.001f;
            startupLog += $"Max speed: {MaxSpeed}, Max height: {_maxHeight}\r\n";

            // Set starting height uniformly.
            _initialHeight = math.min(_maxHeight, _startingHeight);
            startupLog += $"Initial height: {_initialHeight}\r\n";
            for (var nodeIdx = 0; nodeIdx < _height.Length; nodeIdx++)
            {
                _height[nodeIdx] = _solid[nodeIdx] * _initialHeight;
            }

            // Set initial velocity, some factor of the max speed.
            _initialVelocity = new float2((_startingHeight / _maxHeight) * MaxSpeed / math.SQRT2, 0.0f);
            startupLog += $"Initial velocity: {_initialVelocity}\r\n";
            for (var nodeIdx = 0; nodeIdx < _velocity.Length; nodeIdx++)
            {
                _velocity[nodeIdx] = _solid[nodeIdx] * _initialVelocity;
            }

            // The starting distribution will be the equilbrium distribution.
            {
                var computeEquilibriumDistributionJob =
                    new ComputeEquilibriumDistributionJob(
                        _latticeWidth,
                         _e,
                        GravitationalForce,
                        _linkDirection,
                        _solid,
                        _height,
                        _velocity,
                        _equilibriumDistribution);
                var computeEquilibriumDistributionJobHandle = computeEquilibriumDistributionJob.Schedule(_latticeHeight, 1);

                // Copy equilibrium distribution into the starting distribution (_lastDistribution).
                var copyJob = new CopyJob(_equilibriumDistribution, _lastDistribution);
                var copyJobHandle = copyJob.Schedule(computeEquilibriumDistributionJobHandle);
                copyJobHandle.Complete();
            }

            Debug.Log(startupLog);
        }

        private void Update()
        {
            if (_lastJobHandles != null)
            {
                _lastJobHandles.Value.Item1.Complete();
                _lastJobHandles.Value.Item2.Complete();
                _lastJobHandles = null;

                // Copy sim data to results.
                NativeArray<float>.Copy(_height, _heightResult);
                NativeArray<float2>.Copy(_velocity, _velocityResult);

                // And copy latest solid to sim solid. Now is safe to do this.
                NativeArray<byte>.Copy(_solidResult, _solid);

                SimulationStepCompleted?.Invoke(this, EventArgs.Empty);

                // After jobs complete, update any textures and markers.
                UpdateTextures(_maxHeight, MaxSpeed);
                UpdateMarkers(); // TODO: Jobify/move to different component.
                DumpStats(); // TODO: Jobify/move to different component.
            }

            // Compute per-frame terms.
            var inverseESq = 1.0f / (_e * _e);
            var smagorinskyConstantSq = _smagorinskyConstant * _smagorinskyConstant;
            var relaxationTimeSq = _relaxationTime * _relaxationTime;

            // Setup jobs.
            var usePeriodicBoundary = _inletOutletBoundaryCondition == InletOutletBoundaryCondition.Periodic;
            var eddyRelaxationTimeJob =
                new EddyRelaxationTimeJob(
                    _latticeWidth,
                    _e,
                    inverseESq,
                    _relaxationTime,
                    relaxationTimeSq,
                    smagorinskyConstantSq,
                    _linkDirection,
                    _solid,
                    _equilibriumDistribution,
                    _height,
                    _lastDistribution,
                    _inverseEddyRelaxationTime);
            var collideJob =
                new CollideJob(
                    _simulationStepTime,
                    _latticeWidth,
                    _latticeHeight,
                    _e,
                    inverseESq,
                    _relaxationTime,
                    GravitationalForce,
                    _bedSlope,
                    _linkDirection,
                    _linkOffsetX,
                    _linkOffsetY,
                    _solid,
                    _equilibriumDistribution,
                    _height,
                    _velocity,
                    _lastDistribution,
                    _inverseEddyRelaxationTime);
            var streamJob =
                new StreamJob(
                    usePeriodicBoundary,
                    _latticeWidth,
                    _latticeHeight,
                    _linkOffsetX,
                    _linkOffsetY,
                    _lastDistribution,
                    _newDistribution);
            var computeVelocityAndHeightJob =
                new ComputeVelocityAndHeightJob(
                    usePeriodicBoundary,
                    _latticeWidth,
                    _e,
                    _maxHeight,
                    GravitationalForce,
                    _linkDirection,
                    _solid,
                    _newDistribution,
                    _height,
                    _velocity);
            var computeEquilibriumDistributionJob =
                new ComputeEquilibriumDistributionJob(
                    _latticeWidth,
                     _e,
                    GravitationalForce,
                    _linkDirection,
                    _solid,
                    _height,
                    _velocity,
                    _equilibriumDistribution);
            var copyNewDistributionToLastDistributionJob = new CopyJob(_newDistribution, _lastDistribution);
            var fillNewDistributionJob = new FillJob(_newDistribution);

            // Schedule simulation jobs.
            var eddyRelaxationTimeJobHandle = eddyRelaxationTimeJob.Schedule(LatticeHeight, 1);
            var collideJobHandle = collideJob.Schedule(_latticeHeight, 1, eddyRelaxationTimeJobHandle);
            var streamJobHandle = streamJob.Schedule(_latticeHeight, 1, collideJobHandle);
            var computeVelocityAndHeightJobHandle = computeVelocityAndHeightJob.Schedule(_latticeHeight, 1, streamJobHandle);
            var inflowJobHandle =
                usePeriodicBoundary ?
                    computeVelocityAndHeightJobHandle :
                    _inletOutletBoundaryCondition == InletOutletBoundaryCondition.ZeroGradient ?
                        new ZeroGradientInflowJob(
                            _latticeWidth,
                            _latticeHeight,
                            _solid,
                            _initialHeight,
                            _initialVelocity,
                            _newDistribution,
                            _height,
                            _velocity).
                        Schedule(computeVelocityAndHeightJobHandle) :
                new ZouHeInflowJob(
                            _latticeWidth,
                            _latticeHeight,
                            _inverseE,
                            _solid,
                            _initialHeight,
                            _initialVelocity,
                            _newDistribution,
                            _height,
                            _velocity).
                        Schedule(computeVelocityAndHeightJobHandle);
            var outflowJobHandle =
                usePeriodicBoundary ?
                    inflowJobHandle :
                    _inletOutletBoundaryCondition == InletOutletBoundaryCondition.ZeroGradient ?
                        new ZeroGradientOutflowJob(_latticeWidth, _latticeHeight, _solid, _newDistribution, _height, _velocity).Schedule(inflowJobHandle) :
                        new ZouHeOutflowJob(
                            _latticeWidth,
                            _latticeHeight,
                            _inverseE,
                            _solid,
                            _newDistribution,
                            _height,
                            _velocity).
                        Schedule(inflowJobHandle);
            var floodHeightsJobHandle =
                !_fixupSolidHeights ?
                    outflowJobHandle :
                    new FloodSolidHeightsJob(_latticeWidth, _solid, _height).Schedule(_latticeHeight - 1, 1, outflowJobHandle);
            var computeEquilibriumDistributionJobHandle = computeEquilibriumDistributionJob.Schedule(_latticeHeight, 1, floodHeightsJobHandle);

            // And copy new distribution to the last distribution. This can run somewhat parallel to the simulation jobs.
            var copyNewDistributionToLastDistributionJobHandle = copyNewDistributionToLastDistributionJob.Schedule(outflowJobHandle);
            var fillNewDistributionJobHandle = fillNewDistributionJob.Schedule(copyNewDistributionToLastDistributionJobHandle);

            // Stash the job handles.
            _lastJobHandles = (computeEquilibriumDistributionJobHandle, fillNewDistributionJobHandle);
        }

        private void OnDisable()
        {
            if (_lastJobHandles != null)
            {
                _lastJobHandles.Value.Item1.Complete();
                _lastJobHandles.Value.Item2.Complete();
                _lastJobHandles = null;
            }

            foreach (var marker in _markers)
            {
                DestroyImmediate(marker);
            }
            _markers.Clear();

            // Static sim data
            _linkDirection.Dispose();
            _linkOffsetX.Dispose();
            _linkOffsetY.Dispose();

            // Sim data
            _solid.Dispose();
            _velocity.Dispose();
            _height.Dispose();
            _lastDistribution.Dispose();
            _newDistribution.Dispose();
            _equilibriumDistribution.Dispose();
            _inverseEddyRelaxationTime.Dispose();

            // Result data.
            _solidResult.Dispose();
            _velocityResult.Dispose();
            _heightResult.Dispose();
        }

        public void AddSolidNodeCluster(float2 uv)
        {
            // UV -> row,col
            var colRowIdx = math.int2(math.round(math.saturate(uv) * new float2(_latticeWidth - 1, _latticeHeight - 1)));
            var rowIdx = colRowIdx.y;
            var colIdx = colRowIdx.x;

            // Write area of solid pixels.
            AddSolidNode(math.int2(colIdx - 0, rowIdx - 1));
            AddSolidNode(math.int2(colIdx - 1, rowIdx + 0));
            AddSolidNode(math.int2(colIdx - 0, rowIdx + 0));
            AddSolidNode(math.int2(colIdx + 1, rowIdx + 0));
            AddSolidNode(math.int2(colIdx - 0, rowIdx + 1));
        }

        public void AddSolidNode(float2 uv)
        {
            var colRowIdx = math.int2(math.round(math.saturate(uv) * new float2(_latticeWidth - 1, _latticeHeight - 1)));
            AddSolidNode(colRowIdx);
        }
          
        public void AddSolidNode(int2 colRowIdx)
        {
            if (!_solidResult.IsCreated)
            {
                _solidResult = new NativeArray<byte>(_latticeWidth * _latticeHeight, Allocator.Persistent);
                for (var idx = 0; idx < _solidResult.Length; idx++)
                {
                    _solidResult[idx] = 1;
                }
            }

            var clampedColRowIdx = math.clamp(colRowIdx, math.int2(0), math.int2(_latticeWidth - 1, _latticeHeight - 1));
            var nodeIdx = clampedColRowIdx.y * _latticeWidth + clampedColRowIdx.x;
            _solidResult[nodeIdx] = 0;
        }

        private readonly List<ValueTuple<JobHandle?, Texture2D, TextureEventHandler>>  _textureProcessingList = new();
        private void UpdateTextures(float maxHeight, float maxSpeed)
        {
            _textureProcessingList.Clear();
            _textureProcessingList.Add(UpdateMaskTexture());
            _textureProcessingList.Add(UpdateHeightTexture(maxHeight));
            _textureProcessingList.Add(UpdateFlowTexture(maxSpeed));

            foreach (var processedTexture in _textureProcessingList)
            {
                if (processedTexture.Item1 == null)
                {
                    continue;
                }

                processedTexture.Item1.Value.Complete();
                processedTexture.Item2.Apply(false, false);
                processedTexture.Item3.Invoke(this, processedTexture.Item2);
            }

            _textureProcessingList.Clear();
        }

        private ValueTuple<JobHandle?, Texture2D, TextureEventHandler> UpdateMaskTexture()
        {
            if (MaskTextureUpdated == null || MaskTextureUpdated.GetInvocationList().Length <= 0)
            {
                return NoTextureProcessedResult;
            }

            if (_maskTexture == null)
            {
                _maskTexture =
                    new Texture2D(_latticeWidth, _latticeHeight, TextureFormat.Alpha8, mipChain: false, linear: true)
                    {
                        hideFlags = HideFlags.HideAndDontSave,
                        wrapMode = TextureWrapMode.Clamp
                    };
            }

            var maskTextureData = _maskTexture.GetPixelData<byte>(0);
            var updateMaskTextureJob = new UpdateMaskTextureJob(_latticeWidth, _solid, maskTextureData);
            var jobHandle = updateMaskTextureJob.Schedule(_latticeHeight, 1);

            return (jobHandle, _maskTexture, MaskTextureUpdated);
        }

        private ValueTuple<JobHandle?, Texture2D, TextureEventHandler> UpdateFlowTexture(float maxSpeed)
        {
            if (FlowTextureUpdated == null || FlowTextureUpdated.GetInvocationList().Length <= 0)
            {
                return NoTextureProcessedResult;
            }

            if (_flowTexture == null)
            {
                _flowTexture = new Texture2D(_latticeWidth, _latticeHeight, TextureFormat.RG16, mipChain: false, linear: true);
                _flowTexture.hideFlags = HideFlags.HideAndDontSave;
                _flowTexture.wrapMode = TextureWrapMode.Clamp;
            }

            var flowTextureData = _flowTexture.GetPixelData<byte>(0);
            var updateFlowTexture = new UpdateFlowTextureJob(_latticeWidth, maxSpeed, _solid, _velocity, flowTextureData);
            var jobHandle = updateFlowTexture.Schedule(_latticeHeight, 1);

            return (jobHandle, _flowTexture, FlowTextureUpdated);
        }

        private ValueTuple<JobHandle?, Texture2D, TextureEventHandler> UpdateHeightTexture(float maxHeight)
        {
            if (HeightTextureUpdated == null || HeightTextureUpdated.GetInvocationList().Length <= 0)
            {
                return NoTextureProcessedResult;
            }

            if (_heightTexture == null)
            {
                _heightTexture = new Texture2D(_latticeWidth, _latticeHeight, TextureFormat.RHalf, mipChain: false, linear: true);
                _heightTexture.hideFlags = HideFlags.HideAndDontSave;
                _heightTexture.wrapMode = TextureWrapMode.Clamp;
            }

            var pixelData = _heightTexture.GetPixelData<half>(0);
            var updateHeightTextureJob = new UpdateHeightTextureJob(_latticeWidth, maxHeight, _height, pixelData);
            var jobHandle = updateHeightTextureJob.Schedule(_latticeHeight, 1);

            return (jobHandle, _heightTexture, HeightTextureUpdated);
        }

        private void UpdateMarkers()
        {
            _markerTimer -= Time.deltaTime;
            if (_showMarkers && _markerPrefab && _markerTimer < 0.0f)
            {
                const int markers = 10; ;
                const float markerU = 0.02f;
                const float vBorder = 0.05f;
                const float vHeight = 1.0f - 2 * vBorder;
                const float markerScale = 0.02f;

                for (var i = 0; i <= markers; i++)
                {
                    var uv = new float2(markerU, vBorder + vHeight * i / markers);
                    var localPosition = new Vector3(uv.x, uv.y, 0.0f) - new Vector3(0.5f, 0.5f, 0.0f);
                    var marker = Instantiate(_markerPrefab);
                    marker.transform.SetParent(transform, false);
                    marker.transform.localPosition = localPosition;
                    marker.transform.localScale = markerScale * Vector3.one;
                    _markers.Add(marker);
                }

                _markerTimer = _markerFrequency;
            }

            var markersToRemove = new List<GameObject>();
            foreach (var marker in _markers)
            {
                var markerTransform = marker.transform;
                var uv = new float2(markerTransform.localPosition.x, markerTransform.localPosition.y) + new float2(0.5f, 0.5f);
                var rowColIdxPartial = uv * new float2(_latticeWidth - 1, _latticeHeight - 1);
                var rowIdx = (int)math.round(rowColIdxPartial.y);
                var colIdx = (int)math.round(rowColIdxPartial.x);

                var velocity = _velocity[rowIdx * _latticeWidth + colIdx];
                var newPosition = rowColIdxPartial + velocity;
                var newUV = newPosition / new float2(_latticeWidth - 1, _latticeHeight - 1);
                if (newUV.x >= 1.0 || newUV.y >= 1.0 || newUV.y < 0.0 || newUV.x < 0.0)
                {
                    markersToRemove.Add(marker);
                }
                else
                {
                    var newUV01 = math.saturate(newUV);
                    var newLocalPosition = newUV01 - Half;
                    marker.transform.localPosition = new Vector3(newLocalPosition.x, newLocalPosition.y, 0.0f);
                }
            }

            for (var i = _markers.Count - 1; i >= 0; i--)
            {
                if (markersToRemove.Contains(_markers[i]))
                {
                    DestroyImmediate(_markers[i]);
                    _markers.RemoveAt(i);
                }
            }
        }

        private void DumpStats()
        {
            if (!_dumpStats)
            {
                return;
            }

            var maxHeight = float.MinValue;
            var minHeight = float.MaxValue;
            var maxVelocity = float.MinValue;
            var minVelocity = float.MaxValue;
            var maxFr = float.MinValue;
            float minFr = float.MaxValue;
            float maxCelerity = float.MinValue;

            for (var rowIdx = 0; rowIdx < _latticeHeight; rowIdx++)
            {
                var rowStartIdx = rowIdx * _latticeWidth;
                for (var colIdx = 0; colIdx < _latticeWidth; colIdx++)
                {
                    var idx = rowStartIdx + colIdx;
                    var height = _height[idx];
                    var velocity = _velocity[idx];
                    var speed = math.length(velocity);
                    var fr = height == 0.0f ? 0.0f : speed / math.sqrt(GravitationalForce * height);
                    var celerity = GravitationalForce * height;

                    maxHeight = math.max(maxHeight, height);
                    minHeight = math.min(minHeight, height);
                    maxVelocity = math.max(maxVelocity, speed);
                    minVelocity = math.min(minVelocity, speed);
                    maxFr = math.max(maxFr, fr);
                    minFr = math.min(minFr, fr);
                    maxCelerity = math.max(maxCelerity, celerity);
                }
            }

            //Debug.Log($"height: [{minHeight}, {maxHeight}], velocity: [{minVelocity}, {maxVelocity}], celerity: {maxCelerity}, Fr: [{minFr}, {maxFr}]");

            var sentinelX = 0;
            var sentinelY = _latticeHeight / 2;
            var sentinelIdx = sentinelY * _latticeWidth + sentinelX;
            Debug.Log($"({sentinelX}, {sentinelY}): velocity: {_velocity[sentinelIdx]}, speed: {math.length(_velocity[sentinelIdx])}");
        }
    }
}

