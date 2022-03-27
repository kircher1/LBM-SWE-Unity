using System;
using System.Collections.Generic;
using Unity.Collections;
using Unity.Jobs;
using Unity.Mathematics;
using UnityEngine;

namespace LatticeBoltzmannMethods
{
    // TODO: Optimization ideas
    // - Ping-pong sim data and pick up result at the beginning of next frame and then start next sim.
    // - Split link data into separate arrays. Then run equilibrium distribution jobs in parallel.

    [ExecuteInEditMode]
    public class LbmSimulator : MonoBehaviour
    {
        private const float GravitationalForce = 9.8f;
        private const float Sqrt2 = 1.41421356237f;
        private const float PiOverFour = 0.78539816339f;

        private static readonly ValueTuple<JobHandle?, Texture2D, TextureEventHandler> NoTextureProcessedResult = (null, null, null);

        [SerializeField]
        [Range(0.016f, 1.0f)]
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
        [Range(0.51f, 2.0f)]
        private float _relaxationTime = 0.51f;

        [SerializeField]
        private bool _applyEddyRelaxationTime = true;

        [SerializeField]
        private float2 _bedSlope = new float2(-0.005f, 0.0f);

        [SerializeField]
        private bool _applyShearForces = false;

        [SerializeField]
        [Range(0.06f, 0.5f)]
        private float _smagorinskyConstant = 0.18f;

        [SerializeField]
        [Range(0.01f, 1f)]
        private float _textureScale = 1.0f;

        [SerializeField]
        private bool _usePeriodicBoundary = false;

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

        public event TextureEventHandler FlowTextureUpdated;
        public event TextureEventHandler HeightTextureUpdated;
        public event TextureEventHandler MaskTextureUpdated;
        public event TextureEventHandler ForceTextureUpdated;

        public float MaxSpeed { get; private set; } = 1.0f;

        private static readonly float2 Half = new float2(0.5f, 0.5f);

        private NativeArray<float2> _linkDirection;
        public NativeArray<float2> LinkDirection => _linkDirection;
        private NativeArray<int> _linkOffsetX;
        public NativeArray<int> LinkOffsetX => _linkOffsetX;
        private NativeArray<int> _linkOffsetY;
        public NativeArray<int> LinkOffsetY => _linkOffsetY;

        // Simulation data.

        private NativeArray<bool> _solid;
        public NativeArray<bool> Solid => _solid;
        private NativeArray<float2> _velocity;
        public NativeArray<float2> Velocity => _velocity;
        private NativeArray<float2> _force;
        private NativeArray<float2> _forceMinMaxResult;
        private NativeArray<float> _height;
        public NativeArray<float> Height => _height;
        private NativeArray<float> _lastDistribution;
        private NativeArray<float> _newDistribution;
        private NativeArray<float> _equilibriumDistribution;

        private readonly List<GameObject> _markers = new List<GameObject>();
        private float _markerTimer = 1.0f;

        private Texture2D _flowTexture;
        private Texture2D _heightTexture;
        private Texture2D _maskTexture;
        private Texture2D _forceTexture;

        private float _e;
        private float _inverseE;
        private float _maxHeight;
        private float _initialHeight;
        private float2 _initialVelocity;

        private static void FillLinkData(
            out NativeArray<float2> linkDirection,
            out NativeArray<int> linkOffsetX,
            out NativeArray<int> linkOffsetY)
        {
            linkDirection = new NativeArray<float2>(9, Allocator.Persistent);
            linkOffsetX = new NativeArray<int>(9, Allocator.Persistent);
            linkOffsetY = new NativeArray<int>(9, Allocator.Persistent);

            var _linkOffsetX = new[] { 0, 1, 1, 0, -1, -1, -1, 0, 1 };
            var _linkOffsetY = new[] { 0, 0, 1, 1, 1, 0, -1, -1, -1 };

            for (var linkIdx = 1; linkIdx < 9; linkIdx++)
            {
                var angle = PiOverFour * (linkIdx - 1);
                linkDirection[linkIdx] = math.normalize(new float2(math.cos(angle), math.sin(angle)));
                if (linkIdx % 2 == 0)
                {
                    linkDirection[linkIdx] *= Sqrt2;
                }

                linkOffsetX[linkIdx] = _linkOffsetX[linkIdx];
                linkOffsetY[linkIdx] = _linkOffsetY[linkIdx];
            }
        }

        private void OnEnable()
        {
            FillLinkData(out _linkDirection, out _linkOffsetX, out _linkOffsetY);

            _markerTimer = _markerFrequency;

            InitializeSimData();
        }

        private void InitializeNativeArrays()
        {
            if (_solid.IsCreated)
            {
                return;
            }

            _solid = new NativeArray<bool>(_latticeWidth * _latticeHeight, Allocator.Persistent);
            _lastDistribution = new NativeArray<float>(_latticeWidth * _latticeHeight * 9, Allocator.Persistent);
            _newDistribution = new NativeArray<float>(_latticeWidth * _latticeHeight * 9, Allocator.Persistent);
            _equilibriumDistribution = new NativeArray<float>(_latticeWidth * _latticeHeight * 9, Allocator.Persistent);
            _height = new NativeArray<float>(_latticeWidth * _latticeHeight, Allocator.Persistent);
            _velocity = new NativeArray<float2>(_latticeWidth * _latticeHeight, Allocator.Persistent);
            _force = new NativeArray<float2>(_latticeWidth * _latticeHeight, Allocator.Persistent);
        }

        private void InitializeSimData()
        {
            InitializeNativeArrays();

            var startupLog = "LbmSimulator ---\r\n";

            // Set solid rails unless some solid nodes have already been set by someone else.
            if (!_solid.IsCreated)
            {
                startupLog += "Added solid rails\r\n";
                for (var idx = 0; idx < _solid.Length; idx++)
                {
                    _solid[idx] = true;
                }
                for (var rowIdx = 1; rowIdx < _latticeHeight - 1; rowIdx++)
                {
                    var rowStartIdx = rowIdx * _latticeWidth;
                    for (var colIdx = 0; colIdx < _latticeWidth; colIdx++)
                    {
                        _solid[rowStartIdx + colIdx] = false;
                    }
                }
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
                _height[nodeIdx] = _solid[nodeIdx] ? 0.0f : _initialHeight;
            }

            // Set initial velocity, some factor of the max speed.
            _initialVelocity = new float2((_startingHeight / _maxHeight) * MaxSpeed / Sqrt2, 0.0f);
            startupLog += $"Initial velocity: {_initialVelocity}\r\n";
            for (var nodeIdx = 0; nodeIdx < _velocity.Length; nodeIdx++)
            {
                _velocity[nodeIdx] = _solid[nodeIdx] ? float2.zero : _initialVelocity;
            }

            // Compute force from initial height.
            for (var nodeIdx = 0; nodeIdx < _height.Length; nodeIdx++)
            {
                _force[nodeIdx] = _solid[nodeIdx] ? float2.zero : -GravitationalForce * _height[nodeIdx] * _bedSlope;
            }
            _forceMinMaxResult = new NativeArray<float2>(2, Allocator.Persistent);

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
            // Compute per-frame terms.
            var inverseESq = 1.0f / (_e * _e);
            var smagorinskyConstantSq = _smagorinskyConstant * _smagorinskyConstant;
            var relaxationTimeSq = _relaxationTime * _relaxationTime;

            // Setup jobs.
            var collideJob =
                new CollideJob(
                    _simulationStepTime,
                    _latticeWidth,
                    _e,
                    inverseESq,
                    _applyEddyRelaxationTime,
                    _relaxationTime,
                    relaxationTimeSq,
                    smagorinskyConstantSq,
                    _linkDirection,
                    _solid,
                    _equilibriumDistribution,
                    _force,
                    _height,
                    _lastDistribution);
            var streamJob =
                new StreamJob(
                    _usePeriodicBoundary,
                    _latticeWidth,
                    _latticeHeight,
                    _linkOffsetX,
                    _linkOffsetY,
                    _solid,
                    _lastDistribution,
                    _newDistribution);
            var computeVelocityAndHeightJob =
                new ComputeVelocityAndHeightJob(
                    _latticeWidth,
                    _e,
                    _maxHeight,
                    GravitationalForce,
                    _linkDirection,
                    _solid,
                    _newDistribution,
                    _height,
                    _velocity);
            var updateForcesJob =
                new UpdateForcesJob(
                    _latticeWidth,
                    _latticeHeight,
                    GravitationalForce,
                    _bedSlope,
                    _applyShearForces,
                    _linkDirection,
                    _linkOffsetX,
                    _linkOffsetY,
                    _solid,
                    _height,
                    _velocity,
                    _force);
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

            // Schedule main simulation jobs.
            var collideJobHandle = collideJob.Schedule(_latticeHeight, 1);
            var streamJobHandle = streamJob.Schedule(_latticeHeight, 1, collideJobHandle);
            var inflowJobHandle =
                _usePeriodicBoundary ?
                    streamJobHandle :
                    // TODO: Enum to select option.
                    //new ZeroGradientInflowJob(_latticeWidth, _latticeHeight, _solid, _newDistribution).Schedule(streamJobHandle);
                    new ZhouHeInflowJob(_latticeWidth, _latticeHeight, _inverseE, _solid, _initialHeight, _initialVelocity, _newDistribution).Schedule(streamJobHandle);
            var computeVelocityAndHeightJobHandle = computeVelocityAndHeightJob.Schedule(_latticeHeight, 1, inflowJobHandle);
            JobHandle? floodHeightsJobHandle = null;
            if (_fixupSolidHeights)
            {
                var floodHeightsJob = new FloodSolidHeightsJob(_latticeWidth, _solid, _height);
                floodHeightsJobHandle = floodHeightsJob.Schedule(_latticeHeight - 1, 1, computeVelocityAndHeightJobHandle);
            }
            var computeEquilibriumDistributionJobHandle = computeEquilibriumDistributionJob.Schedule(_latticeHeight, 1, floodHeightsJobHandle ?? computeVelocityAndHeightJobHandle);
            var updateForcesJobHandle = updateForcesJob.Schedule(_latticeHeight, 1, computeEquilibriumDistributionJobHandle);

            // Copy new distribution to the last distribution.
            var copyNewDistributionToLastDistributionJobHandle = copyNewDistributionToLastDistributionJob.Schedule(computeVelocityAndHeightJobHandle);
            var fillNewDistributionJobHandle = fillNewDistributionJob.Schedule(copyNewDistributionToLastDistributionJobHandle);

            // Wait for jobs to complete.
            updateForcesJobHandle.Complete();
            fillNewDistributionJobHandle.Complete();

            // After jobs complete, update any textures and markers.
            UpdateTextures(_maxHeight, MaxSpeed);
            UpdateMarkers(); // TODO: Jobify/move to different component.
            DumpStats(); // TODO: Jobify/move to different component.
        }

        private void OnDisable()
        {
            foreach (var marker in _markers)
            {
                DestroyImmediate(marker);
            }
            _markers.Clear();

            // Cleanup native arrays.
            _linkDirection.Dispose();
            _linkOffsetX.Dispose();
            _linkOffsetY.Dispose();
            _solid.Dispose();
            _velocity.Dispose();
            _force.Dispose();
            _forceMinMaxResult.Dispose();
            _height.Dispose();
            _lastDistribution.Dispose();
            _newDistribution.Dispose();
            _equilibriumDistribution.Dispose();
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
            if (!_solid.IsCreated)
            {
                InitializeNativeArrays();
            }

            var clampedColRowIdx = math.clamp(colRowIdx, math.int2(0), math.int2(_latticeWidth - 1, _latticeHeight - 1));
            var nodeIdx = clampedColRowIdx.y * _latticeWidth + clampedColRowIdx.x;

            _solid[nodeIdx] = true;

            for (var linkIdx = 0; linkIdx < 1; linkIdx++)
            {
                _newDistribution[9 * nodeIdx + linkIdx] = 0.0f;
            }

            _velocity[nodeIdx] = float2.zero;
            _height[nodeIdx] = 0.0f;
            _force[nodeIdx] = float2.zero;
        }

        private void UpdateTextures(float maxHeight, float maxSpeed)
        {
            var textureProcessingList =
                new List<(JobHandle?, Texture2D, TextureEventHandler)>
                {
                    UpdateMaskTexture(),
                    UpdateForceTexture(),
                    UpdateHeightTexture(maxHeight),
                    UpdateFlowTexture(maxSpeed)
                };

            foreach (var processedTexture in textureProcessingList)
            {
                if (processedTexture.Item1 == null)
                {
                    continue;
                }

                processedTexture.Item1.Value.Complete();
                processedTexture.Item3.Invoke(this, processedTexture.Item2);
            }
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
            jobHandle.Complete();
            _maskTexture.Apply(false, false);

            MaskTextureUpdated?.Invoke(this, _maskTexture);

            return (jobHandle, _maskTexture, MaskTextureUpdated);
        }

        private ValueTuple<JobHandle?, Texture2D, TextureEventHandler> UpdateForceTexture()
        {
            if (ForceTextureUpdated == null || ForceTextureUpdated.GetInvocationList().Length <= 0)
            {
                return NoTextureProcessedResult;
            }

            if (_forceTexture == null)
            {
                _forceTexture =
                    new Texture2D(_latticeWidth, _latticeHeight, TextureFormat.RG16, mipChain: false, linear: true)
                    {
                        hideFlags = HideFlags.HideAndDontSave,
                        wrapMode = TextureWrapMode.Clamp
                    };
            }

            var computeMinMaxJob = new ComputeMinMaxFloat2Job(_force, _forceMinMaxResult);
            var computeMinMaxJobHandle = computeMinMaxJob.Schedule();
            var updateForceTextureJob = new UpdateForceTextureJob(_latticeWidth, _forceMinMaxResult, _force, _forceTexture.GetPixelData<byte>(0));
            var updateForceTextureJobHandle = updateForceTextureJob.Schedule(_latticeHeight, 1, computeMinMaxJobHandle);
            updateForceTextureJobHandle.Complete();
            _forceTexture.Apply(false, false);

            ForceTextureUpdated?.Invoke(this, _forceTexture);

            return (updateForceTextureJobHandle, _forceTexture, ForceTextureUpdated);
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
            var updateFlowTexture = new UpdateFlowTextureJob(_latticeWidth, maxSpeed, _textureScale, _solid, _velocity, flowTextureData);
            var jobHandle = updateFlowTexture.Schedule(_latticeHeight, 1);
            jobHandle.Complete();
            _flowTexture.Apply(false, false);

            FlowTextureUpdated?.Invoke(this, _flowTexture);

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
            var adjustedMax = _textureScale * maxHeight;
            var updateHeightTextureJob = new UpdateHeightTextureJob(_latticeWidth, adjustedMax, _height, pixelData);
            var jobHandle = updateHeightTextureJob.Schedule(_latticeHeight, 1);
            jobHandle.Complete();
            _heightTexture.Apply(false, false);

            HeightTextureUpdated?.Invoke(this, _heightTexture);

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

