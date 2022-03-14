using LatticeBoltzmannMethods;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.EventSystems;
using UnityEngine.UI;

public class LbmDebugViz : MonoBehaviour
{
    [SerializeField]
    private RawImage _flowRawImage;

    [SerializeField]
    private RawImage _maskRawImage;

    [SerializeField]
    private LbmSimulator _lbmSimulator;

    private EventSystem _eventSystem;
    private GraphicRaycaster _graphicRaycaster;

    private void OnEnable()
    {
        _eventSystem = EventSystem.current;
        _graphicRaycaster = FindObjectOfType<GraphicRaycaster>();

        _lbmSimulator.FlowTextureUpdated -= LbmSimulator_FlowTextureUpdated;
        _lbmSimulator.FlowTextureUpdated += LbmSimulator_FlowTextureUpdated;

        _lbmSimulator.MaskTextureUpdated -= LbmSimulator_MaskTextureUpdated;
        _lbmSimulator.MaskTextureUpdated += LbmSimulator_MaskTextureUpdated;
    }

    private void OnDisable()
    {
        _lbmSimulator.FlowTextureUpdated -= LbmSimulator_FlowTextureUpdated;
        _lbmSimulator.MaskTextureUpdated -= LbmSimulator_MaskTextureUpdated;
    }

    private void Update()
    {
        if (Input.GetKey(KeyCode.Mouse0))
        {
            var pointerEventData =
                new PointerEventData(_eventSystem)
                {
                    position = Input.mousePosition
                };

            var results = new List<RaycastResult>();
            _graphicRaycaster.Raycast(pointerEventData, results);

            foreach (var result in results)
            {
                if (result.gameObject == _flowRawImage.gameObject)
                {
                    RectTransformUtility.ScreenPointToLocalPointInRectangle(_flowRawImage.rectTransform, Input.mousePosition, null, out var localMousePositionXY);
                    var localPositionXY = new Vector2(_flowRawImage.rectTransform.rect.x, _flowRawImage.rectTransform.rect.y);
                    var uv = (localMousePositionXY - localPositionXY) / _flowRawImage.rectTransform.rect.size;
                    _lbmSimulator.AddSolidNodeCluster(uv);
                    break;
                }
            }
        }
    }

    private void LbmSimulator_FlowTextureUpdated(object sender, Texture2D e)
    {
        _flowRawImage.texture = e;
        _flowRawImage.rectTransform.sizeDelta = new Vector2(e.width, e.height);
        _flowRawImage.GetComponent<AspectRatioFitter>().aspectRatio = e.width / (float)e.height;
    }

    private void LbmSimulator_MaskTextureUpdated(object sender, Texture2D e)
    {
        _maskRawImage.texture = e;
        _maskRawImage.rectTransform.sizeDelta = new Vector2(e.width, e.height);
        _maskRawImage.GetComponent<AspectRatioFitter>().aspectRatio = e.width / (float)e.height;
    }
}
