Shader "Lattice Boltzmann Methods/Flow Viz"
{
    Properties
    {
        _MainTex ("Texture", 2D) = "white" {}
        _VelocityPower("Velocity Power", Range(0.0001, 100)) = 0.4
        _FreeChannelScale ("Free Channel Scale", float) = 0.8
    }
    SubShader
    {
        // No culling or depth
        Cull Off ZWrite Off ZTest Always

        Pass
        {
            CGPROGRAM
            #pragma vertex vert
            #pragma fragment frag

            #include "UnityCG.cginc"

            struct appdata
            {
                float4 vertex : POSITION;
                float2 uv : TEXCOORD0;
            };

            struct v2f
            {
                float2 uv : TEXCOORD0;
                float4 vertex : SV_POSITION;
            };

            v2f vert (appdata v)
            {
                v2f o;
                o.vertex = UnityObjectToClipPos(v.vertex);
                o.uv = v.uv;
                return o;
            }

            sampler2D _MainTex;
            float _VelocityPower;
            float _FreeChannelScale;

            fixed4 frag (v2f i) : SV_Target
            {
                fixed4 col = tex2D(_MainTex, i.uv);
                fixed2 velocity = 2.0 * col.rg - 1.0;
                // TODO: Options for different modes.
                // Velocity viz...
                //return pow(_FreeChannelScale * col, 10.0 * _VelocityPower);
                // Speed viz...
                //return float4(4.0 * length(velocity) / sqrt(2), 0.0, 0.0, 1.0);

                // I... don't even know. Looks cool though!
                fixed rescaledLength = pow(length(velocity), _VelocityPower);
                fixed2 newVelocity = rescaledLength * normalize(velocity);
                fixed2 reencodedVelocity = 0.5 * newVelocity + 0.5;
                return fixed4(1.0 - reencodedVelocity.x, 1.0 - pow(1.0 - rescaledLength / sqrt(2), _FreeChannelScale), 1.0 - reencodedVelocity.y, 1.0);
            }
            ENDCG
        }
    }
}
