#version 330 core
uniform sampler2D imageTexture;
uniform float lift;
uniform float gamma;
uniform float gain;
in vec2 TexCoords;

out vec4 FragColor;


// Declaration of all helper methods

vec2 ocio_lut1d_0_computePos(float f)
{
    float dep;
    float abs_f = abs(f);
    if (abs_f > 6.10351562e-05)
    {
        vec3 fComp = vec3(15., 15., 15.);
        float absarr = min( abs_f, 65504.);
        fComp.x = floor( log2( absarr ) );
        float lower = pow( 2.0, fComp.x );
        fComp.y = ( absarr - lower ) / lower;
        vec3 scale = vec3(1024., 1024., 1024.);
        dep = dot( fComp, scale );
    }
    else
    {
        dep = abs_f * 1023.0 / 6.09755516e-05;
    }
    dep += step(f, 0.0) * 32768.0;
    vec2 retVal;
    retVal.y = floor(dep / 4095.);
    retVal.x = dep - retVal.y * 4095.;
    retVal.x = (retVal.x + 0.5) / 4096.;
    retVal.y = (retVal.y + 0.5) / 17.;
    return retVal;
}

// Declaration of the OCIO shader function

vec4 OCIODisplay(vec4 inPixel)
{
    vec4 outColor = inPixel;

    // Add LUT 1D processing for ocio_lut1d_0

    // {
    //     outColor.r = texture(ocio_lut1d_0Sampler, ocio_lut1d_0_computePos(outColor.r)).r;
    //     outColor.g = texture(ocio_lut1d_0Sampler, ocio_lut1d_0_computePos(outColor.g)).r;
    //     outColor.b = texture(ocio_lut1d_0Sampler, ocio_lut1d_0_computePos(outColor.b)).r;
    // }

    // Add Matrix processing

    {
        vec4 res = vec4(outColor.rgb.r, outColor.rgb.g, outColor.rgb.b, outColor.a);
        res = vec4(1.25, 1.25, 1.25, 1.) * res;
        res = vec4(-0.125, -0.125, -0.125, -0.) + res;
        outColor.rgb = vec3(res.x, res.y, res.z);
        outColor.a = res.w;
    }

    return outColor;

}

void main() {
    vec4 color = texture(imageTexture, TexCoords);
    color.rgb = pow(color.rgb * gain + lift, vec3(1.0/gamma));
    FragColor = OCIODisplay(color);
}
