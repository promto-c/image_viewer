#version 330 core
uniform sampler2D imageTexture;
uniform float lift;
uniform float gamma;
uniform float gain;
uniform float saturation;
uniform bool isTexture; // Indicates if the current drawing is based on texture sampling or not
uniform vec4 color;

in vec2 TexCoords;
out vec4 FragColor;

void main() {
    if (isTexture) {
        vec4 color = texture(imageTexture, TexCoords);
        color.rgb = pow(color.rgb * gain + lift, vec3(1.0 / gamma));

        // Convert color to grayscale for desaturation
        float gray = dot(color.rgb, vec3(0.299, 0.587, 0.114));

        // Interpolate between grayscale and original color based on saturation
        color.rgb = mix(vec3(gray), color.rgb, saturation);

        FragColor = color;
    } else {
        // Direct color when not using texture
        FragColor = color;
    }
}
