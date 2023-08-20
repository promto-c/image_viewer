#version 330 core
uniform sampler2D imageTexture;
uniform float lift;
uniform float gamma;
uniform float gain;
uniform bool isTexture; // Indicates if the current drawing is based on texture sampling or not
uniform vec4 color;


in vec2 TexCoords;
out vec4 FragColor;

void main() {
    if (isTexture) {
        vec4 color = texture(imageTexture, TexCoords);
        color.rgb = pow(color.rgb * gain + lift, vec3(1.0/gamma));
        FragColor = color;
    } else {
        FragColor = color;
    }
}
