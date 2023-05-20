#version 330 core
uniform sampler2D imageTexture;
uniform float lift;
uniform float gamma;
uniform float gain;
in vec2 TexCoords;

out vec4 FragColor;

void main() {
    vec4 color = texture(imageTexture, TexCoords);
    color.rgb = pow(color.rgb * gain + lift, vec3(1.0/gamma));
    FragColor = color;
}
