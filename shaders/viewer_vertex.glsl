#version 330 core
layout (location = 0) in vec2 position;
uniform mat4 transformation;  // Uniform variable for transformation matrix

out vec2 TexCoords;

void main() {
    vec4 transformed_position = transformation * vec4(position, 0.0, 1.0);  // Apply the transformation
    gl_Position = transformed_position;
    TexCoords = position * 0.5 + 0.5; // map from [-1,1] to [0,1]
}
