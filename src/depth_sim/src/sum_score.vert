#version 130
#extension GL_ARB_explicit_attrib_location : enable
#extension GL_ARB_explicit_uniform_location : enable

layout (location = 0) in vec3 Position;
layout (location = 1) in vec2 TexCoord;

out vec2 TexCoord0;

void main()
{
  TexCoord0 = TexCoord;
  gl_Position = vec4(Position, 1.0);
}
