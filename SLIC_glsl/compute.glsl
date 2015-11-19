#version 430
layout(rgba32f,binding=0)uniform image2D destTex;
layout(local_size_x=16, local_size_y=16) in;
void main(){
ivec2 storePos = ivec2(gl_GlobalInvocationID.xy);
//ivec2 storePos= ivec2(0,0);
vec4 c = imageLoad(destTex,storePos);
imageStore(destTex, storePos, c*2);
}