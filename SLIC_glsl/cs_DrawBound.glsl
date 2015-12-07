#version 430
#extension GL_ARB_compute_variable_group_size : enable
layout(local_size_variable) in;
//layout(local_size_x = 16,local_size_y=16) in;
layout(rgba32f,binding=0) uniform image2D frame;
layout(binding=3,r32f) uniform image2D labelsMat;
layout(r32f,binding=4) coherent uniform image2D isTaken;

uniform int width,height;

void main(){


	const int dx8[8] = { -1, -1,  0,  1, 1, 1, 0, -1 };
	const int dy8[8] = { 0, -1, -1, -1, 0, 1, 1,  1 };

	ivec2 pxPos = ivec2(gl_GlobalInvocationID.xy);
	float label_ij = imageLoad(labelsMat,pxPos).x;

	int nr_p=0;
	for(int k=0; k<8; k++){
		ivec2 neigh_xy = ivec2(pxPos.x+dx8[k],pxPos.y+dy8[k]);
	
		if(neigh_xy.x>=0 && neigh_xy.x<width && neigh_xy.y>=0 && neigh_xy.y<height){
			float label_xy = imageLoad(labelsMat,neigh_xy).x;
			if(imageLoad(isTaken,neigh_xy).x==0.f && label_ij!=label_xy){
				nr_p +=1;
			}
		}
	}
	if(nr_p>=2){
		imageStore(frame,pxPos,vec4(1,0,0,1));
		imageStore(isTaken,pxPos,vec4(1));
	}

}


