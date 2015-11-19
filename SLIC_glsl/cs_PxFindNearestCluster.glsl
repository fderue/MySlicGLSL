#version 430
#extension GL_ARB_compute_variable_group_size : enable
layout(local_size_variable) in;
layout(std430, binding=0) buffer ssbo{
float att[];
}clusters;

layout(rgba32f,binding=1) readonly uniform image2D frame;
layout(binding=2,r32f)coherent uniform image2D distancesMat;
layout(binding=3,r32f)writeonly uniform image2D labelsMat;

uniform int diamSpx;
uniform float wc2;
uniform int nBloc_per_cluster;
uniform int width,height;
uniform int nSpx;

float computeDistance(vec2 c_xy,vec2 px_xy, vec3 c_Lab,vec3 px_Lab,float diamSpx2){
	vec2 c_p_xy = c_xy-px_xy;
	float ds2 = pow(c_p_xy.x,2)+pow(c_p_xy.y,2);
	vec3 c_p_Lab = c_Lab-px_Lab;
	float dc2 = pow(c_p_Lab.x,2)+pow(c_p_Lab.y,2)+pow(c_p_Lab.z,2);
	float dist = sqrt(dc2+ds2/diamSpx2*wc2);
	return dist;
}

void main(){


	uint cluster_idx = gl_GlobalInvocationID.x; // not correct if nBloc_per_cluster>1

	if(cluster_idx<nSpx){
		vec2 cluster_xy = vec2(clusters.att[cluster_idx*5+3],clusters.att[cluster_idx*5+4]);
		vec3 cluster_Lab = vec3(clusters.att[cluster_idx*5],clusters.att[cluster_idx*5+1],clusters.att[cluster_idx*5+2]);
		if(cluster_xy.x!=-1){
			float diamSpx2 = diamSpx*diamSpx;
			for(int y=int(cluster_xy.y)-diamSpx; y<int(cluster_xy.y)+diamSpx;y++){
				for(int x=int(cluster_xy.x)-diamSpx; x<int(cluster_xy.x)+diamSpx; x++){
					if(x>=0&&x<width&&y>=0&&y<height){
						ivec2 pxPos = ivec2(x,y);
						vec3 pxLab = vec3(imageLoad(frame,pxPos));
						float d = computeDistance(cluster_xy,vec2(pxPos),cluster_Lab,pxLab,diamSpx2);
						float d_old = imageLoad(distancesMat,pxPos).x;
						if(d<d_old){
						
						imageStore(distancesMat,pxPos,vec4(d));
						imageStore(labelsMat,pxPos,vec4(float(cluster_idx)));
						}
					}
				}
			}
		}
	}
	
}