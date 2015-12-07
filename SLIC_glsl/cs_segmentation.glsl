#version 430
#extension GL_ARB_compute_variable_group_size : enable
layout(local_size_variable) in;
layout(std430, binding=5) coherent buffer ssbo{
float att[];
}clusters;

layout(rgba32f,binding=1)uniform image2D frame;
layout(binding=2,r32f)coherent uniform image2D distancesMat;
layout(binding=3,r32f)coherent uniform image2D labelsMat;

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


	uint cluster_idx = gl_GlobalInvocationID.x; 

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

		memoryBarrier(); // synchronization before updating cluster



		//================ update cluster ==============

		clusters.att[cluster_idx*5]=0;
		clusters.att[cluster_idx*5+1]=0;
		clusters.att[cluster_idx*5+2]=0;
		clusters.att[cluster_idx*5+3]=0;
		clusters.att[cluster_idx*5+4]=0;
		

		//float att_tmp[5]={0};
		//look in labelsMat for their labels (not optimal)
		int counter = 0;
		for(int i=0; i<height; i++){
			for(int j=0; j<width; j++){
				if(float(cluster_idx)==imageLoad(labelsMat,ivec2(j,i)).x){
					
					vec4 colorFrame = imageLoad(frame,ivec2(j,i));

					clusters.att[cluster_idx*5]+=colorFrame.x;
					clusters.att[cluster_idx*5+1]+=colorFrame.y;
					clusters.att[cluster_idx*5+2]+=colorFrame.z;
					clusters.att[cluster_idx*5+3]+=j;
					clusters.att[cluster_idx*5+4]+=i;

					counter++;
					

				}
			}
		}
		if(counter!=0){

			clusters.att[cluster_idx*5]/=counter;
			clusters.att[cluster_idx*5+1]/=counter;
			clusters.att[cluster_idx*5+2]/=counter;
			clusters.att[cluster_idx*5+3]/=counter;
			clusters.att[cluster_idx*5+4]/=counter;

		}else{
			clusters.att[cluster_idx*5+3]=-1; //reject this center
		}
	}

	
}