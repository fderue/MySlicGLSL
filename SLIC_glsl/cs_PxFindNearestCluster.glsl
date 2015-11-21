#version 430
#extension GL_ARB_compute_variable_group_size : enable
layout(local_size_variable) in;
layout(std430, binding=5)readonly coherent buffer ssbo{
float att[];
}clusters;

layout(rgba32f,binding=1)uniform image2D frame;
layout(binding=2,r32f)coherent uniform image2D distancesMat;
layout(binding=3,r32f)coherent uniform image2D labelsMat;

shared float shareClusters[125];

uniform int diamSpx;
uniform float wc2;
uniform int nBloc_per_col,nBloc_per_row;
uniform int width,height;
uniform int nSpx;

float computeDistance(vec2 c_p_xy, vec3 c_Lab,vec3 px_Lab,float diamSpx2){
	
	float ds2 = pow(c_p_xy.x,2)+pow(c_p_xy.y,2);
	vec3 c_p_Lab = c_Lab-px_Lab;
	float dc2 = pow(c_p_Lab.x,2)+pow(c_p_Lab.y,2)+pow(c_p_Lab.z,2);
	float dist = sqrt(dc2+ds2/diamSpx2*wc2);

	return dist;
}

vec2 findMin(float dist[10],float label[10] ){
	vec2 dist_label = vec2(1000000,-1);
	for(int i=0; i<10; i++){
		if(dist[i]<dist_label.x){
			dist_label.x = dist[i];
			dist_label.y = label[i];
		}
	}
	return dist_label;
}

int convertIdx(ivec2 wg, int lc_idx){

	ivec2 relPos2D = ivec2(lc_idx%5-2,lc_idx/5-2);
	ivec2 glPos2D = wg+relPos2D;

	return glPos2D.y*nBloc_per_row+glPos2D.x;
}


void main(){

	ivec2 pxPos = ivec2(gl_GlobalInvocationID.xy);
	float diamSpx2 = diamSpx*diamSpx;
	float distanceMin = 2000000.f;
	float labelMin = -5;
	ivec2 wgIdx = ivec2(gl_WorkGroupID.xy);

//gathering 25 clusters
	if(pxPos.x<width && pxPos.y<height)
	{
		vec3 px_Lab = vec3(imageLoad(frame,pxPos));
		vec2 px_xy = vec2(pxPos);
	
		int idShareClust = 0;
		for(int k=-2; k<=2; k++){
			for(int l=-2; l<=2; l++){
				ivec2 blockIdxy = ivec2(wgIdx.x+l,wgIdx.y+k);
				
				int idShareClust5 = idShareClust*5;
				if(blockIdxy.x>=0 && blockIdxy.x<nBloc_per_row && blockIdxy.y>=0 && blockIdxy.y<nBloc_per_col){
					int clusterId = blockIdxy.y*nBloc_per_row+blockIdxy.x;
					int clusterId5 = clusterId*5;
					
					shareClusters[idShareClust5] = clusters.att[clusterId5];
					shareClusters[idShareClust5+1] = clusters.att[clusterId5+1];
					shareClusters[idShareClust5+2] = clusters.att[clusterId5+2];
					shareClusters[idShareClust5+3] = clusters.att[clusterId5+3];
					shareClusters[idShareClust5+4] = clusters.att[clusterId5+4];
					idShareClust++;

				}else{
					//case when out ouf bound
					shareClusters[idShareClust5] = -1; 
					idShareClust++;
				}
			}
		}

		barrier();
		memoryBarrierShared();

		for(int cluster_idx=0; cluster_idx<25; cluster_idx++) // cluster locaux
		{
			int cluster_idx5 = cluster_idx*5;
			if(shareClusters[cluster_idx5]!=-1){
				vec2 cluster_xy = vec2(shareClusters[cluster_idx5+3],shareClusters[cluster_idx5+4]);
				vec3 cluster_Lab = vec3(shareClusters[cluster_idx5],shareClusters[cluster_idx5+1],shareClusters[cluster_idx5+2]);

				vec2 px_c_xy = px_xy-cluster_xy;
				if(abs(px_c_xy.x)<diamSpx && abs(px_c_xy.y)<diamSpx){

					float distTmp = min(computeDistance(px_c_xy, cluster_Lab,px_Lab,diamSpx2),distanceMin);
					
					if(distTmp!=distanceMin){
						distanceMin = distTmp;
						labelMin = convertIdx(wgIdx,cluster_idx);
					}
				}
			}
		}
		imageStore(distancesMat,pxPos,vec4(distanceMin,0,0,0));
	
		imageStore(labelsMat,pxPos,vec4(labelMin));
	}
}