#version 430
#extension GL_ARB_compute_variable_group_size : enable
layout(local_size_variable) in;
layout(std430, binding=0) buffer ssbo{
float att[];
}clusters;

layout(rgba32f,binding=1) readonly uniform image2D frame;
layout(binding=2,r32f)coherent uniform image2D distancesMat;
layout(binding=3,r32f)uniform image2D labelsMat;

uniform int width,height;
uniform int nSpx;

void main(){
	uint cluster_idx = gl_GlobalInvocationID.x;

	if(cluster_idx<nSpx)
	{
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
					/*att_tmp[0]+=colorFrame.x;
					att_tmp[1]+=colorFrame.y;
					att_tmp[2]+=colorFrame.z;
					att_tmp[3]+=j;
					att_tmp[4]+=i;*/

					counter++;
					

				}
			}
		}
		if(counter!=0){
			/*clusters.att[cluster_idx*5]=att_tmp[0]/counter;
			clusters.att[cluster_idx*5+1]=att_tmp[1]/counter;
			clusters.att[cluster_idx*5+2]=att_tmp[2]/counter;
			clusters.att[cluster_idx*5+3]=att_tmp[3]/counter;
			clusters.att[cluster_idx*5+4]=att_tmp[4]/counter;*/

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
