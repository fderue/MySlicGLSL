#version 430
#extension GL_ARB_compute_variable_group_size : enable

/*
Compute Shader : Update step
*/

layout(local_size_variable) in;
layout(std430, binding=0)coherent buffer ssbo{
float att[];
}clusters;
layout(std430, binding=1)coherent buffer ssboAcc{
	int accAtt[];
};



uniform int width,height;
uniform int nSpx;

void main(){
	uint cluster_idx = gl_GlobalInvocationID.x;

	if(cluster_idx<nSpx)
	{
		uint cluster_idx6 = cluster_idx*6;
		uint cluster_idx5 = cluster_idx*5;
		int counter = accAtt[cluster_idx6+5];
		if(counter != 0){
			clusters.att[cluster_idx5] = accAtt[cluster_idx6]/counter;
			clusters.att[cluster_idx5+1] = accAtt[cluster_idx6+1]/counter;
			clusters.att[cluster_idx5+2] = accAtt[cluster_idx6+2]/counter;
			clusters.att[cluster_idx5+3] = accAtt[cluster_idx6+3]/counter;
			clusters.att[cluster_idx5+4] = accAtt[cluster_idx6+4]/counter;

//reset accumulator
			accAtt[cluster_idx6] = 0;
			accAtt[cluster_idx6+1] = 0;
			accAtt[cluster_idx6+2] = 0;
			accAtt[cluster_idx6+3] = 0;
			accAtt[cluster_idx6+4] = 0;
			accAtt[cluster_idx6+5] = 0;
			barrier();
		}
	}
}
