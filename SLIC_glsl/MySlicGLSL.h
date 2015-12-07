#pragma once

#define N_ITER 5
#define NMAX_THREAD 1024

#include <GL/glew.h>
#include <GL/freeglut.h>
#include <opencv2/opencv.hpp>
#include "funUtils.h"
class MySlicGLSL {

private:
	int m_nPx;
	int m_nSpx;
	int m_diamSpx;
	int m_wSpx, m_hSpx, m_areaSpx;
	int m_width, m_height;
	float m_wc;

	float *m_clusters;
	float* m_distances;
	float *m_labels, *m_labelsCPU;
	float* m_isTaken;

	// gpu variable
	GLuint vsProg; //vertex shader
	GLuint fsProg; // fragment shader
	GLuint csProg_PxFindNearestCluster;//compute shader 1
	GLuint csProg_UpdateClusters;//compute shader 2
	GLuint csProg_DrawBound; //compute shader 3



	GLuint ssbo_clusters, ssbo_clustersAcc;

	GLuint text_frameRGB; const int text_unit0 = 0; GLuint pbo_frameRGB;
	GLuint text_frameLab; const int text_unit1 = 1; GLuint pbo_frameLab;
	GLuint text_distances; const int text_unit2 = 2; GLuint pbo_distances;
	GLuint text_labels; const int text_unit3 = 3; GLuint pbo_labels;
	GLuint text_isTaken; const int text_unit4 = 4; GLuint pbo_isTaken;
	

	int nBlock;

	//========= methods ===========

	void InitClusters(cv::Mat & frameLab);

	//==== subroutine ====
	void InitBuffers(); // allocate buffers on gpu
	void ClearBuffers(); // clear buffers on gpu
	void CreateCS(); // create the compute shaders

	void SendFrame(cv::Mat& frame, cv::Mat& frameLab); //transfer frame to buffer on gpu

	 //===== Kernel function ======
	void gpu_PxFindNearestCluster();
	void gpu_UpdateClusters();


public:
	MySlicGLSL(int diamSpx, float wc);
	~MySlicGLSL();

	void Initialize(cv::Mat& frame0);
	void Segment(cv::Mat& frame);

	//===== Display function =====
	void displayBound(cv::Mat& image, cv::Scalar colour); // cpu draw
	void gpu_DrawBound(); // gpu draw

	// enforce connectivity between superpixel, discard orphan (optional)
	void enforceConnectivity();

};
