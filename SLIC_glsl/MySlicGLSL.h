#pragma once

/*
 * Written by Derue Francois-Xavier
 * francois-xavier.derue@polymtl.ca
 *
 * Superpixel oversegmentation
 * GPU implementation of the algorithm SLIC of
 * Achanta et al. [PAMI 2012, vol. 34, num. 11, pp. 2274-2282]
 *
 * Library required :
 * - openCV
 * - openGL 4.3 or higher
 * - GLEW
 * - FREEGLUT
 */


#include <GL/glew.h>
#include <GL/freeglut.h>
#include <opencv2/opencv.hpp>
#include "funUtils.h"


#define N_ITER 5 // Kmean iteration
#define NMAX_THREAD 1024 // depend on gpu


class MySlicGLSL {

private:
	int m_nPx;
	int m_nSpx;
	int m_diamSpx;
	int m_wSpx, m_hSpx, m_areaSpx;
	int m_width, m_height;
	float m_wc;

    //cpu buffer
	float *m_clusters;
	float* m_distances;
	float *m_labels, *m_labelsCPU;
	float* m_isTaken;

	// gpu variable
    // shader
	GLuint vsProg; //vertex shader
	GLuint fsProg; // fragment shader
	GLuint csProg_PxFindNearestCluster;//compute shader 1
	GLuint csProg_UpdateClusters;//compute shader 2
	GLuint csProg_DrawBound; //compute shader 3

    //ssbo
	GLuint ssbo_clusters, ssbo_clustersAcc;

    //texture
	GLuint text_frameRGB; const int text_unit0 = 0; GLuint pbo_frameRGB;
	GLuint text_frameLab; const int text_unit1 = 1; GLuint pbo_frameLab;
	GLuint text_distances; const int text_unit2 = 2; GLuint pbo_distances;
	GLuint text_labels; const int text_unit3 = 3; GLuint pbo_labels;
	GLuint text_isTaken; const int text_unit4 = 4; GLuint pbo_isTaken;

	int nBlock;

	//========= methods ===========
    // init centroids uniformly on a grid spaced by diamSpx
	void InitClusters(cv::Mat & frameLab);
	//=subroutine =
	void InitBuffers(); // allocate buffers on gpu
	void ClearBuffers(); // clear buffers on gpu
	void CreateCS(); // create the compute shaders
	void SendFrame(cv::Mat& frame, cv::Mat& frameLab); //transfer frame to gpu buffer

	 //===== Compute shader Invocation ======
	void gpu_PxFindNearestCluster(); //Assignment
	void gpu_UpdateClusters(); // Update


public:
	MySlicGLSL(int diamSpx, float wc);
	~MySlicGLSL();

	void Initialize(cv::Mat& frame0);
	void Segment(cv::Mat& frame); // gpu superpixel segmentation

	//===== Display function =====
	void displayBound(cv::Mat& image, cv::Scalar colour); // cpu draw
	void gpu_DrawBound(); // gpu draw

	// enforce connectivity between superpixel, discard orphan (optional)
    // implementation from Pascal Mettes : https://github.com/PSMM/SLIC-Superpixels
	void enforceConnectivity();

};
