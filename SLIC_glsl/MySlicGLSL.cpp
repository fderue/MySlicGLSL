#include "MySlicGLSL.h"

using namespace std;
using namespace cv;

MySlicGLSL::MySlicGLSL(int nSpx, float wc) {
	m_nSpx = nSpx;
	m_wc = wc;
}

MySlicGLSL::~MySlicGLSL(){
	delete[] m_clusters;
	delete[] m_distances;
	delete[] m_labels;
	delete[] m_isTaken;
}

void MySlicGLSL::Initialize(Mat& frame0) {
	m_width = frame0.cols;
	m_height = frame0.rows;
	m_diamSpx = (int)sqrt(m_width*m_height / (float)m_nSpx);

	//count available number of spx
	int nSpxAvailable = 0;
	int diamSpx_d2 = m_diamSpx / 2;
	for (int y = diamSpx_d2-1; y < m_height; y += m_diamSpx)
		for (int x = diamSpx_d2-1; x < m_width; x += m_diamSpx)
			nSpxAvailable++;
	m_nSpx = nSpxAvailable;

	nBlock = (m_nSpx%NTHREAD_PER_BLOCK) ? m_nSpx / NTHREAD_PER_BLOCK + 1 : m_nSpx / NTHREAD_PER_BLOCK;
	//===== init cpu buffer ======
	m_clusters = new float[m_nSpx * 5];
	m_distances = new float[m_width*m_height];
	m_labels = new float[m_width*m_height];
	m_isTaken = new float[m_width*m_height];

	//====== each frame =====


	//===== clear value ====
	for (int i = 0; i < m_height*m_width; i++) {
		m_labels[i] = -1;
		m_distances[i] = 10000;
		m_isTaken[i] = 0;
	}

	//allocate buffer on Gpu
	InitBuffers(); //ok

	// create compute shader
	CreateCS();
}

void MySlicGLSL::Segment(Mat& frame) {

	ClearBuffers(); // reset all the buffers with constant value allocated on the GPU
	
	Mat frameLab;
	cvtColor(frame, frameLab, CV_BGR2Lab);
	frameLab.convertTo(frameLab, CV_32FC3);

	SendFrame(frame,frameLab); // transfer frame in the already allocated space ok
	InitClusters(frameLab); // 
	
	for (int i = 0; i < N_ITER; i++) {
		gpu_segmentation();
		glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT | GL_PIXEL_BUFFER_BARRIER_BIT);
	}
}

void MySlicGLSL::InitClusters(Mat& frameLab) {

	int diamSpx_d2 = m_diamSpx / 2;
	for (int y = diamSpx_d2 - 1, n = 0; y < m_height; y += m_diamSpx) {
		Vec3f* frameLab_r = frameLab.ptr<Vec3f>(y);
		for (int x = diamSpx_d2 - 1; x < m_width; x += m_diamSpx) {
			int idx = n * 5;
			m_clusters[idx] = frameLab_r[x][0];
			m_clusters[idx + 1] = frameLab_r[x][1];
			m_clusters[idx + 2] = frameLab_r[x][2];
			m_clusters[idx + 3] = x;
			m_clusters[idx + 4] = y;
			n++;
		}
	}

	glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssbo_clusters);
	glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, 5 * m_nSpx*sizeof(float), m_clusters);
	glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
}

void MySlicGLSL::InitBuffers()
{
	//cluster vector -> ssbo
	glGenBuffers(1, &ssbo_clusters);
	glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssbo_clusters);
	//glBufferData(GL_SHADER_STORAGE_BUFFER, sizeof(float) * 5 * m_nSpx, m_clusters, GL_DYNAMIC_DRAW); //non immutable
	glBufferStorage(GL_SHADER_STORAGE_BUFFER, sizeof(float) * 5 * m_nSpx, m_clusters, GL_DYNAMIC_STORAGE_BIT|GL_MAP_READ_BIT);
	glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 5, ssbo_clusters);

	//FrameRGB
	glActiveTexture(GL_TEXTURE0);
	glGenTextures(1, &text_frameRGB);
	glBindTexture(GL_TEXTURE_2D, text_frameRGB);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, m_width, m_height, 0, GL_BGR, GL_UNSIGNED_BYTE, NULL);
	glBindImageTexture(text_unit0, text_frameRGB, 0, GL_FALSE, 0, GL_READ_WRITE, GL_RGBA32F);


	//FrameLab
	glActiveTexture(GL_TEXTURE1);
	glGenTextures(1, &text_frameLab);
	glBindTexture(GL_TEXTURE_2D, text_frameLab);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, m_width, m_height, 0, GL_RGB, GL_FLOAT, NULL);
	glBindImageTexture(text_unit1, text_frameLab, 0, GL_FALSE, 0, GL_READ_WRITE, GL_RGBA32F);

	//distances Matrix 
	glActiveTexture(GL_TEXTURE2);
	glGenTextures(1, &text_distances);
	glBindTexture(GL_TEXTURE_2D, text_distances);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_R32F, m_width, m_height, 0, GL_RED, GL_FLOAT, m_distances);
	glBindImageTexture(text_unit2, text_distances, 0, GL_FALSE, 0, GL_READ_WRITE, GL_R32F);

	//label Matrix 
	glActiveTexture(GL_TEXTURE3);
	glGenTextures(1, &text_labels);
	glBindTexture(GL_TEXTURE_2D, text_labels);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_R32F, m_width, m_height, 0, GL_RED, GL_FLOAT, m_labels);
	glBindImageTexture(text_unit3, text_labels, 0, GL_FALSE, 0, GL_READ_WRITE, GL_R32F);

	//isTaken 
	glActiveTexture(GL_TEXTURE4);
	glGenTextures(1, &text_isTaken);
	glBindTexture(GL_TEXTURE_2D, text_isTaken);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_R32F, m_width, m_height, 0, GL_RED, GL_FLOAT, m_isTaken);
	glBindImageTexture(text_unit4, text_isTaken, 0, GL_FALSE, 0, GL_READ_WRITE, GL_R32F);
}

void MySlicGLSL::ClearBuffers() {

	glBindTexture(GL_TEXTURE_2D, text_distances);
	glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, m_width, m_height, GL_RED, GL_FLOAT, m_distances);

	glBindTexture(GL_TEXTURE_2D, text_labels);
	glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, m_width, m_height, GL_RED, GL_FLOAT, m_labels);

	glBindTexture(GL_TEXTURE_2D, text_isTaken);
	glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, m_width, m_height, GL_RED, GL_FLOAT, m_isTaken);

	glBindTexture(GL_TEXTURE_2D, 0);
}

void MySlicGLSL::CreateCS() {

	//vertex shader
	vsProg = createProgShader(GL_VERTEX_SHADER, "vertex.glsl");
	//fragment shader
	fsProg = createProgShader(GL_FRAGMENT_SHADER, "fragment.glsl");
	//compute shader
	csProg_segmentation = createProgShader(GL_COMPUTE_SHADER, "cs_segmentation.glsl");
	csProg_DrawBound = createProgShader(GL_COMPUTE_SHADER, "cs_DrawBound.glsl");
}

void MySlicGLSL::SendFrame(Mat& frame,Mat& frameLab)
{
	glBindTexture(GL_TEXTURE_2D, text_frameRGB);
	glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, m_width, m_height, GL_BGR, GL_UNSIGNED_BYTE, frame.data); //transfert from cpu to gpu
	
	cvtColor(frame, frameLab, CV_BGR2Lab);
	frameLab.convertTo(frameLab, CV_32FC3);
	glBindTexture(GL_TEXTURE_2D, text_frameLab);
	glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, m_width, m_height, GL_RGB, GL_FLOAT, (float*)frameLab.data);

}


void MySlicGLSL::gpu_segmentation() {
	glUseProgram(csProg_segmentation);
	glUniform1i(glGetUniformLocation(csProg_segmentation, "nSpx"), m_nSpx);
	glUniform1i(glGetUniformLocation(csProg_segmentation, "width"), m_width);
	glUniform1i(glGetUniformLocation(csProg_segmentation, "height"), m_height);
	glUniform1i(glGetUniformLocation(csProg_segmentation, "diamSpx"), m_diamSpx);
	glUniform1f(glGetUniformLocation(csProg_segmentation, "wc2"), pow(m_wc, 2));
	glDispatchComputeGroupSizeARB(nBlock, 1, 1, NTHREAD_PER_BLOCK, 1, 1);
}

void MySlicGLSL::gpu_DrawBound() {
	glUseProgram(csProg_DrawBound);
	glUniform1i(glGetUniformLocation(csProg_DrawBound, "width"), m_width);
	glUniform1i(glGetUniformLocation(csProg_DrawBound, "height"), m_height);
	int side = sqrt(NTHREAD_PER_BLOCK);
	glDispatchComputeGroupSizeARB(iDivUp(m_width, side), iDivUp(m_height, side), 1, side, side, 1);
	
	glUseProgram(vsProg);
	glUseProgram(fsProg);
	glUniform1i(glGetUniformLocation(fsProg, "frameRGB"), text_unit0);

	displayTexture2D(text_frameRGB);
}


void MySlicGLSL::displayBound(Mat& image, Scalar colour)
{
	float* label_f = new float[m_width*m_height];
	auto start = getTickCount();
	getTexture(text_labels, GL_RED, GL_FLOAT, label_f); // very slow : 0.5 s
	void* clus_test;
	getSSBO(ssbo_clusters, clus_test);
	float* clus_testf = (float*)clus_test;
	auto end = getTickCount();
	cout << "load text " << (end - start) / getTickFrequency() << endl;


	const int dx8[8] = { -1, -1,  0,  1, 1, 1, 0, -1 };
	const int dy8[8] = { 0, -1, -1, -1, 0, 1, 1,  1 };

	/* Initialize the contour vector and the matrix detailing whether a pixel
	* is already taken to be a contour. */
	vector<Point> contours;
	vector<vector<bool> > istaken;
	for (int i = 0; i < image.rows; i++) {
		vector<bool> nb;
		for (int j = 0; j < image.cols; j++) {
			nb.push_back(false);
		}
		istaken.push_back(nb);
	}

	/* Go through all the pixels. */

	for (int i = 0; i<image.rows; i++) {
		for (int j = 0; j < image.cols; j++) {

			int nr_p = 0;

			/* Compare the pixel to its 8 neighbours. */
			for (int k = 0; k < 8; k++) {
				int x = j + dx8[k], y = i + dy8[k];

				if (x >= 0 && x < image.cols && y >= 0 && y < image.rows) {
					if (istaken[y][x] == false && label_f[i*m_width+j] != label_f[y*m_width+x]) {
						nr_p += 1;
					}
				}
			}
			/* Add the pixel to the contour list if desired. */
			if (nr_p >= 2) {
				contours.push_back(Point(j, i));
				istaken[i][j] = true;
			}

		}
	}

	/* Draw the contour pixels. */
	for (int i = 0; i < (int)contours.size(); i++) {
		image.at<Vec3b>(contours[i].y, contours[i].x) = Vec3b(colour[0], colour[1], colour[2]);
	}

}

