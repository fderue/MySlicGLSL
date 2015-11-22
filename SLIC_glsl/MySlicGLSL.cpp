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
	m_nPx = m_width*m_height;
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
		m_distances[i] = FLT_MAX;
		m_isTaken[i] = 0;
	}

	//allocate buffer on Gpu
	InitBuffers(); //ok

	// create compute shader
	CreateCS();


}

void MySlicGLSL::Segment(Mat& frame) {

	ClearBuffers(); // reset all the buffers with constant value allocated on the GPU (distance, label, isTaken)
	
	Mat frameLab;
	cvtColor(frame, frameLab, CV_BGR2Lab);
	frameLab.convertTo(frameLab, CV_32FC3);
	SendFrame(frame,frameLab); // transfer frame in the already allocated space ok
	InitClusters(frameLab); // 
	
	/*GLubyte* label_f = new GLubyte[m_width*m_height * 3];
	getTexture(text_frameRGB, GL_RGB, GL_UNSIGNED_BYTE, label_f); // very slow : 0.5 s
	for (int i = 0; i < 50; i++)cout <<(float)label_f[i] << endl;*/



	for (int i = 0; i < N_ITER; i++) {
		//gpu_segmentation();
		gpu_PxFindNearestCluster();
		glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT | GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);

		/*void* acc;
		getSSBO(ssbo_clusters, acc);
		float* accInt = (float*)acc;
		for (int a = 0; a < 60; a++)cout << accInt[a] << endl;

		*/
		/*float* label_f = new float[m_width*m_height];
		auto start = getTickCount();
		getTexture(text_distances, GL_RED, GL_FLOAT, label_f); // very slow : 0.5 s
		auto end = getTickCount();
		cout << "load text " << (end - start) / getTickFrequency() << endl;*/
		//for (int a = 0; a < 480*640; a++)cout << label_f[a] << endl;
	
		//gpu_UpdateClusters();
		glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT | GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);

		/*getSSBO(ssbo_clusters, acc);
		accInt = (float*)acc;
		for (int a = 0; a < 60; a++)cout <<"after "<< accInt[a] << endl;*/
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
	glBufferStorage(GL_SHADER_STORAGE_BUFFER, sizeof(float) * 5 * m_nSpx, m_clusters, GL_DYNAMIC_STORAGE_BIT|GL_MAP_READ_BIT|GL_MAP_WRITE_BIT);
	glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, ssbo_clusters);

	glGenBuffers(1, &ssbo_clustersAcc);
	glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssbo_clustersAcc);
	glBufferStorage(GL_SHADER_STORAGE_BUFFER, sizeof(int) * 6 * m_nSpx, NULL, GL_DYNAMIC_STORAGE_BIT | GL_MAP_READ_BIT | GL_MAP_WRITE_BIT);
	glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, ssbo_clustersAcc);
	
	glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);




	createTextureImage2D(GL_TEXTURE0, text_unit0, text_frameRGB, GL_RGBA32F, m_width, m_height, GL_BGR, GL_UNSIGNED_BYTE, NULL, GL_READ_WRITE);
	createTextureImage2D(GL_TEXTURE1, text_unit1, text_frameLab, GL_RGBA32F, m_width, m_height, GL_RGB, GL_FLOAT, NULL, GL_READ_WRITE);
	createTextureImage2D(GL_TEXTURE2, text_unit2, text_distances, GL_R32F, m_width, m_height, GL_RED, GL_FLOAT, NULL, GL_READ_WRITE);
	createTextureImage2D(GL_TEXTURE3, text_unit3, text_labels, GL_R32F, m_width, m_height, GL_RED, GL_FLOAT, NULL, GL_READ_WRITE);
	createTextureImage2D(GL_TEXTURE4, text_unit4, text_isTaken, GL_R32F, m_width, m_height, GL_RED, GL_FLOAT, NULL, GL_READ_WRITE);
	
	createTextureImage2D(GL_TEXTURE5, text_unit5, text_labels2, GL_R32F, m_width, m_height, GL_RED, GL_FLOAT, NULL, GL_READ_WRITE);



	createPBO(pbo_frameRGB, GL_PIXEL_UNPACK_BUFFER, m_nPx * 3, NULL, GL_STREAM_DRAW);
	createPBO(pbo_frameLab, GL_PIXEL_UNPACK_BUFFER, m_nPx * 3 * sizeof(float), NULL, GL_STREAM_DRAW);
	createPBO(pbo_distances, GL_PIXEL_UNPACK_BUFFER, m_nPx * sizeof(float), m_distances, GL_STREAM_DRAW);
	createPBO(pbo_labels, GL_PIXEL_UNPACK_BUFFER, m_nPx * sizeof(float), m_labels, GL_STREAM_DRAW);
	createPBO(pbo_isTaken, GL_PIXEL_UNPACK_BUFFER, m_nPx * sizeof(float), m_isTaken, GL_STREAM_DRAW);

	createPBO(pbo_labels2, GL_PIXEL_UNPACK_BUFFER, m_nPx * sizeof(float), m_labels, GL_STREAM_DRAW);

}

void MySlicGLSL::ClearBuffers() {

	glBindTexture(GL_TEXTURE_2D, text_distances);
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo_distances);
	//glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, m_width, m_height, GL_RED, GL_FLOAT, 0);
	glClearTexImage(text_distances, 0, GL_RED, GL_FLOAT, m_distances);

	glBindTexture(GL_TEXTURE_2D, text_labels);
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo_labels);
	//glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, m_width, m_height, GL_RED, GL_FLOAT, 0);
	glClearTexImage(text_labels, 0, GL_RED, GL_FLOAT, m_labels);

	glBindTexture(GL_TEXTURE_2D, text_isTaken);
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo_isTaken);
	//glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, m_width, m_height, GL_RED, GL_FLOAT, 0);
	glClearTexImage(text_isTaken, 0, GL_RED, GL_FLOAT, m_isTaken);

	glBindTexture(GL_TEXTURE_2D, 0);
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
}

void MySlicGLSL::CreateCS() {

	//vertex shader
	vsProg = createProgShader(GL_VERTEX_SHADER, "vertex.glsl");
	//fragment shader
	fsProg = createProgShader(GL_FRAGMENT_SHADER, "fragment.glsl");
	//compute shader
	//csProg_segmentation = createProgShader(GL_COMPUTE_SHADER, "cs_segmentation.glsl");
	csProg_PxFindNearestCluster = createProgShader(GL_COMPUTE_SHADER, "cs_PxFindNearestCluster.glsl");
	csProg_UpdateClusters = createProgShader(GL_COMPUTE_SHADER, "cs_UpdateClusters.glsl");

	csProg_DrawBound = createProgShader(GL_COMPUTE_SHADER, "cs_DrawBound.glsl");
}

void MySlicGLSL::SendFrame(Mat& frame,Mat& frameLab)
{
	//first : transfert new picture from cpu -> pbo
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo_frameRGB);
	glBufferData(GL_PIXEL_UNPACK_BUFFER, m_nPx*3, 0, GL_STREAM_DRAW);
	GLubyte* ptr = (GLubyte*)glMapBuffer(GL_PIXEL_UNPACK_BUFFER, GL_WRITE_ONLY);
	if (ptr)
	{
		memcpy(ptr, frame.data, m_nPx * 3);
		glUnmapBufferARB(GL_PIXEL_UNPACK_BUFFER); // release pointer to mapping buffer
	}

	glBindTexture(GL_TEXTURE_2D, text_frameRGB);
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo_frameRGB);
	glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, m_width, m_height, GL_BGR, GL_UNSIGNED_BYTE, 0); //transfer from pbo to texture


	glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo_frameLab);
	glBufferData(GL_PIXEL_UNPACK_BUFFER, m_nPx * 3 * sizeof(float), 0, GL_STREAM_DRAW);
	GLfloat* ptrf = (GLfloat*)glMapBuffer(GL_PIXEL_UNPACK_BUFFER, GL_WRITE_ONLY);
	if (ptrf)
	{
		memcpy(ptrf, (float*)frameLab.data, m_nPx * 3*sizeof(float));
		glUnmapBufferARB(GL_PIXEL_UNPACK_BUFFER); // release pointer to mapping buffer
	}

	glBindTexture(GL_TEXTURE_2D, text_frameLab);
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo_frameLab);
	glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, m_width, m_height, GL_RGB, GL_FLOAT, 0);

	glBindTexture(GL_TEXTURE_2D, 0);
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
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

void MySlicGLSL::gpu_PxFindNearestCluster() {
	glUseProgram(csProg_PxFindNearestCluster);
	glUniform1i(glGetUniformLocation(csProg_PxFindNearestCluster, "nSpx"), m_nSpx);
	glUniform1i(glGetUniformLocation(csProg_PxFindNearestCluster, "width"), m_width);
	glUniform1i(glGetUniformLocation(csProg_PxFindNearestCluster, "height"), m_height);
	glUniform1i(glGetUniformLocation(csProg_PxFindNearestCluster, "diamSpx"), m_diamSpx);
	glUniform1f(glGetUniformLocation(csProg_PxFindNearestCluster, "wc2"), pow(m_wc, 2));

	glUniform1i(glGetUniformLocation(csProg_PxFindNearestCluster, "nBloc_per_row"), 640 /16);
	glUniform1i(glGetUniformLocation(csProg_PxFindNearestCluster, "nBloc_per_col"), 480/16);


	//int side = sqrt(NTHREAD_PER_BLOCK);
	//glDispatchComputeGroupSizeARB(iDivUp(m_width, side), iDivUp(m_height, side), 1, side, side, 1);
//	glDispatchComputeGroupSizeARB(640/m_diamSpx, 480 / m_diamSpx, 1, m_diamSpx, m_diamSpx, 1);
	glDispatchComputeGroupSizeARB(640 / 16, 480 / 16, 1,16, 16, 1);
}

void MySlicGLSL::gpu_UpdateClusters() {
	glUseProgram(csProg_UpdateClusters);
	glUniform1i(glGetUniformLocation(csProg_UpdateClusters, "nSpx"), m_nSpx);
	glUniform1i(glGetUniformLocation(csProg_UpdateClusters, "width"), m_width);
	glUniform1i(glGetUniformLocation(csProg_UpdateClusters, "height"), m_height);
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


	displayTexture2D(text_frameRGB,GL_TEXTURE0);
}


void MySlicGLSL::displayBound(Mat& image, Scalar colour)
{
	
	float* label_f = new float[m_width*m_height];
	/*auto start = getTickCount();
	getTexture(text_labels, GL_RED, GL_FLOAT, label_f); // very slow : 0.5 s
	auto end = getTickCount();
	cout << "load text " << (end - start) / getTickFrequency() << endl;
	*/

	glBindTexture(GL_TEXTURE_2D, text_labels);
	glBindBuffer(GL_PIXEL_PACK_BUFFER, pbo_labels);
	glGetTexImage(GL_TEXTURE_2D, 0, GL_RED,GL_FLOAT,0); // asynchrone call , texture -> pbo, no time

	glBindBuffer(GL_PIXEL_PACK_BUFFER, pbo_labels);

	glBufferData(GL_PIXEL_PACK_BUFFER, m_nPx * sizeof(float), NULL, GL_STREAM_READ); 
	auto start = getTickCount();
	GLfloat* ptr = (GLfloat *)glMapBuffer(GL_PIXEL_PACK_BUFFER, GL_READ_ONLY); // return pointer on the pbo, take time when gpu is busy and it is the case
	auto end = getTickCount();
	cout << "load text " << (end - start) / getTickFrequency() << endl;

	if (ptr)
	{
		memcpy(label_f,ptr, m_nPx * sizeof(float));
		glUnmapBufferARB(GL_PIXEL_PACK_BUFFER); // release pointer to mapping buffer
	}

	glBindTexture(GL_TEXTURE_2D, 0);
	glBindBuffer(GL_PIXEL_PACK_BUFFER,0);

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

