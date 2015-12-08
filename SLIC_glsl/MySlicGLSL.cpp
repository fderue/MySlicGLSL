#include "MySlicGLSL.h"

using namespace std;
using namespace cv;

// specify shader path !!
char* cs_PxFindNearestCluster_path = "/media/derue/4A30A96F30A962A5/ClionProjects/SLIC_glsl/MySlicGLSL/SLIC_glsl/cs_PxFindNearestCluster.glsl";
char* cs_UpdateClusters_path = "/media/derue/4A30A96F30A962A5/ClionProjects/SLIC_glsl/MySlicGLSL/SLIC_glsl/cs_UpdateClusters.glsl";
char* cs_DrawBound_path = "/media/derue/4A30A96F30A962A5/ClionProjects/SLIC_glsl/MySlicGLSL/SLIC_glsl/cs_DrawBound.glsl";
char* fs_path = "/media/derue/4A30A96F30A962A5/ClionProjects/SLIC_glsl/MySlicGLSL/SLIC_glsl/fragment.glsl";
char* vs_path = "/media/derue/4A30A96F30A962A5/ClionProjects/SLIC_glsl/MySlicGLSL/SLIC_glsl/vertex.glsl";

MySlicGLSL::MySlicGLSL(int diamSpx, float wc) {
	m_diamSpx = diamSpx;
	m_wc = wc;
}

MySlicGLSL::~MySlicGLSL(){
	delete[] m_clusters;
	delete[] m_distances;
	delete[] m_labels;
	delete[] m_labelsCPU;
	delete[] m_isTaken;
}

void MySlicGLSL::Initialize(Mat& frame0) {
	m_width = frame0.cols;
	m_height = frame0.rows;
	m_nPx = m_width*m_height;
	getWlHl(m_width, m_height, m_diamSpx, m_wSpx, m_hSpx); // determine w and h of Spx based on diamSpx
	m_areaSpx = m_wSpx*m_hSpx;
	m_nSpx = m_nPx/m_areaSpx; // should be an integer!!

	//===== init cpu buffer  ======
	m_clusters = new float[m_nSpx * 5];
	m_distances = new float[m_nPx];
	m_labels = new float[m_nPx];
	m_isTaken = new float[m_nPx];
	m_labelsCPU = new float[m_nPx]; 

	//===== buffer clearing value ====
	for (int i = 0; i < m_height*m_width; i++) {
		m_labels[i] = -1;
		m_distances[i] = FLT_MAX;
		m_isTaken[i] = 0;
	}

	//allocate buffer on Gpu
	InitBuffers();

	// create compute shader
	CreateCS();
}

void MySlicGLSL::Segment(Mat& frame) {
	ClearBuffers();
	
	Mat frameLab;
	cvtColor(frame, frameLab, CV_BGR2Lab);
	frameLab.convertTo(frameLab, CV_32FC3);
	SendFrame(frame,frameLab);
	InitClusters(frameLab); 

	// main loop
	for (int i = 0; i < N_ITER; i++) {
		gpu_PxFindNearestCluster();
		glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT | GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);
	
		gpu_UpdateClusters();
		glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT | GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);
	}

	//enforceConnectivity(); // optional (cpu implementation -> slow down) , comment for full speed
}

void MySlicGLSL::InitClusters(Mat& frameLab) {

	int wl_d2 = m_wSpx / 2;
	int hl_d2 = m_hSpx / 2;

	for (int y = hl_d2 - 1, n = 0; y < m_height; y += m_hSpx) {
		Vec3f* frameLab_r = frameLab.ptr<Vec3f>(y);
		for (int x = wl_d2 - 1; x < m_width; x += m_wSpx) {
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
	//cluster vector -> ssbo 0
	glGenBuffers(1, &ssbo_clusters);
	glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssbo_clusters);
	glBufferStorage(GL_SHADER_STORAGE_BUFFER, sizeof(float) * 5 * m_nSpx, m_clusters, GL_DYNAMIC_STORAGE_BIT|GL_MAP_READ_BIT|GL_MAP_WRITE_BIT);
	glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, ssbo_clusters);

	//cluster accumulative vector -> ssbo 1
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
	
	createPBO(pbo_frameRGB, GL_PIXEL_UNPACK_BUFFER, m_nPx * 3, NULL, GL_STREAM_DRAW);
	createPBO(pbo_frameLab, GL_PIXEL_UNPACK_BUFFER, m_nPx * 3 * sizeof(float), NULL, GL_STREAM_DRAW);
	createPBO(pbo_distances, GL_PIXEL_UNPACK_BUFFER, m_nPx * sizeof(float), m_distances, GL_STREAM_DRAW);
	createPBO(pbo_labels, GL_PIXEL_UNPACK_BUFFER, m_nPx * sizeof(float), m_labels, GL_STREAM_DRAW);
	createPBO(pbo_isTaken, GL_PIXEL_UNPACK_BUFFER, m_nPx * sizeof(float), m_isTaken, GL_STREAM_DRAW);
}

void MySlicGLSL::ClearBuffers() {

	glBindTexture(GL_TEXTURE_2D, text_distances);
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo_distances);
	glClearTexImage(text_distances, 0, GL_RED, GL_FLOAT, m_distances);

	glBindTexture(GL_TEXTURE_2D, text_labels);
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo_labels);
	glClearTexImage(text_labels, 0, GL_RED, GL_FLOAT, m_labels);

	glBindTexture(GL_TEXTURE_2D, text_isTaken);
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo_isTaken);
	glClearTexImage(text_isTaken, 0, GL_RED, GL_FLOAT, m_isTaken);

	glBindTexture(GL_TEXTURE_2D, 0);
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
}

void MySlicGLSL::CreateCS() {

	//vertex shader
	vsProg = createProgShader(GL_VERTEX_SHADER, vs_path);
	//fragment shader
	fsProg = createProgShader(GL_FRAGMENT_SHADER,fs_path);
	//compute shader
	csProg_PxFindNearestCluster = createProgShader(GL_COMPUTE_SHADER, cs_PxFindNearestCluster_path);
	csProg_UpdateClusters = createProgShader(GL_COMPUTE_SHADER, cs_UpdateClusters_path);
	csProg_DrawBound = createProgShader(GL_COMPUTE_SHADER, cs_DrawBound_path);
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

void MySlicGLSL::gpu_PxFindNearestCluster() {
	glUseProgram(csProg_PxFindNearestCluster);
	glUniform1i(glGetUniformLocation(csProg_PxFindNearestCluster, "nSpx"), m_nSpx);
	glUniform1i(glGetUniformLocation(csProg_PxFindNearestCluster, "width"), m_width);
	glUniform1i(glGetUniformLocation(csProg_PxFindNearestCluster, "height"), m_height);
	glUniform1i(glGetUniformLocation(csProg_PxFindNearestCluster, "wSpx"), m_wSpx);
	glUniform1i(glGetUniformLocation(csProg_PxFindNearestCluster, "hSpx"), m_hSpx);

	glUniform1f(glGetUniformLocation(csProg_PxFindNearestCluster, "wc2"), pow(m_wc, 2));

	glUniform1i(glGetUniformLocation(csProg_PxFindNearestCluster, "nBloc_per_row"), m_width /m_wSpx);
	glUniform1i(glGetUniformLocation(csProg_PxFindNearestCluster, "nBloc_per_col"), m_height/m_hSpx);

	glDispatchComputeGroupSizeARB(m_width / m_wSpx, m_height / m_hSpx, 1,m_wSpx, m_hSpx, 1);
}

void MySlicGLSL::gpu_UpdateClusters() {
	glUseProgram(csProg_UpdateClusters);
	glUniform1i(glGetUniformLocation(csProg_UpdateClusters, "nSpx"), m_nSpx);
	glUniform1i(glGetUniformLocation(csProg_UpdateClusters, "width"), m_width);
	glUniform1i(glGetUniformLocation(csProg_UpdateClusters, "height"), m_height);

	nBlock = m_nSpx / NMAX_THREAD;
	glDispatchComputeGroupSizeARB(nBlock, 1, 1, NMAX_THREAD, 1, 1);
}

void MySlicGLSL::gpu_DrawBound() {
	glUseProgram(csProg_DrawBound);
	glUniform1i(glGetUniformLocation(csProg_DrawBound, "width"), m_width);
	glUniform1i(glGetUniformLocation(csProg_DrawBound, "height"), m_height);
	int side = sqrt(NMAX_THREAD);
	glDispatchComputeGroupSizeARB(iDivUp(m_width, side), iDivUp(m_height, side), 1, side, side, 1);
	
	glUseProgram(vsProg);
	glUseProgram(fsProg);
	glUniform1i(glGetUniformLocation(fsProg, "frameRGB"), text_unit0);

	displayTexture2D(text_frameRGB,GL_TEXTURE0);
}


void MySlicGLSL::displayBound(Mat& image, Scalar colour)
{

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
					if (istaken[y][x] == false && m_labelsCPU[i*m_width+j] != m_labelsCPU[y*m_width+x]) {
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

void MySlicGLSL::enforceConnectivity()
{
	glBindTexture(GL_TEXTURE_2D, text_labels);
	//getTexture(text_labels,GL_RED,GL_FLOAT,m_labelsCPU);
	glBindBuffer(GL_PIXEL_PACK_BUFFER, pbo_labels);
	glGetTexImage(GL_TEXTURE_2D, 0, GL_RED, GL_FLOAT, 0); //tex->pbo

	glBindBuffer(GL_PIXEL_PACK_BUFFER, pbo_labels);

	GLfloat* ptr = (GLfloat *)glMapBuffer(GL_PIXEL_PACK_BUFFER, GL_READ_ONLY); //pbo->cpu

	if (ptr)
	{
		memcpy(m_labelsCPU, ptr, m_nPx * sizeof(float));
		glUnmapBuffer(GL_PIXEL_PACK_BUFFER); // release pointer to mapping buffer
	}
	else{
		cerr<<"! did not get label from gpu"<<endl;
	}


	glBindTexture(GL_TEXTURE_2D, 0);
	glBindBuffer(GL_PIXEL_PACK_BUFFER, 0);

    int label = 0, adjlabel = 0;
	int lims = (m_width * m_height) / (m_nSpx);
	lims = lims >> 2;


	const int dx4[4] = { -1, 0, 1, 0 };
	const int dy4[4] = { 0, -1, 0, 1 };

	vector<vector<int> >newLabels;
	for (int i = 0; i < m_height; i++)
	{
		vector<int> nv(m_width, -1);
		newLabels.push_back(nv);
	}

	for (int i = 0; i < m_height; i++)
	{
		for (int j = 0; j < m_width; j++)
		{
			if (newLabels[i][j] == -1)
			{
				vector<Point> elements;
				elements.push_back(Point(j, i));
				for (int k = 0; k < 4; k++)
				{
					int x = elements[0].x + dx4[k], y = elements[0].y + dy4[k];
					if (x >= 0 && x < m_width && y >= 0 && y < m_height)
					{
						if (newLabels[y][x] >= 0)
						{
							adjlabel = newLabels[y][x];
						}
					}
				}
				int count = 1;
				for (int c = 0; c < count; c++)
				{
					for (int k = 0; k < 4; k++)
					{
						int x = elements[c].x + dx4[k], y = elements[c].y + dy4[k];
						if (x >= 0 && x < m_width && y >= 0 && y < m_height)
						{
							if (newLabels[y][x] == -1 && m_labelsCPU[i*m_width+j] == m_labelsCPU[y*m_width+x])
							{
								elements.push_back(Point(x, y));
								newLabels[y][x] = label;//m_labels[i][j];
								count += 1;
							}
						}
					}
				}
				if (count <= lims) {
					for (int c = 0; c < count; c++) {
						newLabels[elements[c].y][elements[c].x] = adjlabel;
					}
					label -= 1;
				}
				label += 1;
			}
		}
	}

	//m_nSpx = label;
	for (int i = 0; i<newLabels.size(); i++)
		for (int j = 0; j<newLabels[i].size(); j++)
			m_labelsCPU[i*m_width+j] = newLabels[i][j];


    // send back to gpu (if need to draw with opengl)
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER,pbo_labels);
    glBufferData(GL_PIXEL_UNPACK_BUFFER, m_nPx* sizeof(float), 0, GL_STREAM_DRAW);
    GLfloat * ptrf = (GLfloat*)glMapBuffer(GL_PIXEL_UNPACK_BUFFER, GL_WRITE_ONLY);
    if (ptrf)
    {
        memcpy(ptrf, m_labelsCPU, m_nPx * sizeof(float));
        glUnmapBufferARB(GL_PIXEL_UNPACK_BUFFER); // release pointer to mapping buffer
    }

    glBindTexture(GL_TEXTURE_2D, text_labels);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo_labels);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, m_width, m_height, GL_RED, GL_FLOAT, 0); //transfer from pbo to texture

    glBindTexture(GL_TEXTURE_2D, 0);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);


}
