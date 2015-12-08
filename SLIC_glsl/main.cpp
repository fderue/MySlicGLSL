#include <iostream>
#include <opencv2/opencv.hpp>
#include <GL/glew.h>
#include <GL/freeglut.h>
#include "MySlicGLSL.h"

#define SPX_SIZE 16 // superpixel size (limited)
#define WC 35 // compactness
#define VIDEO 0 // use video or single image

using namespace std;
using namespace cv;

int main(int argc, char* argv[]) {

	//openGL context initialization (display, mouse event, ...)
	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_RGB | GLUT_DEPTH | GLUT_DOUBLE);

	int win_w, win_h;
#if VIDEO
	VideoCapture cap("/media/derue/4A30A96F30A962A5/Videos/Tiger1/img/%04d.jpg");
	win_w = cap.get(CV_CAP_PROP_FRAME_WIDTH);
	win_h = cap.get(CV_CAP_PROP_FRAME_HEIGHT);
#else
	Mat im = imread("/media/derue/4A30A96F30A962A5/Videos/Tiger1/img/0001.jpg"); //input image RGB
	win_w = im.cols;
	win_h = im.rows;
#endif
	glutInitWindowSize(win_w, win_h);
	glutCreateWindow("output image opengl");

	//allow runtime function load (like shader)
	glewInit();

	// ====== Superpixel Segmentation ======

#if !VIDEO
	MySlicGLSL slicIm(SPX_SIZE, WC);
	slicIm.Initialize(im);
	slicIm.Segment(im);
	slicIm.gpu_DrawBound();
	glutMainLoop();
#else

    size_t start,end;
	Mat frame;
	cap >> frame;
	MySlicGLSL slic(SPX_SIZE, WC);
	slic.Initialize(frame);
	while (cap.read(frame))
	{
		start = getTickCount();
		slic.Segment(frame);
		end = getTickCount();
		cout <<"segment total time : "<< (end - start) / getTickFrequency() <<" ms"<<endl;
		slic.gpu_DrawBound();
		waitKey(30);
	}

#endif
	
	return 0;

}

