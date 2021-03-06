#include <iostream>
#include <opencv2/opencv.hpp>
#include <GL/glew.h>
#include <GL/freeglut.h>

#include "MySlicGLSL.h"
#include "funUtils.h"
#define WIN_WIDTH 800
#define WIN_HEIGHT 600


#define NSPX 1000
#define WC 35
using namespace std;
using namespace cv;


void clickAction(int button, int state, int x, int y) {
	if (state == GLUT_DOWN) {

	}
	else {
		// on vient de relacher la souris
	}
}
void toDisplay(void) {
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	//gluPerspective(30.0, (GLfloat)WIN_WIDTH / (GLfloat)WIN_HEIGHT, 1, 1000.0);
	gluOrtho2D(-1, 1, -1, 1);
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	gluLookAt(0, 0, 0, 0, 0, 0, 0, 1, 0);

	glBegin(GL_QUADS);
	//glColor3ub(255, 0, 0);
	glTexCoord2f(0, 1);
	glVertex3f(-1, -1, 0);
	glTexCoord2f(0, 0);
	glVertex3f(-1, 1., 0);
	glTexCoord2f(1, 0);
	glVertex3f(1., 1., 0);
	glTexCoord2f(1, 1);
	glVertex3f(1., -1, 0);
	glEnd();

	//glDisable(GL_TEXTURE_2D);
	glutSwapBuffers();
}



void resizeWindow(GLsizei w, GLsizei h)
{
	glViewport(0, 0, w, h);
	toDisplay();
}

int main(int argc, char* argv[]) {

	//openGL context initialization (display, mouse event, ...)
	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_RGB | GLUT_DEPTH | GLUT_DOUBLE);
	glutInitWindowSize(WIN_WIDTH, WIN_HEIGHT);
	glutCreateWindow("output image opengl");

	//set functions that will loop (task setting)
	//glutDisplayFunc(toDisplay);
	//glutMouseFunc(clickAction);
	//glutReshapeFunc(resizeWindow);
	
	//allow runtime function load (like shader)
	glewInit();

	VideoCapture cap("D:/Videos/Tiger1/img/%04d.jpg");
	Mat frame;
	cap >> frame;
	MySlicGLSL slic(NSPX, WC);
	slic.Initialize(frame);
	
	size_t start, end;
	namedWindow("out");
	while (cap.read(frame))
	{
		start = getTickCount();
		slic.Segment(frame);
		end = getTickCount();
		cout <<"segment total "<< (end - start) / getTickFrequency() << endl;
		slic.displayBound(frame, Scalar(255, 0, 0));
		imshow("out", frame);
		 start= getTickCount();
		slic.gpu_DrawBound();
		end = getTickCount();
		//cout <<"display "<< (end - start) / getTickFrequency() << endl;
		waitKey();
	}
	



	waitKey();
	//start all the tasks
	//glutMainLoop();

	return 0;

}

