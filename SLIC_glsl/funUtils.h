#pragma once

#include <GL/glew.h>
#include <GL/freeglut.h>
#include <opencv2/opencv.hpp>


#define WIN_WIDTH 800
#define WIN_HEIGHT 600

void displayShaderLog(GLuint obj, char * message);

void displayTexture2D(GLuint tex);

 int iDivUp(int a, int b);


void getSSBO(GLuint ssboName, void* &out);
void getTexture(GLuint textureName, GLenum format, GLenum type, void* texDataOut);

char* textFileRead(char * fn);

GLuint createProgShader(GLenum typeShader, char * sourceFile);
