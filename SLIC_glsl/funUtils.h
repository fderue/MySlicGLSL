#pragma once

#include <GL/glew.h>
#include <GL/freeglut.h>
#include <opencv2/opencv.hpp>


/* opengl shortcut function */
GLuint createProgShader(GLenum typeShader, char * sourceFile);
void createTextureImage2D(GLenum textUnitEnum, int textUnitInt, GLuint & textureName, GLint internalFormat, int width, int height, GLenum format, GLenum type, const void * data, GLenum access);
void createPBO(GLuint &pboName, GLenum target, int sizeByte, const void * data, GLenum usage);
void getSSBO(GLuint ssboName, void* &out);
void getTexture(GLuint textureName, GLenum format, GLenum type, void* texDataOut);

/* display image 2D with opengl */
void displayTexture2D(GLuint tex, GLenum textUnitEnum);

/* determine width and height (integer value) of an initial spx from its expected size d. */
void getWlHl(int w, int h, int d, int & wl, int & hl);

/*
 *simple reading and writing for text files
 *www.lighthouse3d.com
 */
char* textFileRead(char * fn);
void displayShaderLog(GLuint obj, char * message);


/* divers */
int iDivUp(int a, int b);