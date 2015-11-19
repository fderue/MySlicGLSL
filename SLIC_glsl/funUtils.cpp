#include "funUtils.h"

using namespace std;
using namespace cv;
/*
format : 
GL_RED_INTEGER -> int 
GL_RED -> float
*/
void getTexture(GLuint textureName, GLenum format, GLenum type, void* texDataOut) {
	//glActiveTexture(GL_TEXTURE1);
	glBindTexture(GL_TEXTURE_2D, textureName);
	glGetTexImage(GL_TEXTURE_2D, 0, format, type, texDataOut);

}

void getSSBO(GLuint ssboName,void* &out)
{
	glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssboName);
	out = glMapBuffer(GL_SHADER_STORAGE_BUFFER, GL_READ_ONLY);
	glUnmapBuffer(GL_SHADER_STORAGE_BUFFER);

}

void displayShaderLog(GLuint obj, string message)
{
	int infologLength = 0;
	int charsWritten = 0;
	char *infoLog;

	// afficher le message d'en-tête
	cout << message << endl;

	// afficher le message d'erreur, le cas échéant
	glGetShaderiv(obj, GL_INFO_LOG_LENGTH, &infologLength);

	if (infologLength > 1)
	{
		infoLog = (char *)malloc(infologLength);
		glGetShaderInfoLog(obj, infologLength, &charsWritten, infoLog);
		printf("%s\n", infoLog);
		free(infoLog);
	}
	else
	{
		printf("Aucune erreur :-)\n\n");
	}
}

void displayTexture2D(GLuint tex) {
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	gluOrtho2D(-1, 1, -1, 1);
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	gluLookAt(0, 0, 1, 0, 0, 0, 0, 1, 0);

	glBindTexture(GL_TEXTURE_2D, tex);
	
	glBegin(GL_QUADS);
	glTexCoord2f(0, 1);
	glVertex3f(-1, -1, 0);
	glTexCoord2f(0, 0);
	glVertex3f(-1, 1., 0);
	glTexCoord2f(1, 0);
	glVertex3f(1., 1., 0);
	glTexCoord2f(1, 1);
	glVertex3f(1., -1, 0);
	glEnd();

	glutSwapBuffers();

}

int iDivUp(int a, int b) {
	return (a%b) ? a / b+1 : a / b;
}

char *textFileRead(char *fn) {
	FILE *fp;
	char *content = NULL;
	size_t count = 0;
	if (fn != NULL) {
		//fp = fopen(fn, "rt");
		fopen_s(&fp, fn, "rt");
		if (fp != NULL) {

			fseek(fp, 0, SEEK_END);
			count = ftell(fp);
			rewind(fp);

			if (count > 0) {
				content = (char *)malloc(sizeof(char) * (count + 1));
				count = fread(content, sizeof(char), count, fp);
				content[count] = '\0';
			}
			fclose(fp);
		}
	}
	return content;
}


GLuint createProgShader(GLenum typeShader, char* sourceFile) {
	
	GLuint shader = glCreateShader(typeShader);
	char* cs = textFileRead(sourceFile);
	const char* cs_ptr = cs;
	glShaderSource(shader, 1, &cs_ptr, NULL);
	free(cs);
	glCompileShader(shader);

	string message = "error compilation " + string(sourceFile);
	displayShaderLog(shader, message);

	GLuint prog = glCreateProgram();
	glAttachShader(prog, shader);
	glLinkProgram(prog);
	return prog;

}