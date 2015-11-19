uniform sampler2D frameRGB;
uniform sampler2D isTaken;
uniform sampler2D labelsMat;


void main(void){

gl_FragColor = texture2D(frameRGB,gl_TexCoord[0].xy);

}