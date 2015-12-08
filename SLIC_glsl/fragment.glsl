uniform sampler2D frameRGB;

void main(void){

gl_FragColor = texture2D(frameRGB,gl_TexCoord[0].xy);

}