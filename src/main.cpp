#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

// http://www.pict uretopeople.org/color_converter.html

using namespace std;
using namespace cv;

void histogramGeneral(const Mat &sourceImage, Mat &histo, int channels, Scalar color, int point_val);
void flipImageBasic(const Mat &sourceImage, Mat &destinationImage);
void mouseCoordinatesExampleCallback(int event, int x, int y, int flags, void* param);
void plotHistograms();
void blackWhite();
void filterImage();
void getFilterRange();
void clickRgbToHsv();
void pixelRgbToHsv(int r, int g, int b, float *hsv);
void clickRgbToYiq();
void pixelRgbToYiq(int r, int g, int b, float *yiq);
void pixelYiqToRgb(float y, float i, float q, int *bgr);
void printPoint(char colormodel);
void imageRgbToHsv(const Mat &sourceImage, Mat &destinationImage);
void imageRgbToYiq(const Mat &sourceImage, Mat &destinationImage);
                                        // VARIABLES GLOBALES
                                        // Contador para refreshear la escala de los 3 histogramas
int counter_hist[3] = {0,0,0};
int countFilter=0;
                                        // La escala de los 3 histogramas
int h_scale[3] = {0,0,0};
Mat currentImageRGB, currentImageHSV, currentImageYIQ, displayedImage;
bool congelado = false;
                                        // r (RGB), h (HSV), y (YIQ)
char modelo =  'r';
bool filtrar = false;
                                        // Valor del pixel-clic en los 3 modelos
int bgr_point[3] = {-1, -1, -1};
float hsv_point[3] = {-1, -1, -1};
float yiq_point[3] = {-1, -1, -1};
int minFilter[3] = {267, 267, 267};
int maxFilter[3] = {-11, -11, -11};

int epsilon = 10;




/*< Main START >*/
int main(int argc, char *argv[])
{
  namedWindow("Image");
  setMouseCallback("Image", mouseCoordinatesExampleCallback);
  VideoCapture camera = VideoCapture(0); //Uncomment for real camera usage
  //VideoCapture camera("Videotest");     //Comment for real camera usage
  bool isCameraAvailable = camera.isOpened();
                                        // Limpia la terminal
  cout << "\033[2J\033[1;1H";
  cout << "Basic Show Image \t|\tUse 'x' or 'Esc' to terminate execution\n";
  bool defaultColor = true;
  while (true)
  {
                                        // Obtiene un nuevo Frame de la camara si "congelado" es falso
                                        // y grafica los 3 histogramas del modelo que esta actualmente
                                        // seleccionado.
    if (isCameraAvailable)
    {
      if(!congelado)
      {
        camera.read(currentImageRGB);
        imageRgbToHsv(currentImageRGB, currentImageHSV);
        imageRgbToYiq(currentImageRGB, currentImageYIQ);
      }
      plotHistograms();
      blackWhite();
      filterImage();
    }
    else
    {
      currentImageRGB = imread("PlaceholderImage.jpg", CV_LOAD_IMAGE_COLOR);
    }


    if (currentImageRGB.size().width <= 0 && currentImageRGB.size().height <= 0) {
      cout << "ERROR: Camera returned blank image, check connection\n";
      break;
    }

    if(defaultColor){
      displayedImage =currentImageRGB;
    }
    imshow("Image", displayedImage);


    char key = waitKey(1);
    switch(key)
    {
      case 'r':
        defaultColor = false;
        modelo = 'r';
        displayedImage =currentImageRGB;
        break;
      case 'h':
        defaultColor = false;
        modelo = 'h';
        displayedImage =currentImageHSV;
        break;
      case 'y':
        defaultColor = false;
        displayedImage =currentImageYIQ;
        modelo = 'y';
        break;
      case 'f':
      if(!filtrar)
      {
        cout << "reinicar filtro" << endl;
      	filtrar = true;
      	countFilter =0;
      	minFilter[2] = 257+epsilon;
      	minFilter[1] = 257+epsilon;
      	minFilter[0] = 257+epsilon;
      	maxFilter[2] = -1-epsilon;
      	maxFilter[1] = -1-epsilon;
      	maxFilter[0] = -1-epsilon;
      }
      else
        filtrar = false;
	break;
      case ' ':
        congelado = !congelado;
        break;

      case 'e':
        cout << "Indica el nuevo epsilon" << endl;
        cin >> epsilon;
        break;
    }
                                        // Si 'x' o ESC es presionada el programa termina
    if(key == 'x' || key == 27 )        // 27 = ESC
    {
      destroyAllWindows(); //Cierra todas las ventanas
      break;
    }

  }
}
/*< Main END >*/


/*< Plot Histograms START >*/
void plotHistograms()
{
  Mat histo_1, histo_2, histo_3;

  switch(modelo)
  {
    case 'r':
      histogramGeneral(currentImageRGB, histo_1, 0, Scalar( 255, 0  , 0  ), bgr_point[0]);
      histogramGeneral(currentImageRGB, histo_2, 1, Scalar( 0  , 255, 0  ), bgr_point[1]);
      histogramGeneral(currentImageRGB, histo_3, 2, Scalar( 0  , 0  , 255), bgr_point[2]);

      imshow("Histo_B", histo_1);
      imshow("Histo_G", histo_2);
      imshow("Histo_R", histo_3);

      destroyWindow("Histo_H");
      destroyWindow("Histo_S");
      destroyWindow("Histo_V");
      destroyWindow("Histo_Y");
      destroyWindow("Histo_I");
      destroyWindow("Histo_Q");
      break;

    case 'h':
      histogramGeneral(currentImageHSV, histo_1, 0, Scalar( 255  , 255  , 127  ), hsv_point[0]);
      histogramGeneral(currentImageHSV, histo_2, 1, Scalar( 127  , 255  , 127  ), hsv_point[1]);
      histogramGeneral(currentImageHSV, histo_3, 2, Scalar( 127  , 127  , 255  ), hsv_point[2]);

      imshow("Histo_H", histo_1);
      imshow("Histo_S", histo_2);
      imshow("Histo_V", histo_3);

      destroyWindow("Histo_B");
      destroyWindow("Histo_G");
      destroyWindow("Histo_R");
      destroyWindow("Histo_Y");
      destroyWindow("Histo_I");
      destroyWindow("Histo_Q");
      break;

    case 'y':
      histogramGeneral(currentImageYIQ, histo_1, 0, Scalar( 255  , 127  , 127  ), yiq_point[0]);
      histogramGeneral(currentImageYIQ, histo_2, 1, Scalar( 127  , 255  , 127  ), yiq_point[1]);
      histogramGeneral(currentImageYIQ, histo_3, 2, Scalar( 127  , 127  , 255  ), yiq_point[2]);

      imshow("Histo_Y", histo_1);
      imshow("Histo_I", histo_2);
      imshow("Histo_Q", histo_3);

      destroyWindow("Histo_B");
      destroyWindow("Histo_G");
      destroyWindow("Histo_R");
      destroyWindow("Histo_H");
      destroyWindow("Histo_S");
      destroyWindow("Histo_V");
      break;
  }

}
/*< Plot Histograms END >*/


/*< Flip image START >*/
void flipImageBasic(const Mat &sourceImage, Mat &destinationImage)
{
  if (destinationImage.empty())
    destinationImage = Mat(sourceImage.rows, sourceImage.cols, sourceImage.type());

  for (int y = 0; y < sourceImage.rows; ++y)
    for (int x = 0; x < sourceImage.cols / 2; ++x)
      for (int i = 0; i < sourceImage.channels(); ++i)
      {
        destinationImage.at<Vec3b>(y, x)[i] = sourceImage.at<Vec3b>(y, sourceImage.cols - 1 - x)[i];
        destinationImage.at<Vec3b>(y, sourceImage.cols - 1 - x)[i] = sourceImage.at<Vec3b>(y, x)[i];
      }
}
/*< Flip image END >*/

/*< Histograma START >*/
void histogramGeneral(const Mat &sourceImage, Mat &histo, int channel, Scalar color, int point_val)
{
  vector<Mat> image_channels;
  int offset = 100;
  int gradient_size = 10;
                                        // Separa la imagen en sus 3 canales
  split( sourceImage, image_channels );

                                        // Establece el numero de bins (rectangulos)
  int histSize = 256;

                                        // Establece el rango
  float range[] = { 0, 256 } ;
  const float* histRange = { range };

  bool uniform = true; bool accumulate = false;
                                        // Esta es la matriz en donde se guardara la
                                        // informacion del histograma.
  Mat hist;

                                        // Calcula el histograma y lo guarda en "hist"
  calcHist( &image_channels[channel], 1, 0, Mat(), hist, 1, &histSize, &histRange, uniform, accumulate );

                                        // El "offset" es para que las labels no se sobrepongan
                                        // en el histograma. El "gradient_size" es la altura del
                                        // gradiente de color que esta abajo del histograma.
  int hist_w = 512 + offset + 10; int hist_h = 400 + gradient_size;
  int bin_w = cvRound( (double) (hist_w - offset)/histSize );

                                        // Esta es la imagen del histograma con fondo negro
  Mat histImage( hist_h, hist_w, CV_8UC3, Scalar( 0,0,0) );


                                        // Obtiene el valor maximo de altura del histograma
  int max_value = 0;
  for(int i=0; i<histSize; i++)
    max_value = max(cvRound(hist.at<float>(i)), max_value);

                                        // El counter_hist previene que la escala vertical del histograma
                                        // cambie rapidamente
  if(counter_hist[channel] == 5)
  {
    h_scale[channel] = ( int( (max_value-1)/1000 ) + 2 ) * 1000;
    counter_hist[channel] = 0;
  }
  counter_hist[channel]++;

                                        // Dibuja los 5 valores del eje Y del histograma
  string text = to_string(h_scale[channel]);
  int fontFace = FONT_HERSHEY_SCRIPT_SIMPLEX;
  double fontScale = 0.5;
  int thickness = 1;
  int separacion = 80;

  Point textOrg(0,15);
  putText(histImage, text, textOrg, fontFace, fontScale, Scalar::all(255), thickness, 8);

  text = to_string(h_scale[channel]*4/5);
  textOrg.y = textOrg.y + separacion;
  putText(histImage, text, textOrg, fontFace, fontScale, Scalar::all(255), thickness, 8);

  text = to_string(h_scale[channel]*3/5);
  textOrg.y = textOrg.y + separacion;
  putText(histImage, text, textOrg, fontFace, fontScale, Scalar::all(255), thickness, 8);

  text = to_string(h_scale[channel]*2/5);
  textOrg.y = textOrg.y + separacion;
  putText(histImage, text, textOrg, fontFace, fontScale, Scalar::all(255), thickness, 8);

  text = to_string(h_scale[channel]/5);
  textOrg.y = textOrg.y + separacion;
  putText(histImage, text, textOrg, fontFace, fontScale, Scalar::all(255), thickness, 8);

                                        // Dibuja la linea del histograma y su gradiente
  for( int i = 1; i < histSize; i++ )
  {
    Scalar gradiente = color;
    

    if(modelo == 'r')
    {
      gradiente[channel] = i-1;
    }
    if(modelo == 'h')
    {
                                        // Convertir el color "gradiente" de HSV a BGR
      color[channel] = i-1;
      color[0] = color[0]/255 * 179;
      Mat temphsv( 1, 1, CV_8UC3, color );
      Mat tempbgr_mat;
      cvtColor(temphsv, tempbgr_mat, CV_HSV2BGR);
      gradiente[0] = tempbgr_mat.at<Vec3b>(0, 0)[0];
      gradiente[1] = tempbgr_mat.at<Vec3b>(0, 0)[1];
      gradiente[2] = tempbgr_mat.at<Vec3b>(0, 0)[2];
    }
    if(modelo == 'y')
    {
                                        // Convertir el color "gradiente" de YIQ a BGR
      int tempbgr[3];
      color[channel] = i-1;
      pixelYiqToRgb(color[0], color[1], color[2], tempbgr);
      gradiente[0] = tempbgr[0];
      gradiente[1] = tempbgr[1];
      gradiente[2] = tempbgr[2];
    }

    line(histImage, Point(offset + bin_w*(i-1), hist_h), Point(offset + bin_w*(i-1), hist_h - gradient_size), gradiente , 3, CV_FILLED);

    line( histImage, Point( bin_w*(i-1) + offset, hist_h - gradient_size - cvRound(hist.at<float>(i-1)*400/h_scale[channel])  ) ,
          Point( bin_w*(i) + offset, hist_h - gradient_size - cvRound(hist.at<float>(i)*400/h_scale[channel]) ),
          Scalar(255,255,255), 2, 8, 0  );
  }

                                        // Dibuja dos lineas que marca el umbral
  if(countFilter >= 2)
  {
    int low_boundary = minFilter[channel] - epsilon < 0 ? 0 : minFilter[channel] - epsilon;
    int high_boundary = maxFilter[channel] + epsilon > 255 ? 255 : maxFilter[channel] + epsilon;
    line(histImage, Point(offset + low_boundary*bin_w, 0), Point(offset + low_boundary*bin_w, hist_h - gradient_size), Scalar( 0, 255, 255) , 2, CV_FILLED);
    line(histImage, Point(offset + high_boundary*bin_w, 0), Point(offset + high_boundary*bin_w, hist_h - gradient_size), Scalar( 0, 255, 255) , 2, CV_FILLED);
  }
                                        // Dibuja una linea en el valor del pixel-clic
  if(point_val != -1)
  {
    line(histImage, Point(offset + point_val*bin_w, 0), Point(offset + point_val*bin_w, hist_h - gradient_size), Scalar( 255, 0, 255) , 2, CV_FILLED);
  }

  histo = histImage;
}
/*< Histograma END >*/


/*< Black and white and binary START >*/
void blackWhite()
{
                                        // Toma la currentImageRGB y la despliega en escala
                                        // de grises y binarizada.
}
/*< Black and white and binary END >*/


// referencia: https://docs.opencv.org/trunk/df/d9d/tutorial_py_colorspaces.html
/*< Filter Image START >*/
void filterImage(){
	if(maxFilter[0] !=-1 && countFilter >= 2){
		int lowRed = minFilter[2] - epsilon;
		int hiRed = maxFilter[2]  + epsilon;
		int lowGr =  minFilter[1] - epsilon;
		int hiGr= maxFilter[1]    + epsilon;
		int lowBl=  minFilter[0]  - epsilon;
		int hiBl= maxFilter[0]    + epsilon;
		Mat mask;
		Mat filter;

		inRange( displayedImage, Scalar(lowBl, lowGr, lowRed),Scalar (hiBl, hiGr, hiRed),mask);
		bitwise_and(currentImageRGB,currentImageRGB, filter,mask= mask);
		imshow("Filter",filter);

	}

}
/*< Filter Image END >*/


/*< RGB to HSV of click START >*/
// Reference https://www.tutorialspoint.com/c-program-to-change-rgb-color-model-to-hsv-color-model
// REsult verified with https://www.rapidtables.com/convert/color/rgb-to-hsv.html
void clickRgbToHsv()
{
  pixelRgbToHsv(bgr_point[2], bgr_point[1], bgr_point[0], hsv_point);
}
/*< RGB to HSV of click END >*/


/*< RGB to HSV of one pixel START >*/
// Reference https://www.tutorialspoint.com/c-program-to-change-rgb-color-model-to-hsv-color-model
// REsult verified with https://www.rapidtables.com/convert/color/rgb-to-hsv.html
void pixelRgbToHsv(int r, int g, int b, float *hsv)
{
  double rgb_point_double[3] = {-1, -1, -1};
  double minMaxColors[2] ={0,0};
  double diff;
  rgb_point_double[0] = r/255.0;
  rgb_point_double[1] = g/255.0;
  rgb_point_double[2] = b/255.0;

  //Move this to a function
  minMaxColors[0] = min(rgb_point_double[0],min(rgb_point_double[1],rgb_point_double[2]));
  minMaxColors[1] = max(rgb_point_double[0],max(rgb_point_double[1],rgb_point_double[2]));
  //Move this to a function

  //minMax(rgb_point_double, minMaxColors);
  diff = minMaxColors[1]-minMaxColors[0];
  hsv[2]  = minMaxColors[1] * 255;
  if (minMaxColors[1] == 0){
    hsv[1] = 0;
  }
  else{
    hsv[1] = (diff / minMaxColors[1]) * 255;
  }
  if(minMaxColors[0] == minMaxColors[1]){
    hsv[0] = 0;
  }else if(minMaxColors[1] == rgb_point_double[0] ){
    hsv[0]  = fmod((60 * ((rgb_point_double[1]  - rgb_point_double[2] ) / diff) + 360), 360.0);
  }else if(minMaxColors[1] == rgb_point_double[1] ){
    hsv[0]  = fmod((60 * ((rgb_point_double[2]  - rgb_point_double[0] ) / diff) + 120), 360.0);
  }else if(minMaxColors[1]== rgb_point_double[2] ){
    hsv[0]  = fmod((60 * ((rgb_point_double[0]  - rgb_point_double[1] ) / diff) + 240), 360.0);
  }
  hsv[0] = hsv[0] *255/360.0;
}
/*< RGB to HSV of one pixel END >*/


/*< RGB to YIQ of click START >*/
void clickRgbToYiq()
{
  pixelRgbToYiq(bgr_point[2], bgr_point[1], bgr_point[0], yiq_point);
}
/*< RGB to YIQ of click END >*/

/*< RGB to YIQ of one pixel START >*/
void pixelRgbToYiq(int r, int g, int b, float *yiq)
{
  yiq[0] = (0.299*r + 0.587*g + 0.114*b);
  yiq[1] = ((0.596*r - 0.275*g - 0.321*b)/255 +0.596)/1.192 *255;
  yiq[2] = ((0.212*r - 0.523*g + 0.311*b)/255 +.523)/1.046 *255;
}
/*< RGB to YIQ of one pixel END >*/


/*< RGB to YIQ of one pixel START >*/
void pixelYiqToRgb(float y, float i, float q, int *bgr)
{
  y = y/255;
  i = (i/255)*1.192 -0.596;
  q = (q/255)*1.046 -0.523;
  bgr[2] = ( 1*y + 0.956*i + 0.621*q) * 255;
  bgr[1] = ((1*y - 0.272*i - 0.647*q)) *255;
  bgr[0] = ((1*y - 1.106*i + 1.703*q)) *255;
}
/*< RGB to YIQ of one pixel END >*/


/*< RGB to HSV image START >*/
void imageRgbToHsv(const Mat &sourceImage, Mat &destinationImage)
{
  cvtColor(sourceImage, destinationImage, CV_BGR2HSV);
  Mat channels[3];
  split(destinationImage, channels);
  channels[0] = channels[0]/180*255;

  merge(channels, 3, destinationImage);
}
/*< RGB to HSV image END >*/


/*< RGB to YIQ image START >*/
void imageRgbToYiq(const Mat &sourceImage, Mat &destinationImage)
{
  if (destinationImage.empty())
    destinationImage = Mat(sourceImage.rows, sourceImage.cols, sourceImage.type());

  for (int y = 0; y < sourceImage.rows; ++y)
    for (int x = 0; x < sourceImage.cols; ++x)
    {
        float temp[3];
        pixelRgbToYiq(sourceImage.at<Vec3b>(y, x)[2],
                      sourceImage.at<Vec3b>(y, x)[1],
                      sourceImage.at<Vec3b>(y, x)[0],
                      temp);
        destinationImage.at<Vec3b>(y, x)[0] = temp[0];
        destinationImage.at<Vec3b>(y, x)[1] = temp[1];
        destinationImage.at<Vec3b>(y, x)[2] = temp[2];
    }
}
/*< RGB to YIQ image END >*/

/* max and min range for filters START*/
void getFilterRange(){
	switch(modelo){
		case 'r':
			/*if(bgr_point[0] < minFilter[0]){
				minFilter[0] = bgr_point[0];
        minFilter[1] = bgr_point[1];
        minFilter[2] = bgr_point[2];
			}
		  if(bgr_point[0] > maxFilter[0]){
				maxFilter[0] = bgr_point[0];
        maxFilter[1] = bgr_point[1];
        maxFilter[2] = bgr_point[2];
			}*/
		      if(bgr_point[0] < minFilter[0] - epsilon){
			       minFilter[0] = bgr_point[0];
		      }
		      if(bgr_point[0] > maxFilter[0] + epsilon){
			       maxFilter[0] = bgr_point[0];
		      }
		      if(bgr_point[1] < minFilter[1] - epsilon){
			       minFilter[1] = bgr_point[1];
		      }
		      if(bgr_point[1] > maxFilter[1] + epsilon){
			       maxFilter[1] = bgr_point[1];
		      }
		      if(bgr_point[2] < minFilter[2] - epsilon){
			       minFilter[2] = bgr_point[2];
		      }
		      if(bgr_point[2] > maxFilter[2] + epsilon){
			       maxFilter[2] = bgr_point[2];
		      }
		break;
		case 'h':
			/*if(hsv_point[0] < minFilter[0]){
				minFilter[2] = hsv_point[2];
				minFilter[1] = hsv_point[1];
				minFilter[0] = hsv_point[0];
			}
		    	if(hsv_point[0] > maxFilter[0]){
				maxFilter[2] = hsv_point[2];
				maxFilter[1] = hsv_point[1];
				maxFilter[0] = hsv_point[0];
			}*/

		      if(hsv_point[0] < minFilter[0] - epsilon){
			       minFilter[0] = hsv_point[0];
		      }
		      if(hsv_point[0] > maxFilter[0] + epsilon){
			       maxFilter[0] = hsv_point[0];
		      }
		      if(hsv_point[1] < minFilter[1] - epsilon){
			       minFilter[1] = hsv_point[1];
		      }
		      if(hsv_point[1] > maxFilter[1] + epsilon){
			       maxFilter[1] = hsv_point[1];
		      }
		      if(hsv_point[2] < minFilter[2] - epsilon){
			       minFilter[2] = hsv_point[2];
		      }
		      if(hsv_point[2] > maxFilter[2] + epsilon){
			       maxFilter[2] = hsv_point[2];
		      }
		break;
		case 'y':
			/*if(yiq_point[0] < minFilter[0]){
				minFilter[2] = yiq_point[2];
				minFilter[1] = yiq_point[1];
				minFilter[0] = yiq_point[0];
			}
		    	if(yiq_point[0] > maxFilter[0]){
				maxFilter[2] = yiq_point[2];
				maxFilter[1] = yiq_point[1];
				maxFilter[0] = yiq_point[0];
			}*/
		      if(yiq_point[0] < minFilter[0] - epsilon){
			       minFilter[0] = yiq_point[0];
		      }
		      if(yiq_point[0] > maxFilter[0] + epsilon){
			       maxFilter[0] = yiq_point[0];
		      }
		      if(yiq_point[1] < minFilter[1] - epsilon){
			       minFilter[1] = yiq_point[1];
		      }
		      if(yiq_point[1] > maxFilter[1] + epsilon){
			       maxFilter[1] = yiq_point[1];
		      }
		      if(yiq_point[2] < minFilter[2] - epsilon){
			       minFilter[2] = yiq_point[2];
		      }
		      if(yiq_point[2] > maxFilter[2] + epsilon){
			       maxFilter[2] = yiq_point[2];
		      }
		break;
	}
		countFilter+=1;
		cout<< countFilter <<endl;
		if(countFilter >2){
			//filtrar=false;
			cout<< "EL umbrall minimo es " << minFilter[0] << " " << minFilter[1] << " " <<minFilter[2] <<endl;
			cout<< "EL umbrall maximo es " << maxFilter[0] << " " << maxFilter[1] << " " <<maxFilter[2] <<endl;
		}
}
/* max and min range for filters END*/

void printPoint(char colormodel){
  switch(colormodel){
    case 'r':
      cout <<"\t\t RGB " << bgr_point[2] << ", " << bgr_point[1] <<", " << bgr_point[0] << endl;
      break;
    case 'h':
      cout << "\t\t HSV " << hsv_point[0] << ", " << hsv_point[1] <<", " << hsv_point[2] << endl;
      break;
    case 'y':
    cout <<"\t\t YIQ " << yiq_point[0] << ", " << yiq_point[1] <<", " << yiq_point[2] << endl;
      break;
  }

}

/*< Mouse callback START >*/
void mouseCoordinatesExampleCallback(int event, int x, int y, int flags, void* param)
{
    switch (event)
    {
        case CV_EVENT_LBUTTONDOWN:
            bgr_point[0] = int(currentImageRGB.at<Vec3b>(y, x)[0]);
            bgr_point[1] = int(currentImageRGB.at<Vec3b>(y, x)[1]);
            bgr_point[2] = int(currentImageRGB.at<Vec3b>(y, x)[2]);

            cout << "  Mouse X, Y: " << x << ", " << y << endl;
            printPoint('r');
            clickRgbToHsv();
            printPoint('h');
            clickRgbToYiq();
            printPoint('y');

	if(filtrar){
	    getFilterRange();
	}
            break;
        case CV_EVENT_MOUSEMOVE:
            break;
        case CV_EVENT_LBUTTONUP:
            break;
    }
}
/*< Mouse callback END >*/
