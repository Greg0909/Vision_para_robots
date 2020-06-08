#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <queue>
#include <math.h>

#define PI 3.14159265
#define espacio 5
// http://www.pict uretopeople.org/color_converter.html

using namespace std;
using namespace cv;

void histogramGeneral(const Mat &sourceImage, Mat &histo, int channels, Scalar color, int point_val);
void flipImageBasic(const Mat &sourceImage, Mat &destinationImage);
void mouseCoordinatesExampleCallback(int event, int x, int y, int flags, void* param);
void plotHistograms();
void blackWhite();
void filterImage(Mat &binarizada);
void getFilterRange();
void clickRgbToHsv();
void pixelRgbToHsv(int r, int g, int b, float *hsv);
void clickRgbToYiq();
void pixelRgbToYiq(int r, int g, int b, float *yiq);
void pixelYiqToRgb(float y, float i, float q, int *bgr);
void printPoint(char colormodel);
void imageRgbToHsv(const Mat &sourceImage, Mat &destinationImage);
void imageRgbToYiq(const Mat &sourceImage, Mat &destinationImage);
void segmentacion(const Mat &sourceImage, Mat &destinationImage);
void dilateImage(const Mat &sourceImage, Mat &destinationImage);
void erodeImage(const Mat &sourceImage, Mat &destinationImage);
int checkpoint(float h, float k, float x, float y, float a, float b, float angle) ;
void detectObjects(Mat &sourceImage);
void drawGraph(float objetos[][5]);
void PrepareParking();
void parkingLotSpace(Mat &theMask);
void Enhancement(const Mat &sourceImage, Mat &destinationImage);
void SobelFilter(const Mat &sourceImage, Mat &destinationImage);
void camino(const Mat &sourceImage, Mat &destinationImage);
void navegacion(Mat &destinationImage, Point inicio, Point final);

                                        // VARIABLES GLOBALES
                                        // Contador para refreshear la escala de los 3 histogramas
int counter_hist[3] = {0,0,0};
int countFilter=0;
bool in = false;

                                        // La escala de los 3 histogramas
int h_scale[3] = {0,0,0};
Mat currentImageRGB, currentImageHSV, currentImageYIQ, displayedImage;
Mat parkingLot,displayingParking, caminosPRM;
Mat kernel;

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
int umbral_bw = 100;

queue<Point> exploracion;
vector< vector<float> > momentosOrdinarios;
vector< vector<float> > momentosCentralizados;
vector< vector<float> > momentosNormalizados;
vector< float > fi1;
vector< float > fi2;

Mat grad_x, grad_y;

vector< vector<short> > conexiones;
vector<Point> estacas;
int xEstacas;
Point puntoFinal(276, 30);
bool update = true;

/*< Main START >*/
int main(int argc, char *argv[])
{
  int id1 = 1, id2 = 2;
  namedWindow("Image");
  namedWindow("Parking");
  setMouseCallback("Image", mouseCoordinatesExampleCallback, &id1);
  setMouseCallback("Parking", mouseCoordinatesExampleCallback, &id2);
  VideoCapture camera = VideoCapture(0); //Uncomment for real camera usage
  //VideoCapture camera("Videotest");     //Comment for real camera usage

  PrepareParking();
  Mat parkingLotMask;
  parkingLotSpace(parkingLotMask);

  bool isCameraAvailable = camera.isOpened();
  camera.read(currentImageRGB);
  Mat mascara( currentImageRGB.rows, currentImageRGB.cols, CV_8UC3, Scalar( 0) );
  Mat segmentedImage( currentImageRGB.rows, currentImageRGB.cols, CV_8UC3, Scalar( 0) );
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
      filterImage(mascara);
      segmentedImage = Mat( currentImageRGB.rows, currentImageRGB.cols, CV_8UC3, Scalar( 0) );
      
      dilateImage(mascara, mascara);
      dilateImage(mascara, mascara);
      dilateImage(mascara, mascara);
      erodeImage(mascara,mascara);
      erodeImage(mascara,mascara);
      erodeImage(mascara,mascara);
      segmentacion(mascara, segmentedImage);
      detectObjects(segmentedImage);

      if(update)
      {

        // Dibujar cuadrito blanco

        caminosPRM = Mat( parkingLotMask.rows, parkingLotMask.cols, CV_8UC3, Scalar( 0) );
        camino(parkingLotMask, caminosPRM);
        navegacion(caminosPRM, Point(276, 48), puntoFinal);
        update = false;
      }
      
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
    imshow("ParkingFiltered", displayingParking);
    imshow("CaminosPRM", caminosPRM);

    imshow("Image segmentada", segmentedImage);


    char key = waitKey(1);
    switch(key)
    {
        case 'B':
        modelo = 'B';
        break;
        case 'R':
        modelo = 'R';
        break;
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
      case 'b':
        defaultColor = false;
        modelo = 'b';
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
      case 'u':
        cout << "Indica el nuevo umbral para blanco y negro, debe de estar en el rango [0 - 255]" << endl;
        cin >> umbral_bw;
        umbral_bw = umbral_bw > 255 ? 255 : umbral_bw;
        umbral_bw = umbral_bw < 0 ? 0 : umbral_bw;
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

void Enhancement(const Mat &sourceImage, Mat &destinationImage){
   Mat kernel = (Mat_<double>(3, 3) << 0, -1, 0,
              -1, 5, -1,
              0, -1, 0);  //Matriox obtained from source
  filter2D(sourceImage, destinationImage, -1, kernel);
}

void SobelFilter(const Mat &sourceImage, Mat &destinationImage){
   Sobel( sourceImage, grad_x, CV_16S, 1, 0, 3, 1, 0, BORDER_DEFAULT );
        convertScaleAbs( grad_x, grad_x );

        /// Gradient Y
        //Scharr( src_gray, grad_y, ddepth, 0, 1, scale, delta, BORDER_DEFAULT );
        Sobel( sourceImage, grad_y, CV_16S, 0, 1, 3, 1, 0, BORDER_DEFAULT );
        convertScaleAbs( grad_y, grad_y);
        addWeighted( grad_x, 0.5, grad_y, 0.5, 0, destinationImage);
}



void PrepareParking(){
 parkingLot = imread("Parking4.jpeg", CV_LOAD_IMAGE_COLOR);
 imshow("Parking", parkingLot);
  displayingParking = parkingLot;
  SobelFilter(displayingParking,displayingParking);
  GaussianBlur(displayingParking, displayingParking, Size(3, 3) , 0);

}



void parkingLotSpace(Mat &theMask){
  int limit = 10;
		int lowR = 25;
		int hiR = 255;
		int lowG = 25 ;
		int hiG= 255;
		int lowB =25;
		int hiB= 255;
		Mat mask;
		Mat filter;
	medianBlur(parkingLot, parkingLot,9);
	inRange( parkingLot, Scalar(lowB, lowG, lowR),Scalar (hiB, hiG, hiR),mask);
	kernel = getStructuringElement(MORPH_RECT, Size(5, 5));
	dilate(mask, mask, kernel);
	kernel = getStructuringElement(MORPH_RECT, Size(15, 15));
	erode(mask, mask,kernel);


  //if(modelo == 'B'){
   /*Mat channels[3];	
    Mat black_white( displayingParking.rows, displayingParking.cols, CV_8UC3, Scalar( 0) );
    Mat mask;

    split(parkingLot, channels);
    black_white = channels[0]*0.1 + channels[1]*0.3 + channels[2]*0.6;
    inRange( black_white, Scalar(110),Scalar (256),mask);
    filterImage(mask);*/

    /*if(in == false){
      medianBlur(mask, mask, 9);
      blur(mask, mask, Size(3, 3));

      kernel = getStructuringElement(MORPH_RECT, Size(3, 3));
      erodeImage(mask, mask);
      medianBlur(mask, mask, 9);

      dilate(mask, mask, kernel);
            kernel = getStructuringElement(MORPH_RECT, Size(5, 5));

      morphologyEx(mask, mask, MORPH_CLOSE, kernel);

        
      in = true;
    }*/
  //}
    imshow("ParkingBinarizado", mask);
    theMask = mask;
}

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
    case 'b':
      destroyWindow("Histo_B");
      destroyWindow("Histo_G");
      destroyWindow("Histo_R");
      destroyWindow("Histo_H");
      destroyWindow("Histo_S");
      destroyWindow("Histo_V");
      destroyWindow("Histo_Y");
      destroyWindow("Histo_I");
      destroyWindow("Histo_Q");
      destroyWindow("Filter");
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
  if(modelo == 'b')
  {
    Mat channels[3];
    Mat black_white( currentImageRGB.rows, currentImageRGB.cols, CV_8UC3, Scalar( 0) );
    Mat mask;

    split(currentImageRGB, channels);
    black_white = channels[0]*0.1 + channels[1]*0.3 + channels[2]*0.6;
    inRange( black_white, Scalar(umbral_bw),Scalar (256),mask);
    displayedImage = black_white;
    imshow("Binarizada", mask);
  }  
else{
      destroyWindow("Binarizada");
    }
}
/*< Black and white and binary END >*/


// referencia: https://docs.opencv.org/trunk/df/d9d/tutorial_py_colorspaces.html
/*< Filter Image START >*/
void filterImage(Mat &binarizada){
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
    binarizada = mask;
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


/*< YIQ to RGB of one pixel START >*/
void pixelYiqToRgb(float y, float i, float q, int *bgr)
{
  y = y/255;
  i = (i/255)*1.192 -0.596;
  q = (q/255)*1.046 -0.523;
  bgr[2] = ( 1*y + 0.956*i + 0.621*q) * 255;
  bgr[1] = ((1*y - 0.272*i - 0.647*q)) *255;
  bgr[0] = ((1*y - 1.106*i + 1.703*q)) *255;
}
/*< YIQ to RGB of one pixel END >*/


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
   int * windowID = (int*)param;
    switch (event)
    {
        case CV_EVENT_LBUTTONDOWN:
            

            cout << "  Mouse X, Y: " << x << ", " << y << endl;
            if(*windowID==1)
            {
              bgr_point[0] = int(currentImageRGB.at<Vec3b>(y, x)[0]);
              bgr_point[1] = int(currentImageRGB.at<Vec3b>(y, x)[1]);
              bgr_point[2] = int(currentImageRGB.at<Vec3b>(y, x)[2]);

              printPoint('r');
              clickRgbToHsv();
              printPoint('h');
              clickRgbToYiq();
              printPoint('y');

            	if(filtrar){
            	    getFilterRange();
            	}
            }
            else{
              puntoFinal.x = x;
              puntoFinal.y = y;
              update = true;
		cout<< int(parkingLot.at<Vec3b>(y, x)[0]) <<endl;
		cout<< int(parkingLot.at<Vec3b>(y, x)[2]) <<endl;
		cout<< int(parkingLot.at<Vec3b>(y, x)[3]) <<endl;
            }
            
            break;
        case CV_EVENT_MOUSEMOVE:
            break;
        case CV_EVENT_LBUTTONUP:
            break;
    }
}
/*< Mouse callback END >*/













Point get_seed(const Mat &sourceImage)
{
  Point seed;
  seed.y = rand() % sourceImage.rows;
  seed.x = rand() % sourceImage.cols;
  return seed;
}

bool check(const Mat &sourceImage, Mat &destinationImage, int x, int y)
{
  Vec3b cero(0,0,0);
  return sourceImage.at<unsigned char>(y, x) == 255 && destinationImage.at<Vec3b>(y, x) == cero;
}

void dilateImage(const Mat &sourceImage, Mat &destinationImage){
  kernel = (Mat_<double>(3, 3) << 1, 1, 1,
                1, 1, 1,
                1, 1, 1);  //Matriox obtained from source
  dilate(sourceImage, destinationImage, kernel, Point(-1,-1));
}

void erodeImage(const Mat &sourceImage, Mat &destinationImage){
 kernel = (Mat_<double>(3, 3) << 1, 1, 1,
                1, 1, 1,
                1, 1, 1);  //Matriox obtained from source
  erode(sourceImage, destinationImage, kernel, Point(-1,-1));
}


void segmentacion(const Mat &sourceImage, Mat &destinationImage)
{
  fi1.clear();
  fi2.clear();
  momentosOrdinarios.clear();
  momentosCentralizados.clear();
  momentosNormalizados.clear();

  srand( (unsigned)time(NULL) );

  Vec3b colores[10] = { Vec3b(255,0,0), Vec3b(0,255,0), Vec3b(0,0,255),
                       Vec3b(255,100,100), Vec3b(100,255,100), Vec3b(100,100,255),
                       Vec3b(255,200,0), Vec3b(0,255,200), Vec3b(200,0,255), Vec3b(255,255,255)};

  int color = 0, intentos = 0;
  

  while(intentos <200)
  {
    vector<float> momentosOrdinariosTemp(6,0); 
    Point pixel = get_seed(sourceImage);

    if(check(sourceImage, destinationImage, pixel.x, pixel.y))
    {

      //intentos = 0;
      exploracion.push(pixel);
      destinationImage.at<Vec3b>(pixel.y, pixel.x) = colores[color];
      momentosOrdinariosTemp[0] = 1; //m00
      momentosOrdinariosTemp[1] = pixel.x; //m10
      momentosOrdinariosTemp[2] = pixel.y; //m01
      momentosOrdinariosTemp[3] = pixel.x * pixel.x; //m20
      momentosOrdinariosTemp[4] = pixel.y * pixel.y; //m02
      momentosOrdinariosTemp[5] = pixel.x * pixel.y; //m11
      while(exploracion.size() > 0)
      {
        pixel = exploracion.front();
        exploracion.pop();

        //Norte
        pixel.y -=1;
        if(pixel.y >= 0 && check(sourceImage, destinationImage, pixel.x, pixel.y))
        {
          exploracion.push(pixel);
          destinationImage.at<Vec3b>(pixel.y, pixel.x) = colores[color];
          momentosOrdinariosTemp[0] += 1; //m00
          momentosOrdinariosTemp[1] += pixel.x; //m10
          momentosOrdinariosTemp[2] += pixel.y; //m01
          momentosOrdinariosTemp[3] += pixel.x * pixel.x; //m20
          momentosOrdinariosTemp[4] += pixel.y * pixel.y; //m02
          momentosOrdinariosTemp[5] += pixel.x * pixel.y; //m11
        }
        pixel.y +=1;

        //Oeste
        pixel.x -=1;
        if(pixel.x >= 0 && check(sourceImage, destinationImage, pixel.x, pixel.y))
        {
          exploracion.push(pixel);
          destinationImage.at<Vec3b>(pixel.y, pixel.x) = colores[color];
          momentosOrdinariosTemp[0] += 1; //m00
          momentosOrdinariosTemp[1] += pixel.x; //m10
          momentosOrdinariosTemp[2] += pixel.y; //m01
          momentosOrdinariosTemp[3] += pixel.x * pixel.x; //m20
          momentosOrdinariosTemp[4] += pixel.y * pixel.y; //m02
          momentosOrdinariosTemp[5] += pixel.x * pixel.y; //m11
        }
        pixel.x +=1;

        //Sur
        pixel.y +=1;
        if(pixel.y < sourceImage.rows && check(sourceImage, destinationImage, pixel.x, pixel.y))
        {
          exploracion.push(pixel);
          destinationImage.at<Vec3b>(pixel.y, pixel.x) = colores[color];
          momentosOrdinariosTemp[0] += 1; //m00
          momentosOrdinariosTemp[1] += pixel.x; //m10
          momentosOrdinariosTemp[2] += pixel.y; //m01
          momentosOrdinariosTemp[3] += pixel.x * pixel.x; //m20
          momentosOrdinariosTemp[4] += pixel.y * pixel.y; //m02
          momentosOrdinariosTemp[5] += pixel.x * pixel.y; //m11
        }
        pixel.y -=1;

        //Este
        pixel.x +=1;
        if(pixel.x < sourceImage.cols && check(sourceImage, destinationImage, pixel.x, pixel.y))
        {
          exploracion.push(pixel);
          destinationImage.at<Vec3b>(pixel.y, pixel.x) = colores[color];
          momentosOrdinariosTemp[0] += 1; //m00
          momentosOrdinariosTemp[1] += pixel.x; //m10
          momentosOrdinariosTemp[2] += pixel.y; //m01
          momentosOrdinariosTemp[3] += pixel.x * pixel.x; //m20
          momentosOrdinariosTemp[4] += pixel.y * pixel.y; //m02
          momentosOrdinariosTemp[5] += pixel.x * pixel.y; //m11
        }
        pixel.x -=1;
      }
      color++;
      momentosOrdinarios.push_back(momentosOrdinariosTemp);
    }
      
    intentos++;
    if(color == 4)
      break;
  }

//  cout << "Shape x " << sourceImage.cols << " y " << sourceImage.rows << endl;

 /* if(sourceImage.channels() == 1)
  {
  Moments momentss;
  momentss = moments(sourceImage, true);
  cout << "Opencv "
          << "\tmiu20: " << momentss.mu20
          << "\tmiu02: " << momentss.mu02
          << "\tmiu11: " << momentss.mu11 << endl;
  cout << "Opencv "
          << "\tnu20: " << momentss.nu20
          << "\tnu02: " << momentss.nu02
          << "\tnu11: " << momentss.nu11 << endl;
  cout << "Opencv "
          << "\tm00: " << momentss.m00
          << "\tm10: " << momentss.m10
          << "\tm01: " << momentss.m01
          << "\tm20: " << momentss.m20
          << "\tm02: " << momentss.m02
          << "\tm11: " << momentss.m11 << endl;
  double huMoments[7];
  HuMoments(momentss, huMoments);
  cout << "OpenCv " << huMoments[0] << " " << huMoments[1] << endl;
  }*/

  //cout << "Se encontraron " << color << " objetos"<< endl;
  for(int i = 0; i<momentosOrdinarios.size(); i++)
  {
    float promedioX = momentosOrdinarios[i][1] / momentosOrdinarios[i][0];
    float promedioY = momentosOrdinarios[i][2] / momentosOrdinarios[i][0];


    line(destinationImage, Point(promedioX-5, promedioY), Point(promedioX+5, promedioY), Scalar(255,255,255), 2, CV_FILLED );
    line(destinationImage, Point(promedioX, promedioY-5), Point(promedioX, promedioY+5), Scalar(255,255,255), 2, CV_FILLED );


    vector<float> momentosCentralizadosTemp;
    // miu20 = m20 - Px^2 * m00
    momentosCentralizadosTemp.push_back( momentosOrdinarios[i][3] -  promedioX*promedioX*momentosOrdinarios[i][0]);

    // miu02 = m02 - Py^2 * m00
    momentosCentralizadosTemp.push_back( momentosOrdinarios[i][4] -  promedioY*promedioY*momentosOrdinarios[i][0]); 
    // miu11 = m11 - PxPym00
    momentosCentralizadosTemp.push_back( momentosOrdinarios[i][5] 
                                          -  promedioY * promedioX * momentosOrdinarios[i][0]); 
    momentosCentralizados.push_back(momentosCentralizadosTemp);

    vector<float> momentosNormalizadosTemp;
    // n20 = miu20/(m00^(2))
    momentosNormalizadosTemp.push_back(momentosCentralizados[i][0]/(momentosOrdinarios[i][0]*momentosOrdinarios[i][0]) );
    // n02 = miu02/(m00^(2))
    momentosNormalizadosTemp.push_back(momentosCentralizados[i][1]/(momentosOrdinarios[i][0]*momentosOrdinarios[i][0]) );
    // n11 = miu11/(m00^(2))
    momentosNormalizadosTemp.push_back(momentosCentralizados[i][2]/(momentosOrdinarios[i][0]*momentosOrdinarios[i][0]) );
    momentosNormalizados.push_back(momentosNormalizadosTemp);

    /*cout << "Objeto " << i
          << "\tmiu20: " << momentosCentralizados[i][0]
          << "\tmiu02: " << momentosCentralizados[i][1]
          << "\tmiu11: " << momentosCentralizados[i][2] << endl;
    cout << "Objeto " << i
          << "\tnu20: " << momentosNormalizados[i][0]
          << "\tnu02: " << momentosNormalizados[i][1]
          << "\tnu11: " << momentosNormalizados[i][2] << endl;
    cout << "Objeto " << i
          << "\tm00: " << momentosOrdinarios[i][0]
          << "\tm10: " << momentosOrdinarios[i][1]
          << "\tm01: " << momentosOrdinarios[i][2]
          << "\tm20: " << momentosOrdinarios[i][3]
          << "\tm02: " << momentosOrdinarios[i][4]
          << "\tm11: " << momentosOrdinarios[i][5] << endl;*/

    //fi1 = n20 + n02
    fi1.push_back(momentosNormalizados[i][0] + momentosNormalizados[i][1]);
    //fi2 = (n20 - n02)^2 + 4(n11)^2
    fi2.push_back( (momentosNormalizados[i][0] - momentosNormalizados[i][1]) * (momentosNormalizados[i][0] - momentosNormalizados[i][1]) 
      + 4*(momentosNormalizados[i][2]) * (momentosNormalizados[i][2]) );

    if(fi2[i] > 0.005)
    {
      float angle = 0.5 * atan2(2*momentosCentralizados[i][2], momentosCentralizados[i][0] - momentosCentralizados[i][1]);
      cout<<"\t\t\t\t\t" << angle << endl;
      int x  = cos( PI/180* 90 * (-angle)/1.57 ) * 40 ;
      int y  = sin( PI/180* 90 * (-angle)/1.57 ) * 40;

      line(destinationImage, Point(promedioX-x, promedioY + y), 
            Point( promedioX+x, promedioY-y), Scalar(255,10,255), 1.4, 8 );
    }

    /*cout << "Objeto " << i
          << "\tfi1: " << fi1[i]
          << "\tfi2: " << fi2[i] << endl;*/

    cout << fi1[i] << "\t" << fi2[i] << endl;
  }
}


void mira(int cuadrante, Mat &sourceImage)
{
  rectangle(sourceImage, Point(520,20), Point(620, 120), Scalar(255,255,255), 1, 4);
  switch(cuadrante)
  {
    case 1:
        rectangle(sourceImage, Point(522,22), Point(568, 68), Scalar(0,0,255), -1, 4);
        break;
    case 2:
        rectangle(sourceImage, Point(572,22), Point(618, 68), Scalar(255,0,0), -1, 4);
        break;
    case 3:
        rectangle(sourceImage, Point(522,72), Point(568, 118), Scalar(0,255,255), -1, 4);
        break;
    case 4:
        rectangle(sourceImage, Point(572,72), Point(618, 118), Scalar(255,0,255), -1, 4);
        break;
  }
}


void detectObjects(Mat &sourceImage)
{
  float objetos[4][5];
  //Dona
  objetos[0][0] = 0.242956361111111; // ph 1
  objetos[0][1] = 0.000584436166667; // ph 2
  objetos[0][2] = 0.0207349943413; // range 1
  objetos[0][3] = 0.000481942; // range 2
  objetos[0][4] = 0; // angulo

  //Manzana
  objetos[1][0] = 0.166114421487603; // ph 1
  objetos[1][1] = 0.000696026644628; // ph 2
  objetos[1][2] = 0.00336031365786; // range 1
  objetos[1][3] = 0.000704323522025; // range 2
  objetos[1][4] = 0; // angulo

  //Vaso
  objetos[2][0] = 0.253294305882353; // ph 1
  objetos[2][1] = 0.034330392352941; // ph 2
  objetos[2][2] = 0.010013904277617; // range 1
  objetos[2][3] = 0.005464262987825; // range 2
  objetos[2][4] = 0; // angulo


  //Cuchillo
  objetos[3][0] = 0.84343 -0.05857; // ph 1
  objetos[3][1] = 0.6828 - 0.1015; // ph 2
  objetos[3][2] = 0.2; // range 1
  objetos[3][3] = 0.03; // range 2
  objetos[3][4] = -60; // angulo


  bool cuchuillo = false;
  bool dona = false;
  bool vaso = false;
  bool manzana = false;

  for(int i=0; i<fi1.size(); i++)
  {
    bool objetoCercano[4];
    float minDistance = 100000;
    int minIndex;
    for(int h=0; h<4; h++)
    {
        float distance = sqrt( pow(fi1[i]-objetos[h][0], 2) + pow(fi2[i]-objetos[h][1], 2) );
        if( distance < minDistance )
        {
          minDistance = distance;
          minIndex = h;
        }
          
        objetoCercano[h] = false;
    }

    objetoCercano[minIndex] = true;

    //Dona (redondo)
    if(checkpoint(objetos[0][0], objetos[0][1], fi1[i], fi2[i], objetos[0][2], objetos[0][3], objetos[0][4]) <= 1 && objetoCercano[0])
    {
      cout << "DONAA" << endl;
      dona = true;
    }

    //Manzana (redondo)
    if(checkpoint(objetos[1][0], objetos[1][1], fi1[i], fi2[i], objetos[1][2], objetos[1][3], objetos[1][4]) <= 1 && objetoCercano[1])
    {
      cout << "Manzana" << endl;
      manzana = true;
    }

    //Vaso (Alargado)
    if(checkpoint(objetos[2][0], objetos[2][1], fi1[i], fi2[i], objetos[2][2], objetos[2][3], objetos[2][4]) <= 1 && objetoCercano[2])
    {
      cout << "Vaso" << endl;
      vaso = true;
    }

    //Cuchillo (Alargado)
    if(checkpoint(objetos[3][0], objetos[3][1], fi1[i], fi2[i], objetos[3][2], objetos[3][3], objetos[3][4]) <= 1 && objetoCercano[3])
    {
      cout << "Cuchillo" << endl;
      cuchuillo = true;
    }
  }

  int cuadrante = 0;
  if(dona && cuchuillo)
    cuadrante = 4;
  if(cuchuillo && manzana)
    cuadrante = 3;
  if(manzana && vaso)
    cuadrante = 1;
  if(vaso && dona)
    cuadrante = 2;
  mira(cuadrante, sourceImage);
	drawGraph(objetos);
	
}

void drawGraph(float objetos[][5]){

	int graph_h =450;
	int graph_w=800;
	int fontFace = FONT_HERSHEY_SCRIPT_SIMPLEX;
	double fontScale = 0.5;
	int thickness = 1;	
	Mat graphs = Mat( graph_h+30,graph_w+30, CV_8UC3, Scalar( 0,0,0) );

	//obtenido manualmente. Cambiar el # de objeto si es necesario
	float x_bound = objetos[3][0]+objetos[3][2]*2;   //el maximo valor de x + 2*desv estandar
	float y_bound = objetos[3][1]+objetos[3][3]*2;   //el máximo valor de y + 2*desv estandar
	Vec3b colores[4] = { Vec3b(255,0,0), Vec3b(0,255,0), Vec3b(0,0,255),
                       Vec3b(100,255,100)};	
	for (int i = 0; i<4; i++){
		int x_center= (objetos[i][0])*graph_w/(x_bound);  //regla de 3
		int y_center=graph_h-(objetos[i][1])*graph_h/(y_bound);  //regla de 3
		Point center = Point(x_center, y_center);
		Size axes = Size((objetos[i][2])*graph_w/(2*x_bound),(objetos[i][3])*graph_h/(2*y_bound));
		//cout<< "\n\n\n\ncenter "<<i <<" : " << x_center <<" , " << y_center <<endl;
		//cout<< "axes: " <<(objetos[i][2])*graph_w/(2*x_bound)<<","<<(objetos[i][3])*graph_h/(2*y_bound) <<endl;
		ellipse(graphs, center, axes, objetos[i][4] ,0, 360, colores[i] ,1,8);
    for(int h=0; h<fi1.size(); h++)
    {
      int x_p = fi1[h]*graph_w/(x_bound);
      int y_p = graph_h-fi2[h]*graph_h/(y_bound);
      if (x_p > 0 && x_p<graph_w && y_p > 0 && y_p < graph_h)
      circle(graphs, Point(x_p,y_p), 3, Scalar::all(255), CV_FILLED,8);
    } 
	}

  

	//Pone los labels
	Point textOrg(0, 30);
	string text = to_string(y_bound);
	putText(graphs, text, textOrg, fontFace, fontScale, Scalar::all(255), thickness, 8);
	textOrg.y = graph_h/2;
	text = to_string(y_bound/2);
	putText(graphs, text, textOrg, fontFace, fontScale, Scalar::all(255), thickness, 8);

	textOrg.x =0;
	textOrg.y=graph_h+25;
	text = to_string(0);
	putText(graphs, text, textOrg, fontFace, fontScale, Scalar::all(255), thickness, 8);
	textOrg.x = graph_w/2;
	text = to_string(x_bound/2);
	putText(graphs, text, textOrg, fontFace, fontScale, Scalar::all(255), thickness, 8);
	textOrg.x = graph_w-20;
	text = to_string(x_bound);
	putText(graphs, text, textOrg, fontFace, fontScale, Scalar::all(255), thickness, 8);

	//dibuja los ejes
	line(graphs, Point(20, graph_h), Point(20,0),Scalar (255,255,255), 1,8);
	line(graphs, Point(20, graph_h), Point(graph_w, graph_h), Scalar(255,255,255), 1,8);

	imshow("Graphs",graphs);
}


int checkpoint(float h, float k, float x, float y, float a, float b, float angle) 
{ 
  
    // checking the equation of 
    // ellipse with the given point 
    int p = pow( cos(angle*PI/180)*(x-h) + sin(angle*PI/180)*(y-k) ,2) / pow(a,2)
            + pow( sin(angle*PI/180)*(x-h) - cos(angle*PI/180)*(y-k) ,2) / pow(b,2);
  
    return p; 
} 



bool checarCamino(Mat &sourceImage, Point inicio, Point final)
{
   if(inicio.x == final.x)
   {
      for(int i=inicio.y; i <final.y; i++)
      {
        if(sourceImage.at<Vec3b>( i , inicio.x) == Vec3b(255,255,255) )
          return false;
        //sourceImage.at<Vec3b>(i , inicio.x) = Vec3b(255, 255, 0);
      }
   }
   else
   {
      float m = ((float)inicio.y - (float)final.y) / ((float)inicio.x - (float)final.x) ;
      float b = (float)inicio.y - m* ((float)inicio.x);

      for(int i=inicio.x; i <final.x; i++)
      {
        if(sourceImage.at<Vec3b>( round(i*m+b) , i) == Vec3b(255,255,255))
          return false;

        //sourceImage.at<Vec3b>( round(i*m+b) , i) = Vec3b(255, 255, 0);
        int altura = round(i*m+b) + 1;

        while(i<final.x && altura < round((i+1)*m+b))
        {
          //sourceImage.at<Vec3b>( altura , i) = Vec3b(255, 255, 0);
          if(sourceImage.at<Vec3b>( altura , i) == Vec3b(255,255,255))
            return false;
          altura++;
        }
        
        altura = round(i*m+b) - 1;

        while(i<final.x && altura > round((i+1)*m+b))
        {
          //sourceImage.at<Vec3b>( altura , i) = Vec3b(255, 255, 0);
          if(sourceImage.at<Vec3b>( altura , i) == Vec3b(255,255,255))
            return false;
          altura--;
        }
        
      }
   }

   return true;
}

void camino(const Mat &sourceImage, Mat &destinationImage)
{
  //Pinta la imagen original en la de salida
  for(int i=0; i<sourceImage.rows; i++)
    for(int j=0; j<sourceImage.cols; j++)
    {
      if(sourceImage.at<unsigned char>(i,j) == 255)
        destinationImage.at<Vec3b>( i , j ) = Vec3b(255,255,255);
    }

  if(conexiones.size() ==0)
  {
    int x = ceil( ((float)sourceImage.cols) / (espacio+1));
    xEstacas = x;
    int y = ceil( ((float)sourceImage.rows) / (espacio+1));
    conexiones = vector<vector<short>>(x*y, vector<short>());

    // Pinta las estacas
    for(int i=0; i<sourceImage.rows; i+=espacio+1)
      for(int j=0; j<sourceImage.cols; j+=espacio+1)
      {
        int tempi = i, tempj = j;

        // Si la estaca esta sobre un objeto la mueve fuera
        // de el
        if(sourceImage.at<unsigned char>(tempi,tempj) == 255)
        {
          short extendX = 1, extendY = 1;
          while(true)
          {
            if(tempi+extendY > 0 && tempi+extendY < sourceImage.rows &&
              sourceImage.at<unsigned char>(tempi+extendY,tempj) != 255)
            {
              tempi+=extendY;
              break;
            }
            if(tempj+extendX > 0 && tempj+extendX < sourceImage.cols &&
              sourceImage.at<unsigned char>(tempi,tempj+extendX) != 255)
            {
              tempj+=extendX;
              break;
            }
            extendX = extendX < 0 ? (-extendX)+1 : (-extendX);
            extendY = extendY < 0 ? (-extendY)+1 : (-extendY);
          }
        }

        //Se guardan las coordenadas de las estacas en el vector de estacas
        estacas.push_back( Point(tempj,tempi) );

        // Conexion con el pixel de la izquierda
        if(tempj!=0 && estacas.size() > 1 && checarCamino(destinationImage, estacas[ estacas.size()-2 ], Point(tempj,tempi))
          && checarCamino(destinationImage, Point(tempj,tempi), estacas[ estacas.size()-2 ]))
        {
          //line( destinationImage, estacas[ estacas.size()-2 ], Point(tempj,tempi), Scalar(255, 255, 0), 1 ,8);
          conexiones[estacas.size()-1].push_back( estacas.size()-2 );
          conexiones[estacas.size()-2].push_back( estacas.size()-1 );
        }

        // Conexion con el pixel de arriba
        if(tempi!=0 && estacas.size() > x && checarCamino(destinationImage, estacas[ estacas.size()-1-x ], Point(tempj,tempi))
          && checarCamino(destinationImage, Point(tempj,tempi), estacas[ estacas.size()-1-x ]))
        {
          //line( destinationImage, estacas[ estacas.size()-1 -x ], Point(tempj,tempi), Scalar(255, 255, 0), 1 ,8);
          conexiones[estacas.size()-1].push_back( estacas.size()-1 -x);
          conexiones[estacas.size()-1 -x].push_back( estacas.size()-1 );
        }
        destinationImage.at<Vec3b>( tempi , tempj ) = Vec3b(0,0,255);
      }
  }
}

int minDistance(int dist[], bool sptSet[]) 
{ 
    // Initialize min value 
    int V = conexiones.size();
    int min = INT_MAX, min_index; 
  
    for (int v = 0; v < V; v++) 
        if (sptSet[v] == false && dist[v] <= min) 
            min = dist[v], min_index = v; 
  
    return min_index; 
} 

void printPath(Mat &destinationImage, int parent[], int j, int src) 
{ 
    while(parent[j] > -1)
    {
      if(parent[j] >= estacas.size())
      {
        cout << "Imposible realizar la conexion" << endl;
        break;
      }
      line( destinationImage, estacas[j], estacas[parent[j]], Scalar(0, 0, 255), 1 ,8);
      j = parent[j]; 
    }
    //line( destinationImage, estacas[j], estacas[src], Scalar(255, 255, 0), 1 ,8);
} 

void navegacion(Mat &destinationImage, Point inicio, Point final)
{
    int src= round( (float)inicio.x/((float)espacio+1)) + round((float)inicio.y/((float)espacio+1)) * xEstacas;
    int fin= round( (float)final.x/((float)espacio+1)) + round((float)final.y/((float)espacio+1)) * xEstacas;
    int V = conexiones.size();
    int dist[V]; // The output array.  dist[i] will hold the shortest 
    // distance from src to i 
  
    bool sptSet[V]; // sptSet[i] will be true if vertex i is included in shortest 
    // path tree or shortest distance from src to i is finalized 

    // Parent array to store 
    // shortest path tree 
    int parent[V]; 
  
    // Initialize all distances as INFINITE and stpSet[] as false 
    for (int i = 0; i < V; i++) 
    {
        dist[i] = INT_MAX; 
        sptSet[i] = false; 
        parent[i] = -1; 
    }
  
    // Distance of source vertex from itself is always 0 
    dist[src] = 0; 
    
    // Find shortest path for all vertices 
    for (int count = 0; count < V - 1; count++) { 
        // Pick the minimum distance vertex from the set of vertices not 
        // yet processed. u is always equal to src in the first iteration. 

        int u = minDistance(dist, sptSet); 

  
        // Mark the picked vertex as processed 
        sptSet[u] = true;
  
        // Update dist value of the adjacent vertices of the picked vertex. 
        for (int v = 0; v < conexiones[u].size(); v++) 
        {

            // Update dist[v] only if is not in sptSet, there is an edge from 
            // u to v, and total weight of path from src to  v through u is 
            // smaller than current value of dist[v]
            int index = conexiones[u][v];
            if (!sptSet[index] && find(conexiones[u].begin(), conexiones[u].end() ,index) != conexiones[u].end()
                && dist[u] != INT_MAX  && dist[u] + 1 < dist[index]) 
            {
                parent[index] = u;
                dist[index] = dist[u] + 1; 
            }

        }

    }
  
    parent[src] = -1;

    line( destinationImage, estacas[fin], final, Scalar(0, 0, 255), 1 ,8);
    printPath(destinationImage, parent, fin, src);
    circle(destinationImage, estacas[src], 4, Scalar(0,255,0), -1, 8);
    circle(destinationImage, final, 4, Scalar(255,0,255), -1, 8);
    // print the constructed distance array 
    //printSolution(dist);
}
