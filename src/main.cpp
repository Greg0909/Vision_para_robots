#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace std;
using namespace cv;

void histogramGeneral(const Mat &sourceImage, Mat &histo, int channels, Scalar color, int point_val);
void flipImageBasic(const Mat &sourceImage, Mat &destinationImage);
void mouseCoordinatesExampleCallback(int event, int x, int y, int flags, void* param);
void plotHistograms();
void imageRgbToHsv();
void imageRgbToYiq();
void blackWhite();
void filterImage();

                                        // VARIABLES GLOBALES
                                        // Contador para refreshear la escala de los 3 histogramas
int counter_hist[3] = {0,0,0};
                                        // La escala de los 3 histogramas
int h_scale[3] = {0,0,0};
Mat currentImageRGB, currentImageHSV, currentImageYIQ;
bool congelado = false;
                                        // r (RGB), h (HSV), y (YIQ)
char modelo =  'r';
                                        // Valor del pixel-clic en los 3 modelos
int bgr_point[3] = {-1, -1, -1};
int hsv_point[3] = {-1, -1, -1};
int yiq_point[3] = {-1, -1, -1};



/*< Main START >*/
int main(int argc, char *argv[]) 
{
  namedWindow("Original");
  setMouseCallback("Original", mouseCoordinatesExampleCallback);
  VideoCapture camera = VideoCapture(0);
  bool isCameraAvailable = camera.isOpened();
  
                                        // Limpia la terminal
  cout << "\033[2J\033[1;1H";
  cout << "Basic Show Image \t|\tUse 'x' or 'Esc' to terminate execution\n";

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
        imageRgbToHsv();
        imageRgbToYiq();
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

    imshow("Original", currentImageRGB);

    char key = waitKey(1);
    switch(key)
    {
      case 'r':
        modelo = 'r';
        break;
      case 'h':
        modelo = 'h';
        break;
      case 'y':
        modelo = 'y';
        break;
      case ' ':
        congelado = !congelado;
        break;
    }
                                        // Si 'x' o ESC es presionada el programa termina
    if(key == 'x' || key == 27 )        // 27 = ESC
    {
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
      histogramGeneral(currentImageHSV, histo_1, 0, Scalar( 0  , 255  , 255  ), hsv_point[0]);
      histogramGeneral(currentImageHSV, histo_2, 1, Scalar( 0  , 255  , 255  ), hsv_point[1]);
      histogramGeneral(currentImageHSV, histo_3, 2, Scalar( 0  , 255  , 255  ), hsv_point[2]);

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
      histogramGeneral(currentImageYIQ, histo_1, 0, Scalar( 255  , 120  , 120  ), yiq_point[0]);
      histogramGeneral(currentImageYIQ, histo_2, 1, Scalar( 120  , 255  , 120  ), yiq_point[1]);
      histogramGeneral(currentImageYIQ, histo_3, 2, Scalar( 120  , 120  , 255  ), yiq_point[2]);

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
    gradiente[channel] = i-1;

    if(modelo == 'h')
    {
                                        // Convertir el color "gradiente" de HSV a BGR
    }
    if(modelo == 'y')
    {
                                        // Convertir el color "gradiente" de YIQ a BGR
    }

    line(histImage, Point(offset + bin_w*(i-1), hist_h), Point(offset + bin_w*(i-1), hist_h - gradient_size), gradiente , 3, CV_FILLED);

    line( histImage, Point( bin_w*(i-1) + offset, hist_h - gradient_size - cvRound(hist.at<float>(i-1)*400/h_scale[channel])  ) ,
          Point( bin_w*(i) + offset, hist_h - gradient_size - cvRound(hist.at<float>(i)*400/h_scale[channel]) ),
          Scalar(255,255,255), 2, 8, 0  );
  }

                                        // Dibuja una linea en el valor del pixel-clic
  if(point_val != -1)
  {
    line(histImage, Point(offset + point_val*bin_w, 0), Point(offset + point_val*bin_w, hist_h), Scalar( 255, 255, 255) , 2, CV_FILLED);
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


/*< Filter Image START >*/
void filterImage()
{
                                        // Toma la imagen currentImageRGB y junto con el
                                        // punto-clic filtra por umbrales para desaparecer
                                        // los colores fuera del umbral. 
                                        // referencia: https://docs.opencv.org/trunk/df/d9d/tutorial_py_colorspaces.html
}
/*< Filter Image END >*/


/*< RGB to HSV point START >*/
void pointRgbToHsv()
{
                                        // Toma los valores del arreglo "bgr_point"
                                        // para convertirlos a valores hsv y guardarlos
                                        // en el arreglo "hsv_point"
}
/*< RGB to HSV point END >*/


/*< RGB to YIQ point START >*/
void pointRgbToYiq()
{
                                        // Toma los valores del arreglo "bgr_point"
                                        // para convertirlos a valores hsv y guardarlos
                                        // en el arreglo "hsv_point"
}
/*< RGB to YIQ point END >*/


/*< RGB to HSV image START >*/
void imageRgbToHsv()
{
                                        // Toma los valores de la matriz "currentImageRGB"
                                        // para convertirlos a valores hsv y guardarlos
                                        // en la matriz "currentImageHSV"
}
/*< RGB to HSV image END >*/


/*< RGB to YIQ image START >*/
void imageRgbToYiq()
{
                                        // Toma los valores del arreglo "bgr_point"
                                        // para convertirlos a valores hsv y guardarlos
                                        // en el arreglo "hsv_point"
}
/*< RGB to YIQ image END >*/


/*< Mouse callback START >*/
void mouseCoordinatesExampleCallback(int event, int x, int y, int flags, void* param)
{
    switch (event)
    {
        case CV_EVENT_LBUTTONDOWN:
            cout << "  Mouse X, Y: " << x << ", " << y << "\tRGB " << 
            int(currentImageRGB.at<Vec3b>(y, x)[2]) << ", " <<
            int(currentImageRGB.at<Vec3b>(y, x)[1]) << ", " <<
            int(currentImageRGB.at<Vec3b>(y, x)[0]) ;
            cout << endl;
            bgr_point[0] = int(currentImageRGB.at<Vec3b>(y, x)[0]);
            bgr_point[1] = int(currentImageRGB.at<Vec3b>(y, x)[1]);
            bgr_point[2] = int(currentImageRGB.at<Vec3b>(y, x)[2]);
            pointRgbToHsv();
            pointRgbToYiq();
            break;
        case CV_EVENT_MOUSEMOVE:
            break;
        case CV_EVENT_LBUTTONUP:
            break;
    }
}
/*< Mouse callback END >*/