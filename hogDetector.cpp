#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "opencv2/imgcodecs.hpp"
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv/cv.h>
#include <opencv/highgui.h>
#include <iostream>
#include <fstream>
#include <string>

using namespace cv;  // The new C++ interface API is inside this namespace. Import it.
using namespace std;

std::string folderpath = "/park/";
std::stringstream path;

FILE *fp;



void gradientHist(Mat &cell_fourPx){
	Mat hist;
    std::basic_ofstream<char> feature("Hist_Values.csv",std::ofstream::app);
    
    // create a 9 bin histogram with range from 0 t0 180 for HOG descriptors.
    int histSize = 9;
    float range[] = {0,180};
    const float *histRange = {range};
    bool uniform = true;
    bool accumulate = false;
    calcHist(&cell_fourPx, 1, 0,Mat(),hist, 1, &histSize, &histRange, uniform, accumulate); //Calculate the 9 bin histogram.
    //normalize(hist, hist, 0, 0, NORM_MINMAX, -1, Mat());

    //printf("%d", count);
    for(int i = 0; i < histSize; i++)
    {
       feature << hist.at<float>(i) << ","; // Output the value of HOG to a csv file             
    }
}

void hist8x8(Mat &direction){
	Size blockSize(16,16);
	Size cellSize(8,8);

	vector<Mat> block;
	Size s =  direction.size();
	Mat cell_eightPx;
	Mat cell_fourPx;

	int count = 0;

	for(int col = 0; col < direction.rows - 8; col += cellSize.height)
    {

        for(int row = 0; row < direction.cols - 8; row += cellSize.width)
        {
        	
            Rect rect= Rect(row, col,blockSize.width,blockSize.height);
            cell_eightPx = Mat(direction,rect); // Get a 16 by 16 cell from the gradient direction matrix.

            //**********COMPUTE 9 bin gradient direction histogram of the 16 by 16 cell here !!! ****
            for(int i = 0; i < blockSize.height; i += cellSize.height)
            {
                for(int j  = 0; j < blockSize.width; j += cellSize.width)
                {
                    Rect rect_fourPx = Rect(i,j,cellSize.width,cellSize.height); // 8 by 8 rectangle size
                    cell_fourPx = Mat(cell_eightPx,rect_fourPx); // create a 8 by 8 cell from gradient direction matrix (cell_eightPx)
                    gradientHist(cell_fourPx); // Calculate gradient histogram of the 8 by 8 cell. (Function call)
                    
                    cell_fourPx.deallocate(); // clear the cell.
                    count += 1;
                }
            }

        }
        
    }
    printf("%d\n", count);
}

int main() {

	Mat imagen, imagenGris, imagenGradiente_x, imagenGradiente_y, imagenGradiente_XY;

	//Mat kern_x = (Mat_<char>(1, 3) << -1, 0, 1);

	//Mat kern_y = (Mat_<char>(3, 1) << -1, 0, 1);

	//Mat kern_x = (Mat_<char>(3, 3) << -1, 0, 1, -1, 0, 1, -1, 0, 1);

	//Mat kern_y = (Mat_<char>(3, 3) << 1, 1, 1, 0, 0, 0,-1, -1, -1);

	//int hbins = 256;
	//int histSize[] = { hbins };
	//float hranges[] = { 0, 255 };
	//const float* ranges[] = { hranges };
	//MatND hist;
	//int channels[] = { 0 };
	double min, max;
	if ((fp = fopen("datosDBP.txt", "w+")) == NULL) {
		printf("Cannot open file.\n");
		exit(1);
	}

	//for (int i = 1; i <= 20; i++) { //Uncomment for reading all database

		//std::stringstream path; //Uncomment for reading all database

		//path << folderpath << "DBP" << i << ".jpg"; //i+1  //Uncomment for reading all database

		//imagen = imread(path.str().c_str()); //Uncomment for reading all database
	    imagen = imread("pedestrian.jpg"); //Comment for reading all database
		cvtColor(imagen, imagenGris, COLOR_BGR2GRAY);
		resize(imagenGris, imagenGris, Size(64, 128), 0, 0, CV_INTER_AREA);

		imagenGradiente_x.create(imagenGris.rows, imagenGris.cols, CV_8UC1);
		for (int j = 0; j<imagenGris.rows; j++) {
			uchar* data = imagenGris.ptr<uchar>(j);
			uchar* data1 = imagenGradiente_x.ptr<uchar>(j);
			for (int i = 1; i<imagenGris.cols - 1; i++) {
				data1[i] = (data[i + 1] - data[i - 1] + 255) / 2;
			}
		}

		imagenGradiente_y.create(imagenGris.rows, imagenGris.cols, CV_8UC1);
		for (int j = 1; j<imagenGris.rows - 1; j++) {
			uchar* data = imagenGris.ptr<uchar>(j);
			uchar* data1 = imagenGradiente_y.ptr<uchar>(j);
			uchar* dataAnt = imagenGris.ptr<uchar>(j - 1);
			uchar* dataSig = imagenGris.ptr<uchar>(j + 1);
			for (int i = 0; i<imagenGris.cols; i++) {
				data1[i] = data[i];
				data1[i] = (dataSig[i] - dataAnt[i] + 255) / 2;
			}
		}
		Mat mag, angle, imagenGradiente_yf, imagenGradiente_xf;
		imagenGradiente_x.convertTo(imagenGradiente_xf, CV_32F);
		imagenGradiente_y.convertTo(imagenGradiente_yf, CV_32F);
		cartToPolar(imagenGradiente_xf, imagenGradiente_yf, mag, angle, true);
		cout << "M = "<< endl << " "  << angle << endl << endl;
		imshow("angle", angle);
		hist8x8(angle);
		imagenGradiente_XY = (imagenGradiente_x + imagenGradiente_y)/2;
		imshow("imagen", imagen);
		imshow("imagen Gradiente", imagenGris);
		imshow("imagen Gradiente X", imagenGradiente_xf);
		imshow("imagen Gradiente Y", imagenGradiente_yf);
		imshow("imagen Gradiente XY", imagenGradiente_XY);
		waitKey(100);
	//} //Uncomment for reading all database

	fclose(fp);
	getchar();
	return 0;
}