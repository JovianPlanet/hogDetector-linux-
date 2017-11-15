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

std::string folderpath = "/media/hoh-6/Datos/David/U/ai/codes/hogDetector/cuts/";
std::stringstream path;
std::basic_ofstream<char> feature("Hist_Values.data", std::ofstream::app);

FILE *fp;

void gradientHist(Mat &magcell_8Px,  Mat &dircell_8Px, int countColumn, int contimg) {
	
	Mat hist;
	//std::basic_ofstream<char> feature("Hist_Values.data", std::ofstream::app);
	//hist.create(9, 1, CV_64F);
	hist = Mat::zeros(9, 1, CV_64F);
	// create a 9 bin histogram with range from 0 to 180 for HOG descriptors.
	int histSize = 9;
	float range[] = { 0,180 };
	const float *histRange = { range };
	bool uniform = true;
	bool accumulate = false;
	cout << magcell_8Px << "\n";
	cout << dircell_8Px << "\n";
	/* Calcular Histograma */
	//calcHist(&dircell_8Px, 1, 0, Mat(), hist, 1, &histSize, &histRange, uniform, accumulate); //Calculate the 9 bin histogram.
	for (int row = 0; row < dircell_8Px.rows; row++) {
		uchar* datamag = magcell_8Px.ptr<uchar>(row);
		for (int col = 0; col < dircell_8Px.cols; col++) {
			if (dircell_8Px.at<char>(row, col) < 20) {
				int porc = (int)dircell_8Px.at<char>(row, col)*100 / 20;
				hist.at<double>(0) += datamag[col]*(100 - porc)/100;
				hist.at<double>(1) += datamag[col]*(porc)/100;
				//printf("%d\n", hist.at<char>(2));
			}
			else if (dircell_8Px.at<char>(row, col) >= 20 && dircell_8Px.at<char>(row, col) < 40) {
				int porc = (int)(dircell_8Px.at<char>(row, col) - 20)*100 / 20;
				printf("porc: %d\n", porc);
				hist.at<double>(1) += datamag[col]*(100 - porc)/100;
				hist.at<double>(2) += datamag[col]*porc/100;
				printf("pointmag: %d\n", datamag[col]*porc/100);
				//printf("mag: %d\n", magcell_8Px.at<char>(row, col));
				//printf("Multip: %d\n", (int(magcell_8Px.at<char>(row, col)))*(porc));
				printf("hist: %d\n", hist.at<char>(2));
			}
			else if (dircell_8Px.at<char>(row, col) >= 40 && dircell_8Px.at<char>(row, col) < 60) {
				int porc = (int)(dircell_8Px.at<char>(row, col) - 40)*100 / 20;
				hist.at<double>(2) += datamag[col]*(100 - porc)/100;
				hist.at<double>(3) += datamag[col]*(porc)/100;
			}
			else if (dircell_8Px.at<char>(row, col) >= 60 && dircell_8Px.at<char>(row, col) < 80) {
				int porc = (int)(dircell_8Px.at<char>(row, col) - 60)*100 / 20;
				hist.at<double>(3) += datamag[col]*(100 - porc)/100;
				hist.at<double>(4) += datamag[col]*(porc)/100;
			}			
			else if (dircell_8Px.at<char>(row, col) >= 80 && dircell_8Px.at<char>(row, col) < 100) {
				int porc = (int)(dircell_8Px.at<char>(row, col) - 80)*100 / 20;
				hist.at<double>(4) += datamag[col]*(100 - porc)/100;
				hist.at<double>(5) += datamag[col]*(porc)/100;
			}			
			else if (dircell_8Px.at<char>(row, col) >= 100 && dircell_8Px.at<char>(row, col) < 120) {
				int porc = (int)(dircell_8Px.at<char>(row, col) - 100)*100 / 20;
				hist.at<double>(5) += datamag[col]*(100 - porc)/100;
				hist.at<double>(6) += datamag[col]*(porc)/100;
			}
			else if (dircell_8Px.at<char>(row, col) >= 120 && dircell_8Px.at<char>(row, col) < 140) {
				int porc = (int)(dircell_8Px.at<char>(row, col) - 120)*100 / 20;
				hist.at<double>(6) += datamag[col]*(100 - porc)/100;
				hist.at<double>(7) += datamag[col]*(porc)/100;
			}
			else if (dircell_8Px.at<char>(row, col) >= 140 && dircell_8Px.at<char>(row, col) < 160) {
				int porc = (int)(dircell_8Px.at<char>(row, col) - 140)*100 / 20;
				hist.at<double>(7) += datamag[col]*(100 - porc)/100;
				hist.at<double>(8) += datamag[col]*(porc)/100;
			}
			else if (dircell_8Px.at<char>(row, col) > 160) {
				int porc = (int)(dircell_8Px.at<char>(row, col) - 160)*100 / 20;
				hist.at<double>(8) += datamag[col]*(100 - porc)/100;
				hist.at<double>(0) += datamag[col]*(porc)/100;
			}
		}
	}

	/*double maxVal = 0;
	double minVal = 0;
	minMaxLoc(hist, &minVal, &maxVal, 0, 0);
	printf("Max: %f\n", maxVal);
	printf("Min: %f\n", minVal);*/
	//normalize(hist, hist, 255, 0, NORM_MINMAX, -1, Mat());

	/*if (contimg < 425) {
		feature << "A,";
	}
	else {
		feature << "B,";
	}*/
	//cout << hist;
	
	for (int i = 0; i < histSize; i++)
	{
		if (i*countColumn == 96) {
			feature << hist.at<double>(i) << "\n"; // Output the value of HOG to data file

		}
		else {
			feature << hist.at<double>(i) << ","; // Output the value of HOG to data file   
		}
		//printf("i = %d\n", i);
	}
	return;
}

void hist8x8(Mat &magnitude, Mat &direction, int contImg) {
	Size blockSize(16, 16);
	Size cellSize(8, 8);

	vector<Mat> block;
	Size s = direction.size();
	Mat dirCell_16Px, magCell_16Px;
	Mat dirCell_8Px, magCell_8Px;

	int count = 0;

	for (int row = 0; row <= direction.rows - 16; row += cellSize.height)
	{
		//for (int col = 0; col <= direction.cols; col += cellSize.width) //No se necesita este ciclo ya que la columna siempre es la cero (La imagen es 16x32 y tomo pedazos de 16x16)
		//{
			Rect rect = Rect(0, row, blockSize.width, blockSize.height);
			dirCell_16Px = Mat(direction, rect); // Get a 16 by 16 cell from the gradient direction matrix.
			magCell_16Px = Mat(magnitude, rect);
		    //**********COMPUTE 9 bin gradient direction histogram of the 16 by 16 cell here !!! ****
			for (int i = 0; i <= blockSize.height - 8; i += cellSize.height)
			{
				for (int j = 0; j <= blockSize.width - 8; j += cellSize.width)
				{
					count += 1;
					Rect rect_8Px = Rect(i, j, cellSize.width, cellSize.height); // 8 by 8 rectangle size
					dirCell_8Px = Mat(dirCell_16Px, rect_8Px); // create a 8 by 8 cell from gradient direction matrix (cell_eightPx)
					magCell_8Px = Mat(magCell_16Px, rect_8Px); // create a 8 by 8 cell from gradient direction matrix (cell_eightPx)
					gradientHist(magCell_8Px, dirCell_8Px, count, contImg); // Calculate gradient histogram of the 8 by 8 cell. (Function call)
					dirCell_8Px.deallocate(); // clear the cell.	
					magCell_8Px.deallocate(); // clear the cell.
					//printf("Llamadas a gradientHist: %d\n", count);
				}
			}
		//}
	}
	
	return;
}

int main() {

	Mat imagen, imagenGris, imagenGradiente_x, imagenGradiente_y, imagenGradiente_XY;

	double min, max;
	if ((fp = fopen("datosDBP.txt", "w+")) == NULL) {
		printf("Cannot open file.\n");
		exit(1);
	}

	for (int i = 1; i <= 1; i++) { //Uncomment for reading all database

		std::stringstream path; //Uncomment for reading all database
		path << folderpath << "A" << i << ".jpg"; //i+1  //Uncomment for reading all database

		if (i < 425) {
			feature << "A,";
		}
		else {
			feature << "B,";
		}

		imagen = imread(path.str().c_str()); //Uncomment for reading all database
		//imagen = imread("pedestrian.jpg"); //Comment for reading all database
		cvtColor(imagen, imagenGris, COLOR_BGR2GRAY);
		resize(imagenGris, imagenGris, Size(16, 32), 0, 0, CV_INTER_AREA);

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
		imagenGradiente_x.convertTo(imagenGradiente_xf, CV_32F);// , 1.0 / 255.0);
		imagenGradiente_y.convertTo(imagenGradiente_yf, CV_32F);// , 1.0 / 255.0);
		cartToPolar(imagenGradiente_xf, imagenGradiente_yf, mag, angle, true);
		mag.convertTo(mag, CV_8UC1);
		angle.convertTo(angle, CV_8UC1);
		//cout << mag;

		hist8x8(mag, angle, i);
		imagenGradiente_XY = (imagenGradiente_x + imagenGradiente_y) / 2;
		/*imshow("imagen", imagen);
		imshow("imagen Gradiente", imagenGris);
		imshow("imagen Gradiente X", imagenGradiente_x);
		imshow("imagen Gradiente Y", imagenGradiente_y);
		imshow("imagen Gradiente XY", imagenGradiente_XY);
		waitKey(100);*/
	} //Uncomment for reading all database

	fclose(fp);
	printf("Done!");
	getchar();
	return 0;
}