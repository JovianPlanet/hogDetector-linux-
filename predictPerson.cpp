#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "opencv2/imgcodecs.hpp"
#include "opencv2/ml/ml.hpp"
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv/cv.h>
#include <opencv/highgui.h>
#include <iostream>
#include <fstream>
#include <string>
#include <cstdio>
#include <vector>

using namespace cv;  // The new C++ interface API is inside this namespace. Import it.
using namespace std;
using namespace cv::ml;

Ptr<ANN_MLP> model;

double minValx2 = 0, maxvalx2 = 0, minValx1 = 0, maxvalx1 = 0;

std::stringstream path;
std::basic_ofstream<char> feature("Hist_Values.data", std::ofstream::app);

FILE *fp;

// This function reads data and responses from the file <filename>
static bool
read_num_class_data(const string& filename, int var_count,
	Mat* _data, Mat* _responses)
{
	const int M = 1024;
	char buf[M + 2];

	Mat el_ptr(1, var_count, CV_32F);
	int i;
	vector<int> responses;

	_data->release();
	_responses->release();

	FILE* f = fopen(filename.c_str(), "rt");
	if (!f)
	{
		cout << "Could not read the database " << filename << endl;
		return false;
	}

	for (;;)
	{
		char* ptr;
		if (!fgets(buf, M, f) || !strchr(buf, ','))
			break;
		responses.push_back((int)buf[0]);
		ptr = buf + 2;
		for (i = 0; i < var_count; i++)
		{
			int n = 0;
			sscanf(ptr, "%f%n", &el_ptr.at<float>(i), &n);
			ptr += n + 1;
		}
		if (i < var_count)
			break;
		_data->push_back(el_ptr);
	}
	fclose(f);
	Mat(responses).copyTo(*_responses);

	cout << "The database " << filename << " is loaded.\n";

	return true;
}

template<typename T>
static Ptr<T> load_classifier(const string& filename_to_load)
{
	// load classifier from the specified file
	Ptr<T> model = StatModel::load<T>(filename_to_load);
	if (model.empty())
		cout << "Could not read the classifier " << filename_to_load << endl;
	else
		cout << "The classifier " << filename_to_load << " is loaded.\n";

	return model;
}

static Ptr<TrainData>
prepare_train_data(const Mat& data, const Mat& responses, int ntrain_samples)
{
	Mat sample_idx = Mat::zeros(1, data.rows, CV_8U);
	Mat train_samples = sample_idx.colRange(0, ntrain_samples);
	train_samples.setTo(Scalar::all(1));

	int nvars = data.cols;
	Mat var_type(nvars + 1, 1, CV_8U);
	var_type.setTo(Scalar::all(VAR_ORDERED));
	var_type.at<uchar>(nvars) = VAR_CATEGORICAL;

	return TrainData::create(data, ROW_SAMPLE, responses,
		noArray(), sample_idx, noArray(), var_type);
}

inline TermCriteria TC(int iters, double eps)
{
	return TermCriteria(TermCriteria::MAX_ITER + (eps > 0 ? TermCriteria::EPS : 0), iters, eps);
}

static void test_and_save_classifier(const Ptr<StatModel>& model,
	const Mat& data, const Mat& responses,
	int ntrain_samples, int rdelta,
	const string& filename_to_save)
{
	int i, nsamples_all = data.rows;
	double train_hr = 0, test_hr = 0;

	// compute prediction error on train and test data
	for (i = 0; i < nsamples_all; i++)
	{
		Mat sample = data.row(i);

		double x1 = (int)data.at<float>(i, 0);//157

		double x2 = (int)data.at<float>(i, 1);//67.74

		float r = model->predict(sample);
		printf("Predict: %f\n", r);
		r = std::abs(r + rdelta - responses.at<int>(i)) <= FLT_EPSILON ? 1.f : 0.f;

		if (i < ntrain_samples)
			train_hr += r;
		else
			test_hr += r;
	}

	test_hr /= nsamples_all - ntrain_samples;
	train_hr = ntrain_samples > 0 ? train_hr / ntrain_samples : 1.;

	printf("Recognition rate: train = %.1f%%, test = %.1f%%\n",
		train_hr*100., test_hr*100.);

	if (!filename_to_save.empty())
	{
		model->save(filename_to_save);
	}
}

static bool
build_mlp_classifier(const string& data_filename,
	const string& filename_to_save,
	const string& filename_to_load)
{
	const int class_count = 2;
	Mat data;
	Mat responses;

	bool ok = read_num_class_data(data_filename, 108, &data, &responses);
	if (!ok)
		return ok;

	int nsamples_all = data.rows;
	int ntrain_samples = (int)(nsamples_all*0.8);

	minMaxLoc(data, &minValx1, &maxvalx1, 0, 0);
	minMaxLoc(data, &minValx2, &maxvalx2, 0, 0);

	// Create or load MLP classifier
	if (!filename_to_load.empty())
	{
		model = load_classifier<ANN_MLP>(filename_to_load);
		if (model.empty())
			return false;
		ntrain_samples = 0;
	}
	else
	{
		// !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
		//
		// MLP does not support categorical variables by explicitly.
		// So, instead of the output class label, we will use
		// a binary vector of <class_count> components for training and,
		// therefore, MLP will give us a vector of "probabilities" at the
		// prediction stage
		//
		// !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

		Mat train_data = data.rowRange(0, ntrain_samples);
		Mat train_responses = Mat::zeros(ntrain_samples, class_count, CV_32F);

		// 1. unroll the responses
		cout << "Unrolling the responses...\n";
		for (int i = 0; i < ntrain_samples; i++)
		{
			int cls_label = responses.at<int>(i) - 'A';
			train_responses.at<float>(i, cls_label) = 1.f;
		}

		// 2. train classifier
		int layer_sz[] = { data.cols, 50, 50, 25, class_count };  // Mejor resultado: 50, 50, 25
		int nlayers = (int)(sizeof(layer_sz) / sizeof(layer_sz[0]));
		Mat layer_sizes(1, nlayers, CV_32S, layer_sz);

#if 0
		int method = ANN_MLP::BACKPROP;
		double method_param = 0.001;
		int max_iter = 300;
#else
		int method = ANN_MLP::RPROP;
		double method_param = 0.1;
		int max_iter = 1000;
#endif

		Ptr<TrainData> tdata = TrainData::create(train_data, ROW_SAMPLE, train_responses);

		cout << "Training the classifier (may take a few minutes)...\n";
		model = ANN_MLP::create();
		model->setLayerSizes(layer_sizes);
		model->setActivationFunction(ANN_MLP::SIGMOID_SYM, 0, 0);
		model->setTermCriteria(TC(max_iter, 0));
		model->setTrainMethod(method, method_param);
		model->train(tdata);
		cout << endl;
	}

	test_and_save_classifier(model, data, responses, ntrain_samples, 'A', filename_to_save);
	return true;
}

void predecir(const Ptr<StatModel>& model,
	const Mat& data, Mat& responses
) {
	int i, nsamples_all = data.rows;
	double train_hr = 0, test_hr = 0;

	// compute prediction error on train and test data
	for (i = 0; i < nsamples_all; i++)
	{
		Mat sample = data.row(i);
		double x1 = (int)data.at<float>(i, 0);//157
		double x2 = (int)data.at<float>(i, 1);//67.74
		float r = model->predict(sample);
		responses.at<int>(i) = r;
	}
}


bool decision(const string& data_filename) {
	Mat dataTest_All = Mat::zeros(Size(2, 256 * 256), CV_32FC1);
	Mat responsesTest = Mat::zeros(Size(1, 256 * 256), CV_32SC1);
	long int cont = 0;
	for (int j = 0; j<256; j++) {
		for (int i = 0; i<256; i++) {
			dataTest_All.at<float>(cont, 0) = j;
			dataTest_All.at<float>(cont, 1) = i;
			cont++;
		}
	}
	normalize(dataTest_All(Range::all(), Range(0, 1)), dataTest_All(Range::all(), Range(0, 1)), minValx1, maxvalx1, NORM_MINMAX, CV_32F);
	normalize(dataTest_All(Range::all(), Range(1, 2)), dataTest_All(Range::all(), Range(1, 2)), minValx2, maxvalx2, NORM_MINMAX, CV_32F);
	predecir(model, dataTest_All, responsesTest);
	Mat plotFeatures = Mat::zeros(Size(256, 256), CV_8UC3);
	cont = 0;
	for (int j = 0; j<256; j++) {
		for (int i = 0; i<256; i++) {
			if (responsesTest.at<int>(cont) == 0) {
				circle(plotFeatures, Point(j, i), 3, Scalar(255, 0, 0));
			}
			else {
				circle(plotFeatures, Point(j, i), 3, Scalar(0, 0, 255));
			}
			cont++;
		}
	}

	imshow("plotFeatures1", plotFeatures);
	waitKey(10000);
	return 0;

}

void gradientHist(Mat &magcell_8Px, Mat &dircell_8Px, int countColumn, int contimg) {

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
	//cout << magcell_8Px << "\n";
	//cout << dircell_8Px << "\n";
	/* Calcular Histograma */
	//calcHist(&dircell_8Px, 1, 0, Mat(), hist, 1, &histSize, &histRange, uniform, accumulate); //Calculate the 9 bin histogram.
	for (int row = 0; row < dircell_8Px.rows; row++) {
		uchar* datamag = magcell_8Px.ptr<uchar>(row);
		for (int col = 0; col < dircell_8Px.cols; col++) {
			if (dircell_8Px.at<char>(row, col) < 20) {
				int porc = (int)dircell_8Px.at<char>(row, col) * 100 / 20;
				hist.at<double>(0) += datamag[col] * (100 - porc) / 100;
				hist.at<double>(1) += datamag[col] * (porc) / 100;
				//printf("%d\n", hist.at<char>(2));
			}
			else if (dircell_8Px.at<char>(row, col) >= 20 && dircell_8Px.at<char>(row, col) < 40) {
				int porc = (int)(dircell_8Px.at<char>(row, col) - 20) * 100 / 20;
				//printf("porc: %d\n", porc);
				hist.at<double>(1) += datamag[col] * (100 - porc) / 100;
				hist.at<double>(2) += datamag[col] * porc / 100;
				//printf("pointmag: %d\n", datamag[col] * porc / 100);
				//printf("mag: %d\n", magcell_8Px.at<char>(row, col));
				//printf("Multip: %d\n", (int(magcell_8Px.at<char>(row, col)))*(porc));
				//cout << "hist: " << hist.at<double>(2) << "\n";
			}
			else if (dircell_8Px.at<char>(row, col) >= 40 && dircell_8Px.at<char>(row, col) < 60) {
				int porc = (int)(dircell_8Px.at<char>(row, col) - 40) * 100 / 20;
				hist.at<double>(2) += datamag[col] * (100 - porc) / 100;
				hist.at<double>(3) += datamag[col] * (porc) / 100;
			}
			else if (dircell_8Px.at<char>(row, col) >= 60 && dircell_8Px.at<char>(row, col) < 80) {
				int porc = (int)(dircell_8Px.at<char>(row, col) - 60) * 100 / 20;
				hist.at<double>(3) += datamag[col] * (100 - porc) / 100;
				hist.at<double>(4) += datamag[col] * (porc) / 100;
			}
			else if (dircell_8Px.at<char>(row, col) >= 80 && dircell_8Px.at<char>(row, col) < 100) {
				int porc = (int)(dircell_8Px.at<char>(row, col) - 80) * 100 / 20;
				hist.at<double>(4) += datamag[col] * (100 - porc) / 100;
				hist.at<double>(5) += datamag[col] * (porc) / 100;
			}
			else if (dircell_8Px.at<char>(row, col) >= 100 && dircell_8Px.at<char>(row, col) < 120) {
				int porc = (int)(dircell_8Px.at<char>(row, col) - 100) * 100 / 20;
				hist.at<double>(5) += datamag[col] * (100 - porc) / 100;
				hist.at<double>(6) += datamag[col] * (porc) / 100;
			}
			else if (dircell_8Px.at<char>(row, col) >= 120 && dircell_8Px.at<char>(row, col) < 140) {
				int porc = (int)(dircell_8Px.at<char>(row, col) - 120) * 100 / 20;
				hist.at<double>(6) += datamag[col] * (100 - porc) / 100;
				hist.at<double>(7) += datamag[col] * (porc) / 100;
			}
			else if (dircell_8Px.at<char>(row, col) >= 140 && dircell_8Px.at<char>(row, col) < 160) {
				int porc = (int)(dircell_8Px.at<char>(row, col) - 140) * 100 / 20;
				hist.at<double>(7) += datamag[col] * (100 - porc) / 100;
				hist.at<double>(8) += datamag[col] * (porc) / 100;
			}
			else if (dircell_8Px.at<char>(row, col) > 160) {
				int porc = (int)(dircell_8Px.at<char>(row, col) - 160) * 100 / 20;
				hist.at<double>(8) += datamag[col] * (100 - porc) / 100;
				hist.at<double>(0) += datamag[col] * (porc) / 100;
			}
		}
	}

	/*double maxVal = 0;
	double minVal = 0;
	minMaxLoc(hist, &minVal, &maxVal, 0, 0);
	printf("Max: %f\n", maxVal);
	printf("Min: %f\n", minVal);*/
	//normalize(hist, hist, 255, 0, NORM_MINMAX, -1, Mat());

	for (int i = 0; i < histSize; i++)
	{
		if (i*countColumn == 96) {
			feature << hist.at<double>(i) << "\n"; // Output the value of HOG to data file

		}
		else {
			feature << hist.at<double>(i) << ","; // Output the value of HOG to data file   
		}
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

void extraerCaracteristicas(Mat imagenGris) {

	Mat imagenGradiente_x, imagenGradiente_y, imagenGradiente_XY;
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
	hist8x8(mag, angle, 0);
	imagenGradiente_XY = (imagenGradiente_x + imagenGradiente_y) / 2;
	return;
}

int main() {

	Mat imagen, imagenGris, imagenPequena;
	Size blockSize(50, 100);
	Mat cell_100x200;
	int iter = 0;

	// Predecir //
	Mat data;
	Mat responses;
	string filename_to_save = " ";
	string filename_to_load = "weights.data";
	string data_filename = "Hist_Values.data";

	model = load_classifier<ANN_MLP>(filename_to_load);

	double min, max;
	if ((fp = fopen("datosDBP.txt", "w+")) == NULL) {
		printf("Cannot open file.\n");
		exit(1);
	}


	//imagen = imread(path.str().c_str()); //Uncomment for reading all database
	imagen = imread("dog-park-people.jpg"); //Comment for reading all database
	cvtColor(imagen, imagenGris, COLOR_BGR2GRAY);
	for (int col = 0; col < imagenGris.rows - 100; col += blockSize.height / 2)
	{
		for (int row = 0; row < imagenGris.cols - 50; row += blockSize.width / 2)
		{
			//std::stringstream pathtosave;
			Rect rect = Rect(row, col, blockSize.width, blockSize.height);
			cell_100x200 = Mat(imagenGris, rect); // Get a 100 by 200 cell from img.
			resize(imagenGris, imagenPequena, Size(16, 32), 0, 0, CV_INTER_AREA);
			extraerCaracteristicas(imagenPequena);
			bool ok = read_num_class_data(data_filename, 108, &data, &responses);
			Mat sample = data.row(iter);
			float r = model->predict(sample);
			if ((int)r == 0){
				rectangle(imagen, rect, Scalar(0, 255, 0), 2);
			}
			printf("Prediccion: %f\n", r);
			imshow("People", cell_100x200);
			waitKey();
			iter++;
		}
	}

	fclose(fp);
	printf("Done!");
	imshow("Clasificada", imagen);
	waitKey();
	getchar();
	return 0;
}