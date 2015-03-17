#include "opencv/cv.h"
#include "opencv/cvaux.h"
#include "opencv/highgui.h"
#include <string>
#include <cmath>
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/ml/ml.hpp"
#include <ctime>
#include <iostream>
#include <fstream>
#include <io.h>

using namespace std;
using namespace cv;



#define OPEN_TEMPLATE_FILE_FAIL 0
#define SAMPLE_NUM 45
#define DATA_NUM 622

#define POINT_WIDTH 480
#define POINT_HEIGHT 640

string face_cascade_name = "haarcascade_frontalface_alt.xml"; 
CascadeClassifier face_cascade; 

struct split_feature_load
{
	unsigned long idx1;
	unsigned long idx2;
	float thresh;
};
struct regression_tree_load
{
	std::vector<split_feature_load> splits;
	std::vector<std::vector<float> > leaf_values;
};
struct deltas_values{
	float a;
	float b;
};
struct LBF_model{
	std::vector<float> initial_shape;
	std::vector<std::vector<unsigned long> > anchor_idx;
	std::vector<std::vector<deltas_values > > deltas;
	std::vector<std::vector<regression_tree_load> > forests;
};


int LBF_Model_Load(string file_name, LBF_model& models/*shape_predictor sp*/){
	FILE *initial_shape_read = fopen(file_name.data(), "rb");
	if(NULL==initial_shape_read){
		cout<<"Can not open this file!"<<endl;
		return OPEN_TEMPLATE_FILE_FAIL;
	}

	/////////////////////////////////////////--------initial_shape
	int initial_shape_size = 0;
	fread(&initial_shape_size, sizeof(int), 1, initial_shape_read);
	std::vector<float> initial_shape(initial_shape_size/*sp.initial_shape.size()*/);
	for (int i = 0; i < initial_shape_size /*sp.initial_shape.size()*/; ++i)
	{
		fread(&initial_shape[i], sizeof(float), 1, initial_shape_read);
	}

	/////////////////////////////////////////--------anchor_idx
	int anchor_idx_size = 0;
	fread(&anchor_idx_size, sizeof(int), 1, initial_shape_read);
	int anchor_idx_size_size = 0;
	std::vector<std::vector<unsigned long> > anchor_idx(anchor_idx_size/*sp.anchor_idx.size()*/);
	for (int i = 0; i < anchor_idx_size /*sp.anchor_idx.size()*/; ++i)
	{
		fread(&anchor_idx_size_size, sizeof(int), 1, initial_shape_read);
		anchor_idx[i].resize(anchor_idx_size_size/*sp.anchor_idx[i].size()*/);
		for (int j = 0; j < anchor_idx_size_size/*sp.anchor_idx[i].size()*/; ++j)
		{
			fread(&anchor_idx[i][j], sizeof(unsigned long), 1, initial_shape_read);
		}
	}

	/////////////////////////////////////////--------deltas
	int deltas_size = 0;
	fread(&deltas_size, sizeof(int), 1, initial_shape_read);
	int deltas_size_size = 0;
	std::vector<std::vector<deltas_values > > deltas(deltas_size);
	for (int i = 0; i < deltas_size /*sp.deltas.size()*/; ++i)
	{
		fread(&deltas_size_size, sizeof(int), 1, initial_shape_read);
		deltas[i].resize(deltas_size_size);
		for (int j = 0; j < deltas_size_size /*sp.deltas[i].size()*/; ++j)
		{
			fread(&deltas[i][j].a, sizeof(float), 1, initial_shape_read);
			fread(&deltas[i][j].b, sizeof(float), 1, initial_shape_read);
		}
	}

	/////////////////////////////////////////--------forest
	int forest_size = 0;
	fread(&forest_size, sizeof(int), 1, initial_shape_read);
	int forests_size_size = 0;
	int forests_size_size_splits_size = 0;
	int forests_size_size_tree_leaf_size = 0;
	int forests_size_size_tree_leaf_size_size = 0;

	std::vector<std::vector<regression_tree_load> > forest_load(forest_size/*sp.forests.size()*/);
	for (int vector_one = 0; vector_one < forest_size /*sp.forests.size()*/; ++vector_one)
	{
		fread(&forests_size_size, sizeof(int), 1, initial_shape_read);
		forest_load[vector_one].resize(forests_size_size/*sp.forests[vector_one].size()*/);
		for (int vector_two = 0; vector_two < forests_size_size /*sp.forests[vector_one].size()*/; ++vector_two)
		{
			fread(&forests_size_size_splits_size, sizeof(int), 1, initial_shape_read);
			regression_tree_load* tree = &forest_load[vector_one][vector_two];
			(*tree).splits.resize(forests_size_size_splits_size/*sp.forests[vector_one][vector_two].splits.size()*/);
			for (int split_size = 0; split_size < forests_size_size_splits_size/*sp.forests[vector_one][vector_two].splits.size()*/; ++split_size)
			{
				fread(&(*tree).splits[split_size].idx1, sizeof(unsigned long), 1, initial_shape_read);
				fread(&(*tree).splits[split_size].idx2, sizeof(unsigned long), 1, initial_shape_read);
				fread(&(*tree).splits[split_size].thresh, sizeof(unsigned long), 1, initial_shape_read);
			}

			fread(&forests_size_size_tree_leaf_size, sizeof(int), 1, initial_shape_read);

			(*tree).leaf_values.resize(forests_size_size_tree_leaf_size/*sp.forests[vector_one][vector_two].leaf_values.size()*/);
			for (int leaf_values_size = 0; leaf_values_size < forests_size_size_tree_leaf_size /*sp.forests[vector_one][vector_two].leaf_values.size()*/; ++leaf_values_size)
			{
				fread(&forests_size_size_tree_leaf_size_size, sizeof(int), 1, initial_shape_read);
				(*tree).leaf_values[leaf_values_size].resize(forests_size_size_tree_leaf_size_size/*sp.forests[vector_one][vector_two].leaf_values[leaf_values_size].size()*/);
				for (int leaf_values_size_in = 0; leaf_values_size_in < forests_size_size_tree_leaf_size_size /*sp.forests[vector_one][vector_two].leaf_values[leaf_values_size].size()*/; ++leaf_values_size_in)
				{
					fread(&((*tree).leaf_values[leaf_values_size][leaf_values_size_in]), sizeof(float), 1, initial_shape_read);
				}

			}
		}
	}

	fclose(initial_shape_read);
	models.initial_shape = initial_shape;
	models.deltas = deltas;
	models.anchor_idx = anchor_idx;
	models.forests = forest_load;
	return 1;
}

template <class T>
struct point{
	T x;
	T y;
};

struct point_transform_affine{
	CvMat* m;
	point<double> b;
};

template <class T_1, class T_2>
double length_squared(point<T_1> point_1, point<T_2> point_2)
{
	return (((T_2)point_1.x - point_2.x)*((T_2)point_1.x - point_2.x) + ((T_2)point_1.y - point_2.y)*((T_2)point_1.y - point_2.y));
}

void find_similarity_transform (std::vector<point<float> >& from_points, std::vector<point<float> >& to_points, point_transform_affine& p)
{
	point<double> mean_from, mean_to;
	mean_from.x = 0;
	mean_from.y = 0;
	mean_to.x = 0;
	mean_to.y = 0;
	double sigma_from = 0, sigma_to = 0;
	CvMat* cov = cvCreateMat(2, 2, CV_64FC1);
	cvZero(cov);

	for (unsigned int i = 0; i < from_points.size(); ++i)
	{
		mean_from.x += from_points[i].x;
		mean_from.y += from_points[i].y;
		mean_to.x += to_points[i].x;
		mean_to.y += to_points[i].y;
	}

	mean_from.x /= from_points.size();
	mean_from.y /= from_points.size();
	mean_to.x /= to_points.size();
	mean_to.y /= to_points.size();

	double cov_temp[4] = {0, 0, 0, 0};
	for (unsigned long i = 0; i < from_points.size(); ++i)
	{
		sigma_from += length_squared<float, double>(from_points[i], mean_from);
		sigma_to += length_squared<float, double>(to_points[i], mean_to);
		cov_temp[0] += (to_points[i].x - mean_to.x)*(from_points[i].x - mean_from.x); 
		cov_temp[1] += (to_points[i].x - mean_to.x)*(from_points[i].y - mean_from.y);
		cov_temp[2] += (to_points[i].y - mean_to.y)*(from_points[i].x - mean_from.x);
		cov_temp[3] += (to_points[i].y - mean_to.y)*(from_points[i].y - mean_from.y);
	}
	sigma_from /= from_points.size();
	sigma_to   /= from_points.size();

	cov_temp[0] /= from_points.size();
	cov_temp[1] /= from_points.size();
	cov_temp[2] /= from_points.size();
	cov_temp[3] /= from_points.size();
	cvmSet(cov, 0, 0, cov_temp[0]);
	cvmSet(cov, 0, 1, cov_temp[1]);
	cvmSet(cov, 1, 0, cov_temp[2]);
	cvmSet(cov, 1, 1, cov_temp[3]);

	CvMat* u = cvCreateMat(2, 2, CV_64FC1);
	CvMat* v = cvCreateMat(2, 2, CV_64FC1);
	CvMat* d = cvCreateMat(2, 2, CV_64FC1);
	cvSVD(cov, d, u, v, CV_SVD_V_T);

	if (cvDet(cov) < 0)
	{
		if (cvmGet(d, 1, 1) < cvmGet(d, 0, 0))
		{
			cvmSet(u, 1, 1, -cvmGet(u, 1, 1));
			cvmSet(u, 0, 1, -cvmGet(u, 0, 1));
			cvmSet(d, 1, 1, -cvmGet(d, 1, 1));
			cvmSet(d, 0, 1, -cvmGet(d, 0, 1));
		}
		else
		{
			cvmSet(u, 0, 0, -cvmGet(u, 0, 0));
			cvmSet(u, 1, 0, -cvmGet(u, 1, 0));
			cvmSet(d, 0, 0, -cvmGet(d, 0, 0));
			cvmSet(d, 1, 0, -cvmGet(d, 1, 1));
		}
	}

	CvMat* r = cvCreateMat(2, 2, CV_64FC1);
	cvZero(r);
	cvMatMul(u, v, r);

	double c = 1; 
	if (sigma_from != 0)
		c = 1.0/sigma_from * (cvmGet(d, 0, 0) + cvmGet(d, 1, 1));
	point<double> t;

	cvmSet(r, 0, 0, cvmGet(r, 0, 0)*c);
	cvmSet(r, 0, 1, cvmGet(r, 0, 1)*c);
	cvmSet(r, 1, 0, cvmGet(r, 1, 0)*c);
	cvmSet(r, 1, 1, cvmGet(r, 1, 1)*c);

	t.x = mean_to.x - (cvmGet(r, 0, 0)*mean_from.x + cvmGet(r, 0, 1)*mean_from.y);
	t.y = mean_to.y - (cvmGet(r, 1, 0)*mean_from.x + cvmGet(r, 1, 1)*mean_from.y);

	p.m = cvCloneMat(r);
	p.b = t;

}

void unnormalizing_tform(double rect[4], point_transform_affine& p)
{
	std::vector<point<float> > from_points, to_points;
	point<float> temp;
	temp.x = rect[0]; temp.y = rect[1]; to_points.push_back(temp);
	temp.x = 0; temp.y = 0; from_points.push_back(temp);
	temp.x = rect[2]; temp.y = rect[1]; to_points.push_back(temp);
	temp.x = 1; temp.y = 0; from_points.push_back(temp);
	temp.x = rect[2]; temp.y = rect[3]; to_points.push_back(temp);
	temp.x = 1; temp.y = 1; from_points.push_back(temp);
	find_similarity_transform(from_points, to_points, p);
}


void find_tform_between_shapes(std::vector<float> from_shape, std::vector<float> to_shape, point_transform_affine& p)
{
	std::vector<point<float> > from_points, to_points;
	const unsigned long num = from_shape.size()/2;
	if (num == 1)
	{
		exit(1);
	}
	from_points.resize(num);
	to_points.resize(num);
	for (unsigned long i = 0; i < num; ++i)
	{
		from_points[i].x = from_shape[2*i];
		from_points[i].y = from_shape[2*i + 1];
		to_points[i].x = to_shape[2*i];
		to_points[i].y = to_shape[2*i + 1];
	}
	find_similarity_transform(from_points, to_points, p);
}

template <typename image_type>
void extract_feature_pixel_values (
	image_type& img_,
	double* rect,
	std::vector<float>& current_shape,
	std::vector<float>& reference_shape,
	std::vector<unsigned long>& reference_pixel_anchor_idx,
	std::vector<deltas_values >& reference_pixel_deltas,
	std::vector<float>& feature_pixel_values
	){
		point_transform_affine tform;
		find_tform_between_shapes(reference_shape, current_shape, tform);

		point_transform_affine tform_to_img;
		unnormalizing_tform(rect, tform_to_img);

		feature_pixel_values.resize(reference_pixel_deltas.size());
		for (unsigned long i = 0; i < feature_pixel_values.size(); ++i)
		{
			// Compute the point in the current shape corresponding to the i-th pixel and
			// then map it from the normalized shape space into pixel space.
			point<double> point_p;

			point_p.x = cvmGet(tform.m, 0, 0)*reference_pixel_deltas[i].a + cvmGet(tform.m, 0, 1)*reference_pixel_deltas[i].b + current_shape[reference_pixel_anchor_idx[i]*2];
			point_p.y = cvmGet(tform.m, 1, 0)*reference_pixel_deltas[i].a + cvmGet(tform.m, 1, 1)*reference_pixel_deltas[i].b + current_shape[reference_pixel_anchor_idx[i]*2 + 1];

			int point_temp_x = cvmGet(tform_to_img.m, 0, 0)*point_p.x + cvmGet(tform_to_img.m, 0, 1)*point_p.y + tform_to_img.b.x;
			int point_temp_y = cvmGet(tform_to_img.m, 1, 0)*point_p.x + cvmGet(tform_to_img.m, 1, 1)*point_p.y + tform_to_img.b.y;

			if (point_temp_x > 0 && point_temp_x < img_.cols && point_temp_y > 0 && point_temp_y < img_.rows)
			{
				//feature_pixel_values[i] = (float)(img_.at<unsigned char>(point_temp_y, point_temp_x));
				feature_pixel_values[i] = img_.template at<Vec3b>(point_temp_y, point_temp_x)[0]*0.2989 + img_.template at<Vec3b>(point_temp_y, point_temp_x)[1]*0.5870 + img_.template at<Vec3b>(point_temp_y, point_temp_x)[2]*0.1140;
				//feature_pixel_values[i] = (img_.at<Vec3b>(point_temp_y, point_temp_x)[0] + img_.at<Vec3b>(point_temp_y, point_temp_x)[1] + img_.at<Vec3b>(point_temp_y, point_temp_x)[2])/3;
			}
			else
			{
				feature_pixel_values[i] = 0;
			}
		}

}

std::vector<float>& get_delta_shape(regression_tree_load& tree,const std::vector<float>& feature_pixel_values)
{
	unsigned long i = 0;
	float temp = 0.0f;
	while (i < tree.splits.size())
	{
		//temp = feature_pixel_values[tree.splits[i].idx1] > feature_pixel_values[tree.splits[i].idx2] ? feature_pixel_values[tree.splits[i].idx1] : feature_pixel_values[tree.splits[i].idx2];
		if ((feature_pixel_values[tree.splits[i].idx1] - feature_pixel_values[tree.splits[i].idx2])/* / temp */> tree.splits[i].thresh)
			i = 2*i + 1;
		else
			i = 2*i + 2;
	}
	return tree.leaf_values[i - tree.splits.size()];
}

void change_shape(std::vector<float>& current_shape, std::vector<float>& delta_shape)
{
	for (int i = 0; i < delta_shape.size(); ++i)
	{
		current_shape[i] += delta_shape[i];
	}
}

template <typename image_type>
void calculate_shape(image_type& img, double* rect, LBF_model& model, std::vector<point<double> >& parts)
{
	std::vector<float> current_shape = model.initial_shape;
	std::vector<float> feature_pixel_values;

	for (int iter = 0; iter < model.forests.size()/* - 12*/; ++iter)
	{
		extract_feature_pixel_values(img, rect, current_shape, model.initial_shape, model.anchor_idx[iter], model.deltas[iter], feature_pixel_values);

		for (unsigned long i = 0; i < model.forests[iter].size(); ++i)
		{
			change_shape(current_shape, get_delta_shape(model.forests[iter][i], feature_pixel_values));
		}
	}

	point_transform_affine tform_to_img;
	unnormalizing_tform(rect, tform_to_img);
	//std::vector<point<double> > parts(current_shape.size()/2);
	parts.resize(current_shape.size()/2);
	for (unsigned long i = 0; i < parts.size(); ++i)
	{
		parts[i].x = cvmGet(tform_to_img.m, 0, 0)*current_shape[2*i] + cvmGet(tform_to_img.m, 0, 1)*current_shape[2*i + 1] + tform_to_img.b.x;
		parts[i].y = cvmGet(tform_to_img.m, 1, 0)*current_shape[2*i] + cvmGet(tform_to_img.m, 1, 1)*current_shape[2*i + 1] + tform_to_img.b.y;
		//cout<<parts[i].x<<"	"<<parts[i].y<<endl;
	}
	//cout<<"ok"<<endl;
}

void get_rectangles(std::vector<point<float> >& temp, double* box)
{
	double min_x = temp[0].x;
	double max_x = temp[0].x;
	double min_y = temp[0].y;
	double max_y = temp[0].y;
	for (int i = 0; i < temp.size(); ++i)
	{
		if (min_x > temp[i].x)
		{
			min_x = temp[i].x;
		}
		if (max_x < temp[i].x)
		{
			max_x = temp[i].x;
		}
		if (min_y > temp[i].y)
		{
			min_y = temp[i].y;
		}
		if (max_y < temp[i].y)
		{
			max_y = temp[i].y;
		}
	}
	box[0] = min_x;
	box[1] = min_y;
	box[2] = max_x;
	box[3] = max_y;

}

double compute_error(std::vector<point<double> >& pre, std::vector<point<float> > gt)
{
	double left_eye_x = 0;
	double left_eye_y = 0;
	for (int i = 36; i <= 41; ++i)
	{
		left_eye_x += gt[i].x;
		left_eye_y += gt[i].y;
	}
	left_eye_x /= 6;
	left_eye_y /= 6;

	double right_eye_x = 0;
	double right_eye_y = 0;
	for (int i = 42; i <= 47; ++i)
	{
		right_eye_x += gt[i].x;
		right_eye_y += gt[i].y;
	}
	right_eye_x /= 6;
	right_eye_y /= 6;

	double scalar = sqrt((left_eye_x - right_eye_x) * (left_eye_x - right_eye_x) + (left_eye_y - right_eye_y) * (left_eye_y - right_eye_y));

	double count = 0;
	for (int i = 0; i < gt.size(); ++i)
	{
		count += sqrt((pre[i].x - gt[i].x) * (pre[i].x - gt[i].x) + (pre[i].y - gt[i].y) * (pre[i].y - gt[i].y)) / scalar;
	}
	count /= gt.size();
	return count;
}

void on_tracker(int, void* )  //滑块所对应的操作函数   对图像进行线性混合 
{
	return;
}
LBF_model model;


void SVM_train()
{

}

void findPointsBorder(vector<Point> &points,Point &ul,Point &dr)
{
	double max_point_x,min_point_x,max_point_y,min_point_y;
	max_point_x=points[0].x;
	min_point_x=points[0].x;
	max_point_y=points[0].y;
	min_point_y=points[0].y;
	for (int i = 0; i < points.size(); i++)
	{
		if (max_point_x<points[i].x)
			max_point_x=points[i].x;
		if (min_point_x>points[i].x)
			min_point_x=points[i].x;
		if (max_point_y<points[i].y)
			max_point_y=points[i].y;
		if (min_point_y>points[i].y)
			min_point_y=points[i].y;
	}
	ul.x=min_point_x;
	ul.y=min_point_y;
	dr.x=max_point_x;
	dr.y=max_point_y;
}
int getTrainFeature(Mat &training_mat,Mat &labels)
{

	string img_list[SAMPLE_NUM];
	int files_count=0;
	int max_point_x,min_point_x,max_point_y,min_point_y;

	// get image list
	string img_path="training_img";
	string exten = "*";
	string fileFolder = img_path+"\\*."+exten;
	char fileName[2000];
	struct  _finddata_t fileInfo;
	long findResult = _findfirst(fileFolder.c_str(),&fileInfo);
	if (findResult == -1)
	{
		_findclose(findResult);
		return -1;
	}

	do
	{
		sprintf(fileName,"%s\\%s",img_path.c_str(),fileInfo.name);
		if (fileInfo.attrib==_A_ARCH)
		{
			img_list[files_count++]=fileName;
		}

	}while (!_findnext(findResult,&fileInfo));
	_findclose(findResult); 



	for (int i = 0; i < SAMPLE_NUM; i++)
	{
		if (img_list[i].find("happy")!=string::npos)
		{
			labels.at<float>(i,0)=0;
		}else if ((img_list[i].find("normal")!=string::npos))
		{
			labels.at<float>(i,0)=1;
		}else if ((img_list[i].find("sad")!=string::npos))
		{
			labels.at<float>(i,0)=2;
		}else
			labels.at<float>(i,0)=0;
	}



	double box[4];
	vector<point<double>> parts;
	vector<Point> fixed_point;
	vector<Point> point_eyebrow;
	vector<Point> point_leye;
	vector<Point> point_reye;
	vector<Point> point_mouth;
	vector<Point> point_nose;

	Mat landMark_brow(300,300,CV_8UC3,Scalar(0,0,0));
	Mat landMark_leye(300,300,CV_8UC3,Scalar(0,0,0));
	Mat landMark_reye(300,300,CV_8UC3,Scalar(0,0,0));
	Mat landMark_mouth(300,300,CV_8UC3,Scalar(0,0,0));
	Mat landMark_nose(300,300,CV_8UC3,Scalar(0,0,0));

	for (int k = 0; k < SAMPLE_NUM; k++)
	{
		fixed_point.clear();
		Mat frame=imread(img_list[k]);
		cout<<img_list[k]<<endl;
		if( !face_cascade.load( face_cascade_name ) ){  
			printf("级联分类器错误，可能未找到文件，拷贝该文件到工程目录下！\n");
			return -1;
		}
		std::vector<Rect> faces;
		face_cascade.detectMultiScale( frame, faces, 1.15, 5, 0);

		for( int i = 0; i < faces.size(); i++ ){

			box[0] = faces[i].x;
			box[1] = faces[i].y;
			box[2] = (box[0] + faces[i].width);
			box[3] = (box[1] + faces[i].height);
			calculate_shape<Mat>(frame, box, model, parts);

			max_point_x=parts[0].x;
			min_point_x=parts[0].x;
			max_point_y=parts[0].y;
			min_point_y=parts[0].y;

			for (int j = 0; j < parts.size(); ++j)
			{
				if (max_point_x<parts[j].x)
					max_point_x=parts[j].x;
				if (min_point_x>parts[j].x)
					min_point_x=parts[j].x;
				if (max_point_y<parts[j].y)
					max_point_y=parts[j].y;
				if (min_point_y>parts[j].y)
					min_point_y=parts[j].y;
				Point center( parts[j].x, parts[j].y ); 
				ellipse( frame, center, Size( 1, 1), 0, 0, 0, Scalar( 255, 0, 255 ), 4, 8, 0); 
				char c[3];
				sprintf(c, "%d", j);
				string words= c;  
				putText( frame, words, center,CV_FONT_HERSHEY_COMPLEX, 0.3, Scalar(255, 0, 0)); 
			}

			int l_width=max_point_x-min_point_x;
			int l_height = max_point_y-min_point_y;
			double point_scale=(POINT_WIDTH*1.0/l_width)>(POINT_HEIGHT*1.0/l_height)?(POINT_WIDTH*1.0/l_width):(POINT_HEIGHT*1.0/l_height);

			Mat landMark_face(POINT_HEIGHT,POINT_WIDTH+300,CV_8UC3,Scalar(0,0,0));
			Mat landMark_face_fixed(POINT_HEIGHT,POINT_WIDTH+300,CV_8UC3,Scalar(0,0,0));



			for (int j = 0; j < parts.size(); j++)
			{
				Point center( (parts[j].x-min_point_x)*point_scale, (parts[j].y-min_point_y)*point_scale); 
				ellipse( landMark_face, center, Size( 1, 1), 0, 0, 0, Scalar( 255, 255, 255 ), 4, 8, 0 ); 
				char c[3];
				sprintf(c, "%d", j);
				string words= c;  
				putText( landMark_face, words, center, CV_FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255, 0, 0)); 
				imshow("landMark_face", landMark_face);

			}

			int  turnDir= parts[36].y>parts[45].y?0:1;
			double abs_x_min = abs(parts[45].x-parts[36].x);
			double abs_y_min = abs(parts[45].y-parts[36].y);
			double abs_h = sqrt(abs_x_min*abs_x_min+abs_y_min*abs_y_min);
			double cosbeta;
			double beta;
			cosbeta = abs_x_min/abs_h;
			beta=acos(cosbeta);   //Roate angle

			//x1=cos(angle)*x-sin(angle)*y;
			//y1=cos(angle)*y+sin(angle)*x;

			double nx,ny;
			double cx,cy;
			cx = (parts[27].x-min_point_x)*point_scale;
			cy = (parts[27].y-min_point_y)*point_scale;

			if (turnDir==0) //image needs to roate clockwise
			{
				nx = cos(beta)*cx-sin(beta)*cy;
				ny = cos(beta)*cy+sin(beta)*cx;
			}
			else //image needs to roate anticlockwise
			{
				nx = cos(-beta)*cx-sin(-beta)*cy;
				ny = cos(-beta)*cy+sin(-beta)*cx;
			}
			double gx = cx-nx;
			double gy = cy-ny;

			for (int j = 0; j < parts.size(); j++)
			{
				Point center( (parts[j].x-min_point_x)*point_scale, (parts[j].y-min_point_y)*point_scale); 
				if (turnDir==0) //image needs to roate clockwise
				{
					nx = cos(beta)*center.x-sin(beta)*center.y;
					ny = cos(beta)*center.y+sin(beta)*center.x;
				}
				else //image needs to roate anticlockwise
				{
					nx = cos(-beta)*center.x-sin(-beta)*center.y;
					ny = cos(-beta)*center.y+sin(-beta)*center.x;
				}

				center.x = nx+gx;
				center.y = ny+gy;

				//fixed_point[j].x=center.x;
				//fixed_point[j].y=center.y;
				fixed_point.push_back(center);
				ellipse( landMark_face_fixed, center, Size( 1, 1), 0, 0, 0, Scalar( 255, 255, 255 ), 4, 8, 0 ); 
				imshow("landMark_face_fixed", landMark_face_fixed);
			}


			double brow_scale;

			point_eyebrow.clear();
			point_leye.clear();
			point_reye.clear();
			point_mouth.clear();
			point_nose.clear();

			//get all point 
			//eyebrow
			for (int j = 17; j < 27; j++)
			{
				//point_eyebrow[j-17].x=fixed_point[j].x;
				//point_eyebrow[j-17].y=fixed_point[j].y;
				point_eyebrow.push_back(fixed_point[j]);

			}

			//Left eye
			for (int j = 36; j < 42; j++)
			{
				//point_leye[j-36].x=fixed_point[j].x;
				//point_leye[j-36].y=fixed_point[j].y;
				point_leye.push_back(fixed_point[j]);
			}

			//right eye
			for (int j = 42; j < 48; j++)
			{
				//point_reye[j-42].x=fixed_point[j].x;
				//point_reye[j-42].y=fixed_point[j].y;
				point_reye.push_back(fixed_point[j]);
			}

			// mouth
			for (int j = 48; j < parts.size(); j++)
			{
				//point_mouth[j-48].x=fixed_point[j].x;
				//point_mouth[j-48].y=fixed_point[j].y;
				point_mouth.push_back(fixed_point[j]);
			}

			//nose
			for (int j = 31; j < 35; j++)
			{
				//point_nose[j-31].x=fixed_point[j].x;
				//point_nose[j-31].y=fixed_point[j].y;
				point_nose.push_back(fixed_point[j]);
			}

			Point xbrow,ybrow,xreye,yreye,xleye,yleye,xmouth,ymouth,xnose,ynose;
			findPointsBorder(point_eyebrow,xbrow,ybrow);
			findPointsBorder(point_reye,xreye,yreye);
			findPointsBorder(point_leye,xleye,yleye);
			findPointsBorder(point_mouth,xmouth,ymouth);
			findPointsBorder(point_nose,xnose,ynose);

			/*Mat landMark_brow(ybrow.y-xbrow.y,ybrow.x-xbrow.x,CV_8UC3,Scalar(0,0,0));
			Mat landMark_leye(yleye.y-xleye.y,200,CV_8UC3,Scalar(0,0,0));
			Mat landMark_reye(yreye.y-xreye.y,200,CV_8UC3,Scalar(0,0,0));
			Mat landMark_mouth(ymouth.y-xmouth.y,ymouth.x-xmouth.x,CV_8UC3,Scalar(0,0,0));
			Mat landMark_nose(ynose.y-xnose.y,200,CV_8UC3,Scalar(0,0,0));*/


			//for image show
			//eyebrow
			for (int j = 0; j < point_eyebrow.size(); j++)
			{
				Point center(point_eyebrow[j].x-xbrow.x,point_eyebrow[j].y-xbrow.y); 
				ellipse( landMark_brow, center, Size( 1, 1), 0, 0, 0, Scalar( 255, 255, 255 ), 4, 8, 0 ); 
				imshow("landMark_brow", landMark_brow);
			}

			//Left eye
			for (int j = 0; j < point_leye.size(); j++)
			{
				Point center(point_leye[j].x-xleye.x,point_leye[j].y-xleye.y); 
				ellipse( landMark_leye, center, Size( 1, 1), 0, 0, 0, Scalar( 255, 255, 255 ), 4, 8, 0 ); 
				imshow("landMark_leye", landMark_leye);
			}

			//right eye
			for (int j = 0; j < point_reye.size(); j++)
			{
				Point center(point_reye[j].x-xreye.x,point_reye[j].y-xreye.y); 
				ellipse( landMark_reye, center, Size( 1, 1), 0, 0, 0, Scalar( 255, 255, 255 ), 4, 8, 0 ); 
				imshow("landMark_reye", landMark_reye);
			}

			// mouth
			for (int j = 0; j < point_mouth.size(); j++)
			{
				Point center(point_mouth[j].x-xmouth.x,point_mouth[j].y-xmouth.y); 
				ellipse( landMark_mouth, center, Size( 1, 1), 0, 0, 0, Scalar( 255, 255, 255 ), 4, 8, 0 ); 
				imshow("landMark_mouth", landMark_mouth);
			}

			//nose
			for (int j = 0; j < point_nose.size(); j++)
			{
				Point center(point_nose[j].x-xnose.x,point_nose[j].y-xnose.y); 
				ellipse( landMark_nose, center, Size( 1, 1), 0, 0, 0, Scalar( 255, 255, 255 ), 4, 8, 0 ); 
				imshow("landMark_nose", landMark_nose);
			}


			//vector<> for point feature
			//	
			vector<double> vec_mouth,vec_leye,vec_reye,vec_eyebrow;
			for (int j = 0; j < point_mouth.size()-1; j++)
			{
				for (int l = j+1; l < point_mouth.size(); l++)
				{
					double distance_x,distance_y,distance_h;
					distance_x=point_mouth[j].x - point_mouth[l].x;
					distance_y=point_mouth[j].y - point_mouth[l].y;
					distance_h=sqrt(distance_x*distance_x+distance_y*distance_y);
					vec_mouth.push_back(distance_x/distance_h);
					vec_mouth.push_back(distance_y/distance_h);
				}
			}

			for (int j = 0; j < point_leye.size()-1; j++)
			{
				for (int l = j+1; l < point_leye.size(); l++)
				{
					double distance_x,distance_y,distance_h;
					distance_x=point_leye[j].x - point_leye[l].x;
					distance_y=point_leye[j].y - point_leye[l].y;
					distance_h=sqrt(distance_x*distance_x+distance_y*distance_y);
					//vec_leye.push_back(distance_x/distance_h);
					//vec_leye.push_back(distance_y/distance_h);
					vec_leye.push_back(distance_x);
					vec_leye.push_back(distance_y);
				}
			}

			for (int j = 0; j < point_reye.size()-1; j++)
			{
				for (int l = j+1; l < point_reye.size(); l++)
				{
					double distance_x,distance_y,distance_h;
					distance_x=point_reye[j].x - point_reye[l].x;
					distance_y=point_reye[j].y - point_reye[l].y;
					distance_h=sqrt(distance_x*distance_x+distance_y*distance_y);
					//vec_reye.push_back(distance_x/distance_h);
					//vec_reye.push_back(distance_y/distance_h);
					vec_reye.push_back(distance_x);
					vec_reye.push_back(distance_y);
				}
			}

			for (int j = 0; j < point_eyebrow.size()-1; j++)
			{
				for (int l = j+1; l < point_eyebrow.size(); l++)
				{
					double distance_x,distance_y,distance_h;
					distance_x=point_eyebrow[j].x - point_eyebrow[l].x;
					distance_y=point_eyebrow[j].y - point_eyebrow[l].y;
					distance_h=sqrt(distance_x*distance_x+distance_y*distance_y);
					//vec_eyebrow.push_back(distance_x/distance_h);
					//vec_eyebrow.push_back(distance_y/distance_h);
					vec_eyebrow.push_back(distance_x);
					vec_eyebrow.push_back(distance_y);
				}
			}


			//int totalDimension=
			//	(point_mouth.size()+point_leye.size()+point_reye.size()+point_eyebrow.size()+point_nose.size())*2+
			//	vec_mouth.size()+vec_leye.size()+vec_reye.size()+vec_eyebrow.size();

			int totalDimension=
				(point_mouth.size()+point_leye.size()+point_reye.size()+point_eyebrow.size()+point_nose.size())*2+
				vec_mouth.size()+vec_leye.size()+vec_reye.size()+vec_eyebrow.size();

			cout<<totalDimension<<endl;
			int currentDim=0;
			
			// push into training Mat
			for (int j = 0; j < point_eyebrow.size(); j++)
			{
				training_mat.at<float>(k,currentDim++)=point_eyebrow[j].x;
				training_mat.at<float>(k,currentDim++)=point_eyebrow[j].y;
			}

			//Left eye
			for (int j = 0; j < point_leye.size(); j++)
			{
				training_mat.at<float>(k,currentDim++)=point_leye[j].x;
				training_mat.at<float>(k,currentDim++)=point_leye[j].y;
			}

			//right eye
			for (int j = 0; j < point_reye.size(); j++)
			{
				training_mat.at<float>(k,currentDim++)=point_reye[j].x;
				training_mat.at<float>(k,currentDim++)=point_reye[j].y;
			}

			// mouth
			for (int j = 0; j < point_mouth.size(); j++)
			{
				training_mat.at<float>(k,currentDim++)=point_mouth[j].x;
				training_mat.at<float>(k,currentDim++)=point_mouth[j].y;
			}

			//nose
			for (int j = 0; j < point_nose.size(); j++)
			{
				training_mat.at<float>(k,currentDim++)=point_nose[j].x;
				training_mat.at<float>(k,currentDim++)=point_nose[j].y;
			}
			

			// vector add into Mat

			for (int j = 0; j < vec_mouth.size(); j++)
			{
				training_mat.at<float>(k,currentDim)=vec_mouth[j];
				currentDim++;
			}
			for (int j = 0; j < vec_eyebrow.size(); j++)
			{
				training_mat.at<float>(k,currentDim)=vec_eyebrow[j];
				currentDim++;
			}
			for (int j = 0; j < vec_leye.size(); j++)
			{
				training_mat.at<float>(k,currentDim)=vec_leye[j];
				currentDim++;
			}
			for (int j = 0; j < vec_reye.size(); j++)
			{
				training_mat.at<float>(k,currentDim)=vec_reye[j];
				currentDim++;
			}

			cout<<currentDim<<endl;

			rectangle(frame, faces[i], Scalar( 255, 0, 0), 2,7, 0);

		}

		imshow("Train",frame);
		waitKey(30);

	}

	ofstream out("out.txt");
	out<<labels<<endl;
	out<<training_mat<<endl;
	out.close();

}

int getPredictFeature(Mat &training_mat,Mat &labels,int &predict_count)
{

	string img_list[100];
	int max_point_x,min_point_x,max_point_y,min_point_y;

	// get image list
	string img_path="predict_img";
	string exten = "*";
	string fileFolder = img_path+"\\*."+exten;
	char fileName[2000];
	struct  _finddata_t fileInfo;
	long findResult = _findfirst(fileFolder.c_str(),&fileInfo);
	if (findResult == -1)
	{
		_findclose(findResult);
		return -1;
	}
	int count=0;
	do
	{
		sprintf(fileName,"%s\\%s",img_path.c_str(),fileInfo.name);
		if (fileInfo.attrib==_A_ARCH)
		{
			img_list[count++]=fileName;
		}

	}while (!_findnext(findResult,&fileInfo));
	_findclose(findResult); 

	predict_count=count;
	for (int i = 0; i < count; i++)
	{
		if (img_list[i].find("happy")!=string::npos)
		{
			labels.at<float>(i,0)=0;
		}else if ((img_list[i].find("normal")!=string::npos))
		{
			labels.at<float>(i,0)=1;
		}else if ((img_list[i].find("sad")!=string::npos))
		{
			labels.at<float>(i,0)=2;
		}else
			labels.at<float>(i,0)=0;
	}

	double box[4];
	vector<point<double>> parts;
	vector<Point> fixed_point;
	vector<Point> point_eyebrow;
	vector<Point> point_leye;
	vector<Point> point_reye;
	vector<Point> point_mouth;
	vector<Point> point_nose;

	Mat landMark_brow(300,300,CV_8UC3,Scalar(0,0,0));
	Mat landMark_leye(300,300,CV_8UC3,Scalar(0,0,0));
	Mat landMark_reye(300,300,CV_8UC3,Scalar(0,0,0));
	Mat landMark_mouth(300,300,CV_8UC3,Scalar(0,0,0));
	Mat landMark_nose(300,300,CV_8UC3,Scalar(0,0,0));

	for (int k = 0; k < predict_count; k++)
	{
		fixed_point.clear();
		Mat frame=imread(img_list[k]);
		cout<<img_list[k]<<endl;
		if( !face_cascade.load( face_cascade_name ) ){  
			printf("级联分类器错误，可能未找到文件，拷贝该文件到工程目录下！\n");
			return -1;
		}
		std::vector<Rect> faces;
		face_cascade.detectMultiScale( frame, faces, 1.15, 5, 0);

		for( int i = 0; i < faces.size(); i++ ){

			box[0] = faces[i].x;
			box[1] = faces[i].y;
			box[2] = (box[0] + faces[i].width);
			box[3] = (box[1] + faces[i].height);
			calculate_shape<Mat>(frame, box, model, parts);

			max_point_x=parts[0].x;
			min_point_x=parts[0].x;
			max_point_y=parts[0].y;
			min_point_y=parts[0].y;

			for (int j = 0; j < parts.size(); ++j)
			{
				if (max_point_x<parts[j].x)
					max_point_x=parts[j].x;
				if (min_point_x>parts[j].x)
					min_point_x=parts[j].x;
				if (max_point_y<parts[j].y)
					max_point_y=parts[j].y;
				if (min_point_y>parts[j].y)
					min_point_y=parts[j].y;
				Point center( parts[j].x, parts[j].y ); 
				ellipse( frame, center, Size( 1, 1), 0, 0, 0, Scalar( 255, 0, 255 ), 4, 8, 0); 
				char c[3];
				sprintf(c, "%d", j);
				string words= c;  
				putText( frame, words, center,CV_FONT_HERSHEY_COMPLEX, 0.3, Scalar(255, 0, 0)); 
			}

			int l_width=max_point_x-min_point_x;
			int l_height = max_point_y-min_point_y;
			double point_scale=(POINT_WIDTH*1.0/l_width)>(POINT_HEIGHT*1.0/l_height)?(POINT_WIDTH*1.0/l_width):(POINT_HEIGHT*1.0/l_height);

			Mat landMark_face(POINT_HEIGHT,POINT_WIDTH+300,CV_8UC3,Scalar(0,0,0));
			Mat landMark_face_fixed(POINT_HEIGHT,POINT_WIDTH+300,CV_8UC3,Scalar(0,0,0));



			for (int j = 0; j < parts.size(); j++)
			{
				Point center( (parts[j].x-min_point_x)*point_scale, (parts[j].y-min_point_y)*point_scale); 
				ellipse( landMark_face, center, Size( 1, 1), 0, 0, 0, Scalar( 255, 255, 255 ), 4, 8, 0 ); 
				char c[3];
				sprintf(c, "%d", j);
				string words= c;  
				putText( landMark_face, words, center, CV_FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255, 0, 0)); 
				imshow("landMark_face", landMark_face);

			}

			int  turnDir= parts[36].y>parts[45].y?0:1;
			double abs_x_min = abs(parts[45].x-parts[36].x);
			double abs_y_min = abs(parts[45].y-parts[36].y);
			double abs_h = sqrt(abs_x_min*abs_x_min+abs_y_min*abs_y_min);
			double cosbeta;
			double beta;
			cosbeta = abs_x_min/abs_h;
			beta=acos(cosbeta);   //Roate angle

			//x1=cos(angle)*x-sin(angle)*y;
			//y1=cos(angle)*y+sin(angle)*x;

			double nx,ny;
			double cx,cy;
			cx = (parts[27].x-min_point_x)*point_scale;
			cy = (parts[27].y-min_point_y)*point_scale;

			if (turnDir==0) //image needs to roate clockwise
			{
				nx = cos(beta)*cx-sin(beta)*cy;
				ny = cos(beta)*cy+sin(beta)*cx;
			}
			else //image needs to roate anticlockwise
			{
				nx = cos(-beta)*cx-sin(-beta)*cy;
				ny = cos(-beta)*cy+sin(-beta)*cx;
			}
			double gx = cx-nx;
			double gy = cy-ny;

			for (int j = 0; j < parts.size(); j++)
			{
				Point center( (parts[j].x-min_point_x)*point_scale, (parts[j].y-min_point_y)*point_scale); 
				if (turnDir==0) //image needs to roate clockwise
				{
					nx = cos(beta)*center.x-sin(beta)*center.y;
					ny = cos(beta)*center.y+sin(beta)*center.x;
				}
				else //image needs to roate anticlockwise
				{
					nx = cos(-beta)*center.x-sin(-beta)*center.y;
					ny = cos(-beta)*center.y+sin(-beta)*center.x;
				}

				center.x = nx+gx;
				center.y = ny+gy;

				//fixed_point[j].x=center.x;
				//fixed_point[j].y=center.y;
				fixed_point.push_back(center);
				ellipse( landMark_face_fixed, center, Size( 1, 1), 0, 0, 0, Scalar( 255, 255, 255 ), 4, 8, 0 ); 
				imshow("landMark_face_fixed", landMark_face_fixed);
			}


			double brow_scale;

			point_eyebrow.clear();
			point_leye.clear();
			point_reye.clear();
			point_mouth.clear();
			point_nose.clear();

			//get all point 
			//eyebrow
			for (int j = 17; j < 27; j++)
			{
				//point_eyebrow[j-17].x=fixed_point[j].x;
				//point_eyebrow[j-17].y=fixed_point[j].y;
				point_eyebrow.push_back(fixed_point[j]);

			}

			//Left eye
			for (int j = 36; j < 42; j++)
			{
				//point_leye[j-36].x=fixed_point[j].x;
				//point_leye[j-36].y=fixed_point[j].y;
				point_leye.push_back(fixed_point[j]);
			}

			//right eye
			for (int j = 42; j < 48; j++)
			{
				//point_reye[j-42].x=fixed_point[j].x;
				//point_reye[j-42].y=fixed_point[j].y;
				point_reye.push_back(fixed_point[j]);
			}

			// mouth
			for (int j = 48; j < parts.size(); j++)
			{
				//point_mouth[j-48].x=fixed_point[j].x;
				//point_mouth[j-48].y=fixed_point[j].y;
				point_mouth.push_back(fixed_point[j]);
			}

			//nose
			for (int j = 31; j < 35; j++)
			{
				//point_nose[j-31].x=fixed_point[j].x;
				//point_nose[j-31].y=fixed_point[j].y;
				point_nose.push_back(fixed_point[j]);
			}

			Point xbrow,ybrow,xreye,yreye,xleye,yleye,xmouth,ymouth,xnose,ynose;
			findPointsBorder(point_eyebrow,xbrow,ybrow);
			findPointsBorder(point_reye,xreye,yreye);
			findPointsBorder(point_leye,xleye,yleye);
			findPointsBorder(point_mouth,xmouth,ymouth);
			findPointsBorder(point_nose,xnose,ynose);

			/*Mat landMark_brow(ybrow.y-xbrow.y,ybrow.x-xbrow.x,CV_8UC3,Scalar(0,0,0));
			Mat landMark_leye(yleye.y-xleye.y,200,CV_8UC3,Scalar(0,0,0));
			Mat landMark_reye(yreye.y-xreye.y,200,CV_8UC3,Scalar(0,0,0));
			Mat landMark_mouth(ymouth.y-xmouth.y,ymouth.x-xmouth.x,CV_8UC3,Scalar(0,0,0));
			Mat landMark_nose(ynose.y-xnose.y,200,CV_8UC3,Scalar(0,0,0));*/



			//vector<> for point feature
			//	
			vector<double> vec_mouth,vec_leye,vec_reye,vec_eyebrow;
			for (int j = 0; j < point_mouth.size()-1; j++)
			{
				for (int l = j+1; l < point_mouth.size(); l++)
				{
					double distance_x,distance_y,distance_h;
					distance_x=point_mouth[j].x - point_mouth[l].x;
					distance_y=point_mouth[j].y - point_mouth[l].y;
					distance_h=sqrt(distance_x*distance_x+distance_y*distance_y);
					vec_mouth.push_back(distance_x/distance_h);
					vec_mouth.push_back(distance_y/distance_h);
				}
			}

			for (int j = 0; j < point_leye.size()-1; j++)
			{
				for (int l = j+1; l < point_leye.size(); l++)
				{
					double distance_x,distance_y,distance_h;
					distance_x=point_leye[j].x - point_leye[l].x;
					distance_y=point_leye[j].y - point_leye[l].y;
					distance_h=sqrt(distance_x*distance_x+distance_y*distance_y);
					//vec_leye.push_back(distance_x/distance_h);
					//vec_leye.push_back(distance_y/distance_h);
					vec_leye.push_back(distance_x);
					vec_leye.push_back(distance_y);
				}
			}

			for (int j = 0; j < point_reye.size()-1; j++)
			{
				for (int l = j+1; l < point_reye.size(); l++)
				{
					double distance_x,distance_y,distance_h;
					distance_x=point_reye[j].x - point_reye[l].x;
					distance_y=point_reye[j].y - point_reye[l].y;
					distance_h=sqrt(distance_x*distance_x+distance_y*distance_y);
					//vec_reye.push_back(distance_x/distance_h);
					//vec_reye.push_back(distance_y/distance_h);
					vec_reye.push_back(distance_x);
					vec_reye.push_back(distance_y);
				}
			}

			for (int j = 0; j < point_eyebrow.size()-1; j++)
			{
				for (int l = j+1; l < point_eyebrow.size(); l++)
				{
					double distance_x,distance_y,distance_h;
					distance_x=point_eyebrow[j].x - point_eyebrow[l].x;
					distance_y=point_eyebrow[j].y - point_eyebrow[l].y;
					distance_h=sqrt(distance_x*distance_x+distance_y*distance_y);
					//vec_eyebrow.push_back(distance_x/distance_h);
					//vec_eyebrow.push_back(distance_y/distance_h);
					vec_eyebrow.push_back(distance_x);
					vec_eyebrow.push_back(distance_y);
				}
			}


			//int totalDimension=
			//	(point_mouth.size()+point_leye.size()+point_reye.size()+point_eyebrow.size()+point_nose.size())*2+
			//	vec_mouth.size()+vec_leye.size()+vec_reye.size()+vec_eyebrow.size();

			int totalDimension=
				(point_mouth.size()+point_leye.size()+point_reye.size()+point_eyebrow.size()+point_nose.size())*2+
				vec_mouth.size()+vec_leye.size()+vec_reye.size()+vec_eyebrow.size();

			cout<<totalDimension<<endl;
			int currentDim=0;

			
			// push into training Mat
			for (int j = 0; j < point_eyebrow.size(); j++)
			{
				training_mat.at<float>(k,currentDim++)=point_eyebrow[j].x;
				training_mat.at<float>(k,currentDim++)=point_eyebrow[j].y;
			}

			//Left eye
			for (int j = 0; j < point_leye.size(); j++)
			{
				training_mat.at<float>(k,currentDim++)=point_leye[j].x;
				training_mat.at<float>(k,currentDim++)=point_leye[j].y;
			}

			//right eye
			for (int j = 0; j < point_reye.size(); j++)
			{
				training_mat.at<float>(k,currentDim++)=point_reye[j].x;
				training_mat.at<float>(k,currentDim++)=point_reye[j].y;
			}

			// mouth
			for (int j = 0; j < point_mouth.size(); j++)
			{
				training_mat.at<float>(k,currentDim++)=point_mouth[j].x;
				training_mat.at<float>(k,currentDim++)=point_mouth[j].y;
			}

			//nose
			for (int j = 0; j < point_nose.size(); j++)
			{
				training_mat.at<float>(k,currentDim++)=point_nose[j].x;
				training_mat.at<float>(k,currentDim++)=point_nose[j].y;
			}
			

			// vector add into Mat

			for (int j = 0; j < vec_mouth.size(); j++)
			{
				training_mat.at<float>(k,currentDim)=vec_mouth[j];
				currentDim++;
			}
			for (int j = 0; j < vec_eyebrow.size(); j++)
			{
				training_mat.at<float>(k,currentDim)=vec_eyebrow[j];
				currentDim++;
			}
			for (int j = 0; j < vec_leye.size(); j++)
			{
				training_mat.at<float>(k,currentDim)=vec_leye[j];
				currentDim++;
			}
			for (int j = 0; j < vec_reye.size(); j++)
			{
				training_mat.at<float>(k,currentDim)=vec_reye[j];
				currentDim++;
			}

			cout<<currentDim<<endl;

			rectangle(frame, faces[i], Scalar( 255, 0, 0), 2,7, 0);

		}

		imshow("Predict",frame);
		waitKey(30);

	}
	ofstream out("out_prd.txt");
	out<<labels<<endl;
	out<<training_mat<<endl;
	out.close();

}

int main( int argc, char** argv ){ 


	Mat training_mat(SAMPLE_NUM,DATA_NUM,CV_32FC1);
	Mat train_labels(SAMPLE_NUM,1,CV_32FC1);

	Mat predict_mat(10,DATA_NUM,CV_32FC1);
	Mat predict_labels(10,1,CV_32FC1);
	int predict_count;
	printf("Loading Model!\nWait......\n");

	string file_name = "shape_1.dat";
	if (!LBF_Model_Load(file_name, model))
	{
		cout<<"Can not load the file!"<<endl;
		return 0;
	}


	int use_model_file=0;

	CvSVM SVM;
	if (!use_model_file)
	{
		getTrainFeature(training_mat,train_labels);
		CvSVMParams params;
		params.svm_type = CvSVM::C_SVC;
		params.kernel_type = CvSVM::LINEAR;
		params.term_crit = cvTermCriteria(CV_TERMCRIT_ITER,100,5e-3);
		params.C=10;
		params.p=5e-3;
		params.gamma=0.01;

		SVM.train(training_mat,train_labels,Mat(),Mat(),params);
		SVM.save("svm_model-new");
	}
	else
	{

		SVM.load("svm_model-new");
	}
	


	

	getPredictFeature(predict_mat,predict_labels,predict_count);

	Mat img_test(1,DATA_NUM,CV_32FC1);
	for (int i = 0; i < predict_count; i++)
	{
		predict_mat.row(i).copyTo(img_test.row(0));
		cout<<i<<" : type: "<<predict_labels.at<float>(i,0)<<" Predict: "<<SVM.predict(img_test)<<endl;
	}
	

	return 0;
}
