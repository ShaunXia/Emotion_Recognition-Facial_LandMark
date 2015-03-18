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
#include "LBFModel.h"
using namespace std;
using namespace cv;



#define OPEN_TEMPLATE_FILE_FAIL 0
#define DATA_NUM 622

#define POINT_WIDTH 480
#define POINT_HEIGHT 640

LBF_model model;
string face_cascade_name = "haarcascade_frontalface_alt.xml"; 
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
int getTrainFeature(Mat &training_mat,Mat &labels,vector<string> img_list)
{
	int max_point_x,min_point_x,max_point_y,min_point_y;

	for (int i = 0; i < img_list.size(); i++)
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

	for (int k = 0; k < img_list.size(); k++)
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

			point_eyebrow.clear();
			point_leye.clear();
			point_reye.clear();
			point_mouth.clear();
			point_nose.clear();

			//get all point 
			//eyebrow
			for (int j = 17; j < 27; j++)
			{
				point_eyebrow.push_back(fixed_point[j]);

			}

			//Left eye
			for (int j = 36; j < 42; j++)
			{
				point_leye.push_back(fixed_point[j]);
			}

			//right eye
			for (int j = 42; j < 48; j++)
			{
				point_reye.push_back(fixed_point[j]);
			}

			// mouth
			for (int j = 48; j < parts.size(); j++)
			{
				point_mouth.push_back(fixed_point[j]);
			}

			//nose
			for (int j = 31; j < 35; j++)
			{
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

int getPredictFeature(Mat &training_mat,Mat &labels,vector<string> &img_list)
{

	
	
	int max_point_x,min_point_x,max_point_y,min_point_y;

	for (int i = 0; i < img_list.size(); i++)
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

	for (int k = 0; k < img_list.size(); k++)
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


			point_eyebrow.clear();
			point_leye.clear();
			point_reye.clear();
			point_mouth.clear();
			point_nose.clear();

			//get all point 
			//eyebrow
			for (int j = 17; j < 27; j++)
			{
				point_eyebrow.push_back(fixed_point[j]);
			}

			//Left eye
			for (int j = 36; j < 42; j++)
			{
				point_leye.push_back(fixed_point[j]);
			}

			//right eye
			for (int j = 42; j < 48; j++)
			{
				point_reye.push_back(fixed_point[j]);
			}

			// mouth
			for (int j = 48; j < parts.size(); j++)
			{
				point_mouth.push_back(fixed_point[j]);
			}

			//nose
			for (int j = 31; j < 35; j++)
			{
				point_nose.push_back(fixed_point[j]);
			}

			Point xbrow,ybrow,xreye,yreye,xleye,yleye,xmouth,ymouth,xnose,ynose;
			findPointsBorder(point_eyebrow,xbrow,ybrow);
			findPointsBorder(point_reye,xreye,yreye);
			findPointsBorder(point_leye,xleye,yleye);
			findPointsBorder(point_mouth,xmouth,ymouth);
			findPointsBorder(point_nose,xnose,ynose);

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

void getImgList(string path,vector<string> &imglist)
{
	// get image list
	string img_path=path;
	string exten = "*";
	string fileFolder = img_path+"\\*."+exten;
	char fileName[2000];
	struct  _finddata_t fileInfo;
	long findResult = _findfirst(fileFolder.c_str(),&fileInfo);
	if (findResult == -1)
	{
		_findclose(findResult);
		return ;
	}

	do
	{
		sprintf(fileName,"%s\\%s",img_path.c_str(),fileInfo.name);
		if (fileInfo.attrib==_A_ARCH)
		{
			imglist.push_back(fileName);
		}

	}while (!_findnext(findResult,&fileInfo));
	_findclose(findResult); 
}

int main( int argc, char** argv ){ 

	string train_path = "male";
	string test_path = "test_img";
	vector<string> trainlist,testlist;
	getImgList(train_path,trainlist);
	getImgList(test_path,testlist);

	Mat training_mat(trainlist.size(),DATA_NUM,CV_32FC1);
	Mat train_labels(trainlist.size(),1,CV_32FC1);

	Mat predict_mat(testlist.size(),DATA_NUM,CV_32FC1);
	Mat predict_labels(testlist.size(),1,CV_32FC1);

	cout<<"training img : "<<trainlist.size()<<endl;
	cout<<"test img : "<<testlist.size()<<endl;

	printf("Loading Model!\nWait......\n");
	string file_name = "shape_1.dat";
	if (!LBF_Model_Load(file_name, model))
	{
		cout<<"Can not load the file!"<<endl;
		return 0;
	}

	cout<<"Now Training "<<endl;
	int use_model_file=1;

	CvSVM SVM;
	if (!use_model_file)
	{
		getTrainFeature(training_mat,train_labels,trainlist);
		CvSVMParams params;
		params.svm_type = CvSVM::C_SVC;
		params.kernel_type = CvSVM::LINEAR;
		params.term_crit = cvTermCriteria(CV_TERMCRIT_ITER,100,5e-3);
		params.C=10;
		params.p=5e-3;
		params.gamma=0.01;

		SVM.train(training_mat,train_labels,Mat(),Mat(),params);
		SVM.save("svm_model-smile");
	}
	else
	{
		SVM.load("svm_model-smile");
	}
	

	getPredictFeature(predict_mat,predict_labels,testlist);

	Mat img_test(1,DATA_NUM,CV_32FC1);
	for (int i = 0; i < testlist.size(); i++)
	{
		predict_mat.row(i).copyTo(img_test.row(0));
		cout<<i<<" : type: "<<predict_labels.at<float>(i,0)<<" Predict: "<<SVM.predict(img_test)<<endl;
	}
	
	return 0;
}
