#include <iostream>
#include <stdlib.h>
#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <string.h>
#include <time.h>
#include <numeric>   
#include <math.h>
#include <algorithm>
#include <omp.h>


using namespace std;
using namespace cv;

struct ground_truth
{
	int x_min;
	int y_min;
	int x_max;
	int y_max;
	bool is_detected;
};

struct bounding_box
{
	Rect rect;
	bool is_false_positive;
	bool is_suppressed;
	double confidence;
};


struct feature_value_struct
{
	Mat* hog_value;
};


class node
{
public:
	node(){ left_child = NULL; right_child = NULL; };

	int depth;

	double distribution_pedestrian;
	double distribution_negative;

	int count_pedestrian;
	int count_negative;

	int index;

	double tau;

	vector<int> save_index;
	vector<struct feature_value_struct> pedestrian_feature_value;
	vector<struct feature_value_struct> negative_feature_value;


	node* left_child;
	node* right_child;

};


void split_node(node* tree_ptr, int depth, int maxDepth, int minCount)
{

	if (tree_ptr->count_pedestrian < minCount || tree_ptr->count_negative < minCount || tree_ptr->depth > maxDepth || tree_ptr->distribution_negative>0.99 || tree_ptr->distribution_pedestrian>0.99)
		return;
	else
	{
		double entropy_original;
		double entropy_left;
		double entropy_right;
		double max_information_gain = -9999.0;
		double information_gain;

		double save_tau;

		int count_iteration = 10;
		int feature_selection = 62;

		double distribution_pedestrian = tree_ptr->distribution_pedestrian;
		double distribution_negative = tree_ptr->distribution_negative;
		entropy_original = -(distribution_pedestrian * log10(distribution_pedestrian) + distribution_negative * log10(distribution_negative));	//Shannon entropy is used

		int count_left_pedestrian = 0;
		int count_right_pedestrian = 0;

		int count_left_negative = 0;
		int count_right_negative = 0;


		double distribution_left_pedestrian;
		double distribution_right_pedestrian;
		double distribution_left_negative;
		double distribution_right_negative;

		double save_distribution_left_pedestrian;
		double save_distribution_right_pedestrian;
		double save_distribution_left_negative;
		double save_distribution_right_negative;

		int save_count_left_pedestrian;
		int save_count_right_pedestrian;
		int save_count_left_negative;
		int save_count_right_negative;



		vector<feature_value_struct> save_feature_left_pedestrian;
		vector<feature_value_struct> save_feature_left_negative;
		vector<feature_value_struct> save_feature_right_pedestrian;
		vector<feature_value_struct> save_feature_right_negative;

		tree_ptr->left_child = new node();
		tree_ptr->right_child = new node();


		int index;
		int save_index;

		//select feature subset
		for (int i = 0; i<feature_selection; i++)
		{

			int flag = 1;

			while (flag)
			{
				flag = 0;
				index = rand() % 3780;

				for (vector<int>::iterator it = tree_ptr->save_index.begin(); it != tree_ptr->save_index.end(); it++)
				{
					if (*it == index)
						flag = 1;
				}
			}



			float minFeature, maxFeature;

			minFeature = 9999.0;
			maxFeature = -9999.0;

			for (vector<feature_value_struct>::iterator it = tree_ptr->pedestrian_feature_value.begin(); it != tree_ptr->pedestrian_feature_value.end(); it++)
			{
				if (minFeature > it->hog_value->at<float>(0, index))
					minFeature = it->hog_value->at<float>(0, index);

				if (maxFeature < it->hog_value->at<float>(0, index))
					maxFeature = it->hog_value->at<float>(0, index);
			}

			for (vector<feature_value_struct>::iterator it = tree_ptr->negative_feature_value.begin(); it != tree_ptr->negative_feature_value.end(); it++)
			{
				if (minFeature > it->hog_value->at<float>(0, index))
					minFeature = it->hog_value->at<float>(0, index);

				if (maxFeature < it->hog_value->at<float>(0, index))
					maxFeature = it->hog_value->at<float>(0, index);

			}

			//in a selected feature subset, choose a proper tau by random value
			for (int j = 0; j < count_iteration; j++)
			{

				count_left_negative = 0;
				count_left_pedestrian = 0;
				count_right_negative = 0;
				count_right_pedestrian = 0;



				tree_ptr->left_child->pedestrian_feature_value.clear();
				tree_ptr->left_child->negative_feature_value.clear();
				tree_ptr->right_child->pedestrian_feature_value.clear();
				tree_ptr->right_child->negative_feature_value.clear();


				tree_ptr->tau = (maxFeature - minFeature)*(((double)rand()) / RAND_MAX) + minFeature;


				for (vector<feature_value_struct>::iterator it = tree_ptr->pedestrian_feature_value.begin(); it != tree_ptr->pedestrian_feature_value.end(); it++)
				{
					struct feature_value_struct tmp_struct;
					tmp_struct.hog_value = it->hog_value;


					if (it->hog_value->at<float>(0, index) < tree_ptr->tau)
					{

						count_left_pedestrian++;
						tree_ptr->left_child->pedestrian_feature_value.push_back(tmp_struct);

					}

					else
					{

						count_right_pedestrian++;
						tree_ptr->right_child->pedestrian_feature_value.push_back(tmp_struct);

					}
				}


				for (vector<feature_value_struct>::iterator it = tree_ptr->negative_feature_value.begin(); it != tree_ptr->negative_feature_value.end(); it++)
				{

					struct feature_value_struct tmp_struct;
					tmp_struct.hog_value = it->hog_value;



					if (it->hog_value->at<float>(0, index) < tree_ptr->tau)
					{

						count_left_negative++;
						tree_ptr->left_child->negative_feature_value.push_back(tmp_struct);

					}


					else
					{
						count_right_negative++;
						tree_ptr->right_child->negative_feature_value.push_back(tmp_struct);

					}

				}




				distribution_left_pedestrian = ((double)(count_left_pedestrian)) / ((double)(count_left_pedestrian)+(double)(count_left_negative));
				distribution_left_negative = ((double)(count_left_negative)) / ((double)(count_left_pedestrian)+(double)(count_left_negative));
				entropy_left = -(distribution_left_pedestrian * log10(distribution_left_pedestrian) + distribution_left_negative * log10(distribution_left_negative));	//Shannon entropy is used




				distribution_right_pedestrian = ((double)(count_right_pedestrian)) / ((double)(count_right_pedestrian)+(double)(count_right_negative));
				distribution_right_negative = ((double)(count_right_negative)) / ((double)(count_right_pedestrian)+(double)(count_right_negative));
				entropy_right = -(distribution_right_pedestrian * log10(distribution_right_pedestrian) + distribution_right_negative * log10(distribution_right_negative));	//Shannon entropy is used

				//calculate information gain
				information_gain = entropy_original - ((((double)(count_left_pedestrian + count_left_negative)) / ((double)(tree_ptr->count_pedestrian + tree_ptr->count_negative)))*entropy_left + (((double)(count_right_pedestrian + count_right_negative)) / ((double)(tree_ptr->count_pedestrian + tree_ptr->count_negative)))*entropy_right);



				if (information_gain > max_information_gain)
				{
					max_information_gain = information_gain;

					save_tau = tree_ptr->tau;

					save_index = index;

					save_distribution_left_pedestrian = distribution_left_pedestrian;
					save_distribution_left_negative = distribution_left_negative;
					save_distribution_right_pedestrian = distribution_right_pedestrian;
					save_distribution_right_negative = distribution_right_negative;

					save_count_left_pedestrian = count_left_pedestrian;
					save_count_left_negative = count_left_negative;
					save_count_right_pedestrian = count_right_pedestrian;
					save_count_right_negative = count_right_negative;



					save_feature_left_pedestrian = tree_ptr->left_child->pedestrian_feature_value;
					save_feature_left_negative = tree_ptr->left_child->negative_feature_value;
					save_feature_right_pedestrian = tree_ptr->right_child->pedestrian_feature_value;
					save_feature_right_negative = tree_ptr->right_child->negative_feature_value;

				}

			}



		}


		//split complete


		tree_ptr->tau = save_tau;
		tree_ptr->index = save_index;
		tree_ptr->negative_feature_value.clear();
		tree_ptr->pedestrian_feature_value.clear();


		tree_ptr->left_child->distribution_pedestrian = save_distribution_left_pedestrian;
		tree_ptr->left_child->distribution_negative = save_distribution_left_negative;
		tree_ptr->left_child->count_pedestrian = save_count_left_pedestrian;
		tree_ptr->left_child->count_negative = save_count_left_negative;
		tree_ptr->left_child->depth = depth;
		tree_ptr->left_child->pedestrian_feature_value = save_feature_left_pedestrian;
		tree_ptr->left_child->negative_feature_value = save_feature_left_negative;
		tree_ptr->left_child->save_index = tree_ptr->save_index;
		tree_ptr->left_child->save_index.push_back(save_index);


		tree_ptr->right_child->distribution_pedestrian = save_distribution_right_pedestrian;
		tree_ptr->right_child->distribution_negative = save_distribution_right_negative;
		tree_ptr->right_child->count_pedestrian = save_count_right_pedestrian;
		tree_ptr->right_child->count_negative = save_count_right_negative;
		tree_ptr->right_child->depth = depth;
		tree_ptr->right_child->pedestrian_feature_value = save_feature_right_pedestrian;
		tree_ptr->right_child->negative_feature_value = save_feature_right_negative;
		tree_ptr->right_child->save_index = tree_ptr->save_index;
		tree_ptr->right_child->save_index.push_back(save_index);


		split_node(tree_ptr->left_child, depth + 1, maxDepth, minCount);

		split_node(tree_ptr->right_child, depth + 1, maxDepth, minCount);



	}


}



void NMS(vector<struct bounding_box> *bb_vector, struct bounding_box bb)
{


	for (vector<struct bounding_box>::iterator it = bb_vector->begin(); it != bb_vector->end(); it++)
	{
		if (it->is_suppressed == false)
		{
			int width_intersect = min((it->rect.x + it->rect.width), (bb.rect.x + bb.rect.width)) - max(it->rect.x, bb.rect.x);
			int height_intersect = min((it->rect.y + it->rect.height), (bb.rect.y + bb.rect.height)) - max(it->rect.y, bb.rect.y);
			int area_intersect = width_intersect * height_intersect;
			int area_union = it->rect.width * it->rect.height + bb.rect.width * bb.rect.height - area_intersect;
			double overlap_ratio = (double)(area_intersect) / (double)(area_union);

			//if overlap ratio >0.5 or included area >0.5 -> delete!
			if (width_intersect > 0 && height_intersect > 0 && (overlap_ratio > 0.5 || area_intersect > 0.5*(it->rect.height*it->rect.width) || area_intersect > 0.5*(bb.rect.height*bb.rect.width)))
				it->is_suppressed = true;


		}

	}

}



void save_tree_recursion(node* tree, FILE* fp)
{

	if (tree->left_child == NULL && tree->right_child == NULL)
	{
		char save[1000];
		sprintf_s(save, "%d %d %lf %lf %d %d %d %lf\n", 0, tree->depth, tree->distribution_pedestrian, tree->distribution_negative, tree->count_pedestrian, tree->count_negative, tree->index, tree->tau);
		fprintf_s(fp, save);
		return;
	}
	else
	{
		char save[1000];
		sprintf_s(save, "%d %d %lf %lf %d %d %d %lf\n", 1, tree->depth, tree->distribution_pedestrian, tree->distribution_negative, tree->count_pedestrian, tree->count_negative, tree->index, tree->tau);
		fprintf_s(fp, save);
		save_tree_recursion(tree->left_child, fp);
		save_tree_recursion(tree->right_child, fp);
	}
}
void save_tree(node tree_head[], int NofTree)
{
	FILE* fp_save;
	char filename[100];

	for (int i = 0; i < NofTree; i++)
	{
		sprintf_s(filename, "save_bagging/%d.txt", i + 1);
		fopen_s(&fp_save, filename, "w");
		save_tree_recursion(&tree_head[i], fp_save);
		fclose(fp_save);
	}

}

void load_tree_recursion(node* tree, FILE* fp)
{
	int stop_criteria;
	char load[1000];
	fgets(load, sizeof(load), fp);
	sscanf_s(load, "%d %d %lf %lf %d %d %d %lf\n", &stop_criteria, &(tree->depth), &(tree->distribution_pedestrian), &(tree->distribution_negative), &(tree->count_pedestrian), &(tree->count_negative), &(tree->index), &(tree->tau));
	if (stop_criteria == 0)
		return;
	else
	{
		tree->left_child = new node();
		tree->right_child = new node();
		load_tree_recursion(tree->left_child, fp);
		load_tree_recursion(tree->right_child, fp);
	}

}


void load_tree(node* tree_head, int NofTree)
{
	FILE* fp_load;
	char filename[100];

	for (int index = 0; index < NofTree; index++)
	{
		sprintf_s(filename, "save_bagging/%d.txt", index + 1);
		fopen_s(&fp_load, filename, "r");
		load_tree_recursion(tree_head + index, fp_load);
		fclose(fp_load);
	}


}


float*** get_hog(vector<float> descriptorValues, Size winSize, Size cellSize)
{


	int gradientBinSize = 9;
	// dividing 180¡Æ into 9 bins, how large (in rad) is one bin?
	float radRangeForOneBin = 3.14 / (float)gradientBinSize;

	// prepare data structure: 9 orientation / gradient strenghts for each cell
	int cells_in_x_dir = winSize.width / cellSize.width;
	int cells_in_y_dir = winSize.height / cellSize.height;
	int totalnrofcells = cells_in_x_dir * cells_in_y_dir;
	float*** gradientStrengths = new float**[cells_in_y_dir];
	int** cellUpdateCounter = new int*[cells_in_y_dir];
	for (int y = 0; y<cells_in_y_dir; y++)
	{
		gradientStrengths[y] = new float*[cells_in_x_dir];
		cellUpdateCounter[y] = new int[cells_in_x_dir];
		for (int x = 0; x<cells_in_x_dir; x++)
		{
			gradientStrengths[y][x] = new float[gradientBinSize];
			cellUpdateCounter[y][x] = 0;

			for (int bin = 0; bin<gradientBinSize; bin++)
				gradientStrengths[y][x][bin] = 0.0;
		}
	}

	// nr of blocks = nr of cells - 1
	// since there is a new block on each cell (overlapping blocks!) but the last one
	int blocks_in_x_dir = cells_in_x_dir - 1;
	int blocks_in_y_dir = cells_in_y_dir - 1;

	// compute gradient strengths per cell
	int descriptorDataIdx = 0;
	int cellx = 0;
	int celly = 0;

	for (int blockx = 0; blockx<blocks_in_x_dir; blockx++)
	{
		for (int blocky = 0; blocky<blocks_in_y_dir; blocky++)
		{
			// 4 cells per block ...
			for (int cellNr = 0; cellNr<4; cellNr++)
			{
				// compute corresponding cell nr
				int cellx = blockx;
				int celly = blocky;
				if (cellNr == 1) celly++;
				if (cellNr == 2) cellx++;
				if (cellNr == 3)
				{
					cellx++;
					celly++;
				}

				for (int bin = 0; bin<gradientBinSize; bin++)
				{
					float gradientStrength = descriptorValues[descriptorDataIdx];
					descriptorDataIdx++;

					gradientStrengths[celly][cellx][bin] += gradientStrength;

				} // for (all bins)


				// note: overlapping blocks lead to multiple updates of this sum!
				// we therefore keep track how often a cell was updated,
				// to compute average gradient strengths
				cellUpdateCounter[celly][cellx]++;

			} // for (all cells)


		} // for (all block x pos)
	} // for (all block y pos)


	// compute average gradient strengths
	for (int celly = 0; celly<cells_in_y_dir; celly++)
	{
		for (int cellx = 0; cellx<cells_in_x_dir; cellx++)
		{

			float NrUpdatesForThisCell = (float)cellUpdateCounter[celly][cellx];

			// compute average gradient strenghts for each gradient bin direction
			for (int bin = 0; bin<gradientBinSize; bin++)
			{
				gradientStrengths[celly][cellx][bin] /= NrUpdatesForThisCell;
			}
		}
	}



	// don't forget to free memory allocated by helper data structures!
	for (int y = 0; y<cells_in_y_dir; y++)
		delete[] cellUpdateCounter[y];

	delete[] cellUpdateCounter;

	return gradientStrengths;
}


float*** resize_hog(float*** original_hog, int original_x_cell, int original_y_cell, int resized_x_cell, int resized_y_cell, double factor)
{

	Mat original_mat(original_y_cell, original_x_cell, CV_32FC1);
	Mat resized_mat(resized_y_cell, resized_x_cell, CV_32FC1);

	float*** resized_hog = new float**[resized_y_cell];

	for (int y = 0; y<resized_y_cell; y++)
	{
		resized_hog[y] = new float*[resized_x_cell];

		for (int x = 0; x<resized_x_cell; x++)
		{
			resized_hog[y][x] = new float[9];

			for (int bin = 0; bin<9; bin++)
				resized_hog[y][x][bin] = 0.0;
		}
	}



	for (int bin = 0; bin < 9; bin++)
	{

		for (int y = 0; y < original_y_cell; y++)
		{
			for (int x = 0; x < original_x_cell; x++)
				original_mat.at<float>(y, x) = original_hog[y][x][bin];
		}

		resize(original_mat, resized_mat, Size(resized_x_cell, resized_y_cell), 0, 0, INTER_CUBIC);

		for (int y = 0; y < resized_y_cell; y++)
		{
			for (int x = 0; x < resized_x_cell; x++)
			{
				resized_hog[y][x][bin] = resized_mat.at<float>(y, x);
				resized_hog[y][x][bin] *= factor;
			}
		}
	}




	return resized_hog;

}


void restore_hog(vector<float>* result, float*** hog_value, int cell_y, int cell_x)
{

	for (int x_index = 0; x_index < cell_x - 1; x_index++)
	{
		for (int y_index = 0; y_index < cell_y - 1; y_index++)
		{
			for (int x = x_index; x < x_index + 2; x++)
			{
				for (int y = y_index; y< y_index + 2; y++)
				{
					for (int bin = 0; bin < 9; bin++)
					{
						result->push_back(hog_value[y][x][bin]);
					}
				}
			}
		}
	}




}