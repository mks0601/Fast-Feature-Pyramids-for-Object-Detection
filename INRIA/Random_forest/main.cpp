#include "RF_architecture.h"

void main()
{

	node tree_head[100];
	node* tree_ptr;
	int M = 5;
	int NofBoot = 20;
	int NofTree = 100;
	int maxDepth = 6;
	int minCount = 10;

	vector<struct feature_value_struct> pedestrian_feature_value;
	vector<struct feature_value_struct> negative_feature_value;

	clock_t begin, end;
	begin = clock();

	srand(time(NULL));

	/*
	//positive HOG load
	for (int i = 0; i < 2416; i++)
	{
	FILE* fp_pos;
	char filename[100];

	sprintf_s(filename, "pos_hog/%d.txt", i+1);

	fopen_s(&fp_pos, filename, "r");


	Mat* sample_matrix = new Mat(1, 15 * 7 * 9 * 4, CV_32FC1);


	for (int index = 0; index < 3780; index++)
	{
	float hog_tmp;
	fscanf_s(fp_pos, "%f", &hog_tmp);
	sample_matrix->at<float>(0, index) = hog_tmp;
	}


	fclose(fp_pos);

	struct feature_value_struct tmp;
	tmp.hog_value = sample_matrix;
	pedestrian_feature_value.push_back(tmp);

	}
	printf("Loading positive HOG complete!!\n");


	//negative HOG load
	for (int i = 0; i < 1218*8; i++)
	{
	FILE* fp_neg;
	char filename[100];

	sprintf_s(filename, "neg_bagging/%d.txt", i+1);

	fopen_s(&fp_neg, filename, "r");


	Mat* sample_matrix = new Mat(1, 15 * 7 * 9 * 4, CV_32FC1);


	for (int index = 0; index < 3780; index++)
	{
	float hog_tmp;
	fscanf_s(fp_neg, "%f", &hog_tmp);
	sample_matrix->at<float>(0, index) = hog_tmp;
	}


	fclose(fp_neg);

	struct feature_value_struct tmp;
	tmp.hog_value = sample_matrix;
	negative_feature_value.push_back(tmp);

	}
	printf("Loading negative HOG complete!!\n\n");


	printf("Training START!!\n");
	for (int boot_number = 0; boot_number < NofBoot; boot_number++)
	{


	for (int tree_number = boot_number*M; tree_number < (boot_number+1)*M; tree_number++)
	{

	tree_ptr = &tree_head[tree_number];

	//tree initialize
	tree_ptr->pedestrian_feature_value = pedestrian_feature_value;
	tree_ptr->negative_feature_value = negative_feature_value;
	tree_ptr->distribution_pedestrian = ((double)(tree_ptr->pedestrian_feature_value.size())) / ((double)(tree_ptr->pedestrian_feature_value.size()) + (double)(tree_ptr->negative_feature_value.size()));
	tree_ptr->distribution_negative = ((double)(tree_ptr->negative_feature_value.size())) / ((double)(tree_ptr->pedestrian_feature_value.size()) + (double)(tree_ptr->negative_feature_value.size()));
	tree_ptr->count_pedestrian = tree_ptr->pedestrian_feature_value.size();
	tree_ptr->count_negative = tree_ptr->negative_feature_value.size();
	tree_ptr->depth = 0;

	printf("%dth tree is in training...\n", tree_number + 1);

	split_node(tree_ptr, 1, maxDepth, minCount);//split!!
	}


	//delete negative features and search a false-positive examples by bootstrapping
	for (vector<struct feature_value_struct>::iterator it = negative_feature_value.begin(); it != negative_feature_value.end(); it++)
	delete it->hog_value;

	negative_feature_value.clear();


	if ((boot_number + 1)*M == NofTree)
	break;


	//Bootstrap in order to collect false-positive images
	printf("\nBootstrap start!\n");
	for (; negative_feature_value.size() < 1218 * 8;)
	{
	int file_index = rand() % 1218 + 1;
	int x, y;
	int width, height;
	char filename[100];
	sprintf_s(filename, "neg_original/%d.jpg", file_index);
	Mat boot_image = imread(filename);
	Mat gray;
	cvtColor(boot_image, gray, CV_BGR2GRAY);
	printf("%dth bootstrapping... negative_feature_value vector size : %d\n", boot_number+1,negative_feature_value.size());

	while (gray.cols > 64 && gray.rows > 128)
	{



	HOGDescriptor boot_hog;
	vector<float> boot_hog_value;
	vector<Point> boot_locations;

	width = gray.cols;
	height = gray.rows;


	while ((width - 16) % 8 != 0)
	width--;
	while ((height - 16) % 8 != 0)
	height--;


	boot_hog.winSize = Size(width, height);

	boot_hog.compute(gray, boot_hog_value, Size(0, 0), Size(0, 0), boot_locations);

	x = 0;
	y = 0;


	for (;; x = x + 8)
	{

	if (x + 64 > width)
	{
	x = 0;
	y = y + 8;
	}

	if (y + 128 > height)
	break;

	Mat sample_matrix(1, 15 * 7 * 9 * 4, CV_32FC1);


	int elem;
	int y_index = y / 8;

	for (int x_index = x / 8; x_index < x / 8 + 7; x_index++)
	{
	elem = (9 * 4 * (height / 8 - 1))*x_index + (9 * 4)*y_index;

	for (int c = (x_index - x / 8) * 15 * 4 * 9; c < (x_index - x / 8 + 1) * 15 * 4 * 9; c++)
	sample_matrix.at<float>(0, c) = boot_hog_value.at(elem++);
	}


	double pedestrian_confidence = 0;
	for (int tree_number = 0; tree_number < (boot_number + 1)*M; tree_number++)
	{

	tree_ptr = &tree_head[tree_number];

	for (;;)
	{

	if (tree_ptr->left_child == NULL && tree_ptr->right_child == NULL)
	break;

	else
	{

	if (sample_matrix.at<float>(0, tree_ptr->index) < tree_ptr->tau)
	tree_ptr = tree_ptr->left_child;
	else
	tree_ptr = tree_ptr->right_child;

	}

	}

	pedestrian_confidence += tree_ptr->distribution_pedestrian;
	}

	pedestrian_confidence /= (double)((boot_number + 1)*M);


	if (pedestrian_confidence > 0.25)
	{
	Mat* boot_matrix = new Mat(1, 15 * 7 * 9 * 4, CV_32FC1);
	*boot_matrix = sample_matrix;
	struct feature_value_struct tmp;
	tmp.hog_value = boot_matrix;
	negative_feature_value.push_back(tmp);
	}
	}

	Mat resized_image;
	resize(gray, resized_image, Size(floor((double)gray.cols / 1.05), floor((double)gray.rows / 1.05)), 0, 0, INTER_LINEAR);
	gray = resized_image;

	}
	}


	}


	//memory deallocation
	for (vector<struct feature_value_struct>::iterator it = pedestrian_feature_value.begin(); it != pedestrian_feature_value.end(); it++)
	delete it->hog_value;

	pedestrian_feature_value.clear();


	save_tree(tree_head, NofTree);
	*/

	load_tree(tree_head, NofTree);


	//training tree complete
	printf("Training complete!!\n\n");

	end = clock();
	clock_t training_time = end - begin;
	cout << (double)(end - begin) / CLOCKS_PER_SEC << "seconds for training" << endl << endl;



	printf("Testing start!\n");

	begin = clock();

	int miss_pedestrian[100];//count missed pedestrian
	int false_positive[100];//count false-positive
	int count_pedestrian = 0;//total number of pedestrian


	for (int i = 0; i < 100; i++)
	{
		miss_pedestrian[i] = 0;
		false_positive[i] = 0;
	}


	vector<struct ground_truth> ground_truth_vector[100];//groundtruth
	vector<struct bounding_box> bounding_box[100];//bounding box
	double scale = pow(0.5, 0.125);
	double lamda = 0.21;
	int count_scale;
	for (int i = 0; i < 288; i++)
	{

		printf("%dth image is in testing...\n", i + 1);

		count_scale = 0;

		char filename[100];
		sprintf_s(filename, "test/%d.jpg", i + 1);


		Mat test_image;
		test_image = imread(filename);
		Mat gray;
		cvtColor(test_image, gray, CV_BGR2GRAY);


		double pedestrian_confidence = 0;
		double negative_confidence = 0;


		int x;
		int y;
		int width;
		int height;


		//modify width and height in order to calculate HOG feature
		int original_width = test_image.cols;
		int original_height = test_image.rows;

		while ((original_width - 16) % 8 != 0)
			original_width--;
		while ((original_height - 16) % 8 != 0)
			original_height--;


		int scaled_x;
		int scaled_y;
		int scaled_width;
		int scaled_height;


		for (int p = 0; p < 100; p++)
		{
			bounding_box[p].clear();
			ground_truth_vector[p].clear();
		}


		//groundtruth parsing
		FILE* fp_groundtruth;
		char filename_groundtruth[100];
		sprintf_s(filename_groundtruth, "groundtruth/%d.txt", i + 1);
		fopen_s(&fp_groundtruth, filename_groundtruth, "r");

		char cordinate[100];

		while (fgets(cordinate, sizeof(cordinate), fp_groundtruth))
		{
			count_pedestrian++;

			char* str = cordinate;
			int tmp;
			int x_min;
			int y_min;
			int x_max;
			int y_max;

			for (tmp = 0; tmp < 4; tmp++)
			{
				const char *begin = str;

				while (*str != ' ' && *str != '\n')
					str++;

				if (tmp == 0)
					x_min = atoi(string(begin, str).c_str());
				if (tmp == 1)
					y_min = atoi(string(begin, str).c_str());
				if (tmp == 2)
					x_max = atoi(string(begin, str).c_str());
				if (tmp == 3)
					y_max = atoi(string(begin, str).c_str());

				str++;
			}

			struct ground_truth tmp_struct;
			tmp_struct.is_detected = false;
			tmp_struct.x_max = x_max;
			tmp_struct.x_min = x_min;
			tmp_struct.y_max = y_max;
			tmp_struct.y_min = y_min;

			for (int p = 0; p < 100; p++)
				ground_truth_vector[p].push_back(tmp_struct);

		}

		fclose(fp_groundtruth);

		HOGDescriptor testing_hog;
		vector<float> testing_hog_value;
		vector<Point> testing_locations;

		width = gray.cols;
		height = gray.rows;


		while ((width - 16) % 8 != 0)
			width--;
		while ((height - 16) % 8 != 0)
			height--;


		testing_hog.winSize = Size(width, height);

		testing_hog.compute(gray, testing_hog_value, Size(0, 0), Size(0, 0), testing_locations);

		float*** hog_value = get_hog(testing_hog_value, Size(width, height), Size(8, 8));


		while (width> 64 && height > 128)
		{
			x = 0;
			y = 0;

			//window stride is 8 pixel in both of x and y cordinate
			for (;; x = x + 8)
			{

				if (x + 64 > width)
				{
					x = 0;
					y = y + 8;
				}

				if (y + 128 > height)
					break;



				pedestrian_confidence = 0;
				for (int tree_number = 0; tree_number < M; tree_number++)
				{


					tree_ptr = &tree_head[tree_number];

					for (;;)
					{

						if (tree_ptr->left_child == NULL && tree_ptr->right_child == NULL)
							break;

						else
						{

							if (testing_hog_value.at((9 * 4 * (height / 8 - 1))*(x / 8 + tree_ptr->index / (15 * 4 * 9)) + (9 * 4)*(y / 8) + tree_ptr->index % (15 * 4 * 9)) < tree_ptr->tau)
								tree_ptr = tree_ptr->left_child;
							else
								tree_ptr = tree_ptr->right_child;

						}

					}

					pedestrian_confidence += tree_ptr->distribution_pedestrian;

				}

				pedestrian_confidence = pedestrian_confidence / (double)(M);
				negative_confidence = negative_confidence / (double)(M);

				int t = M;

				//soft cascade
				while (pedestrian_confidence > 0.1 && t < NofTree)
				{
					tree_ptr = &tree_head[t];

					for (;;)
					{

						if (tree_ptr->left_child == NULL && tree_ptr->right_child == NULL)
							break;

						else
						{

							if (testing_hog_value.at((9 * 4 * (height / 8 - 1))*(x / 8 + tree_ptr->index / (15 * 4 * 9)) + (9 * 4)*(y / 8) + tree_ptr->index % (15 * 4 * 9)) < tree_ptr->tau)
								tree_ptr = tree_ptr->left_child;
							else
								tree_ptr = tree_ptr->right_child;

						}

					}

					double tmp_score = tree_ptr->distribution_pedestrian;


					pedestrian_confidence = ((double)(1)) / ((double)(t + 1))*(pedestrian_confidence*(double)t + tmp_score);
					t++;

				}


				//make bounding box
				int bb_num = 0;
				for (double threshold = -0.3; threshold<0.3; threshold = threshold + 0.006)
				{

					if (pedestrian_confidence > 0.50 + threshold)
					{

						scaled_y = (int)((double)original_height) * (((double)(y + 8)) / ((double)height));

						scaled_height = (int)((0.75)*(((double)original_height) * (((double)(y + 8) + (double)128) / ((double)height)) - (double)scaled_y));

						scaled_x = (int)(((double)original_width) * (((double)(x + 8)) / ((double)width)) + 0.05*(double)scaled_height);
						scaled_width = (int)(0.4*(double)scaled_height);

						struct bounding_box tmp;
						tmp.rect = Rect(scaled_x, scaled_y, scaled_width, scaled_height);
						tmp.confidence = pedestrian_confidence;
						tmp.is_false_positive = true;
						tmp.is_suppressed = false;

						bounding_box[bb_num].push_back(tmp);

					}
					bb_num++;

				}


			}

			//Fast feature pyramid!
			if ((count_scale + 1) % 4 == 0)
			{
				for (int y = 0; y<height / 8; y++)
				{
					for (int x = 0; x<width / 8; x++)
					{
						delete[] hog_value[y][x];
					}
					delete[] hog_value[y];

				}
				delete[] hog_value;

				Mat resized;
				resize(gray, resized, Size(floor((double)gray.cols / sqrt(2)), floor((double)gray.rows / sqrt(2))), 0, 0, INTER_LINEAR);
				gray = resized;


				testing_hog_value.clear();
				testing_locations.clear();

				width = gray.cols;
				height = gray.rows;


				while ((width - 16) % 8 != 0)
					width--;
				while ((height - 16) % 8 != 0)
					height--;


				testing_hog.winSize = Size(width, height);

				testing_hog.compute(gray, testing_hog_value, Size(0, 0), Size(0, 0), testing_locations);

				hog_value = get_hog(testing_hog_value, Size(width, height), Size(8, 8));

			}
			else
			{

				int old_width = width;
				int old_height = height;
				width *= scale;
				height *= scale;
				while ((width - 16) % 8 != 0)
					width--;
				while ((height - 16) % 8 != 0)
					height--;

				float ***resized = resize_hog(hog_value, old_width / 8, old_height / 8, width / 8, height / 8, pow(scale, -lamda));

				for (int y = 0; y<old_height / 8; y++)
				{
					for (int x = 0; x<old_width / 8; x++)
					{
						delete[] hog_value[y][x];
					}
					delete[] hog_value[y];

				}
				delete[] hog_value;
				hog_value = resized;


				testing_hog_value.clear();
				testing_locations.clear();

				restore_hog(&testing_hog_value, hog_value, height / 8, width / 8);


			}

			count_scale++;
		}



		vector<struct bounding_box> suppressed_bounding_box[100];

		//Non-Maximum-Suppresion
		for (int p = 0; p < 100; p++)
		{
			int original_size = -1;
			int suppressed_size = 1;


			while (bounding_box[p].size()>1 && suppressed_size != original_size)
			{
				while (1)
				{
					suppressed_size = 0;
					vector<struct bounding_box>::iterator loc;
					float maxValue = -1;


					for (vector<struct bounding_box>::iterator it = bounding_box[p].begin(); it != bounding_box[p].end(); it++)
					{
						if (it->confidence > maxValue && it->is_suppressed == false)
						{
							maxValue = it->confidence;
							loc = it;
						}

						if (it->is_suppressed == false)
							suppressed_size++;
					}

					if (suppressed_size == 0)
						break;

					struct bounding_box NMS_bb;
					NMS_bb.confidence = loc->confidence;
					NMS_bb.is_false_positive = true;
					NMS_bb.is_suppressed = false;
					NMS_bb.rect = loc->rect;

					NMS(&bounding_box[p], *loc);

					suppressed_bounding_box[p].push_back(NMS_bb);

				}


				original_size = bounding_box[p].size();
				bounding_box[p] = suppressed_bounding_box[p];
				suppressed_bounding_box[p].clear();
				suppressed_size = bounding_box[p].size();

			}
		}


		//determine detected groundtruth and false-positive bounding box
		for (int bb_num = 0; bb_num < 100; bb_num++)
		{
			for (vector<struct ground_truth> ::iterator it = ground_truth_vector[bb_num].begin(); it != ground_truth_vector[bb_num].end(); it++)
			{
				for (vector<struct bounding_box>::iterator it2 = bounding_box[bb_num].begin(); it2 != bounding_box[bb_num].end(); it2++)
				{
					int width_intersect = min(it->x_max, it2->rect.width + it2->rect.x) - max(it2->rect.x, it->x_min);
					int height_intersect = min(it->y_max, it2->rect.height + it2->rect.y) - max(it2->rect.y, it->y_min);
					int area_intersect = width_intersect * height_intersect;
					int area_union = it2->rect.height *it2->rect.width + (it->x_max - it->x_min)*(it->y_max - it->y_min) - area_intersect;
					double overlap_ratio = (double)(area_intersect) / (double)(area_union);

					if (width_intersect > 0 && height_intersect > 0 && overlap_ratio > 0.5)
					{
						it->is_detected = true;
						it2->is_false_positive = false;
					}

				}

			}
		}



		for (int p = 0; p < 100; p++)
		{
			for (vector<struct bounding_box>::iterator it = bounding_box[p].begin(); it != bounding_box[p].end(); it++)
			{

				if (it->is_false_positive == true)
					false_positive[p]++;
			}

			for (vector<struct ground_truth>::iterator it2 = ground_truth_vector[p].begin(); it2 != ground_truth_vector[p].end(); it2++)
			{
				if (it2->is_detected == false)
					miss_pedestrian[p]++;
			}

		}

		char filename_result[100];
		sprintf_s(filename_result, "result/%d.jpg", i + 1);

		//draw bounding box and write image(RED is false-positive, BLUE is true-positive, GREEN is groundtruth)
		for (vector<struct bounding_box>::iterator it = bounding_box[0].begin(); it != bounding_box[0].end(); it++)
		{
			if (it->is_false_positive == true)
				rectangle(test_image, it->rect, CV_RGB(255, 0, 0), 2);
			else
				rectangle(test_image, it->rect, CV_RGB(0, 0, 255), 2);
		}

		for (vector<struct ground_truth>::iterator it = ground_truth_vector[0].begin(); it != ground_truth_vector[0].end(); it++)
			rectangle(test_image, Rect(it->x_min, it->y_min, it->x_max - it->x_min, it->y_max - it->y_min), CV_RGB(0, 255, 0), 2);

		imwrite(filename_result, test_image);



	}
	printf("Testing complete!!\n\n");
	end = clock();
	cout << (double)(end - begin) / CLOCKS_PER_SEC << "seconds for 288 test images" << endl << endl;
	cout << (double)training_time / CLOCKS_PER_SEC << "seconds for training" << endl << endl;

	FILE* result;
	fopen_s(&result, "result.txt", "w");
	for (int p = 0; p < 100; p++)
		fprintf_s(result, "%lf %lf\n", (double)miss_pedestrian[p] / (double)count_pedestrian, (double)false_positive[p] / (double)288);

	fclose(result);



}
