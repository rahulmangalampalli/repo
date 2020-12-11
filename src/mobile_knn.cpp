#include "mobile_knn.hpp" // To get function definitions
#include <vector> // to get vector utility
#include <opencv2/core/core.hpp>     
#include <opencv2/highgui/highgui.hpp> //to get gui related functions from opencv
#include <opencv2/imgproc/imgproc.hpp> //to get image processing functions from opencv
#include <opencv2/ml.hpp> // to get all knn function from opencv
#include <caffe/caffe.hpp> // to deal with caffe basic operations
#include "dirent.h"  // to deal wit directories

using namespace cv::ml;
using namespace std;
using namespace cv;
using namespace caffe;

Classify::Classify(){}
Classify::~Classify(){}

int Classify::create_csv(string prototxt, string caffemodel, string data_path)
{
	boost::shared_ptr<Net<float> > net_; //intialize to shared pointer to read prototxt and caffemodel
	int count = 1; 
	int tt_count =0;
	vector<vector<float>> v;
	vector<std::string> labels;
	vector<int> totals;
	net_ = boost::shared_ptr<Net<float> >(new Net<float>(prototxt, caffe::TEST)); //reads prototxt
	net_->CopyTrainedLayersFrom(caffemodel); //reads caffemodel
    std::string inputDirectory = data_path; 
    DIR *directory = opendir (inputDirectory.c_str());
    struct dirent *_dirent = NULL;  
    if(directory == NULL) // loop over every directory and every image and store it a csv file
    {
        cout << "Cannot open Input Directory"<<endl;
        return 1;
    }
    while((_dirent = readdir(directory)) != NULL)
    {
		if( !strcmp(_dirent->d_name, ".")) continue;
        if( !strcmp(_dirent->d_name, "..")) continue;
        std::string fol = inputDirectory + "/" +std::string(_dirent->d_name);
		DIR *folder = opendir (fol.c_str());
        struct dirent *_dirent_2 = NULL;

		std::string label = std::string(_dirent->d_name); 

		while((_dirent_2 = readdir(folder)) != NULL){
			if( !strcmp(_dirent_2->d_name, ".")) continue;
            if( !strcmp(_dirent_2->d_name, "..")) continue;
			std::string im = fol + "/" +std::string(_dirent_2->d_name);
			Mat image = imread(im.c_str());
			Mat img;
			resize(image, img, Size(224, 224));
			// Load image into buffers for inferencing
   			Blob<float>* input_layer = net_->input_blobs()[0];
    		int num_channels_ = input_layer->channels();
    		cv::Size input_geometry_;
    		input_geometry_ = cv::Size(input_layer->width(), input_layer->height());
    		input_layer->Reshape(1, num_channels_, input_geometry_.height, input_geometry_.width);
    		net_->Reshape();
    		std::vector<cv::Mat> input_channels;
    		Blob<float>* input_layer1 = net_->input_blobs()[0];
    		int width = input_layer1->width();
    		int height = input_layer1->height();
    		float* input_data = input_layer1->mutable_cpu_data();
    		for (int i = 0; i < input_layer1->channels(); ++i) {
     			cv::Mat channel(height, width, CV_32FC1, input_data);
     			input_channels.push_back(channel);
     			input_data += width * height;
    		}

    		cv::Mat sample_float;
   			img.convertTo(sample_float, CV_32FC3);
    		cv::split(sample_float, input_channels);
   
    	CHECK(reinterpret_cast<float*>(input_channels.at(0).data)
        	== net_->input_blobs()[0]->cpu_data())
    	<< "Input channels are not wrapping the input layer of the network.";

    	net_->Forward(); //inference starts

		const float* embeddings = NULL;
    	embeddings = net_->blob_by_name("MobilenetV1/Logits")->cpu_data(); //collect results
        
		vector<float> v1;

		for(int i = 0; i < 1001; i++ ){
			v1.push_back(*embeddings);
			embeddings++;
		}
		tt_count+=1;
		v1.push_back(count);
        v.push_back(v1);
		cout << "Read image:" << im.c_str() << endl;
		}
	   totals.push_back(tt_count);
	   tt_count = 0;
	   count+=1;
	   cout << endl;
	   cout << "Done with label:" << label <<endl; 
	   cout << endl;
	   labels.push_back(label);
	   closedir(folder);	
    }
	std::cout << "Number of Data Points:" << v.size() << std::endl << "Number of Features:" << v[0].size() -1 << endl;
    closedir(directory);

	std::ofstream out("train.csv");

	std::cout << endl;

	std::ofstream output_file("labels.txt");
    std::ostream_iterator<std::string> output_iterator(output_file, "\n");
    std::copy(labels.begin(), labels.end(), output_iterator);

	for (int i = 0; i < count-1; i++)
	{
		cout << "Number of images in label " << labels.at(i) <<" are "<< totals.at(i) << endl;

	}
	
	cout << endl;
	cout << "Saved lables.txt" << endl;
	std::cout << "Saving train.csv....." << endl;

	for (auto& row : v) {
  		for (auto col : row)
    		out << col <<',';
  		out << '\n';
		}
	return 0;
}

int Classify::train_csv(string csv_file)
{
	cout << "Training the model" << endl; 
  	Ptr<TrainData> TrainData = TrainData::loadFromCSV(csv_file,1,-1,-1); //Loads csv file
	cv::Mat matResults(0,0,cv::DataType<int>::type);
	TrainData->setTrainTestSplitRatio(0.8,true);  	//Setting train-test split
	Ptr<KNearest> model = KNearest::create(); // Call KNN object from opencv
	model->setDefaultK(3); // Set deafult neighbors
	model->train(TrainData); //training
	model->save("train.knn"); //Saving trained model
	cout<<"Training on :"<< TrainData->getNTrainSamples() << "datapoints:" <<endl; // Total number of train samples
	cout<<endl;	
	cout<<"Accuracy on Test Set of size "<< TrainData->getNTestSamples() <<" datapoints is:" << model->calcError(TrainData, true, matResults)<<endl; //Calculate accuracy on test set. 
	cout<<endl;	
	return 0;
}


int Classify::infer(string prototxt, string caffemodel, string data_path, string knn_path, string labels)
{

    boost::shared_ptr<Net<float> > net_;
	vector<vector<float>> v;
    net_ = boost::shared_ptr<Net<float> >(new Net<float>(prototxt, caffe::TEST)); //read prototxt 
	net_->CopyTrainedLayersFrom(caffemodel); // read caffemodel
	Mat image = imread(data_path); //read image
	Mat img;
	resize(image, img, Size(224, 224)); //resize image
   	Blob<float>* input_layer = net_->input_blobs()[0];
   	
   	// Load image into buffers for inferencing
    int num_channels_ = input_layer->channels();
    cv::Size input_geometry_;
    input_geometry_ = cv::Size(input_layer->width(), input_layer->height());
    input_layer->Reshape(1, num_channels_, input_geometry_.height, input_geometry_.width);
    net_->Reshape();
    std::vector<cv::Mat> input_channels;
    Blob<float>* input_layer1 = net_->input_blobs()[0];
    int width = input_layer1->width();
    int height = input_layer1->height();
    float* input_data = input_layer1->mutable_cpu_data();
    	for (int i = 0; i < input_layer1->channels(); ++i) {
     		cv::Mat channel(height, width, CV_32FC1, input_data);
     		input_channels.push_back(channel);
     		input_data += width * height;
    	}

    	cv::Mat sample_float;
   		img.convertTo(sample_float, CV_32FC3);
    	cv::split(sample_float, input_channels);
   
    CHECK(reinterpret_cast<float*>(input_channels.at(0).data)
        == net_->input_blobs()[0]->cpu_data())
    << "Input channels are not wrapping the input layer of the network.";

    net_->Forward(); //Inference starts

	const float* embeddings = NULL;
    embeddings = net_->blob_by_name("MobilenetV1/Logits")->cpu_data(); //Collect last layer output
        
	vector<float> v1;

	for(int i = 0; i < 1001; i++ ){
		v1.push_back(*embeddings);
		embeddings++;
	}

	v.push_back(v1);// data is in format of rows x col but classifier shape is col x rows

	cv::Mat data(0, 1, cv::DataType<float>::type); // Hence till line 65, the code reshapes data to col x row

    for (unsigned int i = 0; i < v.size(); ++i)
	{
  		cv::Mat Sample(1, v[0].size(), cv::DataType<float>::type, v[i].data());
  		data.push_back(Sample);
	}

	cv::Mat result;
	
	Ptr<KNearest> knn_classify = Algorithm::load<KNearest>(knn_path); //load knn model
    knn_classify-> findNearest(data,knn_classify->getDefaultK(),result); //compare the collected result

	std::ifstream in(labels);

	std::string str;

	vector<std::string> names;

	while (std::getline(in, str))
	{
    	if(str.size() > 0)
        	names.push_back(str);
	}

    cout << "The class of this image is:" << names.at((int)result.at<float>(0)-1) << endl; //show the image class
    return 0;
}

