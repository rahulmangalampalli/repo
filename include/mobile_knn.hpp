#include <iostream> //for std i/o operations in c++
#include <stdio.h> //for std i/o operations in c++
#include <string> //for dealing with strings

using namespace std;   // stdout library for printing values 

class Classify{ // declaration of class classify
public:
    Classify(); //constructor
    ~Classify(); //destructor
    int create_csv(string prototxt, string caffemodel, string data_path); //for creating csv 
    int train_csv(string csv_file); //for creating train.knn
    int infer(string prototxt, string caffemodel, string data_path, string knn_path, string labels); //for inference
};

