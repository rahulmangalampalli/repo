#include "include/mobile_knn.hpp"
#include "initiate.hpp"

int main(int argc, char** argv)
{
    Classify cl;
  /* Check if command-line arguments are correct */
  if(argc != 4 & argc !=6)
  {
     cout<<"\nWrong number of arguments: 3 or 5 number of arguments are valid" << endl;
     exit(0);
  }
  else if(argc == 4) {
  /* Reading arguments from command line */
  string proto = argv[1];
  string caffem = argv[2];
  string data = argv[3];

  cl.create_csv(proto,caffem,data);
  string ch;
  cout << "Created csv file and saved in the present working directory.Do you want to train it (y/n)?";
  cin >> ch;
  if(ch == "y")
    cl.train_csv("train.csv");
  
  else if(ch == "n")
  cout << "No file trained for now"<<endl;

  else{
  cout<<"Wrong input.Hence exiting"<<endl;
  return 0;
  }
  }
  else{
  cout << "Inferencing....." <<endl;
  cl.infer(argv[1],argv[2],argv[3],argv[4],argv[5]);
  }

return 0;
}

