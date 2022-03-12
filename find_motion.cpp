#include <iostream>
#include <vector>
#include "rigid.hpp"

using namespace std;

int main() {
  string input_file = "data/keyframes.txt";
  string output_file = "data/motions.txt";

  int num_se3 = 2, num_so3 = 3;
  vector<Trajectory> trajectories =  read_keyframes(input_file);
  output_trajectories(trajectories, output_file);
  cout << "Done processing input keyframes and output calculated motion." << endl;

  return 0;
}