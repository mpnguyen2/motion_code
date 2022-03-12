#ifndef RIGID_HPP
#define RIGID_HPP

#include <iostream>
#include <fstream>
#include <vector>
#include <Eigen/Core>
#include "sophus/geometry.hpp"
#include "rigid.hpp"

// tmp
using namespace std;

/* A general class to describe the pose of rigid object. The pose can be described by an element in the Lie group
SE(3)^c1 * SO(3)^c2 * SO(2)^c3 (* SE(2)^c4). Ex: hand model = SE(3)*SO(3)^10 (actually is SE(3)*SO(3)^5*SO(2)^5)
*/
const double kPi = Sophus::Constants<double>::pi();

class Rigid{
    private:
        int num_se3 = 0;
        int num_so3 = 0;
        int num_so2 = 0;
        int num_se2 = 0;
        vector<Sophus::SE3d> se3_elems;
        vector<Sophus::SO3d> so3_elems;

    public:
        Rigid(){

        }
        Rigid(vector<vector<double>>& se3Pos, vector<vector<double>>& se3Ang, vector<vector<double>>& so3Ang){
            this->feedData(se3Pos, se3Ang, so3Ang);
        }
        void feedData(vector<vector<double>>& se3Pos, vector<vector<double>>& se3Ang, vector<vector<double>>& so3Ang);
        // Get methods
        int get_num_se3(){
            return num_se3;
        }
        int get_num_so3(){
            return num_so3;
        }
        vector<Sophus::SE3d> get_se3_elements(){
            return se3_elems;
        }
        vector<Sophus::SO3d> get_so3_elements(){
            return so3_elems;
        }
        // Print rigid object
        void print_obj(ofstream& output);
};

/* Class for a trajectory/motion travel by a rigid object. To specify motion, only keyframes of the position are needed.
The whole motion is then interpolated, and velocity (Lie algebra) is calculated.
*/
class Trajectory{
    private:
        // Number of points on this trajectory
        int num_pts;
        // Number of SE3 component of the rigid object
        int num_se3;
        // Number of SO3 component of the rigid object
        int num_so3;
        vector<Rigid> pos;
        vector<double> times;
        vector<vector<Eigen::Vector3d>> se3_velocities;
        vector<vector<Eigen::Vector3d>> so3_velocities;

    public:
        Trajectory(){
        }
        Trajectory(int num_se3, int num_so3){
            num_pts = 0;
            this->num_se3 = num_se3;
            this->num_so3 = num_so3;
        }

        Trajectory(int num_se3, int num_so3, vector<vector<vector<double>>>& se3Pos, 
            vector<vector<vector<double>>>& se3Ang, vector<vector<vector<double>>>& so3Ang, 
            vector<double>& ts, vector<int>& numPtsBtw){
            num_pts = 0;
            this->num_se3 = num_se3;
            this->num_so3 = num_so3;
            feedData(se3Pos, se3Ang, so3Ang, ts, numPtsBtw);
        }

        void feedData(vector<vector<vector<double>>>& se3Pos, vector<vector<vector<double>>>& se3Ang, 
            vector<vector<vector<double>>>& so3Ang, vector<double>& ts, vector<int>& numPtsBtw);

        // Output (interpolated) position and velocity
        void output_motion(ofstream& output);
};

vector<Trajectory> read_keyframes(string input_file);

void output_trajectories(vector<Trajectory>& trajectories, string output_file);

#endif