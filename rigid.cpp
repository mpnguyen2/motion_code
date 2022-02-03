#include <iostream>
#include <vector>
#include "sophus/geometry.hpp"
#include <Eigen/Core>

// tmp
using namespace std;

Sophus::SO3d R1 = Sophus::SO3d::rotX(kPi / 4);
  Sophus::SO3d R2 = Sophus::SO3d::rotY(kPi / 6);
  Sophus::SO3d R3 = Sophus::SO3d::rotZ(-kPi / 3);
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
        // TODO
    }
    void feedData(vector<vector<double>>& se3Pos, vector<vector<double>>& se3Ang, vector<vector<double>>& so3Ang){
        this->num_se3 = int(se3Pos.size());
        this->num_so3 = int(so3Ang.size());
        Sophus::SO3d rx, ry, rz;
        // Feed SE3 elements for rigid object from position and angle (orientation) data
        for(int i = 0; i < this->num_se3; i++){
            rx = Sophus::SO3d::rotX(kPi*se3Ang[i][0]);
            ry = Sophus::SO3d::rotY(kPi*se3Ang[i][1]);
            rz = Sophus::SO3d::rotZ(kPi*se3Ang[i][2]);
            Eigen::Vector3d v(se3Pos[i][0], se3Pos[i][1], se3Pos[i][2]);
            this->se3_elems.push_back(Sophus::SE3d(rx*ry*rz, v));
        }
        // Feed SE3 elements for rigid object from position and angle (orientation) data
        for(int i = 0; i < this->num_so3; i++){
            rx = Sophus::SO3d::rotX(kPi*so3Ang[i][0]);
            ry = Sophus::SO3d::rotY(kPi*so3Ang[i][1]);
            rz = Sophus::SO3d::rotZ(kPi*so3Ang[i][2]);
            this->so3_elems.push_back(rx*ry*rz);
        }
    }
};


class Trajectory{
    private:
        int num_points = 0;
        vector<Rigid> pos;
        vector<Sophus::Vector3d> velocities;

        Trajectory(string input_key_frames){
            // Read file
            // Feed data
        }
        void feedData(vector<vector<vector<double>>>& se3Pos, vector<vector<vector<double>>>& se3Ang, 
            vector<vector<vector<double>>>& so3Ang, vector<double>& ts, vector<int>& numPtsBtw){

        }

        // Output both position and velocity
        void output(string out_file){

        }
    /*

    Calculate derivative:
    Vector3<Scalar> finiteDifferenceRotationalVelocity(
    std::function<SE3<Scalar>(Scalar)> const& foo_T_bar, Scalar t,
    Scalar h = Constants<Scalar>::epsilon())

    Vector3<Scalar> finiteDifferenceRotationalVelocity(
    std::function<SO3<Scalar>(Scalar)> const& foo_R_bar, Scalar t,
    Scalar h = Constants<Scalar>::epsilon())

    */
};