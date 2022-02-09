#include <iostream>
#include <fstream>
#include <vector>
#include <Eigen/Core>
#include "sophus/geometry.hpp"
#include "sophus/interpolate.hpp"
#include "sophus/velocities.hpp"
#include "rigid.hpp"

// tmp
using namespace std;

void Rigid::feedData(vector<vector<double>>& se3Pos, vector<vector<double>>& se3Ang, vector<vector<double>>& so3Ang){
    this->num_se3 = int(se3Pos.size());
    this->num_so3 = int(so3Ang.size());
    Sophus::SO3d rx, ry, rz;
    // Feed SE3 elements for rigid object from position and angle (orientation) data
    for(int i = 0; i < this->num_se3; i++){
        rx = Sophus::SO3d::rotX(kPi*se3Ang[i][0]);
        ry = Sophus::SO3d::rotY(kPi*se3Ang[i][1]);
        rz = Sophus::SO3d::rotZ(kPi*se3Ang[i][2]);
        Eigen::Vector3d p(se3Pos[i][0], se3Pos[i][1], se3Pos[i][2]);
        this->se3_elems.push_back(Sophus::SE3d(rx*ry*rz, p));
    }
    // Feed SE3 elements for rigid object from position and angle (orientation) data
    for(int i = 0; i < this->num_so3; i++){
        rx = Sophus::SO3d::rotX(kPi*so3Ang[i][0]);
        ry = Sophus::SO3d::rotY(kPi*so3Ang[i][1]);
        rz = Sophus::SO3d::rotZ(kPi*so3Ang[i][2]);
        this->so3_elems.push_back(rx*ry*rz);
    }
}

/* Print the following: 
    First line: num_se3 tuples of 3 double numbers for each SE3 component position
    Second line: num_se3 tuples of 3 double numbers for each SE3 component angle
    Third line: num_so3 tuples of 3 double numbers for each SO3 component angle
*/
void Rigid::print_obj(ofstream& output){
    vector<Sophus::SE3d> se3_elems = this->get_se3_elements();
    vector<Sophus::SO3d> so3_elems = this->get_so3_elements();
    int num_se3 = this->get_num_se3(), num_so3 = this->get_num_so3();
    for(int i = 0; i < num_se3; i++)
        output << se3_elems[i].translation()[0] << " " << se3_elems[i].translation()[1] << " " << se3_elems[i].translation()[2] << " ";
    output << endl;
    for(int i = 0; i < num_se3; i++)
        output << se3_elems[i].angleX() << " " << se3_elems[i].angleY() << " " << se3_elems[i].angleZ() << " ";
    output << endl;
    for(int i = 0; i < num_so3; i++)
        output << so3_elems[i].angleX() << " " << so3_elems[i].angleY() << " " << so3_elems[i].angleZ() << " ";
    output << endl;

}

// Helper interpolation functions
vector<double> interpolate1d(double start, double end, int num_pts){
    vector<double> results(num_pts);
    for(int i = 0; i < num_pts; i++)
        results[i] = start + (i*(end-start))/num_pts;
    return results;
}
vector<Rigid> interpolate_pose(vector<vector<double>>& se3Pos_start, vector<vector<double>>& se3Pos_end, vector<vector<double>>& se3Ang_start, 
            vector<vector<double>>& se3Ang_end, vector<vector<double>>& so3Ang_start, vector<vector<double>>& so3Ang_end, int num_pts){
    int num_se3 = se3Pos_start.size(), num_so3 = so3Ang_start.size();
    vector<vector<vector<double>>> se3Pos_interp(num_pts, vector<vector<double>>(num_se3, vector<double>(3)));
    vector<vector<vector<double>>> se3Ang_interp(num_pts, vector<vector<double>>(num_se3, vector<double>(3)));
    vector<vector<vector<double>>> so3Ang_interp(num_pts, vector<vector<double>>(num_so3, vector<double>(3)));
    vector<double> interpolate_results;
    for(int i = 0; i < num_se3; i++){
        for(int j = 0; j < 3; j++){
            // se3 position
            interpolate_results = interpolate1d(se3Pos_start[i][j], se3Pos_end[i][j], num_pts);
            for(int k = 0; k < num_pts; k++)
                se3Pos_interp[k][i][j] = interpolate_results[k];
            // se3 angle
            interpolate_results = interpolate1d(se3Ang_start[i][j], se3Ang_end[i][j], num_pts);
            for(int k = 0; k < num_pts; k++)
                se3Ang_interp[k][i][j] = interpolate_results[k];
            // so3 angle
            interpolate_results = interpolate1d(so3Ang_start[i][j], so3Ang_end[i][j], num_pts);
            for(int k = 0; k < num_pts; k++)
                so3Ang_interp[k][i][j] = interpolate_results[k];
        }
    }
    vector<Rigid> interpolate_rigid_objs(num_pts);
    for(int k = 0; k < num_pts; k++){
        interpolate_rigid_objs[k] = Rigid(se3Pos_interp[k], se3Ang_interp[k], so3Ang_interp[k]);
    }
    return interpolate_rigid_objs;
}

void Trajectory::feedData(vector<vector<vector<double>>>& se3Pos, vector<vector<vector<double>>>& se3Ang, 
    vector<vector<vector<double>>>& so3Ang, vector<double>& ts, vector<int>& numPtsBtw){
    // Interpolate poses and times to get a trajectory of poses
    for(int i = 0; i < ts.size()-1; i++){
        vector<Rigid> next_positions = interpolate_pose(se3Pos[i], se3Pos[i+1], se3Ang[i], 
                                            se3Ang[i+1], so3Ang[i], so3Ang[i+1], numPtsBtw[i]);
        vector<double> next_times = interpolate1d(ts[i], ts[i+1], numPtsBtw[i]);
        pos.insert(pos.end(), next_positions.begin(), next_positions.end());
        times.insert(times.end(), next_times.begin(), next_times.end());
        num_pts += numPtsBtw[i];
    }
    int last_ind = ts.size()-1; num_pts++;
    times.push_back(ts[last_ind]);
    pos.push_back(Rigid(se3Pos[last_ind], se3Ang[last_ind], so3Ang[last_ind]));
    // Compute each pose of rigid object's motion/trajectory. Each SE3/SO3 component is calculated separately
    vector<vector<Sophus::SE3d>> se3_elems_on_traj(num_pts, vector<Sophus::SE3d>(num_se3));
    vector<vector<Sophus::SO3d>> so3_elems_on_traj(num_pts, vector<Sophus::SO3d>(num_so3));
    vector<Sophus::SE3d> se3_elems;
    vector<Sophus::SO3d> so3_elems;
    for(int i = 0; i < num_pts; i++){
        se3_elems = pos[i].get_se3_elements();
        so3_elems = pos[i].get_so3_elements();
        for(int j = 0; j < num_se3; j++)
            se3_elems_on_traj[i][j] = se3_elems[j];
        for(int j = 0; j < num_so3; j++)
            so3_elems_on_traj[i][j] = so3_elems[j];
    }
    // Calculate (interpolative) function in time that represents the motion/trajectory over separate so3/se3 components
    vector<function<Sophus::SE3d(double)>> se3_traj_fct(num_se3);
    vector<function<Sophus::SO3d(double)>> so3_traj_fct(num_so3);
    int num_times = times.size();
    int ind;
    for(int i = 0; i < num_se3; i++){
        ind = i;
        se3_traj_fct[i] = [&](double t)->Sophus::SE3d{
            if (t <= times[0])
                return se3_elems_on_traj[0][ind];
            else if (t >= times[num_pts-1])
                return se3_elems_on_traj[num_pts-1][ind];

            int time_ind = upper_bound(times.begin(), times.end(), t)-times.begin()-1;
            double time_interpolate = double(t-times[time_ind])/(times[time_ind+1]-times[time_ind]);
            Sophus::SE3d tmp = interpolate(se3_elems_on_traj[time_ind][ind], se3_elems_on_traj[time_ind+1][ind], time_interpolate);
            return tmp;
        };
    }
    for(int i = 0; i < num_so3; i++){
        ind = i;
        so3_traj_fct[i] = [&](double t)->Sophus::SO3d{
            if (t <= times[0])
                return so3_elems_on_traj[0][ind];
            else if (t >= times[num_pts-1])
                return so3_elems_on_traj[num_pts-1][ind];

            int time_ind = upper_bound(times.begin(), times.end(), t)-times.begin()-1;
            double time_interpolate = double(t-times[time_ind])/(times[time_ind+1]-times[time_ind]);
            return interpolate(so3_elems_on_traj[time_ind][ind], so3_elems_on_traj[time_ind+1][ind], time_interpolate);
        };
    }
    // Numerically calculate velocities at each time instance over each direct (se3 or so3) component
    se3_velocities = vector<vector<Eigen::Vector3d>>(num_pts, vector<Eigen::Vector3d>(num_se3));
    so3_velocities = vector<vector<Eigen::Vector3d>>(num_pts, vector<Eigen::Vector3d>(num_so3));
    double t = 0;
    const double eps = 1e-6;
    const double finite_diff_eps = 3e-3;
    for(int i = 0; i < num_pts; i++){
        t = times[i];
        for(int j = 0; j < num_se3; j++){
            se3_velocities[i][j] = Sophus::experimental::finiteDifferenceRotationalVelocity(se3_traj_fct[j], t, finite_diff_eps);
            for(int l = 0; l < 3; l++)
                if(abs(se3_velocities[i][j][l]) < eps)
                    se3_velocities[i][j][l] = 0;
        }
        for(int j = 0; j < num_so3; j++){
            so3_velocities[i][j] = Sophus::experimental::finiteDifferenceRotationalVelocity(so3_traj_fct[j], t, finite_diff_eps);
            for(int l = 0; l < 3; l++)
                if(abs(so3_velocities[i][j][l]) < eps)
                    so3_velocities[i][j][l] = 0;
        }
    }
}

// Output (interpolated) position and velocity
void Trajectory::output_motion(ofstream& output){
    output << num_pts << " " << num_se3 << " " << num_so3 << endl;
    for(int i = 0; i < num_pts; i++){
        pos[i].print_obj(output);
    } 
    for(int i = 0; i < num_pts; i++){
        for(int j = 0; j < num_se3; j++){
            output << se3_velocities[i][j][0] << " " << se3_velocities[i][j][1] << " " << se3_velocities[i][j][2] << " ";
        }
        output << endl;
        for(int j = 0; j < num_so3; j++){
            output << so3_velocities[i][j][0] << " " << so3_velocities[i][j][1] << " " << so3_velocities[i][j][2] << " ";
        }
        output << endl;
    }
}

// Fcts for reading keyframes for some trajectories and for outputting motion (Lie position/velocity)
vector<Trajectory> read_keyframes(string input_file){
    ifstream input;
    input.open(input_file);
    /* Structure of input file:
        Number of trajectories Number of se3 components (num_se3) Number of so3 components (num_so3)
        Next we have paragraphs where each corresponds to a trajectory
            First line: num_pts
            Second line: time stamps (num_pts double numbers)
            Third line: number of points to be interpolated between two poses (num_pts-1 integer numbers))
            Each of the next num_pts line include:
                num_se3 tuples of 3 double numbers for each SE3 component position
                num_se3 tuples of 3 double numbers for each SE3 component angle
                num_so3 tuples of 3 double numbers for each SO3 component angle,
    */
    int num_traj, num_se3, num_so3; 
    input >> num_traj >> num_se3 >> num_so3;
    vector<Trajectory> trajs(num_traj);
    int num_pts = 0;
    for(int k = 0; k < num_traj; k++){
        input >> num_pts;
        vector<vector<vector<double>>> se3Pos(num_pts, vector<vector<double>>(num_se3, vector<double>(3)));
        vector<vector<vector<double>>> se3Ang(num_pts, vector<vector<double>>(num_se3, vector<double>(3)));
        vector<vector<vector<double>>> so3Ang(num_pts, vector<vector<double>>(num_so3, vector<double>(3)));
        vector<double> ts(num_pts);
        vector<int> numPtsBtw(num_pts-1);
        for(int i = 0; i < num_pts; i++){
            input >> ts[i];
        }
        for(int i = 0; i < num_pts-1; i++){
            input >> numPtsBtw[i];
        }
        for(int i = 0; i < num_pts; i++){
            for(int j = 0; j < num_se3; j++)
                input >> se3Pos[i][j][0] >> se3Pos[i][j][1] >> se3Pos[i][j][2];
            for(int j = 0; j < num_se3; j++)
                input >> se3Ang[i][j][0] >> se3Ang[i][j][1] >> se3Ang[i][j][2];
            for(int j = 0; j < num_so3; j++)
                input >> so3Ang[i][j][0] >> so3Ang[i][j][1] >> so3Ang[i][j][2];
        }
        trajs[k] = Trajectory(num_se3, num_so3, se3Pos, se3Ang, so3Ang, ts, numPtsBtw);
    }
    input.close();

    return trajs;
}

void output_trajectories(vector<Trajectory>& trajectories, string output_file){
    ofstream output;
    output.open(output_file);
    int num_traj = trajectories.size();
    output << num_traj << endl;
    for(int k = 0; k < num_traj; k++)
        trajectories[k].output_motion(output);
    output.close();
}

/*
while (std::getline(input, line)){
}
*/