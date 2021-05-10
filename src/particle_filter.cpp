/**
 * particle_filter.cpp
 *
 * Created on: Dec 12, 2016
 * Author: Tiffany Huang
 */

#include "particle_filter.h"

#include <math.h>
#include <algorithm>
#include <iostream>
#include <iterator>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "helper_functions.h"

using std::string;
using std::vector;
using std::normal_distribution;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
  /**
   * TODO: Set the number of particles. Initialize all particles to 
   *   first position (based on estimates of x, y, theta and their uncertainties
   *   from GPS) and all weights to 1. 
   * TODO: Add random Gaussian noise to each particle.
   * NOTE: Consult particle_filter.h for more information about this method 
   *   (and others in this file).
   */
  //num_particles = number;  // Move to main.cpp, read input from command line, default=100; TODO: Set the number of particles
  std::default_random_engine gen;
  normal_distribution<double> dist_x(x, std[0]);
  normal_distribution<double> dist_y(y, std[1]);
  normal_distribution<double> dist_theta(theta, std[2]);

  for(int i=0; i<num_particles; i++){
    Particle particle;
    particle.id = i;
    particle.x = dist_x(gen);
    particle.y = dist_y(gen);
    particle.theta = dist_theta(gen);
    particle.weight = 1;
    particles.push_back(particle);
  }
  weights = vector<double>(num_particles, 0.0);
  is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], 
                                double velocity, double yaw_rate) {
  /**
   * TODO: Add measurements to each particle and add random Gaussian noise.
   * NOTE: When adding noise you may find std::normal_distribution 
   *   and std::default_random_engine useful.
   *  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
   *  http://www.cplusplus.com/reference/random/default_random_engine/
   */
  for(int i=0; i<num_particles; i++){

    std::default_random_engine gen;
        
    double x_f, y_f, theta_f;
    double x_0 = particles[i].x;
    double y_0 = particles[i].y;
    double yaw_0 = particles[i].theta;
    double delta_yaw = yaw_rate*delta_t;

    if(abs(yaw_rate) < 0.000001){
      x_f = x_0 + velocity*cos(yaw_0);
      y_f = y_0 + velocity*sin(yaw_0);
      theta_f = yaw_0;
    }else{
      theta_f = yaw_0 + delta_yaw;
      x_f = x_0 + velocity*(sin(theta_f) - sin(yaw_0))/yaw_rate;
      y_f = y_0 + velocity*(cos(yaw_0) - cos(theta_f))/yaw_rate;
      
    }

    normal_distribution<double> dist_x(x_f, std_pos[0]);
    normal_distribution<double> dist_y(y_f, std_pos[1]);
    normal_distribution<double> dist_theta(theta_f, std_pos[2]);

    particles[i].x = dist_x(gen);
    particles[i].y = dist_y(gen);
    particles[i].theta = dist_theta(gen);
  }
}

void ParticleFilter::dataAssociation(vector<LandmarkObs> predicted, 
                                     vector<LandmarkObs>& observations) {
  /**
   * TODO: Find the predicted measurement that is closest to each 
   *   observed measurement and assign the observed measurement to this 
   *   particular landmark.
   * NOTE: this method will NOT be called by the grading code. But you will 
   *   probably find it useful to implement this method and use it as a helper 
   *   during the updateWeights phase.
   */

}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
                                   const vector<LandmarkObs> &observations, 
                                   const Map &map_landmarks) {
  /**
   * TODO: Update the weights of each particle using a mult-variate Gaussian 
   *   distribution. You can read more about this distribution here: 
   *   https://en.wikipedia.org/wiki/Multivariate_normal_distribution
   * NOTE: The observations are given in the VEHICLE'S coordinate system. 
   *   Your particles are located according to the MAP'S coordinate system. 
   *   You will need to transform between the two systems. Keep in mind that
   *   this transformation requires both rotation AND translation (but no scaling).
   *   The following is a good resource for the theory:
   *   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
   *   and the following is a good resource for the actual equation to implement
   *   (look at equation 3.33) http://planning.cs.uiuc.edu/node99.html
   */
  for(int i=0; i<num_particles; i++){
    //Transform, observations VEH(Particle) -> MAP
    double x_p = particles[i].x;
    double y_p = particles[i].y;
    double theta_p = particles[i].theta;

    vector<int> associations;
    vector<double> sense_x;
    vector<double> sense_y;
    double weight_particle = 1.0;
    for(int j=0; j<observations.size(); j++){
      double min_dis = -1;
      double x_obs = observations[j].x;
      double y_obs = observations[j].y;

      double x_obs_map = x_obs*cos(theta_p) - y_obs*sin(theta_p) + x_p;
      double y_obs_map = x_obs*sin(theta_p) + y_obs*cos(theta_p) + y_p;

      // find the association for j-th observation
      int association;
      double x_association, y_association;
      for(int k=0; k<map_landmarks.landmark_list.size(); k++){
        double x_landmark = map_landmarks.landmark_list[k].x_f;
        double y_landmark = map_landmarks.landmark_list[k].y_f;
        double dis = sqrt(pow(x_obs_map - x_landmark, 2) 
                        + pow(y_obs_map - y_landmark, 2));

        if((min_dis < 0) || (min_dis > dis)){
          min_dis = dis;
          association = map_landmarks.landmark_list[k].id_i;
          x_association = x_landmark;
          y_association = y_landmark;
        }
      }

      // asign one landmark
      associations.push_back(association);
      sense_x.push_back(x_association);
      sense_y.push_back(y_association);

      //get weight for j-th observation
      double std_x = std_landmark[0];
      double std_y = std_landmark[1];
      double gauss_norm = 1.0/(2.0*M_PI*std_x*std_y);
      double exponent = (pow(x_obs_map - x_association, 2) / (2.0 * pow(std_x, 2)))
                       + (pow(y_obs_map - y_association, 2) / (2.0 * pow(std_y, 2)));
      double weight_obs = gauss_norm * exp(-exponent);
      //
      weight_particle = weight_particle*weight_obs;
      std::cout<<"weight_obs:"<<weight_obs;
    }

    //Associate observation landmark <-> map landmark
    SetAssociations(particles[i], associations, sense_x, sense_y);

    //Update weights
    particles[i].weight = weight_particle;
    weights[i] = weight_particle;
  }
}

void ParticleFilter::resample() {
  /**
   * TODO: Resample particles with replacement with probability proportional 
   *   to their weight. 
   * NOTE: You may find std::discrete_distribution helpful here.
   *   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
   */

  std::random_device rd;
  std::mt19937 gen(rd());
  std::discrete_distribution<> id_particle(weights.begin(), weights.end());
  std::vector<Particle> temple_particles;
  for(int n=0; n<num_particles; n++){
    int id = id_particle(gen);
    temple_particles.push_back(particles[id]);
    std::cout<<"resample id:"<<id;
  }
  particles = temple_particles;
}

void ParticleFilter::SetAssociations(Particle& particle, 
                                     const vector<int>& associations, 
                                     const vector<double>& sense_x, 
                                     const vector<double>& sense_y) {
  // particle: the particle to which assign each listed association, 
  //   and association's (x,y) world coordinates mapping
  // associations: The landmark id that goes along with each listed association
  // sense_x: the associations x mapping already converted to world coordinates
  // sense_y: the associations y mapping already converted to world coordinates
  particle.associations= associations;
  particle.sense_x = sense_x;
  particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best) {
  vector<int> v = best.associations;
  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<int>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}

string ParticleFilter::getSenseCoord(Particle best, string coord) {
  vector<double> v;

  if (coord == "X") {
    v = best.sense_x;
  } else {
    v = best.sense_y;
  }

  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<float>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}