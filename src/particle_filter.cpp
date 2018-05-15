/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"

using namespace std;
default_random_engine gen;
void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).
	num_particles = 10;
	
	// This line creates a normal (Gaussian) distribution for x,y and theta.
	normal_distribution<double> dist_x(0, std[0]);
	normal_distribution<double> dist_y(0,std[1]);
	normal_distribution<double> dist_theta(0,std[2]);
	for (int i = 0;i<num_particles;i++)
	{
		Particle p = {};
		p.id = i;
		p.x = x + dist_x(gen);
		p.y = y + dist_y(gen);
		p.theta =theta + dist_theta(gen);
		p.weight = 1.0;

		particles.push_back(p);
	}
	is_initialized=true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/
	
	// This line creates a normal (Gaussian) distribution for x,y and theta.
	normal_distribution<double> dist_noise_x(0, std_pos[0]);
	normal_distribution<double> dist_noise_y(0,std_pos[1]);
	normal_distribution<double> dist_noise_theta(0,std_pos[2]);

	for (int i = 0;i<num_particles;i++)
	{
		Particle &p = particles[i];
		double delta_x, delta_y, delta_theta;

		//If yaw rate id greater than zero,then prediction equations are different 
        if(fabs(yaw_rate) > 0.001){
            delta_x = (velocity/yaw_rate)*(sin(p.theta+ yaw_rate*delta_t) - sin(p.theta));
            delta_y = (velocity/yaw_rate)*(cos(p.theta) - cos(p.theta+ yaw_rate*delta_t));
			delta_theta = yaw_rate*delta_t;
        }else{
            delta_x = velocity*delta_t*cos(p.theta);
            delta_y = velocity*delta_t*sin(p.theta);
            delta_theta = yaw_rate*delta_t;
        }

        // Add noise for delta_x,delta_y and delta_theta
        delta_x += dist_noise_x(gen);
		delta_y += dist_noise_y(gen);
		delta_theta += dist_noise_theta(gen);

		//Updating particle coordinates with predicte values and noise
        p.x += delta_x;
        p.y += delta_y;
        p.theta += delta_theta;

	}

}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.

	for (unsigned int i = 0 ; i<observations.size();i++)
	{
		//Each observation
		LandmarkObs obs = observations[i];

		//initializing minimum distance by maximum value.
		double min_distance = numeric_limits<double>::max();

		//landmark id which will be associated with observation
		int associated_id = -1;

		//Associating each observation with a landmark positions within predicted sensor range
		for (unsigned int j = 0 ; j<predicted.size();j++)
		{
			LandmarkObs pred = predicted[j];
			double current_distance = dist(obs.x,obs.y,pred.x,pred.y);
			if(current_distance < min_distance)
			{
				min_distance = current_distance;
				associated_id = pred.id;
			}
		}
		
		observations[i].id = associated_id;
	}
	
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		const std::vector<LandmarkObs> &observations, const Map &map_landmarks) {
	// TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation 
	//   3.33
	//   http://planning.cs.uiuc.edu/node99.html

	//Updating particle weight of all the particles

	//Normalization term used in multi-variate gaussian probability for updating weights
	double sig_x = std_landmark[0];
	double sig_y = std_landmark[1];

	float gauss_norm  = (1/(2*M_PI*sig_x*sig_y));

	for (int i =0;i<num_particles;i++)
	{
		//x,y and theta for each particle
		double p_x = particles[i].x;
		double p_y=particles[i].y;
		double p_theta=particles[i].theta;

		//List for transformed observation from vehicle coordinates to map  coordinates
		vector<LandmarkObs> transform_obs;

		//Looping on all the observation to convert to map coordinates
		for (unsigned int j = 0 ; j < observations.size();j++)
		{
			double obs_x = observations[j].x;
			double obs_y = observations[j].y;
	
			double tf_x =  p_x + cos(p_theta)*obs_x-sin(p_theta)*obs_y;
			double tf_y = p_y + sin(p_theta)*obs_x+cos(p_theta)*obs_y;

			transform_obs.push_back(LandmarkObs{observations[j].id,tf_x,tf_y});
		}
		

		//List to hold landmark position predicted within the sensor range
		vector<LandmarkObs> predictions;

		//Taking landmarks within the sensor range
		for (unsigned int k =0;k < map_landmarks.landmark_list.size();k++)
		{
			float l_x = map_landmarks.landmark_list[k].x_f;
			float l_y = map_landmarks.landmark_list[k].y_f;
			int l_id = map_landmarks.landmark_list[k].id_i;

			if(fabs(l_x - p_x)<= sensor_range && fabs(l_y - p_y)<= sensor_range)
			{
				predictions.push_back(LandmarkObs{l_id,l_x,l_y});
			}
		}

		//Associate each transformed observation to a landmark position.
		dataAssociation(predictions,transform_obs);

		//Reinitializing weights
		particles[i].weight = 1.0;

		//Updating weight of the particle by using multivariate guassian on each observation.
		for (unsigned int j=0;j<transform_obs.size();j++)
		{
			double mu_x,mu_y;
			double x_obs = transform_obs[j].x;
			double y_obs = transform_obs[j].y;
			int associate_id = transform_obs[j].id;
			
			

			for (unsigned int k =0;k<predictions.size();k++)
			{
				if(associate_id == predictions[k].id)
				{
					mu_x = predictions[k].x;
					mu_y = predictions[k].y;
					
				}		
			}
			double dx = (pow(x_obs-mu_x,2)/(2*pow(sig_x,2)));
			double dy = (pow(y_obs-mu_y,2)/(2*pow(sig_y,2)));
			double exponent = exp(-(dx+dy));
			double weight = gauss_norm*exponent;
			particles[i].weight *= weight; 
		} 
	}
}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution

	vector<Particle> new_particles;
	vector<double> weights;
	for (int i = 0;i<num_particles;i++){
		weights.push_back(particles[i].weight);
		
	}

    random_device rd;
    mt19937 gen(rd());
    discrete_distribution<> distribution(weights.begin(), weights.end());

    for(int i = 0; i < num_particles; i++){
        Particle p = particles[distribution(gen)];
        new_particles.push_back(p);
    }
    particles = new_particles;

}

void ParticleFilter::SetAssociations(Particle& particle, const std::vector<int>& associations, 
                                     const std::vector<double>& sense_x, const std::vector<double>& sense_y)
{
    //particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
    // associations: The landmark id that goes along with each listed association
    // sense_x: the associations x mapping already converted to world coordinates
    // sense_y: the associations y mapping already converted to world coordinates

    particle.associations= associations;
    particle.sense_x = sense_x;
    particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
