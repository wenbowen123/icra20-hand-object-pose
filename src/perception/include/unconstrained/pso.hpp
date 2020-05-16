/*################################################################################
  ##
  ##   Copyright (C) 2016-2018 Keith O'Hara
  ##
  ##   This file is part of the OptimLib C++ library.
  ##
  ##   Licensed under the Apache License, Version 2.0 (the "License");
  ##   you may not use this file except in compliance with the License.
  ##   You may obtain a copy of the License at
  ##
  ##       http://www.apache.org/licenses/LICENSE-2.0
  ##
  ##   Unless required by applicable law or agreed to in writing, software
  ##   distributed under the License is distributed on an "AS IS" BASIS,
  ##   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
  ##   See the License for the specific language governing permissions and
  ##   limitations under the License.
  ##
  ################################################################################*/

/*
 * Particle Swarm Optimization (PSO)
 */

#ifndef _optim_pso_HPP
#define _optim_pso_HPP





class ArgPasser
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

    std::string name;
    PointCloudRGBNormal::Ptr model, scene_hand_region, scene_hand_region_removed_noise, scene_remove_swivel;
    Eigen::Matrix4f model_in_cam, model2handbase, finger_out2parent;
    boost::shared_ptr<pcl::KdTreeFLANN<pcl::PointXYZRGBNormal> > kdtree_scene;
    float dist_thres, outter_pt_dist;
    FingerProperty finger_property, finger_out_property;
    float objval;
    Eigen::Vector4f pair_tip1, pair_tip2;
    ConfigParser cfg;
	SDFchecker sdf;
    ArgPasser()
    {
		model_in_cam.setIdentity();
		model2handbase.setIdentity();
        finger_out2parent.setIdentity();
		objval = std::numeric_limits<float>::max();
		pair_tip1.setZero();
        pair_tip2.setZero();
        model = boost::make_shared<pcl::PointCloud<pcl::PointXYZRGBNormal>>();
        scene_hand_region = boost::make_shared<pcl::PointCloud<pcl::PointXYZRGBNormal>>();
        scene_hand_region_removed_noise = boost::make_shared<pcl::PointCloud<pcl::PointXYZRGBNormal>>();
        scene_remove_swivel = boost::make_shared<pcl::PointCloud<pcl::PointXYZRGBNormal>>();
        kdtree_scene = boost::make_shared<pcl::KdTreeFLANN<pcl::PointXYZRGBNormal> >();

    };

    ~ArgPasser()
    {
    }

    void reset()
    {
        model_in_cam.setIdentity();
        model2handbase.setIdentity();
        finger_out2parent.setIdentity();
        objval = std::numeric_limits<float>::max();
        pair_tip1.setZero();
        pair_tip2.setZero();
        model = boost::make_shared<pcl::PointCloud<pcl::PointXYZRGBNormal>>();
        scene_hand_region = boost::make_shared<pcl::PointCloud<pcl::PointXYZRGBNormal>>();
        scene_hand_region_removed_noise = boost::make_shared<pcl::PointCloud<pcl::PointXYZRGBNormal>>();
        scene_remove_swivel = boost::make_shared<pcl::PointCloud<pcl::PointXYZRGBNormal>>();
        kdtree_scene = boost::make_shared<pcl::KdTreeFLANN<pcl::PointXYZRGBNormal> >();
        sdf.reset();
    }

    ArgPasser(const ArgPasser &other)   //NOTE: deep copy
    {
        name = other.name;
        model = boost::make_shared<PointCloudRGBNormal>();
        *model = *(other.model);
        scene_hand_region = boost::make_shared<PointCloudRGBNormal>();
        *scene_hand_region = *(other.scene_hand_region);
        scene_hand_region_removed_noise = boost::make_shared<PointCloudRGBNormal>();
        *scene_hand_region_removed_noise = *(other.scene_hand_region_removed_noise);
        scene_remove_swivel = boost::make_shared<PointCloudRGBNormal>();
        *scene_remove_swivel = *(other.scene_remove_swivel);
        model_in_cam = other.model_in_cam;
        model2handbase = other.model2handbase;
        finger_out2parent = other.finger_out2parent;
        kdtree_scene = boost::make_shared<pcl::KdTreeFLANN<pcl::PointXYZRGBNormal> >();
        *kdtree_scene = *(other.kdtree_scene);
        dist_thres = other.dist_thres;
        outter_pt_dist = other.outter_pt_dist;
        finger_property = other.finger_property;
        finger_out_property = other.finger_out_property;
        objval = other.objval;
        pair_tip1 = other.pair_tip1;
        pair_tip2 = other.pair_tip2;
        cfg = other.cfg;
        sdf = other.sdf;
    };

    bool operator = (const ArgPasser &other)
    {
        name = other.name;
        model = boost::make_shared<PointCloudRGBNormal>();
        *model = *(other.model);
        scene_hand_region = boost::make_shared<PointCloudRGBNormal>();
        *scene_hand_region = *(other.scene_hand_region);
        scene_hand_region_removed_noise = boost::make_shared<PointCloudRGBNormal>();
        *scene_hand_region_removed_noise = *(other.scene_hand_region_removed_noise);
        scene_remove_swivel = boost::make_shared<PointCloudRGBNormal>();
        *scene_remove_swivel = *(other.scene_remove_swivel);
        model_in_cam = other.model_in_cam;
        model2handbase = other.model2handbase;
        finger_out2parent = other.finger_out2parent;
        kdtree_scene = boost::make_shared<pcl::KdTreeFLANN<pcl::PointXYZRGBNormal> >();
        *kdtree_scene = *(other.kdtree_scene);
        dist_thres = other.dist_thres;
        outter_pt_dist = other.outter_pt_dist;
        finger_property = other.finger_property;
        finger_out_property = other.finger_out_property;
        objval = other.objval;
        pair_tip1 = other.pair_tip1;
        pair_tip2 = other.pair_tip2;
        cfg = other.cfg;
        sdf = other.sdf;
    }

};

bool pso_int(arma::vec& init_out_vals, std::function<double (const arma::vec& vals_inp, arma::vec* grad_out, ArgPasser* opt_data)> opt_objfn, ArgPasser* opt_data, algo_settings_t* settings_inp);

bool pso(arma::vec& init_out_vals, std::function<double (const arma::vec& vals_inp, arma::vec* grad_out, ArgPasser* opt_data)> opt_objfn, ArgPasser* opt_data);
bool pso(arma::vec& init_out_vals, std::function<double (const arma::vec& vals_inp, arma::vec* grad_out, ArgPasser* opt_data)> opt_objfn, ArgPasser* opt_data, algo_settings_t& settings);

//

inline
bool
pso_int(arma::vec& init_out_vals, std::function<double (const arma::vec& vals_inp, arma::vec* grad_out, ArgPasser* opt_data)> opt_objfn, ArgPasser* opt_data, algo_settings_t* settings_inp)
{
    bool success = false;
    // arma::arma_rng::set_seed_random();
    arma::arma_rng::set_seed(0);
    const size_t n_vals = init_out_vals.n_elem;

    //
    // PSO settings

    algo_settings_t settings;

    if (settings_inp) {
        settings = *settings_inp;
    }

    const double err_tol = settings.err_tol;

    const bool center_particle = settings.pso_center_particle;

    const size_t n_pop = (center_particle) ? settings.pso_n_pop + 1 : settings.pso_n_pop;
    const size_t n_gen = settings.pso_n_gen;
    const uint_t check_freq = (settings.pso_check_freq > 0) ? settings.pso_check_freq : n_gen ;

    const uint_t inertia_method = settings.pso_inertia_method;

    double par_w = settings.pso_par_initial_w;
    const double par_w_max = settings.pso_par_w_max;
    const double par_w_min = settings.pso_par_w_min;
    const double par_damp = settings.pso_par_w_damp;

    const uint_t velocity_method = settings.pso_velocity_method;

    double par_c_cog = settings.pso_par_c_cog;   // Scaling factor to search towards the particle's best known position
    double par_c_soc = settings.pso_par_c_soc;  // Scaling factor to search towards the swarm's best known position

    const double par_initial_c_cog = settings.pso_par_initial_c_cog;
    const double par_final_c_cog = settings.pso_par_final_c_cog;
    const double par_initial_c_soc = settings.pso_par_initial_c_soc;
    const double par_final_c_soc = settings.pso_par_final_c_soc;

    const arma::vec par_initial_lb = (settings.pso_initial_lb.n_elem == n_vals) ? settings.pso_initial_lb : init_out_vals - 0.5;
    const arma::vec par_initial_ub = (settings.pso_initial_ub.n_elem == n_vals) ? settings.pso_initial_ub : init_out_vals + 0.5;

    const bool vals_bound = settings.vals_bound;

    const arma::vec lower_bounds = settings.lower_bounds;
    const arma::vec upper_bounds = settings.upper_bounds;

    const arma::uvec bounds_type = determine_bounds_type(vals_bound, n_vals, lower_bounds, upper_bounds);

    // lambda function for box constraints

    std::function<double (const arma::vec& vals_inp, arma::vec* grad_out, ArgPasser* box_data)> box_objfn \
    = [opt_objfn, vals_bound, bounds_type, lower_bounds, upper_bounds] (const arma::vec& vals_inp, arma::vec* grad_out, ArgPasser* opt_data) \
    -> double
    {
        if (vals_bound)
        {
            arma::vec vals_inv_trans = invLinearTransform(vals_inp, bounds_type, lower_bounds, upper_bounds);

            return opt_objfn(vals_inv_trans,nullptr,opt_data);
        }
        else
        {
            return opt_objfn(vals_inp,nullptr,opt_data);
        }
    };


    //
    // initialize
    arma::vec objfn_vals(n_pop);
    arma::mat P = arma::randu(n_pop, n_vals);

// omp_set_dynamic(0);     // Explicitly disable dynamic teams
// omp_set_num_threads(1); // Use 1 threads for all consecutive parallel regions
    //NOTE: % is element-wise product in arma.  P in range [0,1]
    if (center_particle)
    {
        P.row(n_pop - 1) = arma::sum(P.rows(0, n_pop - 2), 0) / static_cast<double>(n_pop - 1); // center vector
    }
#pragma omp parallel
    {
        ArgPasser opt_data_local = *opt_data;
#pragma omp for schedule(dynamic)
        for (size_t i = 0; i < n_pop; i++)
        {
            arma::rowvec particle = P.row(i).t();
            double prop_objfn_val = box_objfn(particle, nullptr, &opt_data_local);

            if (!std::isfinite(prop_objfn_val))
            {
                prop_objfn_val = std::numeric_limits<double>::max();
            }

            objfn_vals(i) = prop_objfn_val;
        }
    }
    arma::vec best_vals = objfn_vals;
    arma::mat best_vecs = P;

    double cur_global_best_val = objfn_vals.min();
    double global_best_val_check = cur_global_best_val;
    arma::rowvec global_best_vec = P.row( objfn_vals.index_min() );

    //
    // begin loop

    uint_t iter = 0;
    double err = 2.0*err_tol;

    arma::mat V = arma::randu(n_pop,n_vals);   //Initial velocity

    while (err > err_tol && iter < n_gen)
    {
        iter++;

        //
        // parameter updating
        if (velocity_method == 2)
        {
            par_c_cog = par_initial_c_cog - (par_initial_c_cog - par_final_c_cog) * (iter + 1) / n_gen;
            par_c_soc = par_initial_c_soc - (par_initial_c_soc - par_final_c_soc) * (iter + 1) / n_gen;
        }

        // population loop
        arma::mat global_best_mat(n_pop, n_vals);
        global_best_mat.each_row() = global_best_vec;
        V = par_w * V + par_c_cog * arma::randu(n_pop, n_vals) % (best_vecs - P) + par_c_soc * arma::randu(n_pop, n_vals) % (global_best_mat - P);
        P += V;
        if (center_particle)
        {
            P.row(n_pop-1) = arma::sum(P.rows(0, n_pop - 2), 0) / static_cast<double>(n_pop - 1); // center vector
        }
        P = arma::clamp(P, 0.0, 1.0);
#pragma omp parallel
        {
            ArgPasser opt_data_local = *opt_data;
#pragma omp for schedule(dynamic)
            for (size_t i = 0; i < n_pop; i++)
            {
                arma::rowvec particle = P.row(i).t();
                double prop_objfn_val = box_objfn(particle, nullptr, &opt_data_local);

                if (!std::isfinite(prop_objfn_val))
                {
                    prop_objfn_val = std::numeric_limits<double>::max();
                }

                objfn_vals(i) = prop_objfn_val;

#pragma omp critical
                {
                    if (objfn_vals(i) < best_vals(i))
                    {
                        best_vals(i) = objfn_vals(i);
                        best_vecs.row(i) = P.row(i);
                    }
                }
            }
        }
        uint_t min_objfn_val_index = best_vals.index_min();
        double min_objfn_val = best_vals(min_objfn_val_index);
        if (min_objfn_val < cur_global_best_val)
        {

            cur_global_best_val = min_objfn_val;
            global_best_vec = best_vecs.row( min_objfn_val_index );
        }

        if (iter%check_freq == 0)
        {
            err = std::abs(cur_global_best_val - global_best_val_check) / (1e-20 + std::abs(global_best_val_check));
        }

        if (cur_global_best_val < global_best_val_check)
        {
            global_best_val_check = cur_global_best_val;
        }

        if (inertia_method == 1) {
            par_w = par_w_min + (par_w_max - par_w_min) * (iter + 1) / n_gen;
        } else {
            par_w *= par_damp;
        }

    }

    //
    if (iter==n_gen)
    {
        std::cout<<"PSO reach max iter\n";
    }
    if (vals_bound) {
        global_best_vec = arma::trans( invLinearTransform(global_best_vec.t(), bounds_type, lower_bounds, upper_bounds) );
    }

    init_out_vals = arma::conv_to< arma::colvec >::from(global_best_vec);
    opt_data->objval = global_best_val_check;
    //

    return true;
}

inline
bool
pso(arma::vec& init_out_vals, std::function<double (const arma::vec& vals_inp, arma::vec* grad_out, ArgPasser* opt_data)> opt_objfn, ArgPasser* opt_data)
{
    return pso_int(init_out_vals,opt_objfn,opt_data,nullptr);
}

inline
bool
pso(arma::vec& init_out_vals, std::function<double (const arma::vec& vals_inp, arma::vec* grad_out, ArgPasser* opt_data)> opt_objfn, ArgPasser* opt_data, algo_settings_t& settings)
{
    return pso_int(init_out_vals,opt_objfn,opt_data,&settings);
}

#endif
