//
// Created by Sandra Alfaro on 24/05/18.
//

#include <vector>
#include <atomic>
#include <chrono>

#ifdef OpenGR_USE_OPENMP
#include <omp.h>
#endif

#include "gr/shared.h"
#include "gr/sampling.h"
#include "gr/accelerators/kdtree.h"
#include "gr/utils/logger.h"

#include "gr/algorithms/congruentSetExplorationBase.h"
#ifdef TEST_GLOBAL_TIMINGS
#   include "../utils/timer.h"
#endif


namespace gr {
  class Timer
  {
  public:
      Timer() : beg_(clock_::now()) {}
      void reset() { beg_ = clock_::now(); }
      double elapsed()  {
          return std::chrono::duration_cast<second_>
              (clock_::now() - beg_).count(); }
      void print(std::string message = ""){
          double t = elapsed();
          reset();
      }
  private:
      typedef std::chrono::high_resolution_clock clock_;
      typedef std::chrono::duration<double, std::ratio<1> > second_;
      std::chrono::time_point<clock_> beg_;
  };


template <typename Traits, typename TransformVisitor,
          typename PairFilteringFunctor,
          template < class, class > typename ... OptExts >
  CongruentSetExplorationBase<Traits, TransformVisitor, PairFilteringFunctor, OptExts ...>::CongruentSetExplorationBase(
          const typename CongruentSetExplorationBase<Traits, TransformVisitor, PairFilteringFunctor, OptExts ...>::OptionsType& options
        , const Utils::Logger& logger )
    : MatchBaseType(options, logger)
    , number_of_trials_(0)
    , best_LCP_(0.0)
//    , options_(options)
{

}

template <typename Traits, typename TransformVisitor,
          typename PairFilteringFunctor,
          template < class, class > class ... OptExts >
CongruentSetExplorationBase<Traits, TransformVisitor, PairFilteringFunctor, OptExts ...>::~CongruentSetExplorationBase(){}


// The main 4PCS function. Computes the best rigid transformation and transfoms
// Q toward P by this transformation
template <typename Traits, typename TransformVisitor,
          typename PairFilteringFunctor,
          template < class, class > class ... OptExts >
template <typename Sampler>
typename CongruentSetExplorationBase<Traits, TransformVisitor, PairFilteringFunctor, OptExts ...>::Scalar
CongruentSetExplorationBase<Traits, TransformVisitor, PairFilteringFunctor, OptExts ...>::ComputeTransformation(
        const std::vector<Point3D>& P,
        const std::vector<Point3D>& Q,
        Eigen::Ref<typename CongruentSetExplorationBase<Traits, TransformVisitor, PairFilteringFunctor, OptExts ...>::MatrixType> transformation,
        const Sampler& sampler,
        TransformVisitor& v) {
  const Scalar kSmallError = 0.00001;
  const int kMinNumberOfTrials = 30;
  const Scalar kDiameterFraction = 0.3;

#ifdef TEST_GLOBAL_TIMINGS
    kdTreeTime = 0;
    totalTime  = 0;
    verifyTime = 0;
#endif

  if (P.empty() || Q.empty()) return kLargeNumber;

  // RANSAC probability and number of needed trials.
  Scalar first_estimation =
          std::log(kSmallError) / std::log(1.0 - pow(MatchBaseType::options_.getOverlapEstimation(),
                                                     static_cast<Scalar>(kMinNumberOfTrials)));
  // We use a simple heuristic to elevate the probability to a reasonable value
  // given that we don't simply sample from P, but instead, we bound the
  // distance between the points in the base as a fraction of the diameter.
  number_of_trials_ =
          static_cast<int>(first_estimation * (MatchBaseType::P_diameter_ / kDiameterFraction) /
                           MatchBaseType::max_base_diameter_);
  if (number_of_trials_ < kMinNumberOfTrials)
      number_of_trials_ = kMinNumberOfTrials;

  // MatchBaseType::template Log<LogLevel::Verbose>( "norm_max_dist: ", MatchBaseType::options_.delta );
  current_trial_ = 0;
  best_LCP_ = 0.0;

  for (int i = 0; i < Traits::size(); ++i) {
      base_[i] = 0;
      current_congruent_[i] = 0;
  }

  MatchBaseType::init(P, Q, sampler);

  int success_quadrilaterals_times = 0;
  Perform_N_steps(number_of_trials_, transformation, v, success_quadrilaterals_times);

#ifdef TEST_GLOBAL_TIMINGS
  MatchBaseType::template Log<LogLevel::Verbose>( "----------- Timings (msec) -------------" );
  MatchBaseType::template Log<LogLevel::Verbose>( " Total computation time  : ", totalTime   );
  MatchBaseType::template Log<LogLevel::Verbose>( " Total verify time       : ", verifyTime  );
  MatchBaseType::template Log<LogLevel::Verbose>( "    Kdtree query         : ", kdTreeTime  );
  MatchBaseType::template Log<LogLevel::Verbose>( "----------------------------------------" );
#endif

  return best_LCP_;
}

// Performs N RANSAC iterations and compute the best transformation. Also,
// transforms the set Q by this optimal transformation.
template <typename Traits, typename TransformVisitor,
          typename PairFilteringFunctor,
          template < class, class > class ... OptExts >
bool
CongruentSetExplorationBase<Traits, TransformVisitor, PairFilteringFunctor, OptExts ...>::Perform_N_steps(
        int n,
        Eigen::Ref<typename CongruentSetExplorationBase<Traits, TransformVisitor, PairFilteringFunctor, OptExts ...>::MatrixType> transformation,
        TransformVisitor &v,
        int &success_quadrilaterals_times) {
  using std::chrono::system_clock;

#ifdef TEST_GLOBAL_TIMINGS
    Timer t (true);
#endif




  // The transformation has been computed between the two point clouds centered
  // at the origin, we need to recompute the translation to apply it to the original clouds
  auto getGlobalTransform = [this](Eigen::Ref<MatrixType> transformation){
    Eigen::Matrix<Scalar, 3, 3> rot, scale;
    Eigen::Transform<Scalar, 3, Eigen::Affine> (MatchBaseType::transform_).computeRotationScaling(&rot, &scale);
    transformation = MatchBaseType::transform_;
    scale.setIdentity();
    transformation.col(3) = (MatchBaseType::qcentroid1_ + MatchBaseType::centroid_P_ -
            ( rot * scale * (MatchBaseType::qcentroid2_ + MatchBaseType::centroid_Q_))).homogeneous();
  };

  v(0, best_LCP_, transformation);

  bool ok = false;
  std::chrono::time_point<system_clock> t0 = system_clock::now(), end;
  for (int i = 0; i < n; ++i)
  {

    // {
    //   std::ofstream ff("/home/bowen/debug/prob_dist_trial"+std::to_string(i)+".txt");
    //   for (int i=0;i<MatchBaseType::_point_probs.size();i++)
    //   {
    //     auto pt = MatchBaseType::sampled_P_3D_[i];
    //     ff<<pt.x()<<" "<<pt.y()<<" "<<pt.z()<<" "<<MatchBaseType::_point_probs[i]<<std::endl;
    //   }
    //   ff.close();
    // }

    ok = TryOneBase(v, success_quadrilaterals_times);

    Scalar fraction_try  = Scalar(i) / Scalar(number_of_trials_);
    Scalar fraction_time =
        std::chrono::duration_cast<std::chrono::seconds>
        (system_clock::now() - t0).count() /
                          MatchBaseType::options_.max_time_seconds;
    Scalar fraction = std::max(fraction_time, fraction_try);


    // ok means that we already have the desired LCP.
    if (i > number_of_trials_ || fraction >= 0.99 || success_quadrilaterals_times>=MatchBaseType::options_.success_quadrilaterals) break;
  }

#ifdef TEST_GLOBAL_TIMINGS
    totalTime += Scalar(t.elapsed().count()) / Scalar(CLOCKS_PER_SEC);
#endif

  return ok;
}



template <typename Traits, typename TransformVisitor,
          typename PairFilteringFunctor,
          template < class, class > class ... OptExts >
bool CongruentSetExplorationBase<Traits, TransformVisitor, PairFilteringFunctor, OptExts ...>::TryOneBase(
        TransformVisitor &v, int &success_quadrilaterals_times) {
        CongruentBaseType base;
        Set congruent_quads;
        if (!generateCongruents(base,congruent_quads))
            return false;
        success_quadrilaterals_times++;
        size_t nb = 0;


        bool match = TryCongruentSet(base,congruent_quads,v,nb);
        //if (nb != 0)
        //  MatchBaseType::Log<LogLevel::Verbose>( "Congruent quads: (", nb, ")    " );

        return match;
}

template <typename Traits, typename TransformVisitor,
          typename PairFilteringFunctor,
          template < class, class > class ... OptExts >
bool CongruentSetExplorationBase<Traits, TransformVisitor, PairFilteringFunctor, OptExts ...>::TryCongruentSet(
        typename CongruentSetExplorationBase<Traits, TransformVisitor, PairFilteringFunctor, OptExts ...>::CongruentBaseType& base,
        typename CongruentSetExplorationBase<Traits, TransformVisitor, PairFilteringFunctor, OptExts ...>::Set& set,
        TransformVisitor &v,
        size_t &nbCongruent)
{
  static const Scalar pi = std::acos(-1);

  // get references to the basis coordinate
  Coordinates references;
  //    std::cout << "Process congruent set for base: \n";
  for (int i = 0; i != Traits::size(); ++i)
  {
    references[i] = MatchBaseType::sampled_P_3D_[base[i]];
    //        std::cout << "[" << base[i] << "]: " << references[i].pos().transpose() << "\n";
  }
  Scalar targetAngle = (references[1].pos() - references[0].pos()).normalized().dot((references[3].pos() - references[2].pos()).normalized());
  //    std::cout << "Target Angle : " << std::acos(targetAngle)*Scalar(180)/pi << std::endl;

  // Centroid of the basis, computed once and using only the three first points
  Eigen::Matrix<Scalar, 3, 1> centroid1 = (references[0].pos() + references[1].pos() + references[2].pos()) / Scalar(3);

  // std::cout << "Congruent set size: " << set.size() <<  std::endl;

#pragma omp parallel for schedule(dynamic) firstprivate(references,centroid1)
  for (int i = 0; i < int(set.size()); ++i)
  {
    Coordinates congruent_candidate;
    const auto &congruent_ids = set[i];
    for (int j = 0; j != Traits::size(); ++j)
      congruent_candidate[j] = MatchBaseType::sampled_Q_3D_[congruent_ids[j]];

    Eigen::Matrix<Scalar, 4, 4> transform;
    // Centroid of the sets, computed in the loop using only the three first points
    Eigen::Matrix<Scalar, 3, 1> centroid2;

#ifdef STATIC_BASE
    MatchBaseType::Log<LogLevel::Verbose>("Ids: ");
    for (int j = 0; j != Traits::size(); ++j)
      MatchBaseType::Log<LogLevel::Verbose>(base[j], "\t");
    MatchBaseType::Log<LogLevel::Verbose>("     ");
    for (int j = 0; j != Traits::size(); ++j)
      MatchBaseType::Log<LogLevel::Verbose>(congruent_ids[j], "\t");
#endif

    centroid2 = (congruent_candidate[0].pos() +
                 congruent_candidate[1].pos() +
                 congruent_candidate[2].pos()) /
                Scalar(3.);

    Scalar rms = -1;

    const bool ok =
        this->ComputeRigidTransformation(references,                 // input congruent quad
                                         congruent_candidate, // tested congruent quad
                                         centroid1,           // input: basis centroid
                                         centroid2,           // input: candidate quad centroid
                                         transform,           // output: transformation
                                         rms,                 // output: rms error of the transformation between the basis and the congruent quad
#ifdef MULTISCALE
                                         true
#else
                                         false
#endif
        ); // state: compute scale ratio ?

    if (ok && rms >= Scalar(0.))
    {

      // We give more tolerant in computing the best rigid transformation.
      if (rms < distance_factor * MatchBaseType::options_.delta)
      {

        //                std::cout << "congruent candidate: [";
        //                for (int j = 0; j!= Traits::size(); ++j)
        //                    std::cout << congruent_ids[j] << " ";
        //                std::cout << "]";

        // The transformation is computed from the point-clouds centered inn [0,0,0]
        // std::cout<<"Computed transform in ComputeRigidTransformation is:\n"<<transform<<std::endl<<std::endl;
        // Verify the rest of the points in Q against P.

        //                std::cout << " " << lcp << " - Angle: ";
        Scalar lcp = 0;
        // compute angle between pairs of points: for debug only
        //                Scalar angle = (congruent_candidate[1].pos() - congruent_candidate[0].pos()).normalized().dot(
        //                      (congruent_candidate[3].pos() - congruent_candidate[2].pos()).normalized());
        //                std::cout << std::acos(angle)*Scalar(180)/pi << " (error: "
        //                          << (std::acos(targetAngle) - std::acos(angle))*Scalar(180)/pi << ")\n";

        // transformation has been computed between the two point clouds centered
        // at the origin, we need to recompute the translation to apply it to the original clouds
        auto getGlobalTransform =
        [this, transform, centroid1, centroid2](Eigen::Ref<MatrixType> transformation) {
            Eigen::Matrix<Scalar, 3, 3> rot, scale;
            Eigen::Transform<Scalar, 3, Eigen::Affine>(transform).computeRotationScaling(&rot, &scale);
            transformation = transform;
            transformation.col(3) = (centroid1 + MatchBaseType::centroid_P_ -
                                     (rot * scale * (centroid2 + MatchBaseType::centroid_Q_)))
                                        .homogeneous();
        };
        lcp = Verify(transform);

        if (lcp > 0.0 /*&& lcp_hash[lcp]==false*/)
        {
            // lcp_hash[lcp] = true;
            getGlobalTransform(transform);
#pragma omp critical
            {
                MatchBaseType::_pose_hypo.push_back(transform);
                MatchBaseType::_pose_lcp_scores.push_back(lcp);
            }
        }
      }
    }
  }

// If we reached here we do not have yet the desired LCP.
return MatchBaseType::_pose_lcp_scores.size() > 0;
} // namespace gr

// Verify a given transformation by computing the number of points in P at
// distance at most (normalized) delta from some point in Q. In the paper
// we describe randomized verification. We apply deterministic one here with
// early termination. It was found to be fast in practice.
template <typename Traits, typename TransformVisitor,
          typename PairFilteringFunctor,
          template < class, class > class ... OptExts >
typename CongruentSetExplorationBase<Traits, TransformVisitor, PairFilteringFunctor, OptExts ...>::Scalar
CongruentSetExplorationBase<Traits, TransformVisitor, PairFilteringFunctor, OptExts ...>::Verify(
        const Eigen::Ref<const typename CongruentSetExplorationBase<Traits, TransformVisitor, PairFilteringFunctor, OptExts ...>::MatrixType> &mat) const {
    using RangeQuery = typename gr::KdTree<Scalar>::template RangeQuery<>;

#ifdef TEST_GLOBAL_TIMINGS
    Timer t_verify (true);
#endif

    // We allow factor 2 scaling in the normalization.
    const Scalar epsilon = MatchBaseType::options_.delta;
#ifdef OPENGR_USE_WEIGHTED_LCP
    std::atomic<float> good_points(0);

    auto kernel = [](Scalar x) {
        return std::pow(std::pow(x,4) - Scalar(1), 2);
    };

    auto computeWeight = [kernel](Scalar sqx, Scalar th) {
        return kernel( std::sqrt(sqx) / th );
    };
#else
    std::atomic_uint good_points(0);
#endif
    const size_t number_of_points = MatchBaseType::sampled_Q_3D_.size();

    const Scalar sq_eps = epsilon*epsilon;
#ifdef OPENGR_USE_WEIGHTED_LCP
    const Scalar    eps = std::sqrt(sq_eps);
#endif

    // #pragma omp parallel for schedule(dynamic) firstprivate(kd_tree_local)
    for (size_t i = 0; i < number_of_points; ++i)
    {

      // Use the kdtree to get the nearest neighbor
#ifdef TEST_GLOBAL_TIMINGS
      Timer t(true);
#endif

      RangeQuery query;
      query.queryPoint = (mat * MatchBaseType::sampled_Q_3D_[i].pos().homogeneous()).template head<3>();
      query.sqdist = sq_eps;

      auto result = MatchBaseType::kd_tree_.doQueryRestrictedClosestIndex(query);

#ifdef TEST_GLOBAL_TIMINGS
      kdTreeTime += Scalar(t.elapsed().count()) / Scalar(CLOCKS_PER_SEC);
#endif

      if (result.first != gr::KdTree<Scalar>::invalidIndex())
      {
        // Point3D query_pt = sampled_Q_3D_local[i];
        // Point3D neighbor_pt = MatchBaseType::sampled_P_3D_[result.first];
        // const float cos_angle_thres = std::cos(45/180.0*M_PI);   // cannot expect too much for this global registration
        // if ( query_pt.normal().dot(neighbor_pt.normal())<cos_angle_thres ) continue;

        //      Point3D& q = sampled_P_3D_[near_neighbor_index[0]];
        //      bool rgb_good =
        //          (p.rgb()[0] >= 0 && q.rgb()[0] >= 0)
        //              ? cv::norm(p.rgb() - q.rgb()) < options_.max_color_distance
        //              : true;
        //      bool norm_good = norm(p.normal()) > 0 && norm(q.normal()) > 0
        //                           ? fabs(p.normal().ddot(q.normal())) >= cos_dist
        //                           : true;
        //      if (rgb_good && norm_good) {
#ifdef OPENGR_USE_WEIGHTED_LCP
        assert(result.second <= query.sqdist);
        good_points = good_points + computeWeight(result.second, eps);
#else
        good_points++;
#endif
        //      }
      }

      // We can terminate if there is no longer chance to get better than the
      // current best LCP.
      // if (number_of_points - i + good_points < terminate_value) {
      //     break;
      // }
    }

#ifdef TEST_GLOBAL_TIMINGS
    verifyTime += Scalar(t_verify.elapsed().count()) / Scalar(CLOCKS_PER_SEC);
#endif
    return Scalar(good_points) / Scalar(number_of_points);
}

}
