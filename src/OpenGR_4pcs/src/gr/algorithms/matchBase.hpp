//
// Created by Sandra Alfaro on 24/05/18.
//

#include <vector>
#include <atomic>
#include <chrono>

#ifdef OpenGR_USE_OPENMP
#include <omp.h>
#endif

#include "gr/algorithms/matchBase.h"
#include "gr/shared.h"
#include "gr/sampling.h"
#include "gr/accelerators/kdtree.h"
#include "gr/utils/logger.h"

#ifdef TEST_GLOBAL_TIMINGS
#   include "../utils/timer.h"
#endif


#define MATCH_BASE_TYPE MatchBase<TransformVisitor, OptExts ... >




namespace gr {

int ppfClosestBin(int value, int discretization)
{

    int lower_limit = value - (value % discretization);
    int upper_limit = lower_limit + discretization;

    int dist_from_lower = value - lower_limit;
    int dist_from_upper = upper_limit - value;

    int closest = (dist_from_lower < dist_from_upper) ? lower_limit : upper_limit;

    return closest;
}



void computePPF(Point3D pt1, Point3D pt2, std::vector<int> &ppf)
{
    const float DIST_DISCRET = 5;
    const float ANGLE_DISCRET = 10;
    ppf.clear();
    ppf.resize(4);
    Eigen::Vector3f n1 = pt1.normal().normalized();
    Eigen::Vector3f n2 = pt2.normal().normalized();
    n1.normalize();
    n2.normalize();
    Eigen::Vector3f p1 = pt1.pos();
    Eigen::Vector3f p2 = pt2.pos();
    int dist = static_cast<int>((p1 - p2).norm() * 1000);
    Eigen::Vector3f p1p2 = p2 - p1;
    int n1_p1p2 = static_cast<int>(std::acos(n1.dot(p1p2.normalized())) / M_PI * 180);
    int n2_p1p2 = static_cast<int>(std::acos(n2.dot(p1p2.normalized())) / M_PI * 180);
    int n1_n2 = static_cast<int>(std::acos(n1.dot(n2)) / M_PI * 180);
    ppf[0] = (ppfClosestBin(dist, DIST_DISCRET));
    ppf[1] = (ppfClosestBin(n1_p1p2, ANGLE_DISCRET));
    ppf[2] = (ppfClosestBin(n2_p1p2, ANGLE_DISCRET));
    ppf[3] = (ppfClosestBin(n1_n2, ANGLE_DISCRET));
    }

    template <typename TransformVisitor, template <class, class> typename... OptExts>
    MATCH_BASE_TYPE::MatchBase(const typename MATCH_BASE_TYPE::OptionsType &options,
                               const Utils::Logger &logger)
        : max_base_diameter_(-1), P_mean_distance_(1.0), randomGenerator_(options.randomSeed), logger_(logger), options_(options)
    {
        std::random_device rd;
        _point_index_engine = std::mt19937(0);   // initialize random generator to smaple points according to their probs
    }

template <typename TransformVisitor, template < class, class > typename ... OptExts>
MATCH_BASE_TYPE::~MatchBase(){}


template <typename TransformVisitor, template < class, class > typename ... OptExts>
typename MATCH_BASE_TYPE::Scalar
MATCH_BASE_TYPE::MeanDistance() const {
    const Scalar kDiameterFraction = 0.2;
    using RangeQuery = gr::KdTree<Scalar>::RangeQuery<>;

    int number_of_samples = 0;
    Scalar distance = 0.0;

    for (size_t i = 0; i < sampled_P_3D_.size(); ++i) {

        RangeQuery query;
        query.sqdist = P_diameter_ * kDiameterFraction;
        query.queryPoint = sampled_P_3D_[i].pos().template cast<Scalar>();

        auto resId = kd_tree_.doQueryRestrictedClosestIndex(query , i).first;

        if (resId != gr::KdTree<Scalar>::invalidIndex()) {
            distance += (sampled_P_3D_[i].pos() - sampled_P_3D_[resId].pos()).norm();
            number_of_samples++;
        }
    }

    return distance / number_of_samples;
}


//@sample_pool: will be returned as candidate pool for 4th point sampling
template <typename TransformVisitor, template < class, class > typename ... OptExts>
bool
MATCH_BASE_TYPE::SelectRandomTriangle(int &base1, int &base2, int &base3, std::vector<int> &sample_pool) {
    int number_of_points = sampled_P_3D_.size();
    base1 = base2 = base3 = -1;
    std::discrete_distribution<> sampler(_point_probs.begin(), _point_probs.end());

    // Pick the first point at random.
    int first_point = sampler(_point_index_engine);
    _point_probs[first_point] *= options_.sample_dispersion;        // decrease the chance to sample this point again
    sample_pool.clear();
    sample_pool.reserve(number_of_points);
    std::vector<float> point_probs;
    point_probs.reserve(number_of_points);

    #pragma parallel for schedule(dynamic)
    for (int i=0;i<number_of_points;i++)
    {
        if (i==first_point) continue;
        std::vector<int> ppf;
        computePPF(sampled_P_3D_[first_point], sampled_P_3D_[i], ppf);

        #pragma omp critical
        if (_ppfs.find(ppf) != _ppfs.end())
        {
            sample_pool.push_back(i);
            point_probs.push_back(_point_probs[i]);
        }

    }

    if (sample_pool.size()<3)
    {
        return false;
    }

    const Scalar sq_max_base_diameter_ = max_base_diameter_*max_base_diameter_;

    //NOTE: point_probs is annealing and cannot parallel. Try fixed number of times retaining the best other two.
    Scalar best_wide = 0.0;
    for (int i = 0; i < sample_pool.size()*sample_pool.size()/4; ++i) {
        // Pick and compute
        std::discrete_distribution<> sampler1(point_probs.begin(), point_probs.end());
        const int second_point = sampler1(_point_index_engine);
        const int third_point = sampler1(_point_index_engine);
        if (second_point==third_point) continue;
        std::vector<int> ppf;
        computePPF(sampled_P_3D_[sample_pool[second_point]], sampled_P_3D_[sample_pool[third_point]], ppf);
        if (_ppfs.find(ppf) == _ppfs.end())
        {
            continue;
        }
        point_probs[second_point] *= options_.sample_dispersion;
        point_probs[third_point] *= options_.sample_dispersion;
        const VectorType u =
                sampled_P_3D_[sample_pool[second_point]].pos() -
                sampled_P_3D_[first_point].pos();
        const VectorType w =
                sampled_P_3D_[sample_pool[third_point]].pos() -
                sampled_P_3D_[first_point].pos();
        // We try to have wide triangles but still not too large.
        Scalar how_wide = u.normalized().dot(w.normalized());   // cosine
        if (std::abs(how_wide) <= std::cos(45*M_PI/180.0) &&
                u.squaredNorm() < sq_max_base_diameter_ &&
                w.squaredNorm() < sq_max_base_diameter_) {
            best_wide = how_wide;
            base1 = first_point;
            base2 = sample_pool[second_point];
            base3 = sample_pool[third_point];
            break;
        }
    }

    if (base2==-1 || base3==-1) return false;

    // sample pool for 4th point
    std::vector<int> sample_pool_backup = sample_pool;
    sample_pool.clear();

    #pragma parallel for schedule(dynamic)
    for (int i = 0; i < sample_pool_backup.size(); i++)
    {
        if (sample_pool_backup[i] == base2 || sample_pool_backup[i] == base3 || sample_pool_backup[i] == base1)
            continue;
        std::vector<int> ppf_base2;
        computePPF(sampled_P_3D_[base2], sampled_P_3D_[sample_pool_backup[i]], ppf_base2);
        std::vector<int> ppf_base3;
        computePPF(sampled_P_3D_[base3], sampled_P_3D_[sample_pool_backup[i]], ppf_base3);

        #pragma omp critical
        if (_ppfs.find(ppf_base2) != _ppfs.end() && _ppfs.find(ppf_base3) != _ppfs.end())
        {
            sample_pool.push_back(i);
        }
    }

    if (sample_pool.size() < 1)
    {
        return false;
    }
    return base1 != -1 && base2 != -1 && base3 != -1;
}

template <typename TransformVisitor, template < class, class > typename ... OptExts>
void
MATCH_BASE_TYPE::initKdTree(){
    size_t number_of_points = sampled_P_3D_.size();

    // Build the kdtree.
    kd_tree_ = gr::KdTree<Scalar>(number_of_points);

    for (size_t i = 0; i < number_of_points; ++i) {
        kd_tree_.add(sampled_P_3D_[i].pos());
    }
    kd_tree_.finalize();
}


template <typename TransformVisitor, template < class, class > typename ... OptExts>
template <typename Coordinates>
bool
MATCH_BASE_TYPE::ComputeRigidTransformation(const Coordinates& ref,
        const Coordinates& candidate,
        const Eigen::Matrix<Scalar, 3, 1>& centroid1,
        Eigen::Matrix<Scalar, 3, 1> centroid2,
        Eigen::Ref<MatrixType> transform,
        Scalar& rms_,
        bool computeScale ) const {
    static const Scalar pi = std::acos(-1);

    rms_ = std::numeric_limits<Scalar>::max();

    Scalar kSmallNumber = 1e-6;

    // We only use the first 3 pairs. This simplifies the process considerably
    // because it is the planar case.

    const VectorType& p0 = ref[0].pos();
    const VectorType& p1 = ref[1].pos();
    const VectorType& p2 = ref[2].pos();
    VectorType  q0 = candidate[0].pos();
    VectorType  q1 = candidate[1].pos();
    VectorType  q2 = candidate[2].pos();

    Scalar scaleEst (1.);
    // Compute scale factor if needed
    if (computeScale){
        const VectorType& p3 = ref[3].pos();
        const VectorType& q3 = candidate[3].pos();

        const Scalar ratio1 = (p1 - p0).norm() / (q1 - q0).norm();
        const Scalar ratio2 = (p3 - p2).norm() / (q3 - q2).norm();

        const Scalar ratioDev  = std::abs(ratio1/ratio2 - Scalar(1.));  // deviation between the two
        const Scalar ratioMean = (ratio1+ratio2)/Scalar(2.);            // mean of the two

        if ( ratioDev > Scalar(0.1) )
            return std::numeric_limits<Scalar>::max();


        //Log<LogLevel::Verbose>( ratio1, " ", ratio2, " ", ratioDev, " ", ratioMean);
        scaleEst = ratioMean;

        // apply scale factor to q
        q0 = q0*scaleEst;
        q1 = q1*scaleEst;
        q2 = q2*scaleEst;
        centroid2 *= scaleEst;
    }

    VectorType vector_p1 = p1 - p0;
    if (vector_p1.squaredNorm() == 0) return std::numeric_limits<Scalar>::max();
    vector_p1.normalize();
    VectorType vector_p2 = (p2 - p0) - ((p2 - p0).dot(vector_p1)) * vector_p1;
    if (vector_p2.squaredNorm() == 0) return std::numeric_limits<Scalar>::max();
    vector_p2.normalize();
    VectorType vector_p3 = vector_p1.cross(vector_p2);

    VectorType vector_q1 = q1 - q0;
    if (vector_q1.squaredNorm() == 0) return std::numeric_limits<Scalar>::max();
    vector_q1.normalize();
    VectorType vector_q2 = (q2 - q0) - ((q2 - q0).dot(vector_q1)) * vector_q1;
    if (vector_q2.squaredNorm() == 0) return std::numeric_limits<Scalar>::max();
    vector_q2.normalize();
    VectorType vector_q3 = vector_q1.cross(vector_q2);

    //cv::Mat rotation = cv::Mat::eye(3, 3, CV_64F);
    Eigen::Matrix<Scalar, 3, 3> rotation = Eigen::Matrix<Scalar, 3, 3>::Identity();

    Eigen::Matrix<Scalar, 3, 3> rotate_p;
    rotate_p.row(0) = vector_p1;
    rotate_p.row(1) = vector_p2;
    rotate_p.row(2) = vector_p3;

    Eigen::Matrix<Scalar, 3, 3> rotate_q;
    rotate_q.row(0) = vector_q1;
    rotate_q.row(1) = vector_q2;
    rotate_q.row(2) = vector_q3;

    rotation = rotate_p.transpose() * rotate_q;


    // Discard singular solutions. The rotation should be orthogonal.
    // if (((rotation * rotation).diagonal().array() - Scalar(1) > kSmallNumber).any())
    //     return false;

    if ( !(rotation.transpose()*rotation).isIdentity(kSmallNumber) )
    {
        return false;
    }

    //Filter transformations.
    // \fixme Need to consider max_translation_distance and max_scale too
    if (options_.max_angle >= 0) {
        Scalar mangle = options_.max_angle * pi / 180.0;
        // Discard too large solutions (todo: lazy evaluation during boolean computation
        if (! (
                    std::abs(std::atan2(rotation(2, 1), rotation(2, 2)))
                    <= mangle &&

                    std::abs(std::atan2(-rotation(2, 0),
                                        std::sqrt(std::pow(rotation(2, 1),2) +
                                                  std::pow(rotation(2, 2),2))))
                    <= mangle &&

                    std::abs(atan2(rotation(1, 0), rotation(0, 0)))
                    <= mangle
                    ))
            return false;
    }


    //FIXME
    // Compute rms and return it.
    rms_ = Scalar(0.0);
    {
        VectorType first, transformed;

        //cv::Mat first(3, 1, CV_64F), transformed;
        for (int i = 0; i < 3; ++i) {
            first = scaleEst*candidate[i].pos() - centroid2;
            transformed = rotation * first;
            rms_ += (transformed - ref[i].pos() + centroid1).norm();
        }
    }

    rms_ /= Scalar(ref.size());

    Eigen::Transform<Scalar, 3, Eigen::Affine> etrans (Eigen::Transform<Scalar, 3, Eigen::Affine>::Identity());

    if (fabs(rotation.block(0,0,3,1).norm()-1.0)>0.1 ||
    fabs(rotation.block(0,1,3,1).norm()-1.0)>0.1 ||
    fabs(rotation.block(0,2,3,1).norm()-1.0)>0.1)
    {
        std::cout<<"rotation is not orthogonal!: \n"<<rotation<<"\n\n";
        std::cout<<(rotation * rotation).diagonal()<<"\n\n";
    }

    transform = etrans
            .scale(scaleEst)
            .translate(centroid1)
            .rotate(rotation)
            .translate(-centroid2)
            .matrix();

    return true;
}


template <typename TransformVisitor, template < class, class > typename ... OptExts>
template <typename Sampler>
void MATCH_BASE_TYPE::init(const std::vector<Point3D>& P,
                     const std::vector<Point3D>& Q,
                     const Sampler& sampler){

    centroid_P_ = VectorType::Zero();
    centroid_Q_ = VectorType::Zero();

    sampled_P_3D_.clear();
    sampled_Q_3D_.clear();

    sampled_P_3D_ = P;

    // prepare Q, Q should be CAD model
    if (Q.size() > options_.sample_size){
        std::vector<Point3D> uniform_Q;
        sampler(Q, options_, uniform_Q);


        std::shuffle(uniform_Q.begin(), uniform_Q.end(), randomGenerator_);
        size_t nbSamples = std::min(uniform_Q.size(), options_.sample_size);
        auto endit = uniform_Q.begin();
        std::advance(endit, nbSamples);
        std::copy(uniform_Q.begin(), endit, std::back_inserter(sampled_Q_3D_));
    }
    else
    {
        Log<LogLevel::ErrorReport>( "(Q) More samples requested than available: use whole cloud" );
        std::cout<<"Q size: "<<Q.size()<<"  while requested sample size: "<<options_.sample_size<<std::endl;
        sampled_Q_3D_ = Q;
    }

    std::cout<<"sampled_P_3D_ size: "<<sampled_P_3D_.size()<<std::endl;
    std::cout<<"sampled_Q_3D_ size: "<<sampled_Q_3D_.size()<<std::endl;

    _point_probs.clear();
    _point_probs.resize(sampled_P_3D_.size());
    for (int i=0;i<sampled_P_3D_.size();i++)
    {
        auto pt=sampled_P_3D_[i];
        _point_probs[i] = pt._prob;
    }

    // center points around centroids
    auto centerPoints = [](std::vector<Point3D>&container,
            VectorType& centroid){
        for(const auto& p : container) centroid += p.pos();
        centroid /= Scalar(container.size());
        for(auto& p : container) p.pos() -= centroid;
    };
    centerPoints(sampled_P_3D_, centroid_P_);
    centerPoints(sampled_Q_3D_, centroid_Q_);


    initKdTree();
    // Compute the diameter of P approximately (randomly). This is far from being
    // Guaranteed close to the diameter but gives good results for most common
    // objects if they are densely sampled.
    P_diameter_ = 0.0;
    for (int i = 0; i < kNumberOfDiameterTrials; ++i) {
        int at = randomGenerator_() % sampled_Q_3D_.size();
        int bt = randomGenerator_() % sampled_Q_3D_.size();

        Scalar l = (sampled_Q_3D_[bt].pos() - sampled_Q_3D_[at].pos()).norm();
        if (l > P_diameter_) {
            P_diameter_ = l;
        }
    }

    // Mean distance and a bit more... We increase the estimation to allow for
    // noise, wrong estimation and non-uniform sampling.
    P_mean_distance_ = MeanDistance();

    // Normalize the delta (See the paper) and the maximum base distance.
    // delta = P_mean_distance_ * delta;
    max_base_diameter_ = P_diameter_;  // * estimated_overlap_;

    transform_ = Eigen::Matrix<Scalar, 4, 4>::Identity();

    // call Virtual handler
    Initialize(P,Q);
}

}

#undef MATCH_BASE_TYPE
