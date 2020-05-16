#ifndef _OPENGR_ALGO_PAIRCREATIONFUNCTOR_H
#define _OPENGR_ALGO_PAIRCREATIONFUNCTOR_H

#include <iostream>
#include <vector>
#include "gr/shared.h"

#include "gr/accelerators/pairExtraction/bruteForceFunctor.h"
#include "gr/accelerators/pairExtraction/intersectionFunctor.h"
#include "gr/accelerators/pairExtraction/intersectionPrimitive.h"
#include "gr/algorithms/match4pcsBase.h"

namespace gr {

template <typename _Scalar, typename FilterFunctor, typename Options>
struct PairCreationFunctor{

public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
  using Scalar      = _Scalar;
  using PairsVector = std::vector<std::pair<int, int>>;
  using VectorType  = typename Point3D::VectorType;
  using BaseCoordinates = Traits4pcs::Coordinates;
  using OptionType  = Options;

  // Shared data
  OptionType options_;
  double pair_distance;
  double pair_normals_angle;
  double pair_distance_epsilon;
  const std::vector<Point3D>& Q_;

  PairsVector pairs;

  std::vector<unsigned int> ids;


  // Internal data
  typedef Eigen::Matrix<Scalar, 3, 1> Point;
  typedef Accelerators::PairExtraction::HyperSphere
  < typename PairCreationFunctor::Point, 3, Scalar> Primitive;

  std::vector< /*Eigen::Map<*/typename PairCreationFunctor::Point/*>*/ > points;
  std::vector< Primitive > primitives;

private:
  VectorType segment1;
  BaseCoordinates base_3D_;
  int base_point1_, base_point2_;

  typename PairCreationFunctor::Point _gcenter;
  Scalar _ratio;
  static const typename PairCreationFunctor::Point half;

public:
  inline PairCreationFunctor(
    const OptionType& options,
    const std::vector<Point3D>& Q)
    :options_(options), Q_(Q),
     _ratio(1.f)
    { }

  inline PairCreationFunctor(const PairCreationFunctor& other):Q_(other.Q_)
  { 
    options_ = other.options_;
    pair_distance = other.pair_distance;
    pair_normals_angle = other.pair_normals_angle;
    pair_distance_epsilon = other.pair_distance_epsilon;
    pairs = other.pairs;
    ids = other.ids;
    points = other.points;
    primitives = other.primitives;
    segment1 = other.segment1;
    base_3D_ = other.base_3D_;
    base_point1_ = other.base_point1_;
    base_point2_ = other.base_point2_;
    _gcenter = other._gcenter;
    _ratio = other._ratio;
    // half = other.half;
  }

  // PairCreationFunctor& operator= (const PairCreationFunctor& other)
  // { 
  //   options_ = other.options_;
  //   pair_distance = other.pair_distance;
  //   pair_normals_angle = other.pair_normals_angle;
  //   pair_distance_epsilon = other.pair_distance_epsilon;
  //   Q_ = other.Q_;
  //   pairs = other.pairs;
  //   ids = other.ids;
  //   points = other.pointspoints;
  //   primitives = other.primitives;
  //   segment1 = other.segment1;
  //   base_3D_ = other.base_3D_;
  //   base_point1_ = other.base_point1_;
  //   base_point2_ = other.base_point2_;
  //   _gcenter = other._gcenter;
  //   _ratio = other._ratio;
  //   half = other.half;
  //   return *this;
  // }

private:
  inline Point worldToUnit(
    const Eigen::MatrixBase<typename PairCreationFunctor::Point> &p) const {
    static const Point half = Point::Ones() * Scalar(0.5f);
    return (p-_gcenter) / _ratio + half;
  }


public:
  inline Point unitToWorld(
    const Eigen::MatrixBase<typename PairCreationFunctor::Point> &p) const {
    static const Point half = Point::Ones() * Scalar(0.5f);
    return (p - half) * _ratio + _gcenter;
  }


  inline Scalar unitToWorld( Scalar d) const {
    return d * _ratio;
  }


  inline Point getPointInWorldCoord(int i) const {
    return unitToWorld(points[i]);
  }


  inline void synch3DContent(){
    points.clear();
    primitives.clear();

    Eigen::AlignedBox<_Scalar, 3> bbox;

    unsigned int nSamples = Q_.size();

    points.reserve(nSamples);
    primitives.reserve(nSamples);

    // Compute bounding box on fine data to be SURE to have all points in the
    // unit bounding box
    for (unsigned int i = 0; i < nSamples; ++i) {
      const VectorType &q = Q_[i].pos();
      points.push_back(q);
      bbox.extend(q);
    }

    _gcenter = bbox.center();
    // add a delta to avoid to have elements with coordinate = 1
    _ratio = bbox.diagonal().maxCoeff() + 0.001;

    // update point cloud (worldToUnit use the ratio and gravity center
    // previously computed)
    // Generate primitives
    for (unsigned int i = 0; i < nSamples; ++i) {
      points[i] = worldToUnit(points[i]);

      primitives.emplace_back(points[i], Scalar(1.));
      ids.push_back(i);
    }
  }

  inline void setRadius(Scalar radius) {
    const Scalar nRadius = radius/_ratio;
    for(typename std::vector< Primitive >::iterator it = primitives.begin();
        it != primitives.end(); ++it)
      (*it).radius() = nRadius;
  }

  inline Scalar getNormalizedEpsilon(Scalar eps){
    return eps/_ratio;
  }

  inline void setBase( int base_point1, int base_point2,
                       const BaseCoordinates& base_3D){
    base_3D_     = base_3D;
    base_point1_ = base_point1;
    base_point2_ = base_point2;

    segment1 = (base_3D_[base_point2_].pos() -
                base_3D_[base_point1_].pos()).normalized();
  }


  inline void beginPrimitiveCollect(int /*primId*/){ }
  inline void endPrimitiveCollect(int /*primId*/){ }


  inline void process(int i, int j)
  {
    if (i > j)
    {
      const Point3D &p = Q_[j];
      const Point3D &q = Q_[i];

      // Compute the distance and two normal angles to ensure working with
      // wrong orientation. We want to verify that the angle between the
      // normals is close to the angle between normals in the base. This can be
      // checked independent of the full rotation angles which are not yet
      // defined by segment matching alone..
      const Scalar distance = (q.pos() - p.pos()).norm();
#ifndef MULTISCALE
      if (std::abs(distance - pair_distance) > pair_distance_epsilon)
        return;
#endif
      FilterFunctor fun;
      std::pair<bool, bool> res = fun(p, q, pair_normals_angle, base_3D_[base_point1_], base_3D_[base_point2_], options_);
      if (res.first)
        pairs.emplace_back(i, j);
      if (res.second)
        pairs.emplace_back(j, i);

    }
  }
};

} // namespace Super4PCS

#endif // PAIRCREATIONFUNCTOR_H
