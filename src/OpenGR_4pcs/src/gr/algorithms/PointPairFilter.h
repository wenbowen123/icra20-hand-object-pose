//
// Created by Sandra Alfaro on 26/04/18.
//

#ifndef OPENGR_FUNCTORFEATUREPOINTTEST_H
#define OPENGR_FUNCTORFEATUREPOINTTEST_H

#include <gr/shared.h>
#include <vector>

namespace gr {

    ///Bowen: use PPF for verify pair
    //p,q are two end points in line segment from P. b0 b1 are from Q
    //p <--> b0  
    //q <--> b1
    bool pairPPFisGood(const Point3D& p, const Point3D &q, const Point3D &b0, const Point3D &b1)
    {
        using VectorType  = typename Point3D::VectorType;
        float length1 = (p.pos()-q.pos()).norm();
        float length2 = (b0.pos()-b1.pos()).norm();
        if (std::abs(length1-length2)>5e-3) return false;

        VectorType pq = (q.pos() - p.pos()).normalized();
        VectorType b0_b1 = (b1.pos() - b0.pos()).normalized();
        float pq_np = std::acos(std::abs(pq.dot(p.normal()))) / M_PI * 180;   // angle in deg
        float pq_nq = std::acos(std::abs(pq.dot(q.normal()))) / M_PI * 180;
        float b0_b1_n0 = std::acos(std::abs(b0_b1.dot(b0.normal()))) / M_PI * 180;
        float b0_b1_n1 = std::acos(std::abs(b0_b1.dot(b1.normal()))) / M_PI * 180;
        float np_nq = std::acos(p.normal().dot(q.normal())) / M_PI * 180;
        float n0_n1 = std::acos(b0.normal().dot(b1.normal())) / M_PI * 180;

        if (std::abs(pq_np - b0_b1_n0)>30 || std::abs(pq_nq - b0_b1_n1)>30 || std::abs(np_nq - n0_n1)>30) 
        {
            return false;
        }
        return true;
    }

    /// \brief Functor used in n-pcs algorithm to filters pairs of points according
    ///        to the exploration basis,
    /// \tparam
    /// \implements PairFilterConcept
    ///
    struct DummyPointFilter {
    template < class Derived, class TBase>
    struct Options : public TBase {
      bool dummyFilteringResponse;
      enum { IS_DUMMYPOINTFILTER_OPTIONS = true };
    };
    template <typename WantedOptionsAndMore>
    inline std::pair<bool,bool> operator() (const Point3D& /*p*/,
                                            const Point3D& /*q*/,
                                            typename Point3D::Scalar /*pair_normals_angle*/,
                                            const Point3D& /*b0*/,
                                            const Point3D& /*b1*/,
                                            const WantedOptionsAndMore &options) {
        return std::make_pair(options.dummyFilteringResponse, options.dummyFilteringResponse);
    }
    };

    /// \brief Functor used in n-pcs algorithm to filters pairs of points according
    ///        to the exploration basis. Uses normal, colors and max motion when
    ///        available
    ///
    /// \implements PairFilterConcept
    ///
    struct AdaptivePointFilter {
      template < class Derived, class TBase>
      struct Options : public TBase {
        using Scalar = typename TBase::Scalar;

        /// Maximum normal difference.
        Scalar max_normal_difference = -1;
        /// Maximum color RGB distance between corresponding vertices. Set negative to ignore
        Scalar max_color_distance = -1;

        enum { IS_ADAPTIVEPOINTFILTER_OPTIONS = true };
      };

        /// Verify that the 2 points found in Q are similar to 2 of the points in the base.
        /// A filter by point feature : normal, distance, translation distance, angle and color.
        /// Return a pair of bool, according of the right addition of the pair (p,q) or (q,p) in the congruent set on Q.
        /// b0 and b1 are from model P.
        /// p <---> b0
        /// q <---> b1
        template <typename WantedOptionsAndMore>
        inline std::pair<bool,bool> operator() (const Point3D& p,
                                                const Point3D& q,
                                                typename Point3D::Scalar pair_normals_angle,
                                                const Point3D& b0,
                                                const Point3D& b1,
                                                const WantedOptionsAndMore &options) {
            static_assert( WantedOptionsAndMore::IS_ADAPTIVEPOINTFILTER_OPTIONS,
                           "Options passed to AdaptivePointFilter must inherit AdaptivePointFilter::Options" );
            using Scalar      = typename Point3D::Scalar;
            using PairsVector = std::vector< std::pair<int, int> >;
            using VectorType  = typename Point3D::VectorType;


            std::pair<bool,bool> res;
            res.first = false;
            res.second = false;

            if (!pairPPFisGood(p,q,b0,b1)) return res;

            VectorType segment1 = (b1.pos() - b0.pos()).normalized();

            if ( options.max_normal_difference > 0 &&
                 q.normal().squaredNorm() > 0 &&
                 p.normal().squaredNorm() > 0) {

                
                const Scalar norm_threshold =
                        0.5 * options.max_normal_difference * M_PI / 180.0;
                const double first_normal_angle = (q.normal() - p.normal()).norm();
                const double second_normal_angle = (q.normal() + p.normal()).norm();
                // Take the smaller normal distance in case normal is flipped when computing
                const Scalar first_norm_distance =
                        std::min(std::abs(first_normal_angle - pair_normals_angle),
                                 std::abs(second_normal_angle - pair_normals_angle));
                // Verify appropriate angle between normals and distance.
                if (first_norm_distance > norm_threshold) return res;
                

                /*
                float angle_deg = options.max_normal_difference/180.0*M_PI;
                std::cout<<"angle_deg="<<angle_deg<<"\n";
                std::cout<<"p.normal().dot(b0.normal()) = "<<p.normal().dot(b0.normal())<<"\n";
                std::cout<<"q.normal().dot(b1.normal()) = "<<q.normal().dot(b1.normal())<<"\n";
                if (p.normal().dot(b0.normal()) < std::cos(angle_deg) ||
                    q.normal().dot(b1.normal()) < std::cos(angle_deg) ) return res;
                */
            }
            // Verify restriction on the rotation angle, translation and colors.
            if (options.max_color_distance > 0) {
                const bool use_rgb = (p.rgb()[0] >= 0 && q.rgb()[0] >= 0 &&
                                      b0.rgb()[0] >= 0 &&
                                      b1.rgb()[0] >= 0);
                bool color_good = (p.rgb() - b0.rgb()).norm() <
                                  options.max_color_distance &&
                                  (q.rgb() - b1.rgb()).norm() <
                                  options.max_color_distance;

                if (use_rgb && ! color_good) return res;
            }

            if (options.max_translation_distance > 0) {
                const bool dist_good = (p.pos() - b0.pos()).norm() <
                                       options.max_translation_distance &&
                                       (q.pos() - b1.pos()).norm() <
                                       options.max_translation_distance;
                if (! dist_good) return res;
            }

            // need cleaning here
            if (options.max_angle > 0){
                VectorType segment2 = (q.pos() - p.pos()).normalized();
                if (std::acos(segment1.dot(segment2)) <= options.max_angle * M_PI / 180.0) {
                    res.second = true;
                }

                if (std::acos(segment1.dot(- segment2)) <= options.max_angle * M_PI / 180.0) {
                    // Add ordered pair.
                    res.first = true;
                }
            }else {
                res.first = true;
                res.second = true;
            }
            return res;
        }
    };
}

#endif //OPENGR_FUNCTORFEATUREPOINTTEST_H
