//
// Created by Sandra Alfaro on 24/05/18.
//

#ifndef OPENGR_MATCH4PCSBASE_H
#define OPENGR_MATCH4PCSBASE_H

#include <vector>

#ifdef OpenGR_USE_OPENMP
#include <omp.h>
#endif

#include "gr/shared.h"
#include "gr/sampling.h"
#include "gr/accelerators/kdtree.h"
#include "gr/utils/logger.h"
#include "gr/algorithms/congruentSetExplorationBase.h"

#ifdef TEST_GLOBAL_TIMINGS
#   include "gr/utils/timer.h"
#endif

namespace gr {

    struct Traits4pcs {
        static constexpr int size() { return 4; }
        using Base = std::array<int,4>;
        using Set = std::vector<Base>;
        using Coordinates = std::array<Point3D, 4>;
    };

    /// Class for the computation of the 4PCS algorithm. This is the one also used in super4pcs
    /// \param Functor use to determinate the use of Super4pcs or 4pcs algorithm.
    template <template <typename, typename> typename _Functor,
              typename _TransformVisitor,
              typename _PairFilteringFunctor,  /// <\brief Must implements PairFilterConcept
              template < class, class > typename PairFilteringOptions >
    class Match4pcsBase : public CongruentSetExplorationBase<Traits4pcs, _TransformVisitor, _PairFilteringFunctor, PairFilteringOptions> {
    public:
        using Scalar            = typename Point3D::Scalar;
        using PairFilteringFunctor = _PairFilteringFunctor;
        using MatchBaseType     = CongruentSetExplorationBase<Traits4pcs, _TransformVisitor, _PairFilteringFunctor, PairFilteringOptions>;
        using VectorType        = typename MatchBaseType::VectorType;
        using MatrixType        = typename MatchBaseType::MatrixType;
        using TransformVisitor  = typename MatchBaseType::TransformVisitor;
        using CongruentBaseType = typename MatchBaseType::CongruentBaseType;
        using Set               = typename MatchBaseType::Set;
        using Coordinates       = typename MatchBaseType::Coordinates;
        using OptionsType       = typename MatchBaseType::OptionsType;
        using Functor           = _Functor<PairFilteringFunctor, OptionsType>;

    protected:
        Functor fun_;

    public:

        inline Match4pcsBase (const OptionsType& options
                , const Utils::Logger& logger);

        virtual ~Match4pcsBase();

        inline const Functor& getFunctor() const { return fun_; }

        /// Takes quadrilateral as a base, computes robust intersection point
        /// (approximate as the lines might not intersect) and returns the invariants
        /// corresponding to the two selected lines. The method also updates the order
        /// of the base base_3D_.
        inline bool TryQuadrilateral(Scalar &invariant1, Scalar &invariant2,
                                     int &id1, int &id2, int &id3, int &id4);

        /// Selects a random triangle in the set P (then we add another point to keep the
        /// base as planar as possible). We apply a simple heuristic that works in most
        /// practical cases. The idea is to accept maximum distance, computed by the
        /// estimated overlap, multiplied by the diameter of P, and try to have
        /// a triangle with all three edges close to this distance. Wide triangles helps
        /// to make the transformation robust while too large triangles makes the
        /// probability of having all points in the inliers small so we try to trade-off.
        inline bool SelectQuadrilateral(Scalar &invariant1, Scalar &invariant2,
                                        int& base1, int& base2, int& base3, int& base4);

        /// Initializes the data structures and needed values before the match
        /// computation.
        /// @param [in] point_P First input set.
        /// @param [in] point_Q Second input set.
        /// expected to be in the inliers.
        /// This method is called once the internal state of the Base class as been
        /// set.
        void Initialize(const std::vector<Point3D>& /*P*/,
                        const std::vector<Point3D>& /*Q*/) override;

        /// Find all the congruent set similar to the base in the second 3D model (Q).
        /// It could be with a 3 point base or a 4 point base.
        /// \param base use to find the similar points congruent in Q.
        /// \param congruent_set a set of all point congruent found in Q.
        bool generateCongruents (CongruentBaseType& base,Set& congruent_quads) override;

    private:
        static inline Scalar distSegmentToSegment( const VectorType& p1, const VectorType& p2,
                                                   const VectorType& q1, const VectorType& q2,
                                                   Scalar& invariant1, Scalar& invariant2);
    };
}

#include "match4pcsBase.hpp"

#endif //OPENGR_MATCH4PCSBASE_H
