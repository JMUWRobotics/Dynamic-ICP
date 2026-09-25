// Thin NumPy interface to the rebuilt Open3D registration implementation.
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>
#include <open3d/geometry/PointCloud.h>
#include <open3d/pipelines/registration/DopplerPt2PlaneICP.h>
#include <open3d/geometry/KDTreeSearchParam.h>

namespace py = pybind11;
namespace reg = open3d::pipelines::registration;
using MatrixX3 = Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor>;

std::vector<Eigen::Vector3d> vectors(const MatrixX3& x) {
    std::vector<Eigen::Vector3d> result(x.rows());
    for (Eigen::Index i = 0; i < x.rows(); ++i) result[i] = x.row(i);
    return result;
}

open3d::geometry::PointCloud cloud(const MatrixX3& points, const Eigen::VectorXd& s) {
    if (points.rows() != s.size() || !points.allFinite() || !s.allFinite())
        throw std::invalid_argument("Point/Doppler shapes or values are invalid");
    open3d::geometry::PointCloud out;
    out.points_ = vectors(points);
    out.dopplers_.assign(s.data(), s.data() + s.size());
    return out;
}

PYBIND11_MODULE(_native, m) {
    m.attr("solver_sha256") = DYNAMIC_ICP_SOLVER_SHA;
    m.attr("backend") = "dynamic-open3d-cpp";
    m.def("register", [](const MatrixX3& p, const Eigen::VectorXd& sp, const MatrixX3& up,
                          const MatrixX3& q, const Eigen::VectorXd& sq, const MatrixX3& uq,
                          const MatrixX3& normals, const Eigen::Matrix4d& init,
                          double distance, double lambda, double kg, double kv,
                          double tolerance, int iterations, double ego_weight,
                          const Eigen::Vector3d& displacement, const Eigen::Matrix3d& information) {
        if (up.rows() != p.rows() || uq.rows() != q.rows() || normals.rows() != q.rows())
            throw std::invalid_argument("LOS/normal lengths must match the scans");
        auto source = cloud(p, sp), target = cloud(q, sq);
        target.normals_ = vectors(normals);
        auto source_dirs = vectors(up), target_dirs = vectors(uq);
        reg::TransformationEstimationForDopplerPt2PlaneICP estimator(
            lambda, 1.0, 0, 0, std::make_shared<reg::TukeyLoss>(kg),
            std::make_shared<reg::TukeyLoss>(kv));
        estimator.ego_translation_weight_ = ego_weight;
        estimator.displacement_prior_ = displacement;
        estimator.translation_information_ = information;
        reg::RegistrationResult result;
        {
            py::gil_scoped_release release;
            result = reg::RegistrationDopplerPt2PlaneICP(source, target, source_dirs,
                target_dirs, distance, init, estimator,
                reg::ICPConvergenceCriteria(tolerance, tolerance, iterations), 0.1);
        }
        py::dict out;
        out["transformation"] = result.transformation_;
        out["converged"] = result.converged_;
        out["iterations"] = result.num_iterations_;
        out["fitness"] = result.fitness_;
        out["rmse"] = result.inlier_rmse_;
        return out;
    });
    // Uses the pinned Dynamic-ICP solver with fixed index correspondences. This
    // exposes incorrect handling of nonidentity R without nearest-neighbor noise.
    m.def("step", [](const MatrixX3& p, const Eigen::VectorXd& sp, const MatrixX3& up,
                      const MatrixX3& q, const Eigen::VectorXd& sq, const MatrixX3& uq,
                      const MatrixX3& normals, const Eigen::Matrix4d& transform,
                      double lambda, double kg, double kv, double ego_weight,
                      const Eigen::Vector3d& displacement, const Eigen::Matrix3d& information) {
        if (p.rows() != q.rows() || up.rows() != p.rows() || uq.rows() != q.rows()
            || normals.rows() != q.rows()) throw std::invalid_argument("Mismatched step arrays");
        auto source = cloud(p, sp), target = cloud(q, sq);
        source.Transform(transform);
        target.normals_ = vectors(normals);
        reg::CorrespondenceSet pairs;
        for (int i = 0; i < p.rows(); ++i) pairs.push_back(Eigen::Vector2i(i, i));
        reg::TransformationEstimationForDopplerPt2PlaneICP estimator(
            lambda, 1.0, 0, 0, std::make_shared<reg::TukeyLoss>(kg),
            std::make_shared<reg::TukeyLoss>(kv));
        estimator.ego_translation_weight_ = ego_weight;
        estimator.displacement_prior_ = displacement;
        estimator.translation_information_ = information;
        return estimator.ComputeTransformation(source, target, pairs, vectors(up), vectors(uq),
                                                0.1, transform, 0);
    });
}
