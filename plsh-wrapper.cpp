#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "plsh.hpp"

namespace py = pybind11;

class PLSHWrapper {
   public:
    PLSHWrapper(size_t dimensions, int k, int m, unsigned int num_threads)
        : index_(dimensions, k, m, num_threads), is_built_(false) {}

    // ---------------------------
    // 构建
    // ---------------------------
    void build(py::array_t<float, py::array::c_style | py::array::forcecast> X,
               size_t n_points, std::vector<uint32_t> ids) {
        auto buf = X.unchecked<2>();
        size_t dim = buf.shape(1);

        std::vector<SparseVector> data;
        data.reserve(n_points);

        for (size_t i = 0; i < n_points; i++) {
            SparseVector sv;
            sv.indices.reserve(dim);
            sv.values.reserve(dim);
            for (size_t j = 0; j < dim; j++) {
                float v = buf(i, j);
                if (v != 0.0f) {
                    sv.indices.push_back(j);
                    sv.values.push_back(v);
                }
            }
            data.push_back(std::move(sv));
        }
        index_.build(data);
        is_built_ = true;
    }

    // ---------------------------
    // 插入
    // ---------------------------
    void insert(py::array_t<float, py::array::c_style | py::array::forcecast> X,
                std::vector<uint32_t> ids) {
        auto buf = X.unchecked<2>();
        size_t n_points = buf.shape(0);
        size_t dim = buf.shape(1);

        for (size_t i = 0; i < n_points; i++) {
            SparseVector sv;
            sv.indices.reserve(dim);
            sv.values.reserve(dim);
            for (size_t j = 0; j < dim; j++) {
                float v = buf(i, j);
                if (v != 0.0f) {
                    sv.indices.push_back(j);
                    sv.values.push_back(v);
                }
            }
            index_.insert(sv);
        }
    }

    // ---------------------------
    // 查询 Top-K (实现: radius search + 排序 + 截断)
    // ---------------------------
    std::pair<std::vector<uint32_t>, std::vector<float>> query_topk(
        py::array_t<float, py::array::c_style | py::array::forcecast> q,
        int k) {
        auto buf = q.unchecked<1>();
        size_t dim = buf.shape(0);

        // 转 SparseVector
        SparseVector sv;
        sv.indices.reserve(dim);
        sv.values.reserve(dim);
        for (size_t j = 0; j < dim; j++) {
            float v = buf(j);
            if (v != 0.0f) {
                sv.indices.push_back(j);
                sv.values.push_back(v);
            }
        }

        // 调用 PLSH 内部的 query_topk 方法，获取前 k 个最近邻
        auto results = index_.query_topk(sv, k);

        // 转成 Python 可用格式
        std::vector<uint32_t> ids;
        std::vector<float> dists;
        ids.reserve(results.size());
        dists.reserve(results.size());

        for (const auto& result : results) {
            ids.push_back(result.id + 1);  // 保持 +1 规则
            dists.push_back(result.distance);
        }

        return {ids, dists};
    }

    // ---------------------------
    // 查询 radius (额外接口，用于调试/对齐 C++ demo)
    // ---------------------------
    std::pair<std::vector<uint32_t>, std::vector<float>> query_radius(
        py::array_t<float, py::array::c_style | py::array::forcecast> q,
        float radius) {
        auto buf = q.unchecked<1>();
        size_t dim = buf.shape(0);

        // 转 SparseVector
        SparseVector sv;
        sv.indices.reserve(dim);
        sv.values.reserve(dim);
        for (size_t j = 0; j < dim; j++) {
            float v = buf(j);
            if (v != 0.0f) {
                sv.indices.push_back(j);
                sv.values.push_back(v);
            }
        }

        // radius search
        auto results = index_.query_radius(sv, radius);

        // 转成 Python 可用格式
        std::vector<uint32_t> ids;
        std::vector<float> dists;
        ids.reserve(results.size());
        dists.reserve(results.size());

        for (auto& r : results) {
            ids.push_back(r.id + 1);  // 保持 +1 规则
            dists.push_back(r.distance);
        }
        return {ids, dists};
    }

    // ---------------------------
    // 合并
    // ---------------------------
    void merge_delta_to_static() { index_.merge_delta_to_static(); }

   private:
    PLSHIndex index_;
    bool is_built_;
};

// ---------------------------
// pybind11 模块定义
// ---------------------------
PYBIND11_MODULE(plsh_python, m) {
    py::class_<PLSHWrapper>(m, "Index")
        .def(py::init<size_t, int, int, unsigned int>(), py::arg("dimensions"),
             py::arg("k"), py::arg("m"), py::arg("num_threads") = 1)
        .def("build", &PLSHWrapper::build, py::arg("X"), py::arg("n_points"),
             py::arg("ids"))
        .def("insert", &PLSHWrapper::insert, py::arg("X"), py::arg("ids"))
        .def("query_topk", &PLSHWrapper::query_topk, py::arg("q"), py::arg("k"))
        .def("query_radius", &PLSHWrapper::query_radius, py::arg("q"),
             py::arg("radius"))
        .def("merge_delta_to_static", &PLSHWrapper::merge_delta_to_static);
}
