#include "../vamana.h"
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

PYBIND11_MODULE(vamanalib, m) {
  using namespace vamana;
  m.doc() = "Python bindings for Vamana graph-based index";

  py::class_<vamana::Vamana>(m, "Vamana")
      .def(py::init<uint32_t, uint32_t, float, uint32_t, uint32_t, uint32_t>(),
           py::arg("dim"), py::arg("max_points"), py::arg("alpha"),
           py::arg("R"), py::arg("L"), py::arg("ef_search"))
      .def(py::init<const std::string &>())
      .def("add_point",
           [](vamana::Vamana &self, py::array_t<float> point, uint32_t id) {
             py::buffer_info buf = point.request();
             if (buf.ndim != 1) {
               throw std::runtime_error("Point must be a 1-dimensional array");
             }
             return self.AddPoint(static_cast<float *>(buf.ptr), id);
           })
      .def("build_index",
           [](vamana::Vamana &self) {
             //  py::gil_scoped_release release;
             uint32_t res = self.BuildIndex();
             // uint32_t res = self.BuildIndexWithoutParallel();
             //  py::gil_scoped_acquire acquire;
             return res;
           })
      .def("save_index", &vamana::Vamana::SaveIndex)
      .def("search",
           [](const vamana::Vamana &self, py::array_t<float> query, uint32_t k,
              uint32_t ef_search) {
             py::buffer_info buf = query.request();
             if (buf.ndim != 1) {
               throw std::runtime_error("Query must be a 1-dimensional array");
             }
             if (buf.size != self.GetDimension()) {
               throw std::runtime_error(
                   "Query dimension does not match index dimension");
             }

             std::vector<uint32_t> result_ids(k);
             std::vector<float> result_distances(k);

             int num_results =
                 self.Search(static_cast<float *>(buf.ptr), k,
                             result_ids.data(), result_distances.data());

             if (num_results < 0) {
               throw std::runtime_error("Search failed");
             }

             auto ids = py::array_t<uint32_t>(num_results);
             auto distances = py::array_t<float>(num_results);

             auto ids_buf = ids.request();
             auto distances_buf = distances.request();

             std::memcpy(ids_buf.ptr, result_ids.data(),
                         num_results * sizeof(uint32_t));
             std::memcpy(distances_buf.ptr, result_distances.data(),
                         num_results * sizeof(float));

             return py::make_tuple(ids, distances);
           })
      .def("get_point",
           [](const vamana::Vamana &self, uint32_t id) {
             std::vector<float> point(self.GetDimension());
             if (self.GetPoint(id, point.data()) < 0) {
               throw std::runtime_error("Failed to get point");
             }
             auto result = py::array_t<float>(point.size());
             auto buf = result.request();
             std::memcpy(buf.ptr, point.data(), point.size() * sizeof(float));
             return result;
           })
      .def("get_dimension", &vamana::Vamana::GetDimension)
      .def("get_max_points", &vamana::Vamana::GetMaxPoints)
      .def("get_alpha", &vamana::Vamana::GetAlpha)
      .def("get_r", &vamana::Vamana::GetR)
      .def("get_l", &vamana::Vamana::GetL);
}
