// GPU solver pybind11 binding module
#include <torch/extension.h>

#define CHECK_GPU_DOUBLE_TENSOR(x) \
    TORCH_CHECK((x).is_cuda(), #x " must be a CUDA/ROCm GPU tensor"); \
    TORCH_CHECK((x).is_contiguous(), #x " must be contiguous in memory"); \
    TORCH_CHECK((x).scalar_type() == torch::kFloat64, #x " must be float64 (double precision)")

#define CHECK_GPU_INT_TENSOR(x) \
    TORCH_CHECK((x).is_cuda(), #x " must be a CUDA/ROCm GPU tensor"); \
    TORCH_CHECK((x).is_contiguous(), #x " must be contiguous in memory"); \
    TORCH_CHECK((x).scalar_type() == torch::kInt32, #x " must be int32")

// Forward declarations of launch wrappers defined in the HIP file
extern "C" void launch_general_transient_kernel_double(
    int dev_idx,
    double t_start, double t_end, double min_dt, double tol, int maxIter, int max_steps,
    const double* master_schedule, const int* is_subsample, int num_edges,
    double* device_states_soa,
    const double* device_params_soa,
    int numNodes, int num_devices,
    const int* dev_top_full_idx, const int* dev_bot_full_idx,
    const int* dev_top_idx, const int* dev_bot_idx,
    const int* gMat_row_indices, const int* gMat_col_indices, const int* iRes_indices,
    const int* top_indices, const int* bot_indices, const int* both_indices, const int* dev_both_idx,
    int L_top, int L_bot, int L_both,
    const int* d_nodes, const int* dyn_indices, const int* dev_to_dyn_src,
    const int* v_nodes_gpu, const double* v_pwl_t_matrix, const double* v_pwl_v_matrix, const int* v_pwl_mask_gpu,
    const int* dyn_nodes_gpu, const double* dyn_pwl_t_matrix, const double* dyn_pwl_v_matrix, const int* dyn_pwl_mask_gpu,
    bool is_1t1r, bool recordHistory, double leakage_g,
    int num_v_sources, int num_dyn_sources, int num_all_nodes,
    // History buffers
    double* tArr_buffer, double* vHist_buffer, double* iHist_buffer,
    double* stateHist_buffer, double* f22Hist_buffer, double* gHist_buffer,
    double* tempHist_buffer,
    // System dimensions
    int num_gMat_vals,
    int max_pwl_points,
    int batch_size,
    const double* y_i,
    const double* y_v,
    const double* y_fp,
    const double* y_fd,
    int lookup_size,
    double* global_workspace_cg,
    double* global_workspace_t,
    double* d_out,
    void* stream_ptr
);

void run_gpu_simulation_double(
    double t_start, double t_end, double min_dt, double tol, int maxIter, int max_steps,
    torch::Tensor master_schedule,
    torch::Tensor is_subsample,
    torch::Tensor device_states_soa,
    torch::Tensor device_params_soa,
    int numNodes, int num_devices,
    torch::Tensor dev_top_full_idx, torch::Tensor dev_bot_full_idx,
    torch::Tensor dev_top_idx, torch::Tensor dev_bot_idx,
    torch::Tensor gMat_row_indices, torch::Tensor gMat_col_indices, torch::Tensor iRes_indices,
    torch::Tensor top_indices, torch::Tensor bot_indices, torch::Tensor both_indices, torch::Tensor dev_both_idx,
    torch::Tensor d_nodes, torch::Tensor dyn_indices, torch::Tensor dev_to_dyn_src,
    torch::Tensor v_nodes_gpu, torch::Tensor v_pwl_t_matrix, torch::Tensor v_pwl_v_matrix, torch::Tensor v_pwl_mask_gpu,
    torch::Tensor dyn_nodes_gpu, torch::Tensor dyn_pwl_t_matrix, torch::Tensor dyn_pwl_v_matrix, torch::Tensor dyn_pwl_mask_gpu,
    bool is_1t1r, bool recordHistory, double leakage_g,
    int num_v_sources, int num_dyn_sources,
    // History buffers
    torch::Tensor tArr_buffer, torch::Tensor vHist_buffer, torch::Tensor iHist_buffer,
    torch::Tensor stateHist_buffer, torch::Tensor f22Hist_buffer, torch::Tensor gHist_buffer,
    torch::Tensor tempHist_buffer,
    int num_gMat_vals,
    int max_pwl_points,
    int batch_size,
    // Lookup tables
    torch::Tensor y_i,
    torch::Tensor y_v,
    torch::Tensor y_fp,
    torch::Tensor y_fd,
    int lookup_size,
    torch::Tensor global_workspace_cg,
    torch::Tensor global_workspace_t,
    // Pre-allocated GPU output tensor to return hist_idx and t_final
    torch::Tensor out_tensor,
    int64_t stream_ptr
) {
    CHECK_GPU_DOUBLE_TENSOR(master_schedule);
    CHECK_GPU_INT_TENSOR(is_subsample);
    CHECK_GPU_DOUBLE_TENSOR(device_states_soa);
    CHECK_GPU_DOUBLE_TENSOR(device_params_soa);
    CHECK_GPU_INT_TENSOR(dev_top_full_idx);
    CHECK_GPU_INT_TENSOR(dev_bot_full_idx);
    CHECK_GPU_INT_TENSOR(dev_top_idx);
    CHECK_GPU_INT_TENSOR(dev_bot_idx);
    CHECK_GPU_INT_TENSOR(gMat_row_indices);
    CHECK_GPU_INT_TENSOR(gMat_col_indices);
    CHECK_GPU_INT_TENSOR(iRes_indices);
    CHECK_GPU_INT_TENSOR(top_indices);
    CHECK_GPU_INT_TENSOR(bot_indices);
    CHECK_GPU_INT_TENSOR(both_indices);
    CHECK_GPU_INT_TENSOR(dev_both_idx);
    CHECK_GPU_INT_TENSOR(d_nodes);
    CHECK_GPU_INT_TENSOR(dyn_indices);
    CHECK_GPU_INT_TENSOR(dev_to_dyn_src);
    CHECK_GPU_INT_TENSOR(v_nodes_gpu);
    CHECK_GPU_DOUBLE_TENSOR(v_pwl_t_matrix);
    CHECK_GPU_DOUBLE_TENSOR(v_pwl_v_matrix);
    CHECK_GPU_INT_TENSOR(v_pwl_mask_gpu);
    CHECK_GPU_INT_TENSOR(dyn_nodes_gpu);
    CHECK_GPU_DOUBLE_TENSOR(dyn_pwl_t_matrix);
    CHECK_GPU_DOUBLE_TENSOR(dyn_pwl_v_matrix);
    CHECK_GPU_INT_TENSOR(dyn_pwl_mask_gpu);
    CHECK_GPU_DOUBLE_TENSOR(tArr_buffer);
    CHECK_GPU_DOUBLE_TENSOR(vHist_buffer);
    CHECK_GPU_DOUBLE_TENSOR(iHist_buffer);
    CHECK_GPU_DOUBLE_TENSOR(stateHist_buffer);
    CHECK_GPU_DOUBLE_TENSOR(f22Hist_buffer);
    CHECK_GPU_DOUBLE_TENSOR(gHist_buffer);
    CHECK_GPU_DOUBLE_TENSOR(tempHist_buffer);
    CHECK_GPU_DOUBLE_TENSOR(y_i);
    CHECK_GPU_DOUBLE_TENSOR(y_v);
    CHECK_GPU_DOUBLE_TENSOR(y_fp);
    CHECK_GPU_DOUBLE_TENSOR(y_fd);
    CHECK_GPU_DOUBLE_TENSOR(out_tensor);
    if (global_workspace_cg.numel() > 0) { CHECK_GPU_DOUBLE_TENSOR(global_workspace_cg); }
    if (global_workspace_t.numel() > 0) { CHECK_GPU_DOUBLE_TENSOR(global_workspace_t); }

    double* workspace_cg_ptr = (global_workspace_cg.numel() > 0) ? static_cast<double*>(global_workspace_cg.data_ptr()) : nullptr;
    double* workspace_t_ptr = (global_workspace_t.numel() > 0) ? static_cast<double*>(global_workspace_t.data_ptr()) : nullptr;

    int dev_idx = (int)master_schedule.get_device();

    int L_top = top_indices.size(0);
    int L_bot = bot_indices.size(0);
    int L_both = both_indices.size(0);
    int num_edges = master_schedule.size(0);

    if (batch_size <= 0 || numNodes < 0 || num_devices <= 0 || max_steps <= 0 || num_edges <= 0) {
        return;
    }

    int num_all_nodes = (vHist_buffer.dim() >= 2) ? (int)vHist_buffer.size(1) : (numNodes + num_v_sources);
    if (num_all_nodes <= 0) {
        num_all_nodes = numNodes + num_v_sources;
    }

    launch_general_transient_kernel_double(
        dev_idx,
        t_start, t_end, min_dt, tol, maxIter, max_steps,
        static_cast<double*>(master_schedule.data_ptr()),
        static_cast<int*>(is_subsample.data_ptr()), num_edges,
        static_cast<double*>(device_states_soa.data_ptr()),
        static_cast<double*>(device_params_soa.data_ptr()),
        numNodes, num_devices,
        static_cast<int*>(dev_top_full_idx.data_ptr()), static_cast<int*>(dev_bot_full_idx.data_ptr()),
        static_cast<int*>(dev_top_idx.data_ptr()), static_cast<int*>(dev_bot_idx.data_ptr()),
        static_cast<int*>(gMat_row_indices.data_ptr()), static_cast<int*>(gMat_col_indices.data_ptr()), static_cast<int*>(iRes_indices.data_ptr()),
        static_cast<int*>(top_indices.data_ptr()), static_cast<int*>(bot_indices.data_ptr()), static_cast<int*>(both_indices.data_ptr()), static_cast<int*>(dev_both_idx.data_ptr()),
        L_top, L_bot, L_both,
        static_cast<int*>(d_nodes.data_ptr()), static_cast<int*>(dyn_indices.data_ptr()), static_cast<int*>(dev_to_dyn_src.data_ptr()),
        static_cast<int*>(v_nodes_gpu.data_ptr()), static_cast<double*>(v_pwl_t_matrix.data_ptr()), static_cast<double*>(v_pwl_v_matrix.data_ptr()), static_cast<int*>(v_pwl_mask_gpu.data_ptr()),
        static_cast<int*>(dyn_nodes_gpu.data_ptr()), static_cast<double*>(dyn_pwl_t_matrix.data_ptr()), static_cast<double*>(dyn_pwl_v_matrix.data_ptr()), static_cast<int*>(dyn_pwl_mask_gpu.data_ptr()),
        is_1t1r, recordHistory, leakage_g,
        num_v_sources, num_dyn_sources, num_all_nodes,
        static_cast<double*>(tArr_buffer.data_ptr()), static_cast<double*>(vHist_buffer.data_ptr()), static_cast<double*>(iHist_buffer.data_ptr()),
        static_cast<double*>(stateHist_buffer.data_ptr()), static_cast<double*>(f22Hist_buffer.data_ptr()), static_cast<double*>(gHist_buffer.data_ptr()),
        static_cast<double*>(tempHist_buffer.data_ptr()),
        num_gMat_vals,
        max_pwl_points,
        batch_size,
        static_cast<double*>(y_i.data_ptr()),
        static_cast<double*>(y_v.data_ptr()),
        static_cast<double*>(y_fp.data_ptr()),
        static_cast<double*>(y_fd.data_ptr()),
        lookup_size,
        workspace_cg_ptr,
        workspace_t_ptr,
        static_cast<double*>(out_tensor.data_ptr()),
        (void*)stream_ptr
    );
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("run_gpu_simulation_double", &run_gpu_simulation_double, "Run double-precision memristive circuit simulation on GPU.");
}
