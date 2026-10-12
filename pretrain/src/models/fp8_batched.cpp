#include <torch/extension.h>
#include <ATen/cuda/CUDABlas.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/Exceptions.h>
#include <c10/cuda/CUDAGuard.h>
#include <cublasLt.h>
#include <map>
#include <tuple>

struct Key {
    int64_t batch, m, n, k;
    cudaDataType_t a, b;
    bool interleaved;
    bool operator<(const Key& other) const {
        return std::tie(batch, m, n, k, a, b, interleaved) <
               std::tie(other.batch, other.m, other.n, other.k, other.a, other.b, other.interleaved);
    }
};

struct Layout {
    cublasLtMatrixLayout_t value;
    Layout(cudaDataType_t type, int64_t rows, int64_t cols, int64_t ld,
           int32_t batches, int64_t stride) {
        TORCH_CUDABLAS_CHECK(cublasLtMatrixLayoutCreate(&value, type, rows, cols, ld));
        TORCH_CUDABLAS_CHECK(cublasLtMatrixLayoutSetAttribute(
            value, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batches, sizeof(batches)));
        TORCH_CUDABLAS_CHECK(cublasLtMatrixLayoutSetAttribute(
            value, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &stride, sizeof(stride)));
    }
    ~Layout() { cublasLtMatrixLayoutDestroy(value); }
};

struct Descriptor {
    cublasLtMatmulDesc_t value;
    Descriptor() {
        TORCH_CUDABLAS_CHECK(cublasLtMatmulDescCreate(&value, CUBLAS_COMPUTE_32F, CUDA_R_32F));
    }
    ~Descriptor() { cublasLtMatmulDescDestroy(value); }
    template <typename T> void set(cublasLtMatmulDescAttributes_t attribute, T value_) {
        TORCH_CUDABLAS_CHECK(cublasLtMatmulDescSetAttribute(value, attribute, &value_, sizeof(T)));
    }
};

struct Preference {
    cublasLtMatmulPreference_t value;
    Preference(size_t bytes) {
        TORCH_CUDABLAS_CHECK(cublasLtMatmulPreferenceCreate(&value));
        TORCH_CUDABLAS_CHECK(cublasLtMatmulPreferenceSetAttribute(
            value, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &bytes, sizeof(bytes)));
    }
    ~Preference() { cublasLtMatmulPreferenceDestroy(value); }
};

cudaDataType_t fp8_type(const at::Tensor& value) {
    if (value.scalar_type() == at::kFloat8_e4m3fn) return CUDA_R_8F_E4M3;
    TORCH_CHECK(value.scalar_type() == at::kFloat8_e5m2, "Expected E4M3 or E5M2");
    return CUDA_R_8F_E5M2;
}

at::Tensor matmul(const at::Tensor& a, const at::Tensor& b,
                  const at::Tensor& sa, const at::Tensor& sb, const at::Tensor& scratch,
                  bool interleaved) {
    TORCH_CHECK(a.is_cuda() && b.device() == a.device() && sa.device() == a.device() &&
                sb.device() == a.device() && scratch.device() == a.device(), "Expected same CUDA device");
    TORCH_CHECK(a.dim() == 3 && b.dim() == 3 && a.size(0) == b.size(0) && a.size(2) == b.size(1),
                "Expected matching batched matrices");
    TORCH_CHECK(a.is_contiguous() && b.transpose(-2, -1).is_contiguous(),
                "Expected row-major A and column-major B");
    TORCH_CHECK(sa.scalar_type() == at::kFloat && sb.scalar_type() == at::kFloat &&
                sa.numel() == 1 && sb.numel() == 1, "Expected scalar FP32 scales");
    TORCH_CHECK(scratch.scalar_type() == at::kByte && scratch.is_contiguous(), "Expected byte workspace");
    c10::cuda::CUDAGuard guard(a.device());
    const int64_t m = a.size(1), k = a.size(2), n = b.size(2);
    const int32_t batches = a.size(0);
    auto out = at::empty({batches, m, n}, a.options().dtype(at::kBFloat16));
    if (interleaved) out = out.as_strided({batches, m, n}, {n, batches * n, 1});
    // Compute C^T = B^T A^T in the TN layout required by Hopper FP8.
    Layout left(fp8_type(b), k, n, k, batches, b.stride(0));
    Layout right(fp8_type(a), k, m, k, batches, a.stride(0));
    Layout result(CUDA_R_16BF, n, m, interleaved ? batches * n : n,
                  batches, interleaved ? n : m * n);
    Descriptor operation;
    operation.set(CUBLASLT_MATMUL_DESC_TRANSA, CUBLAS_OP_T);
    operation.set(CUBLASLT_MATMUL_DESC_TRANSB, CUBLAS_OP_N);
    operation.set(CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, sb.data_ptr<float>());
    operation.set(CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, sa.data_ptr<float>());
    operation.set(CUBLASLT_MATMUL_DESC_FAST_ACCUM, int8_t(0));
    Preference preference(scratch.nbytes());
    static std::map<Key, cublasLtMatmulAlgo_t> algorithms;
    Key key{batches, m, n, k, fp8_type(a), fp8_type(b), interleaved};
    auto algorithm = algorithms.find(key);
    auto handle = at::cuda::getCurrentCUDABlasLtHandle();
    if (algorithm == algorithms.end()) {
        cublasLtMatmulHeuristicResult_t heuristic{};
        int count = 0;
        TORCH_CUDABLAS_CHECK(cublasLtMatmulAlgoGetHeuristic(
            handle, operation.value, left.value, right.value, result.value, result.value,
            preference.value, 1, &heuristic, &count));
        TORCH_CHECK(count > 0, "No cuBLASLt strided-batch FP8 algorithm for this shape/type");
        algorithm = algorithms.emplace(key, heuristic.algo).first;
    }
    const float alpha = 1.0f, beta = 0.0f;
    TORCH_CUDABLAS_CHECK(cublasLtMatmul(
        handle, operation.value, &alpha, b.data_ptr(), left.value, a.data_ptr(), right.value,
        &beta, out.data_ptr(), result.value, out.data_ptr(), result.value, &algorithm->second,
        scratch.data_ptr(), scratch.nbytes(), at::cuda::getCurrentCUDAStream()));
    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
    module.def("matmul", &matmul);
}
