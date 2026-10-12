import torch


class GradientProjector:
    def __init__(
        self, rank, verbose=False, update_proj_gap=200, alpha=1.0
    ):
        self.rank = rank
        self.verbose = verbose
        self.update_proj_gap = update_proj_gap
        self.alpha = alpha
        self.ortho_matrix = None

    def project(self, full_rank_grad, iter):

        if full_rank_grad.shape[0] >= full_rank_grad.shape[1]:
            if self.ortho_matrix is None or iter % self.update_proj_gap == 0:
                self.ortho_matrix = self.get_orthogonal_matrix(
                    full_rank_grad, self.rank, type="right"
                )
            low_rank_grad = torch.matmul(full_rank_grad, self.ortho_matrix.t())
        else:
            if self.ortho_matrix is None or iter % self.update_proj_gap == 0:
                self.ortho_matrix = self.get_orthogonal_matrix(
                    full_rank_grad, self.rank, type="left"
                )
            low_rank_grad = torch.matmul(self.ortho_matrix.t(), full_rank_grad)

        return low_rank_grad

    def project_back(self, low_rank_grad):

        if low_rank_grad.shape[0] >= low_rank_grad.shape[1]:
            full_rank_grad = torch.matmul(low_rank_grad, self.ortho_matrix)
        else:
            full_rank_grad = torch.matmul(self.ortho_matrix, low_rank_grad)

        return full_rank_grad * self.alpha

    # svd decomposition
    def get_orthogonal_matrix(self, weights, rank, type):
        module_params = weights

        if module_params.data.dtype != torch.float:
            float_data = False
            original_type = module_params.data.dtype
            original_device = module_params.data.device
            matrix = module_params.data.float()
        else:
            float_data = True
            matrix = module_params.data

        U, s, Vh = torch.linalg.svd(matrix, full_matrices=False)

        # make the smaller matrix always to be orthogonal matrix
        if type == "right":
            B = Vh[:rank, :]
            if not float_data:
                B = B.to(original_device).type(original_type)
            return B
        elif type == "left":
            A = U[:, :rank]
            if not float_data:
                A = A.to(original_device).type(original_type)
            return A
        else:
            raise ValueError("type should be left or right")
