import torch

class GaLoreProjector:
    def __init__(self, density, grad_shape):
        self.ortho_matrix = None
        # Project the smaller side: keep the right singular vectors of tall/square
        # gradients and the left singular vectors of wide ones.
        if grad_shape[0] >= grad_shape[1]:
            self.proj_side = "right"
            self.rank = round(grad_shape[1] * density)
        else:
            self.proj_side = "left"
            self.rank = round(grad_shape[0] * density)

    def update_proj(self, full_rank_grad):
        self.ortho_matrix = self._get_orthogonal_matrix(full_rank_grad, self.rank, type=self.proj_side)

    def project_down(self, full_rank):
        
        if self.proj_side == "right":
            low_rank = torch.matmul(full_rank, self.ortho_matrix.t())
        else:
            low_rank = torch.matmul(self.ortho_matrix.t(), full_rank)

        return low_rank

    def project_up(self, low_rank):
        
        if self.proj_side == 'right':
            full_rank = torch.matmul(low_rank, self.ortho_matrix)
        else:
            full_rank = torch.matmul(self.ortho_matrix, low_rank)
        
        return full_rank
        
    # svd decomposition
    def _get_orthogonal_matrix(self, weights, rank, type):
        module_params = weights

        if module_params.data.dtype != torch.float:
            float_data = False
            original_type = module_params.data.dtype
            original_device = module_params.data.device
            matrix = module_params.data.float()
        else:
            float_data = True
            matrix = module_params.data
            
        U, s, Vh = torch.linalg.svd(matrix, full_matrices = False)
        
        if type=='right':
            B = Vh[:rank, :]
            
            if not float_data:
                B = B.to(original_device).type(original_type)
            return B
        else:
            A = U[:, :rank]
            if not float_data:
                A = A.to(original_device).type(original_type)
            return A
