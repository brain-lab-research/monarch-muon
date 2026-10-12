# Activation
# Utils
from ._division import fp8_division
from ._division_transpose import fp8_division_transpose
from ._quantize import fp8_quantize
from ._quantize_pertensor import fp8_quantize_pertensor
from .add_fwd import fp8_add_Ifp_Ifp_Ofp_Og16
# Normalization
from .func_quantize import Coat_quantize_bgn, Coat_quantize_end
from .func_rmsnorm import fp8_rmsnorm_backward, fp8_rmsnorm_forward
# mul and SiLU
from .mul_bwd import fp8_mul_backward
from .mul_fwd import fp8_mul_forward
from .silu_bwd import fp8_silu_backward
from .silu_fwd import fp8_silu_forward
