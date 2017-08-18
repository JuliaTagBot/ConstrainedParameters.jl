module ConstrainedParameters

using StaticArrays, SparseQuadratureGrids

import  Base:   show,
                getindex,
                setindex!,
                size,
                IndexStyle,
                +,
                *,
                convert,
                Val,
                length


export  Data,
        parameter,
        parameters,
        CovarianceMatrix,
        ConstrainedVector,
        PositiveVector,
        ProbabilityVector,
        RealVector,
        construct,
        negative_log_density,
        negative_log_density!,
        log_jacobian,
        quad_form,
        inv_det,
        inv_root_det,
        root_det,
        log_root_det,
        trace_inverse,
        lpdf_InverseWishart,
        lpdf_normal,
        logit,
        logistic,
        sigmoid,
        update_Σ!,
        update!,
        htrace_AΣinv,
        hlogdet,
        nhlogdet

include("helper_functions.jl")
include("constrained_types/constrained_types.jl")
include("constrained_types/constrained_vectors.jl")
include("constrained_types/covariance_matrices.jl")
include("log_density_functions.jl")

end # module
