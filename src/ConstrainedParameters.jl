module ConstrainedParameters

using StaticArrays

import  Base.show,
        Base.getindex,
        Base.setindex!,
        Base.size,
        Base.IndexStyle,
        Base.+,
        Base.*,
        Base.convert
        Base.Val


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
        log_jacobian!,
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
include("constrained_types/covariance_matrices.jl")
include("constrained_types/constrained_vectors.jl")
include("log_density_functions.jl")

end # module
