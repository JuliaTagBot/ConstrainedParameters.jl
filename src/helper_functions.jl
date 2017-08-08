struct Val{p} end

logit(x::Real) = log( x / (1 - x) )
logistic(x::Real) = 1 / ( 1 + exp( - x ) )
sigmoid(x::Real) = log( (1 + x)/(1 - x) )
