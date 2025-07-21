# vectorlize operator for symmetric matrices
function vech(A::AbstractMatrix{T}) where {T<:Real}
    return A[triu(trues(size(A)))]
end

# vech^{-1}
function avech(a::Vector{T}) where {T<:Real} 
    m = length(a)
    d = (isqrt(1+8m) - 1)÷2
    d + (d^2-d)÷2 == m || throw(DimensionMismatch("The length of argument is not valid."))

    A = zeros(d,d)
    k = 1
    for i in 1:d
        for j in 1:i
            A[i,j] = a[k]
            A[j,i] = a[k]
            k += 1
        end
    end
    return Symmetric(A)
end