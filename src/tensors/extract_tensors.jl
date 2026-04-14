using ForwardDiff

# Polynomial tensor extraction via nested ForwardDiff.
# For a single element with displacements u ∈ R^ndof_e:
#   K1e = ∂f/∂u                       (Jacobian)
#   K2e = ½ ∂²f_i/∂u∂u                (Hessian per output / 2)
#   K3e = ⅙ ∂³f_i/∂u∂u∂u              (Jacobian of vec(Hessian) / 6)
# all evaluated at u = 0. For SVK these are exact (f is cubic in u).

# Extract the three local tensors for one element via nested ForwardDiff.

# Index layout (column-major vec + reshape):
#   J1[i, j]                         = ∂f_i/∂u_j
#   J2[i + (j-1)n, k]                = ∂²f_i/∂u_j∂u_k
#   J3[i + (j-1)n + (k-1)n², l]      = ∂³f_i/∂u_j∂u_k∂u_l
function extract_elem_tensors(geom::ElementGeom, mat::SVKMaterial)
    ndof = length(geom.gdofs)
    u0   = zeros(ndof)
    f(u) = elem_force_pure(u, geom, mat)
    g(u) = vec(ForwardDiff.jacobian(f, u))
    h(u) = vec(ForwardDiff.jacobian(g, u))

    K1e = ForwardDiff.jacobian(f, u0)
    K2e = reshape(ForwardDiff.jacobian(g, u0), ndof, ndof, ndof) ./ 2
    K3e = reshape(ForwardDiff.jacobian(h, u0), ndof, ndof, ndof, ndof) ./ 6

    return K1e, K2e, K3e
end
