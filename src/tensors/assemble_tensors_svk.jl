using SparseArrays

# Analytical SVK polynomial-tensor assembly.
# Mirrors assemble_poly_tensors but calls extract_elem_tensors_svk per element
# instead of ForwardDiff. Output layout (K1::SparseMatrixCSC, K2::SparseT3,
# K3::SparseT4) is identical, so downstream code (BCs, export, evaluation)
# works unchanged.
function assemble_poly_tensors_svk(elems::Vector{ElementGeom}, mat::SVKMaterial,
                                   nDOF::Int; tol=1e-15)
    Ir1 = Int[];     Ic1 = Int[];     V1  = Float64[]
    Ir2 = Int[];     Ic2 = Int[];     Is2 = Int[];     V2 = Float64[]
    Ir3 = Int[];     Ic3 = Int[];     Is3 = Int[];     It3 = Int[];     V3 = Float64[]

    nelem = length(elems)
    for (eidx, geom) in enumerate(elems)
        if eidx == 1 || eidx == nelem || eidx % max(1, nelem ÷ 5) == 0
            println("  Element $eidx / $nelem")
        end

        K1e, K2e, K3e = extract_elem_tensors_svk(geom, mat)
        gd   = geom.gdofs
        ndof = length(gd)

        for a in 1:ndof, b in 1:ndof
            v = K1e[a, b]
            abs(v) > tol || continue
            push!(Ir1, gd[a]); push!(Ic1, gd[b]); push!(V1, v)
        end

        for a in 1:ndof, b in 1:ndof, c in 1:ndof
            v = K2e[a, b, c]
            abs(v) > tol || continue
            push!(Ir2, gd[a]); push!(Ic2, gd[b]); push!(Is2, gd[c]); push!(V2, v)
        end

        for a in 1:ndof, b in 1:ndof, c in 1:ndof, d in 1:ndof
            v = K3e[a, b, c, d]
            abs(v) > tol || continue
            push!(Ir3, gd[a]); push!(Ic3, gd[b])
            push!(Is3, gd[c]); push!(It3, gd[d]); push!(V3, v)
        end
    end

    K1 = sparse(Ir1, Ic1, V1, nDOF, nDOF)
    K2 = SparseT3(Ir2, Ic2, Is2, V2, nDOF)
    K3 = SparseT4(Ir3, Ic3, Is3, It3, V3, nDOF)
    return K1, K2, K3
end
