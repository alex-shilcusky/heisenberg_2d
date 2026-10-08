module hami_svd

using LinearAlgebra
using SparseArrays
using KrylovKit



using ..bits
using ..lattice
using ..block
using ..heis_chain
using ..model

mutable struct Heisenberg_svd <: AbstractModel
    N::Int
    L0::Int
    bc::BoundaryCondition
    Jz::Float64
    Jxy::Float64
    bond_terms::Vector{Tuple{Int,Int,Matrix{Float64}}}
    local_terms::Vector{Tuple{Int,Matrix{Float64}}}
    block::Chain
    dim::Int
    ops::Vector{Matrix{Float64}}
    new_ops::Vector{Matrix{Float64}}
    neighbors

    function Heisenberg_svd(N, L0, bc, Jz, Jxy, sign, dmax; add_flips=false)
        bond_terms = []
        local_terms = []
        neighbors = lattice.Chain(N,bc)
        dim = 2^L0
        ################ CREATE OPERATORS AND ROTATE TO EIGENBASIS
        block_obj = Chain(L0, OBC)

        aux_hami = Heisenberg(3,L0,PBC,Jz,Jxy,sign,dim; use_new_ops = true)

        aux_basis = full_basis(aux_hami)
        println(aux_basis)
        dim3 = length(aux_basis)
        Hmatrix = zeros(dim3,dim3)
        n = 1
        rows = Int[]
        cols = Int[]
        vals = Float64[]
        Szop = aux_hami.new_ops[end]
        for state in @view aux_basis[:]  # @view avoids copying
            Hx = apply_H(aux_hami,state)
            for (i,j,ni,nj,coef) in Hx
               new_state = copy(state)
               new_state[i] = ni
               new_state[j] = nj
               idx = findfirst(==(new_state),aux_basis)
               Hmatrix[idx,n] += coef
               push!(rows, idx)
               push!(cols, n)
               push!(vals, coef)
            end
            n += 1
        end
        hami_sparse = sparse(rows, cols, vals, dim3, dim3)
        eig, vecs, info = eigsolve(hami_sparse, randn(dim3), 1, :SR)
        e0 = eig[1]
        psi = @view vecs[1][:]

# 2. Build reduced density matrix

        # Pre-allocate Coordinate lists for building a sparse matrix
        rows = Int[]
        cols = Int[]
        vals = Float64[]

        # Loop over all pairs of states in your restricted basis
        for k in 1:dim3
            i_row, j2_row, j3_row = aux_basis[k]
            c_row = psi[k]
    
            for m in 1:dim3
                i_col, j2_col, j3_col = aux_basis[m]
                c_col = psi[m]
        
                # PARTIAL TRACE CONDITION:
                # The environment (subsystems 2 and 3) must match exactly!
                if j2_row == j2_col && j3_row == j3_col
                    push!(rows, i_row+1)
                    push!(cols, i_col+1)
                    push!(vals, c_row * c_col)
                end
            end
        end

        # Build the sparse matrix (Julia automatically sums duplicate row/col indices)
        rho_sparse = sparse(rows, cols, vals, dim, dim)

        # Convert to standard dense matrix for physical measurements
        rho = Matrix(rho_sparse)

# 3. Diagonalize rho
        w, v = eigen(Hermitian(rho))
        println(sum(w[:]))

# 4. Truncate
        if dmax == -1
            dmax = dim
        end

        # Global ordering by singular value
        idx = sortperm(w, rev=true)
        println(w[idx])

        # Keep only the dmax eigenstates
        V = @view v[:, idx[1:dmax]]
        println(V[:,1])

        # Rotate and project every operator
        new_ops =  [V' * op * V for op in aux_hami.new_ops]
        println(diag(new_ops[1]))
        w, v = eigen(Hermitian(new_ops[1]))
        println(w)
        
# 5. Create Hamiltonian terms 

        szL = new_ops[2]
        szR = new_ops[2*L0]
        spL = new_ops[3]
        spR = new_ops[2*L0+1]

        Hbond = kron(szR, szL) * Jz
        Hbond += kron(spR, transpose(spL)) * (0.5 * sign * Jxy)
        Hbond += kron(transpose(spR), spL) * (0.5 * sign * Jxy)

        xmax = N
        if bc == OBC
            xmax = N - 1
        end

        for i in 1:xmax
            j = mod1(i + 1, N)  # mod1 for 1-based indexing
            push!(bond_terms, (i, j, copy(Hbond)))
        end

        for i in 1:N
            push!(local_terms, (i, copy(new_ops[1])))
        end
        new(N, L0, bc, Jz, Jxy, bond_terms, local_terms, block_obj, dmax, aux_hami.ops, new_ops, neighbors)
    end
end

function calc_operators_svd1d(L0, Jz, Jxy, dmax; sign = 1)
    neighbors = lattice.Chain(4, OBC)
    dim0 = 2^L0

#    hami = Heisenberg(3,L0,PBC,Jz,Jxy,sign,dim0)

#    basis = full_basis(hami)
#    dim = length(basis)
#    Hmatrix = zeros(dim,dim)
#    n = 1
#    for state in @view basis[:]  # @view avoids copying
#        Hx = apply_H(hami,state)
#        for (i,j,ni,nj,coef) in Hx
#           new_state = copy(state)
#           new_state[i] = ni
#           new_state[j] = nj
#           idx = findfirst(==(new_state),basis)
#           Hmatrix[idx,n] += coef
#        end
#        n += 1
#    end
#    w, v = eigen(Hermitian(Hmatrix))

# 1. We build and diagonalize the 3-block Hamiltonian  
    aux_hami = Heisenberg(3,L0,PBC,Jz,Jxy,sign,dim0)

    aux_basis = full_basis(aux_hami)
    println(aux_basis)
    dim = length(aux_basis)
    Hmatrix = zeros(dim,dim)
    n = 1
    rows = Int[]
    cols = Int[]
    vals = Float64[]
    Szop = aux_hami.new_ops[end]
    sz_vals = diag(Szop)
    for state in @view aux_basis[:]  # @view avoids copying
        Hx = apply_H(aux_hami,state)
        for (i,j,ni,nj,coef) in Hx
           new_state = copy(state)
           new_state[i] = ni
           new_state[j] = nj
           idx = findfirst(==(new_state),aux_basis)
           Hmatrix[idx,n] += coef
           push!(rows, idx)
           push!(cols, n)
           push!(vals, coef)
        end
        n += 1
    end
    hami_sparse = sparse(rows, cols, vals, dim, dim)
    eig, vecs, info = eigsolve(hami_sparse, randn(dim), 1, :SR)
    e0 = eig[1]
    psi = @view vecs[1][:]

# 2. Build reduced density matrix

    # Pre-allocate Coordinate lists for building a sparse matrix
    rows = Int[]
    cols = Int[]
    vals = Float64[]

    # Loop over all pairs of states in your restricted basis
    for k in 1:dim
        i_row, j2_row, j3_row = aux_basis[k]
        c_row = psi[k]

        for m in 1:dim
            i_col, j2_col, j3_col = aux_basis[m]
            c_col = psi[m]

            # PARTIAL TRACE CONDITION:
            # The environment (subsystems 2 and 3) must match exactly!
            if j2_row == j2_col && j3_row == j3_col
                push!(rows, i_row+1)
                push!(cols, i_col+1)
                push!(vals, c_row * c_col)
            end
        end
    end

    # Build the sparse matrix (Julia automatically sums duplicate row/col indices)
    rho_sparse = sparse(rows, cols, vals, dim0, dim0)

    # Convert to standard dense matrix for physical measurements
    rho = Matrix(rho_sparse)

# 3. Diagonalize rho
    w, v = eigen(rho)
# 4. Truncate
    if dmax == -1
        dmax = dim0
    end

    # Global ordering by energy
    idx = sortperm(w, rev=true)
 
    # Keep only the dmax eigenstates
    V = @view v[:, idx[1:dmax]]

    # Rotate and project every operator
    return [V' * op * V for op in aux_hami.new_ops]

end


export Heisenberg_svd, calc_operators_svd1

end
