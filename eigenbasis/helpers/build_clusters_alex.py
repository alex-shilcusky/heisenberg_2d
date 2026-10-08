#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 29 15:41:33 2024

@author: ashilcusky
"""
import numpy as np



    
def rotate(op, v):
    return np.around((np.conj(v.T)@ op @ v ),12)
    # return (np.conj(v.T)@ op @ v )

    
sz = 0.5*np.array([[1,0],[0,-1]])
sx = 0.5*np.array([[0,1],[1,0]])
sy = 0.5*np.array([[0,-1j],[1j,0]])

sp = sx + 1j*sy
sm = sx - 1j*sy
sp = np.real(sp)
sm = np.real(sm)


# print('\nS+ = \n', sp)
# print('\nS- = \n', sm)


h12 = 1*(np.kron(sz, sz) + 1/2*(np.kron(sp, sm) + np.kron(sm, sp)))

# print('\n H= \n', h12)
e, v = np.linalg.eigh(h12)
    
I2 = np.eye(2)

def add_site(H, i):
    I2 = np.eye(2)
    I = np.eye(2**(i-2))
    return np.kron(H, I2) + np.kron(np.kron(I, sz), sz) + 1/2*(
        np.kron(np.kron(I, sp), sm) + np.kron(np.kron(I, sm), sp))

from functools import reduce
def sz_i(i, N):
    """
    Compute the Sz operator for site i in a chain of N sites.
    
    Sz_i = I ⊗ ... ⊗ Sz ⊗ ... ⊗ I
                       ^
                    site i
    Parameters
    ----------
    i : int - site index (0-based)
    N : int - total number of sites
    """
    Sz = np.array([[0.5,  0.],
                   [0., -0.5]])
    I  = np.eye(2)
    
    ops = [Sz if site == i else I for site in range(N)]
    return reduce(np.kron, ops)

def Sz_total(N):
    return sum(sz_i(i, N) for i in range(N))


def get_ops_Alex(b, sign_rule, J1, J2):
    I2 = np.eye(2)

    def rotate(op, v):
        return np.around((np.conj(v.T) @ op @ v), 12)

    sz = 0.5*np.array([[1,0],[0,-1]])
    sx = 0.5*np.array([[0,1],[1,0]])
    sy = 0.5*np.array([[0,-1j],[1j,0]])

    sp = sx + 1j*sy
    sm = sx - 1j*sy
    sp = np.real(sp)
    sm = np.real(sm)

    def site_op(single_op, k):
        """Place single_op on site k (0-indexed) in a chain of b sites."""
        ops = [single_op if site == k else I2 for site in range(b)]
        return reduce(np.kron, ops)

    sz_list = [site_op(sz, k) for k in range(b)]
    sp_list = [site_op(sp, k) for k in range(b)]
    sm_list = [site_op(sm, k) for k in range(b)]

    sz_total = Sz_total(b)

    # Build Hamiltonian from site operators
    h = np.zeros((2**b, 2**b))

    # J1: nearest-neighbor pairs (i, i+1)
    for i in range(b - 1):
        bond = sz_list[i] @ sz_list[i+1]
        exch = 0.5 * (sp_list[i] @ sm_list[i+1] + sm_list[i] @ sp_list[i+1])
        h += J1 * (bond - exch if sign_rule[0] else bond + exch)

    # J2: next-nearest-neighbor pairs (i, i+2)
    for i in range(b - 2):
        bond = sz_list[i] @ sz_list[i+2]
        exch = 0.5 * (sp_list[i] @ sm_list[i+2] + sm_list[i] @ sp_list[i+2])
        h += J2 * (bond - exch if sign_rule[1] else bond + exch)

    # Perturb to lift degeneracy before diagonalising
    epsilon = 1e-4
    h_perturbed = h + epsilon * sz_total
    e, v = np.linalg.eigh(h_perturbed)

    # ops layout: [h, sz_1, ..., sz_b, sp_1, ..., sp_b, sm_1, ..., sm_b, sz_total]
    ops = [h] + sz_list + sp_list + sm_list + [sz_total]

    new_ops = [rotate(op, v) for op in ops]

    return ops, new_ops

## SAME DEBUG FOR B=2

# b= 4
# old_ops, new_ops = get_ops_Alex(b=4, sign_rule=[False, False], J1=1.0, J2=0.0)

# # GOOD DEBUG FOR B=4
# b= 4
# old_ops, new_ops = get_ops_Alex(b=4, sign_rule=[False, False])

# print(len(old_ops))
# print(len(new_ops))

# H4, sz1,sz2,sz3,sz4,sp1,sp2,sp3,sp4, sm1,sm2,sm3,sm4, sztot = new_ops

# J1 = 1.0
# J2 = 0.1
# sign_rule = [False, False]


# H12 = np.matmul(sz1, sz2) 
# exch = 1/2 * np.matmul(sp1, sm2)
# exch += exch.T
# H12 = J1 * (H12 - exch if sign_rule[0] else H12 + exch)

# H23 = np.matmul(sz2, sz3) 
# exch = 1/2 * np.matmul(sp2, sm3)
# exch += exch.T
# H23 = J1 * (H23 - exch if sign_rule[0] else H23 + exch)

# H34 = np.matmul(sz3, sz4) 
# exch = 1/2 * np.matmul(sp3, sm4)
# exch += exch.T
# H34 = J1 * (H34 - exch if sign_rule[0] else H34 + exch)



# ###
# # J2 terms
# H13 = np.matmul(sz1, sz3)
# exch = 1/2 * np.matmul(sp1, sm3)
# exch += exch.T
# H13 = J2 * (H13 - exch if sign_rule[1] else H13 + exch)

# H24 = np.matmul(sz2, sz4)
# exch = 1/2 * np.matmul(sp2, sm4)
# exch += exch.T
# H24 = J2 * (H24 - exch if sign_rule[1] else H24 + exch)
# ###
# term = H12 + H23 + H34
# term += H13 + H24
# print(np.diag(term))


if 0:
    term = H12 + H23 + H34
    term += H13 + H24

    evals1, evevs1 = np.linalg.eigh(term)
    print(evals1)

    evals2, evevs2 = np.linalg.eigh(H4)
    print(evals2)


    # print(new_ops[0])
    off_diag = new_ops[0] - np.diag(np.diag(new_ops[0]))
    print(np.max(np.abs(off_diag))) 

if 0:
    b=2
    ops2, new_ops2 = get_ops_Alex(b=b, sign_rule=False)


    # i=2
    # print(ops[i])
    # print(ops2[i])


    ops4, new_ops4 = get_ops_Alex(b=4, sign_rule=False)

    # print(ops4[0])

    print(new_ops4[0])
    print(np.diag(new_ops4[0]))
# # For debug _________________________
# print(new_ops[-1])
# sztot = new_ops[-1]
# print(np.allclose(sztot - np.diag(np.diag(sztot)), 0))
# off_diag = sztot - np.diag(np.diag(sztot))
# print(np.max(np.abs(off_diag))) 

# print(np.diag(sztot))
# H, sz_L, sz_R = get_ops(2, sign_rule=True)

# print

# print('sz_L = \n', rotate(sz_L, v))
# print('sz_R = \n', rotate(sz_R, v))
    
# print(rotate(H, v))
# print('H = \n', H)


if 0: # for just 2 sites
    sz_tot = np.kron(I2, sz)+ np.kron(sz, I2)
    sx_tot = np.kron(I2, sx)+ np.kron(sx, I2)
    sy_tot = np.kron(I2, sy)+ np.kron(sy, I2)
    
    
    sz_L = np.kron(sz, I2)
    sz_R = np.kron(I2, sz)
    sz_L = rotate(sz_L, v)
    sz_R = rotate(sz_R, v)
    # sz_tot = rotate(sz_tot, v)
    print(sz_tot.shape)
    print(np.diag(sz_tot))
    sz_tot = sz_tot 
    
    
if 0: # EVERYTHING BELOW HERE IS FOR 4 SITES
    h3 = add_site(h12, 3)
    h4 = add_site(h3, 4)
    
    e4, v4 = np.linalg.eig(h4)
    
    
    
    sztot4 = np.kron(sz, np.kron(I2, np.kron(I2,I2))) + \
        np.kron(I2,np.kron(sz, np.kron(I2,I2))) +\
            np.kron(I2,np.kron(I2, np.kron(sz,I2))) + \
                np.kron(I2,np.kron(I2, np.kron(I2,sz)))
                
    sxtot4 = np.kron(sx, np.kron(I2, np.kron(I2,I2))) + \
        np.kron(I2,np.kron(sx, np.kron(I2,I2))) +\
            np.kron(I2,np.kron(I2, np.kron(sx,I2))) + \
                np.kron(I2,np.kron(I2, np.kron(I2,sx)))
                
    sytot4 = np.kron(sy, np.kron(I2, np.kron(I2,I2))) + \
        np.kron(I2,np.kron(sy, np.kron(I2,I2))) +\
            np.kron(I2,np.kron(I2, np.kron(sy,I2))) + \
                np.kron(I2,np.kron(I2, np.kron(I2,sy)))
                
    sq4 = (np.matmul(sztot4,sztot4) + np.matmul(sxtot4,sxtot4)+ np.matmul(sytot4,sytot4)  ).real
    
    def is_diagonal(matrix):
        return np.all(matrix == np.diag(np.diagonal(matrix)))
    
    print('\n S^2 = S(S+1) = ')
    print(np.diag(sq4))
    sq4 = rotate(sq4, v4)
    print(np.diag(sq4))
    
    print('\n S^z_tot = ')
    print(np.diag(sztot4))
    sztot4 = rotate(sztot4, v4)
    print(np.diag(sztot4))
# print(np.diag(sztot4))


# print(is_diagonal(sq4))

# sq4_rot = rotate(sq4,v4)
# print(is_diagonal(rotate(sq4,v4)))


# print(is_diagonal(sztot4))
# sztot4_rot = rotate(sztot4,v4)
# print(is_diagonal(rotate(sztot4,v4)))


# print()
# sq4 = sq4.real
# sq4 = rotate(sq4, v4)
# # print('\n sq4 = \n',sq4 )
# print(np.diag(sq4))
# print('S^2 = S(S+1)')

# print(np.diag(sq4))
# print()
# print('sz4 tot')
# print(np.diag(sztot4))
# print(np.diag(rotate(sztot4, v4)))



# sztot4 = rotate(sztot4, v4)
# print(np.diag(sztot))

    
    # print('\n###############\n')
if 0:  
    sz_tot1 = np.kron(sz, np.eye(2)) + np.kron(np.eye(2), sz)
    print('sz_tot1 = \n', sz_tot1)
    print('rotate it')
    print(rotate(sz_tot1, v))
    
    sx_tot = np.kron(sx, np.eye(2)) + np.kron(np.eye(2), sx)
    sy_tot = np.kron(sy, np.eye(2)) + np.kron(np.eye(2), sy)
    
    sq2 = np.matmul(sz_tot,sz_tot) + np.matmul(sx_tot,sx_tot)+ np.matmul(sy_tot,sy_tot)  
    # sq2 = sq2.real
    # sq2 = rotate(sq2, v)
    print('\n sq2 =\n', sq2)
    print('rotate it')
    print(rotate(sq2, v))
    # sztot = np.kron(sz, np.kron(I2, np.kron(I2,I2))) + \
    
    
    
    