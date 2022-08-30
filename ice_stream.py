"""
Simple ice stream model. Shear-thinning flow in a rectangular duct.
No slip on sides. No stress on top. Plastic bed condition on bottom.
"""

import numpy as np
import matplotlib.pyplot as plt
import dedalus.public as d3
import logging
logger = logging.getLogger(__name__)

# Discretization parameters
nx = 24
ny = 24
Lx = 4
Ly = 1
dealias = 4
dtype = np.float64

# Physical parameters
τd = 1
τb = 0.8
εu = 5e-2
εη = 5e-2
n = 3

# Tau parameters
k_tau_int = 2
k_lift_int = 2
k_lift_edge = 2

# Iteration parameters
damping = 0.8
ncc_cutoff = 1e-3
newton_tolerance = 1e-5
load = False
save = True

# Bases
coords = d3.CartesianCoordinates('x', 'y')
dist = d3.Distributor(coords, dtype=dtype)
xb = d3.ChebyshevT(coords['x'], nx*Lx, bounds=(0, Lx), dealias=dealias)
yb = d3.ChebyshevT(coords['y'], ny*Ly, bounds=(0, Ly), dealias=dealias)
xb_tau_int = xb.derivative_basis(k_tau_int)
yb_tau_int = yb.derivative_basis(k_tau_int)
xb_lift_int = xb.derivative_basis(k_lift_int)
yb_lift_int = yb.derivative_basis(k_lift_int)
xb_lift_edge = xb.derivative_basis(k_lift_edge)
yb_lift_edge = yb.derivative_basis(k_lift_edge)

# Variables
u = dist.Field(name='u', bases=(xb, yb))
tx = [dist.Field(name=f'τx{i}', bases=xb_tau_int) for i in range(2)]
ty = [dist.Field(name=f'τy{i}', bases=yb_tau_int) for i in range(2)]
tc = [dist.Field(name=f'τ{i}') for i in range(4)]
vars = [u] + tx + ty + tc

# Tau terms
tau_u = (d3.Lift(tx[0], yb_lift_int, -1) + d3.Lift(tx[1], yb_lift_int, -2) +
         d3.Lift(ty[0], xb_lift_int, -1) + d3.Lift(ty[1], xb_lift_int, -2))
tau_T = d3.Lift(tc[2], xb_lift_edge, -1) + d3.Lift(tc[0], xb_lift_edge, -2)
tau_B = d3.Lift(tc[3], xb_lift_edge, -1) + d3.Lift(tc[1], xb_lift_edge, -2)

# Substitutions
x, y = dist.local_grids(xb, yb)
dx = lambda A: d3.Differentiate(A, coords['x'])
dy = lambda A: d3.Differentiate(A, coords['y'])
u2 = dx(u)**2 + dy(u)**2
η = (εη + u2) ** ((1-n)/n/2)

# Problem
problem = d3.NLBVP(vars, namespace=locals())
problem.add_equation("div(η*grad(u)) + tau_u = - τd")
problem.add_equation("u(x=0) = 0")
problem.add_equation("dx(u)(x=Lx) = 0")
problem.add_equation("dy(u)(y=Ly) + tau_T = 0")
problem.add_equation("(η*dy(u))(y=0) - np.tanh(u(y=0)/εu)*τb + tau_B = 0")
# Interior tau degeneracies
problem.add_equation("tx[0](x=0) = 0")
problem.add_equation("tx[0](x=Lx) = 0")
problem.add_equation("tx[1](x=0) = 0")
problem.add_equation("tx[1](x=Lx) = 0")
solver = problem.build_solver(ncc_cutoff=ncc_cutoff)

# Plot solution
def plot_solution() :
    fig, axs = plt.subplots(3, 2, figsize=(10, 8))
    # u(x,y)
    ax = axs[0,0]
    u.change_scales(1)
    im = ax.pcolormesh(x.ravel(), y.ravel(), u['g'].T, cmap='viridis', shading='gouraud', rasterized=True)
    ax.set_aspect('equal')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    fig.colorbar(im, ax=ax, location='top', label='u')
    # Spectrum of u
    ax = axs[1,0]
    im = ax.imshow(np.log10(1e-30+np.abs(u['c'].T)), origin='lower', cmap='viridis', clim=(-10, 0))
    ax.set_xlabel('kx')
    ax.set_ylabel('ny')
    fig.colorbar(im, ax=ax, location='top', label='log10(|uc|)')
    # η(x,y)
    ax = axs[0,1]
    ηi = η.evaluate()
    ηi.change_scales(1)
    im = ax.pcolormesh(x.ravel(), y.ravel(), ηi['g'].T, cmap='viridis', shading='gouraud', rasterized=True)
    ax.set_aspect('equal')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    fig.colorbar(im, ax=ax, location='top', label='η')
    # Spectrum of η
    ax = axs[1,1]
    im = ax.imshow(np.log10(1e-30+np.abs(ηi['c'].T)), origin='lower', cmap='viridis', clim=(-10, 0))
    ax.set_xlabel('kx')
    ax.set_ylabel('ny')
    fig.colorbar(im, ax=ax, location='top', label='log10(|ηc|)')
    # bottom slip
    ax = axs[2,0]
    u0 = u(y=0).evaluate()
    u0.change_scales(1)
    ax.plot(x.ravel(), u0['g'], color='C0')
    ax.set_ylabel('u', color='C0')
    ax.set_xlabel('x')
    ax.set_title('y = 0')
    # bottom stress
    ax = ax.twinx()
    s0 = (η*dy(u))(y=0).evaluate()
    s0.change_scales(1)
    ax.plot(x.ravel(), s0['g'], color='C1')
    ax.set_ylabel('η*dy(u)', color='C1')
    # blank
    ax = axs[2,1]
    ax.axis('off')
    fig.tight_layout()
    fig.savefig(f'frames/shear_thinning_{solver.iteration}.pdf')

# Initial condition
if load:
    with np.load('solution.npz') as data:
        u.load_from_global_grid_data(data['u'])
else:
    u['g'] = y * (2*Ly - y) * x * (2*Lx - x) / Lx**2 / 10
plot_solution()

# Newton iterations
pert_norm = np.inf
while pert_norm > newton_tolerance:
    solver.ncc_cutoff = min(ncc_cutoff, pert_norm)
    solver.newton_iteration(damping)
    pert_norm = sum(pert.allreduce_data_norm('c', 2) for pert in solver.perturbations)
    logger.info(f'Iteration: {solver.iteration}, Perturbation norm: {pert_norm:.3e}')
    plot_solution()

# Save solution
if save:
    np.savez('solution.npz', u=u['g'], x=x.ravel(), y=y.ravel())
