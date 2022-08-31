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
nx = 32
ny = 32
nmod = 2
Lx = 4
Ly = 1
dealias = 4
dtype = np.float64

# Physical parameters
τd = 1
τb = 0.9
sx = 0.5
εη = 1e-1
n = 3

# Tau parameters
k_tau_int = 2
k_lift_int = 2
k_lift_edge = 2

# Iteration parameters
damping = 1.0
ncc_cutoff = 1e-3
run_cutoff = False
newton_tolerance = 1e-8
load = False
save = True

# Bases
coords = d3.CartesianCoordinates('x', 'y')
dist = d3.Distributor(coords, dtype=dtype)
xb1 = d3.ChebyshevT(coords['x'], int(nx*sx*nmod), bounds=(0, sx), dealias=dealias)
xb2 = d3.ChebyshevT(coords['x'], int(nx*(Lx-sx)/nmod), bounds=(sx, Lx), dealias=dealias)
yb = d3.ChebyshevT(coords['y'], ny*Ly, bounds=(0, Ly), dealias=dealias)
xb1_tau_int = xb1.derivative_basis(k_tau_int)
xb2_tau_int = xb2.derivative_basis(k_tau_int)
yb_tau_int = yb.derivative_basis(k_tau_int)
xb1_lift_int = xb1.derivative_basis(k_lift_int)
xb2_lift_int = xb2.derivative_basis(k_lift_int)
yb_lift_int = yb.derivative_basis(k_lift_int)
xb1_lift_edge = xb1.derivative_basis(k_lift_edge)
xb2_lift_edge = xb2.derivative_basis(k_lift_edge)
yb_lift_edge = yb.derivative_basis(k_lift_edge)

# Variables
u1 = dist.Field(name='u', bases=(xb1, yb))
u2 = dist.Field(name='u', bases=(xb2, yb))
tx1 = [dist.Field(name=f'τx{i}', bases=xb1_tau_int) for i in range(2)]
tx2 = [dist.Field(name=f'τx{i}', bases=xb2_tau_int) for i in range(2)]
ty = [dist.Field(name=f'τy{i}', bases=yb_tau_int) for i in range(4)]
tc = [dist.Field(name=f'τ{i}') for i in range(9)]
vars = [u1, u2] + tx1 + tx2 + ty + tc

# Tau terms
tau_u1 = (d3.Lift(tx1[0], yb_lift_int, -1) + d3.Lift(tx1[1], yb_lift_int, -2) +
          d3.Lift(ty[0], xb1_lift_int, -1) + d3.Lift(ty[1], xb1_lift_int, -2))
tau_u2 = (d3.Lift(tx2[0], yb_lift_int, -1) + d3.Lift(tx2[1], yb_lift_int, -2) +
          d3.Lift(ty[2], xb2_lift_int, -1) + d3.Lift(ty[3], xb2_lift_int, -2))
tau_T1 = d3.Lift(tc[0], xb1_lift_edge, -1) + d3.Lift(tc[1], xb1_lift_edge, -2)
tau_T2 = d3.Lift(tc[2], xb2_lift_edge, -1) + d3.Lift(tc[3], xb2_lift_edge, -2)
tau_B1 = d3.Lift(tc[4], xb1_lift_edge, -1) + d3.Lift(tc[5], xb1_lift_edge, -2)
tau_B2 = d3.Lift(tc[6], xb2_lift_edge, -1) + d3.Lift(tc[7], xb2_lift_edge, -2)
tau_L1 = 0#d3.Lift(tc[4], yb_lift_edge, -1) + d3.Lift(tc[10], yb_lift_edge, -2)
tau_L2 = 0#d3.Lift(tc[5], yb_lift_edge, -1) + d3.Lift(tc[1], yb_lift_edge, -2)
tau_R1 = 0#d3.Lift(tc[6], yb_lift_edge, -1) + d3.Lift(tc[2], yb_lift_edge, -2)
tau_R2 = 0#d3.Lift(tc[5], yb_lift_edge, -1) + d3.Lift(tc[11], yb_lift_edge, -2)

# Substitutions
x1 = dist.local_grid(xb1)
x2 = dist.local_grid(xb2)
y = dist.local_grid(yb)
dx = lambda A: d3.Differentiate(A, coords['x'])
dy = lambda A: d3.Differentiate(A, coords['y'])
η1 = (εη + dx(u1)**2 + dy(u1)**2) ** ((1-n)/n/2)
η2 = (εη + dx(u2)**2 + dy(u2)**2) ** ((1-n)/n/2)
fx1 = η1 * dx(u1)
fx2 = η2 * dx(u2)
fy1 = η1 * dy(u1)
fy2 = η2 * dy(u2)

# Problem
problem = d3.NLBVP(vars, namespace=locals())
problem.add_equation("div(η1*grad(u1)) + tau_u1 = - τd")
problem.add_equation("div(η2*grad(u2)) + tau_u2 = - τd")
problem.add_equation("u1(x=0) = 0")
problem.add_equation("u1(x=sx) = u2(x=sx)")
problem.add_equation("dx(u1)(x=sx) = dx(u2)(x=sx)")
problem.add_equation("dx(u2)(x=Lx) = 0")
problem.add_equation("dy(u1)(y=Ly) + tau_T1 = 0")
problem.add_equation("dy(u2)(y=Ly) + tau_T2 = 0")
problem.add_equation("u1(y=0) + tau_B1 = 0")
problem.add_equation("fy2(y=0) + tau_B2 = tc[-1]")
#problem.add_equation("fy1(x=sx,y=0) = tc[-1]")
#problem.add_equation("dx(u2)(x=sx,y=0) = 0")
problem.add_equation("u1(x=sx,y=0) = 0")
# Interior tau degeneracies
problem.add_equation("tx1[0](x=0) = 0")
problem.add_equation("tx1[0](x=sx) = 0")
problem.add_equation("tx1[1](x=0) = 0")
problem.add_equation("tx1[1](x=sx) = 0")
problem.add_equation("tx2[0](x=sx) = 0")
problem.add_equation("tx2[0](x=Lx) = 0")
problem.add_equation("tx2[1](x=sx) = 0")
problem.add_equation("tx2[1](x=Lx) = 0")
solver = problem.build_solver(ncc_cutoff=ncc_cutoff)

# Plot solution
def plot_solution() :
    fig, axs = plt.subplots(3, 2, figsize=(10, 8))
    # u(x,y)
    ax = axs[0,0]
    u1.change_scales(1)
    u2.change_scales(1)
    umax = max(u1['g'].max(), u2['g'].max())
    im1 = ax.pcolormesh(x1.ravel(), y.ravel(), u1['g'].T, cmap='viridis', shading='gouraud', rasterized=True, clim=(0,umax))
    im2 = ax.pcolormesh(x2.ravel(), y.ravel(), u2['g'].T, cmap='viridis', shading='gouraud', rasterized=True, clim=(0,umax))
    ax.set_aspect('equal')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    fig.colorbar(im2, ax=ax, location='top', label='u')
    # Spectrum of u
    ax = axs[1,0]
    uc = np.concatenate([u1['c'], u2['c']], axis=0)
    im = ax.imshow(np.log10(1e-30+np.abs(uc.T)), origin='lower', cmap='viridis', clim=(-10, 0))
    ax.set_xlabel('kx')
    ax.set_ylabel('ny')
    fig.colorbar(im, ax=ax, location='top', label='log10(|uc|)')
    # η(x,y)
    ax = axs[0,1]
    η1i = η1.evaluate()
    η2i = η2.evaluate()
    η1i.change_scales(1)
    η2i.change_scales(1)
    ηmax = max(η1i['g'].max(), η2i['g'].max())
    im1 = ax.pcolormesh(x1.ravel(), y.ravel(), η1i['g'].T, cmap='viridis', shading='gouraud', rasterized=True, clim=(0,ηmax))
    im2 = ax.pcolormesh(x2.ravel(), y.ravel(), η2i['g'].T, cmap='viridis', shading='gouraud', rasterized=True, clim=(0,ηmax))
    ax.set_aspect('equal')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    fig.colorbar(im2, ax=ax, location='top', label='η')
    # Spectrum of η
    ax = axs[1,1]
    ηic = np.concatenate([η1i['c'], η2i['c']], axis=0)
    im = ax.imshow(np.log10(1e-30+np.abs(ηic.T)), origin='lower', cmap='viridis', clim=(-10, 0))
    ax.set_xlabel('kx')
    ax.set_ylabel('ny')
    fig.colorbar(im, ax=ax, location='top', label='log10(|ηc|)')
    # bottom slip
    ax = axs[2,0]
    u1i = u1(y=0).evaluate()
    u2i = u2(y=0).evaluate()
    u1i.change_scales(1)
    u2i.change_scales(1)
    ax.plot(x1.ravel(), u1i['g'], color='C0')
    ax.plot(x2.ravel(), u2i['g'], color='C0')
    ax.set_ylabel('u', color='C0')
    ax.set_xlabel('x')
    ax.set_title('y = 0')
    # bottom stress
    ax = ax.twinx()
    fy1i = fy1(y=0).evaluate()
    fy2i = fy2(y=0).evaluate()
    fy1i.change_scales(1)
    fy2i.change_scales(1)
    ax.plot(x1.ravel(), fy1i['g'], color='C1')
    ax.plot(x2.ravel(), fy2i['g'], color='C1')
    ax.set_ylabel('η*dy(u)', color='C1')
    # stats
    ax = axs[2,1]
    ax.text(0.5, 0.5, f"sx = {sx}\nτd = {tc[-1]['g'][0,0]:.10f}", ha='center', va='center')
    ax.axis('off')
    fig.tight_layout()
    fig.savefig(f'frames_sem/shear_thinning_{solver.iteration}.pdf')

# Initial condition
if load:
    with np.load('solution.npz') as data:
        u1.load_from_global_grid_data(data['u1'])
        u2.load_from_global_grid_data(data['u2'])
else:
    u1['g'] = y * (2*Ly - y) * x1 * (2*Lx - x1) / Lx**2 / 10
    u2['g'] = y * (2*Ly - y) * x2 * (2*Lx - x2) / Lx**2 / 10
    tc[8]['g'] = τb
plot_solution()

# Newton iterations
pert_norm = np.inf
while pert_norm > newton_tolerance:
    if run_cutoff:
        solver.ncc_cutoff = min(ncc_cutoff, pert_norm)
    solver.newton_iteration(damping)
    pert_norm = sum(pert.allreduce_data_norm('c', 2) for pert in solver.perturbations)
    logger.info(f'Iteration: {solver.iteration}, Perturbation norm: {pert_norm:.3e}')
    logger.info(f"τb: {tc[-1]['g'][0,0]:.10f}")
    for t in tc[:-1]:
        print(f"{t['g'][0,0]:.2e}")
    plot_solution()

# Save solution
if save:
    np.savez('solution.npz', u1=u1['g'], u2=u2['g'], x1=x1.ravel(), x2=x2.ravel(), y=y.ravel())
