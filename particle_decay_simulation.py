#!/usr/bin/env python
# coding: utf-8

"""
Particle Decay Beam Simulation
==============================

Author: Newton Fernihough

A Monte Carlo simulation of a beam of unstable particles travelling along
the z-axis, decaying exponentially in flight, and emitting daughter
particles isotropically in their rest frame. The daughter particles are
tracked to four downstream tracking stations, where their (x, y) hit
positions are recorded with realistic Gaussian measurement smearing and
finite detector acceptance.

The script:
    - Samples beam velocities from a Gaussian distribution
    - Samples decay times from an exponential distribution
    - Generates isotropic daughter-particle directions
    - Propagates trajectories to each tracking station
    - Applies measurement resolution and acceptance cuts
    - Produces hit maps, hit-multiplicity distributions, angular spectra,
      and a set of validation plots comparing sampled distributions to
      their analytic forms.
"""

# ### **Setup** ###

# Imports

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm


# Parameters

# In[128]:


LIFETIME = 2.5e-3          # Mean lifetime τ (s)
V_Z_MEAN = 2000.0          # Beam mean velocity (m/s)
V_Z_SIGMA = 100.0          # Velocity spread (m/s)
STN_Z = np.array([30, 35, 40, 45])  # Detector z-positions (m)
STN_HALF_WIDTH = 5.0       # Half-width of station active area (m)
POS_SIGMA = 0.01           # Hit resolution (m)
N_EVENTS = 10000           # Number of simulated beam particles


# Functions

# In[130]:


def sample_beam_velocities(n=N_EVENTS, mean=V_Z_MEAN, spread=V_Z_SIGMA):
    """
    Draws longitudinal beam velocities from a Gaussian distribution.
    Only the z-component is sampled since motion is forward along z.
    """
    return np.random.normal(mean, spread, n)


def sample_decay_times(tau=LIFETIME, n=N_EVENTS):
    """
    Samples decay times according to the exponential law.
    """
    return np.random.exponential(tau, n)


def compute_decay_vertices(vel_z, t_decay):
    """
    Converts decay times into 3D decay vertex coordinates.
    Since motion is non-relativistic and strictly along z,
    x=y=0 and z = v * t.
    """
    z = vel_z * t_decay
    zeros = np.zeros_like(z)
    return np.column_stack((zeros, zeros, z))


def sample_isotropic_directions(n=N_EVENTS):
    """
    Generates isotropic unit vectors using uniform cos(theta)
    and uniform phi distributions.
    """
    cosT = np.random.uniform(-1, 1, n)
    phi = np.random.uniform(0, 2*np.pi, n)
    sinT = np.sqrt(1 - cosT**2)

    dx = sinT * np.cos(phi)
    dy = sinT * np.sin(phi)
    dz = cosT

    return np.column_stack((dx, dy, dz))


def intercept_with_stations(vtx, directions, z_planes):
    """
    Computes true (x,y) intercepts at each tracking station.
    The straight-line path is parameterised and solved for z=z_station.
    """
    x0, y0, z0 = vtx[:,0], vtx[:,1], vtx[:,2]
    dx, dy, dz = directions[:,0], directions[:,1], directions[:,2]

    hit_list = []
    for z_stn in z_planes:
        t = (z_stn - z0) / dz
        x = x0 + dx * t
        y = y0 + dy * t
        hit_list.append(np.column_stack((x, y)))

    return np.stack(hit_list, axis=1)


def smear_positions(true_xy, sigma=POS_SIGMA):
    """
    Applies Gaussian measurement noise to hit coordinates.
    """
    return true_xy + np.random.normal(0, sigma, true_xy.shape)


def inside_active_area(hit_xy, half_width=STN_HALF_WIDTH):
    """
    Boolean mask: whether each smeared hit falls inside active detector bounds.
    """
    x_in = np.abs(hit_xy[:,:,0]) <= half_width
    y_in = np.abs(hit_xy[:,:,1]) <= half_width
    return x_in & y_in


# ### **Simulation** ###

# In[132]:


# Beam and decay
beam_vel = sample_beam_velocities(N_EVENTS)
decay_t = sample_decay_times(n=N_EVENTS)
vtx_pos = compute_decay_vertices(beam_vel, decay_t)

# Directions
direc = sample_isotropic_directions(N_EVENTS)

# True and measured hit positions
true_xy = intercept_with_stations(vtx_pos, direc, STN_Z)
meas_xy = smear_positions(true_xy)

# Detector acceptance
hit_mask = inside_active_area(meas_xy)
nhits = hit_mask.sum(axis=1)


# Results and Plots

# In[134]:


# Hit maps at each detector station

station_counts = []
for s in range(len(STN_Z)):
    m = hit_mask[:, s]
    H, _, _ = np.histogram2d(meas_xy[m,s,0], meas_xy[m,s,1],bins=50, range=[[-5,5],[-5,5]])
    station_counts.append(H)

# Common colour scale for all heatmaps
vmax = np.max([np.max(h) for h in station_counts])

fig, axs = plt.subplots(2, 2, figsize=(12, 10))
axs = axs.flatten()

for i, zpos in enumerate(STN_Z):
    m = hit_mask[:, i]
    h = axs[i].hist2d(meas_xy[m,i,0], meas_xy[m,i,1],bins=50, range=[[-5,5],[-5,5]],cmap='plasma', vmin=0, vmax=vmax)
    axs[i].set_title(f"Station at z = {zpos} m")
    axs[i].set_xlabel("x (m)")
    axs[i].set_ylabel("y (m)")
    plt.colorbar(h[3], ax=axs[i])

plt.tight_layout()
plt.show()


# In[135]:


# Distribution of nhits

plt.figure(figsize=(6,4))
plt.hist(nhits, bins=np.arange(-0.5,5.5,1), rwidth=0.9,edgecolor='black')
plt.xticks(range(5))
plt.xlabel("Number of hit stations")
plt.ylabel("Counts")
plt.title("nhits Distribution")
plt.grid(alpha=0.3)
plt.show()

print("Mean nhits:", nhits.mean())
for k in range(5):
    print(f"  nhits = {k}: {np.mean(nhits == k):.4f}")


# In[136]:


# Total hits per station

hits_by_station = hit_mask.sum(axis=0)
plt.figure(figsize=(7,5))
plt.bar([f"Station {i+1}" for i in range(len(STN_Z))],hits_by_station, color='skyblue', edgecolor='black')
plt.ylabel("Counts")
plt.title("Total Hits per Station")
plt.grid(axis='y', alpha=0.3)
plt.show()

for i in range(len(STN_Z)):
    print(f"  Station {i}: {hits_by_station[i]}")


# In[137]:


# Fraction of backwards detections

dz = direc[:,2]
back_mask = dz < 0
detected_mask = nhits > 0

back_dets = np.sum(back_mask & detected_mask)
tot_dets = detected_mask.sum()

print("Total detected events:", tot_dets)
print("Backward detections:", back_dets)
print("Fraction:", back_dets / tot_dets)


# In[138]:


# Forward only events

fwd_mask = dz >= 0
vtx_fwd = vtx_pos[fwd_mask]
direc_fwd = direc[fwd_mask]
true_fwd = true_xy[fwd_mask]
meas_fwd = meas_xy[fwd_mask]
mask_fwd = hit_mask[fwd_mask]
nhits_fwd = nhits[fwd_mask]

print("Original events:", len(nhits))
print("Forward-only events:", len(nhits_fwd))


# In[139]:


# Angular distribution at stations

theta_deg = np.degrees(np.arccos(dz))

fig, axs = plt.subplots(2, 2, figsize=(12, 10))
axs = axs.flatten()

num_stations = len(STN_Z)

for s in range(num_stations):
    hit_s = hit_mask[:, s]

    f_vals = theta_deg[hit_s & fwd_mask]
    b_vals = theta_deg[hit_s & back_mask]

    ax = axs[s]
    ax.hist(f_vals, bins=40, alpha=0.7, label="Forward", color="blue")
    ax.hist(b_vals, bins=40, alpha=0.7, label="Backward", color="red")
    ax.set_title(f"Station {s} (z={STN_Z[s]} m)")
    ax.set_xlabel("θ (deg)")
    ax.set_ylabel("Counts")
    ax.legend()

plt.tight_layout()
plt.show()


# ### **Validation Checks** ###

# Check the normal function works

# In[142]:


# Beam velocity distribution

vel_samples = np.random.normal(V_Z_MEAN, V_Z_SIGMA, N_EVENTS)
xs = np.linspace(V_Z_MEAN - 4*V_Z_SIGMA, V_Z_MEAN + 4*V_Z_SIGMA, 500)

plt.figure(figsize=(6,4))
plt.hist(vel_samples, bins=50, density=True, alpha=0.6,edgecolor='black', label="Sampled")
plt.plot(xs, norm.pdf(xs, V_Z_MEAN, V_Z_SIGMA), 'r-', lw=2,label="Expected PDF")
plt.xlabel("Velocity (m/s)")
plt.ylabel("Density")
plt.legend()
plt.title("Beam Velocity Check")
plt.show()


# Check Exponential Survival Curve 

# In[144]:


# Exponential decay distance check

z_decay = beam_vel * decay_t

plt.figure(figsize=(6,4))
counts, edges, _ = plt.hist(z_decay, bins=100, alpha=0.6,edgecolor='black', label="Simulated")

centers = 0.5 * (edges[:-1] + edges[1:])
lambda_decay = V_Z_MEAN * LIFETIME

theory = len(beam_vel) * (np.exp(-edges[:-1]/lambda_decay)- np.exp(-edges[1:]/lambda_decay))

plt.plot(centers, theory, 'r--', lw=2, label="Theory")
plt.xlabel("Decay Distance z (m)")
plt.ylabel("Counts")
plt.title("Decay Distance Validation")
plt.legend()
plt.grid(alpha=0.3)
plt.show()


# Isotropy Check

# In[162]:


# 3. Isotropy validation
iso = sample_isotropic_directions(N_EVENTS)
px, py, pz = iso[:,0], iso[:,1], iso[:,2]

phi = np.arctan2(py, px)
phi[phi < 0] += 2*np.pi
cosT = pz
theta = np.arccos(cosT)

plt.figure(figsize=(6,4))
plt.hist(phi, bins=50, density=True, edgecolor='black')
plt.xlabel("φ (radians)")
plt.ylabel("Normalised count")
plt.title("Uniform φ distribution")
plt.show()

plt.figure(figsize=(6,4))
plt.hist(cosT, bins=50, density=True, edgecolor='black')
plt.xlabel("cosθ")
plt.ylabel("Normalised count")
plt.title("Uniform cosθ distribution")
plt.show()

plt.figure(figsize=(6,4))
plt.hist(theta, bins=50, density=True, edgecolor='black', label="Simulated")
t_vals = np.linspace(0, np.pi, 200)
plt.plot(t_vals, 0.5*np.sin(t_vals), lw=2, label="Expected ∝ sinθ")
plt.legend()
plt.xlabel("θ (radians)")
plt.ylabel("Normalised count")
plt.title("θ distribution")
plt.show()

fig = plt.figure(figsize=(5,5))
ax = fig.add_subplot(111, projection='3d')
sample = np.random.choice(len(px), 5000, False)
ax.scatter(px[sample], py[sample], pz[sample], s=2, alpha=0.3)
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")
ax.set_title("3D Isotropy Scatter")
plt.show()



# In[ ]:




