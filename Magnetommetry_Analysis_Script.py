###NV MAGNETOMETRY FULL ANALYSIS SCRIPT###
import numpy as np
from scipy.optimize import brentq, curve_fit
import matplotlib.pyplot as plt
 
 
###CONSTANTS###
D = 2870.0        # zero-field splitting, MHz
gamma = 28.025    # gyromagnetic ratio, MHz per mT
mu0M = 1.345      # mu_0 * M of the magnet, T
L = 20.0          # magnet length, mm
R = 5.0           # magnet radius, mm
 
model_error = 0.028   # 2.8 % uncertainty on both models (magnetisation, radius, length)
 
 
###THEORY###
 
# exact field of the magnet along its axis, z in mm, result in mT
def B_cylinder(z, mu0M=mu0M):
    # exact finite-cylinder model, Eq. (2)
    term1 = (z + L / 2) / np.sqrt((z + L / 2)**2 + R**2)
    term2 = (z - L / 2) / np.sqrt((z - L / 2)**2 + R**2)
    return mu0M / 2 * (term1 - term2) * 1000 #Output is in mT
 
 # point-dipole approximation
def B_dipole(z, mu0M=mu0M):
    return mu0M * R**2 * L / (2 * z**3) * 1000 #also in mT
 
 
#Functions for B -> lower and higher peak
def esr_frequencies_linear(B):
# ESR frequencies, weak-field (linear) approximation
# f = D +- gamma * B * cos(theta)
# For a field along [100] all the NV's see an angle theta with cos(theta) = 1/sqrt(3)
    lower_peak = D - gamma * B / np.sqrt(3)
    upper_peak = D + gamma * B / np.sqrt(3)
    return lower_peak, upper_peak
 
def frequencies_fullsolution(B):
# ESR frequencies, full solution (characteristic polynomial from the RP reader)
# For a field along [100], cos(2*theta) = -1/3, so then
# f^3 - (D^2/3 + (gamma*B)^2) * f + (2/27) * D^3 = 0
# Fro this we can see the coefficients
    a3 = 1
    a2 = 0
    a1 = -(D**2 / 3 + (gamma * B)**2)
    a0 = 2 / 27 * D**3
 
    solutions = np.roots([a3, a2, a1, a0])
    # Three f1 < f2 < f3.
    # lower peak  f_l = f2 - f1
    # upper peak  f_u = f3 - f1
    
    solutions = np.sort(solutions.real) #small to large sort
    f1 = solutions[0]
    f2 = solutions[1]
    f3 = solutions[2]
 
    low_peak = f2 - f1
    high_peak = f3 - f1
    return low_peak, high_peak
 
#Functions for higher and lower peak -> B
def B_from_splitting_linear(splitting):
    # B = (f_upper - f_lower) * sqrt(3) / (2 * gamma) = splitting * sqrt(3) / (2 * gamma)
    return np.sqrt(3) * splitting / (2 * gamma)
 
def B_from_lower_peak(f_measured):
###FROM A MEASURED FREQUENCY BACK TO B###
# We try values of B until the calculated frequency matches the
# measured one. brentq finds a value where the
# function  is zero when the match is perfect, using a range of B
# to search in.
    def mismatch(B):
        lower, _ = frequencies_fullsolution(B)
        return lower - f_measured
    optimized_B = brentq(mismatch, 0.0, 30.0) #up to 30 B, because see the plot, after that point there's multiple possible solutions!
    return optimized_B
 
def B_from_upper_peak(f_measured):
    def mismatch(B):
        _, upper = frequencies_fullsolution(B)
        return upper - f_measured
    optimized_B = brentq(mismatch, 0.0, 100.0)
    return optimized_B
 
 
###THE MEASUREMENTS###
 
# both peaks measured
# columns: z, f_lower, f_upper, uncertainty
both = np.array([
    [53, 2833,   2911,   1],
    [51, 2830.3, 2915.3, 1],
    [47, 2817,   2931,   1],
    [45, 2811,   2939,   1],
    [43, 2800,   2951,   1],
    [41, 2792,   2965,   1],
    [39, 2781,   2983,   1],
])
 
# only the lower peak measured
# columns: z, f_lower, uncertainty
lower_only = np.array([
    [37, 2767,   1],
    [35, 2749,   1],
    [33, 2731,   1],
    [31, 2708,   1],
    [29, 2680.5, 1],
    [27, 2655.5, 1],
])
 
# only the upper peak measured
# columns: z, f_upper, uncertainty
upper_only = np.array([
    [25, 3518, 2.6],   
    [23, 3775, 5],     
])
 
 
###FROM THE MEASUREMENTS TO B ARRAY###
 
# One function per kind of measurement. Each gives B and its uncertainty.
# The uncertainty is found by a literal shift: calculate B again with the
# frequency shifted by its uncertainty, and take the difference.
 
def B_for_both(f_lower, f_upper, f_error):
    splitting = f_upper - f_lower
 
    # two independent errors for both peaks together so sqrt(1^2 + 1^2) = 1.41 MHz
    splitting_error = np.sqrt(f_error**2 + f_error**2)
 
    B = B_from_splitting_linear(splitting)
    B_shifted = B_from_splitting_linear(splitting + splitting_error)
    return B, abs(B_shifted - B)
 
 
def B_for_lower(f_lower, f_error):
    # full solution
    # brentq can only do one at a time so we have to loop over the values instead of just running it on an array
    B = []
    B_error = []
 
    for i in range(len(f_lower)):
        b = B_from_lower_peak(f_lower[i])
        b_shifted = B_from_lower_peak(f_lower[i] + f_error[i])  
        
        B.append(b)
        B_error.append(abs(b_shifted - b))
    
    return np.array(B), np.array(B_error)
 
 
def B_for_upper(f_upper, f_error):
    #same thing but for the upper peak
    B = []
    B_error = []
 
    for i in range(len(f_upper)):
        b = B_from_upper_peak(f_upper[i])
        b_shifted = B_from_upper_peak(f_upper[i] + f_error[i])
 
        B.append(b)
        B_error.append(abs(b_shifted - b))
 
    return np.array(B), np.array(B_error)
 
 
B_both, B_both_error = B_for_both(both[:, 1], both[:, 2], both[:, 3])
B_lower, B_lower_error = B_for_lower(lower_only[:, 1], lower_only[:, 2])
B_upper, B_upper_error = B_for_upper(upper_only[:, 1], upper_only[:, 2])
 
# put everything together 
z_measured = np.concatenate([both[:, 0], lower_only[:, 0], upper_only[:, 0]])
B_measured = np.concatenate([B_both, B_lower, B_upper])
B_measured_error = np.concatenate([B_both_error, B_lower_error, B_upper_error])
 
 
###GET B FOR THEORETICAL MODELS###
B_theoretical_cyl = B_cylinder(z_measured)
B_theoretical_dip = B_dipole(z_measured)
 
B_theoretical_cyl_error = model_error * B_theoretical_cyl #2.8 % for both models
B_theoretical_dip_error = model_error * B_theoretical_dip
 
 
 
#N SIGMA
n_cyl = (B_measured - B_theoretical_cyl) / np.sqrt(B_measured_error**2 + B_theoretical_cyl_error**2)
n_dip = (B_measured - B_theoretical_dip) / np.sqrt(B_measured_error**2 + B_theoretical_dip_error**2)
print("cylinder: points with |n_sigma| > 2:", np.sum(np.abs(n_cyl) > 2), "of", len(n_cyl))
print("dipole:   points with |n_sigma| > 2:", np.sum(np.abs(n_dip) > 2), "of", len(n_dip))
 
 
 
 
 
###OUTLIER CHECK AT 49 MM###
# the point at 49 mm (2842 and 2922 MHz) was left out because of its huge sigma
# for both models. THis means that it wouldnt favor any model and just mess up the fit. Here we check how many sigma it is away from each model.
 
B_outlier, B_outlier_error = B_for_both(np.array([2842]), np.array([2922]), np.array([1]))
B_out = B_outlier[0]
error_out = B_outlier_error[0]
 
n_cyl_outlier = (B_out - B_cylinder(49)) / np.sqrt(error_out**2 + (model_error * B_cylinder(49))**2)
n_dip_outlier = (B_out - B_dipole(49)) / np.sqrt(error_out**2 + (model_error * B_dipole(49))**2)
 
print("outlier at 49 mm: n_sigma cylinder =", round(n_cyl_outlier, 1),
      ", n_sigma dipole =", round(n_dip_outlier, 1))
 
 
 
 
 
###FIT USING MORE AND MORE POINTS###
# curve_fit tries values of mu0M  
# Minimizes chi^2 = sum( ((B - model) / error)^2 ), least square fitting!
# z_measured is ordered from far to close already because the experimental data matrices were too
 
closest_distance = []       # the closest z used in that fit (mm)
mu0M_cylinder = []          # fitted mu0M   cylinder model (T)
mu0M_cylinder_error = []
mu0M_dipole = []            # fitted mu0M   dipole model (T)
mu0M_dipole_error = []
 
 
# Start with 2 points and end with all of them.
for number_of_points in range(2, len(z_measured) + 1):
 
    z_in_fit = z_measured[:number_of_points] #(slicing is exclusive of the last index)
    B_in_fit = B_measured[:number_of_points]
    error_in_fit = B_measured_error[:number_of_points]
 
    # Fit each model. curve_fit gives back two things:
    #   fit_values: the best mu0M
    #   covariance: variance of mu0M,
    # p0=[1.3] is the starting guess for mu0M (close to the datasheet value).
    # absolute_sigma=True: uses our error bars instead of estimating them from datapoint spread
    fit_values_cyl, covariance_cyl = curve_fit(B_cylinder, z_in_fit, B_in_fit,
                                               p0=[1.3], sigma=error_in_fit, absolute_sigma=True)
    fit_values_dip, covariance_dip = curve_fit(B_dipole, z_in_fit, B_in_fit,
                                               p0=[1.3], sigma=error_in_fit, absolute_sigma=True)
 
    #Get the actual results
    best_mu0M_cyl = fit_values_cyl[0] #its a 1 number array
    best_mu0M_cyl_error = np.sqrt(covariance_cyl[0, 0]) #its a 1 number 2D 1x1 array
 
    best_mu0M_dip = fit_values_dip[0]
    best_mu0M_dip_error = np.sqrt(covariance_dip[0, 0])
 
    closest_z_in_fit = z_in_fit[-1]      # last point = the closest one
 
    #save for the figure
    closest_distance.append(closest_z_in_fit)
    mu0M_cylinder.append(best_mu0M_cyl)
    mu0M_cylinder_error.append(best_mu0M_cyl_error)
    mu0M_dipole.append(best_mu0M_dip)
    mu0M_dipole_error.append(best_mu0M_dip_error)
 
 
 
 
###FIGURES###
 
# figure style (same for all figures)
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
    "mathtext.fontset": "stix",      # Times-style maths ($B$, $\mu_0 M$, ...)
    "font.size": 9,
})
 
# colours
blue = "#2a78d6"     # figures 2 and 3: cylinder
orange = "#eb6834"   # figures 2 and 3: dipole
green = "#2e9e44"    # figure 1: lower peak
red = "#d62728"      # figure 1: upper peak
light_grey = "0.85"  # a number between 0 (black) and 1 (white) means grey
 
# text size of the legends
legend_size = 7.5
 
 
###FIGURE 1: THE ESR FREQUENCIES AS A FUNCTION OF B###
# solid lines  = full solution
# dashed lines = linear approximation
 
# the B values we draw the lines for
B_values = np.linspace(0, 50, 500)     # 500 values from 0 to 50 mT
 
# full solution: one B at a time, because np.roots does one at a time
lower_full = []
upper_full = []
for B in B_values:
    lower, upper = frequencies_fullsolution(B)
    lower_full.append(lower)
    upper_full.append(upper)
 
# linear approximation: works on all B values at once
lower_linear, upper_linear = esr_frequencies_linear(B_values)
 
# make an empty figure
fig, ax = plt.subplots(figsize=(5.2, 3.6))     # width, height in inches
 
# full solution, solid lines
ax.plot(B_values, upper_full, color=red, linestyle="-", linewidth=1.5, label=r"upper peak $f_+$")
ax.plot(B_values, lower_full, color=green, linestyle="-", linewidth=1.5, label=r"lower peak $f_-$")
 
# linear approximation, dashed lines
# (only one of the two gets a label, so it shows up once in the legend)
ax.plot(B_values, upper_linear, color=red, linestyle="--", linewidth=1.0, label="linear approximation")
ax.plot(B_values, lower_linear, color=green, linestyle="--", linewidth=1.0)
 
# axes and legend
ax.set_xlabel("magnetic field strength $B$ (mT)")
ax.set_ylabel("ESR frequency (MHz)")
ax.set_xlim(0, 50)
ax.legend(fontsize=legend_size, frameon=False, loc="upper left")   # frameon=False: no box around it
 
# save
fig.savefig("fig1_esr_vs_B.png", dpi=200, bbox_inches="tight")    # bbox_inches="tight": cut off empty edges
fig.savefig("fig1_esr_vs_B.pdf", bbox_inches="tight")
 
 
###FIGURE 2: MEASURED B COMPARED WITH BOTH MODELS###
 
# the z values we draw the model lines for
z_line = np.linspace(21, 60, 400)      # 400 values from 21 to 60 mm
 
# the two models along that line
B_line_cyl = B_cylinder(z_line)
B_line_dip = B_dipole(z_line)
 
# the 2.8 % uncertainty band around each model: a lower and an upper edge
B_line_cyl_low = B_line_cyl * (1 - model_error)
B_line_cyl_high = B_line_cyl * (1 + model_error)
B_line_dip_low = B_line_dip * (1 - model_error)
B_line_dip_high = B_line_dip * (1 + model_error)
 
# measured B divided by the model (1 = perfect match), and the error on that
ratio_cyl = B_measured / B_theoretical_cyl
ratio_cyl_error = B_measured_error / B_theoretical_cyl
ratio_dip = B_measured / B_theoretical_dip
ratio_dip_error = B_measured_error / B_theoretical_dip
 
# make a figure with two panels on top of each other
#   sharex=True:   both panels use the same z axis
#   height_ratios: the top panel is 1.4 times as tall as the bottom one
#   hspace:        the gap between the two panels
fig, (top, bottom) = plt.subplots(2, 1, figsize=(5.2, 6.0), sharex=True,
                                  gridspec_kw={"height_ratios": [1.4, 1], "hspace": 0.07})
 
 
# TOP PANEL
 
# the uncertainty bands: fill the space between the low and high edge
# alpha=0.2 makes the colour see-through
top.fill_between(z_line, B_line_cyl_low, B_line_cyl_high, color=blue, alpha=0.2, linewidth=0)
top.fill_between(z_line, B_line_dip_low, B_line_dip_high, color=orange, alpha=0.2, linewidth=0)
 
# the model lines
top.plot(z_line, B_line_cyl, color=blue, linestyle="-", linewidth=1.4, label="finite cylinder (exact), Eq. (2)")
top.plot(z_line, B_line_dip, color=orange, linestyle="--", linewidth=1.4, label="point dipole, Eq. (4)")
 
# the measurements, black, with a different marker for each kind:
#   o = circle, v = triangle down, ^ = triangle up
# linestyle="none": only the dots, no line connecting them
top.errorbar(both[:, 0], B_both, yerr=B_both_error,
             marker="o", linestyle="none", markersize=4.5, color="k", elinewidth=0.7,
             label="both peaks")
top.errorbar(lower_only[:, 0], B_lower, yerr=B_lower_error,
             marker="v", linestyle="none", markersize=4.5, color="k", elinewidth=0.7,
             label="lower peak only")
top.errorbar(upper_only[:, 0], B_upper, yerr=B_upper_error,
             marker="^", linestyle="none", markersize=4.5, color="k", elinewidth=0.7,
             label="upper peak only")
 
# axes and legend
top.set_ylim(0, 42)
top.set_ylabel(r"$|B|$ (mT)")
top.legend(fontsize=legend_size, frameon=False, loc="upper right")
 
# the "(a)" in the bottom left corner
# transform=top.transAxes: the position is a fraction of the panel (0 = left/bottom, 1 = right/top)
top.text(0.02, 0.05, "(a)", transform=top.transAxes)
 
 
# BOTTOM PANEL
 
# grey band = the 2.8 % model uncertainty around 1
bottom.axhspan(1 - model_error, 1 + model_error, color=light_grey, linewidth=0,
               label=r"model uncertainty $\pm$2.8%")
 
# thin black line at 1 (perfect match)
bottom.axhline(1, color="k", linewidth=0.6)
 
# the ratios
#   "s" = square, "D" = diamond
#   markerfacecolor="white": the diamonds are hollow
bottom.errorbar(z_measured, ratio_cyl, yerr=ratio_cyl_error,
                marker="s", linestyle="none", markersize=4, color=blue, elinewidth=0.8,
                label="finite cylinder")
bottom.errorbar(z_measured, ratio_dip, yerr=ratio_dip_error,
                marker="D", linestyle="none", markersize=4, color=orange, elinewidth=0.8,
                markerfacecolor="white", markeredgewidth=0.9,
                label="point dipole")
 
# theory: cylinder divided by dipole (where the dipole points should lie if the cylinder is right)
bottom.plot(z_line, B_line_cyl / B_line_dip, color=orange, linestyle="--", linewidth=1.0,
            label="theory: cylinder / dipole")
 
# axes and legend
bottom.set_ylabel(r"$B_\mathrm{meas}/B_\mathrm{model}$")
bottom.set_xlabel(r"distance from magnet centre to diamond $z$ (mm), incl. 1 mm offset")
bottom.set_xlim(21, 60)
bottom.legend(fontsize=legend_size, frameon=False, loc="upper right")
 
# the "(b)" in the bottom right corner (ha="right": the text ends at that point)
bottom.text(0.98, 0.05, "(b)", transform=bottom.transAxes, ha="right")
 
# save
fig.savefig("fig2_field_comparison.png", dpi=200, bbox_inches="tight")
fig.savefig("fig2_field_comparison.pdf", bbox_inches="tight")
 
 
###FIGURE 3: FITTED mu0*M WITH MORE AND MORE POINTS###
# each point is one fit from the "FIT USING MORE AND MORE POINTS" part above
 
# make an empty figure
fig, ax = plt.subplots(figsize=(5.2, 3.4))
 
# grey band = the datasheet range of the magnet
ax.axhspan(1.32, 1.37, color=light_grey, label="datasheet N45 (1.32–1.37 T)")
 
# the fitted values, with a line through them
#   cylinder: squares, solid line
#   dipole:   hollow diamonds, dashed line
ax.errorbar(closest_distance, mu0M_cylinder, yerr=mu0M_cylinder_error,
            marker="s", linestyle="-", markersize=4, linewidth=0.8, color=blue,
            label="finite cylinder")
ax.errorbar(closest_distance, mu0M_dipole, yerr=mu0M_dipole_error,
            marker="D", linestyle="--", markersize=4, linewidth=0.8, color=orange,
            markerfacecolor="white",
            label="point dipole")
 
# axes and legend
ax.set_xlim(53, 22)   # reversed: left = only far points, right = all points
ax.set_xlabel("closest distance included in the fit (mm)")
ax.set_ylabel(r"fitted $\mu_0 M$ (T)")
ax.legend(fontsize=legend_size, frameon=False, loc="upper left")
 
# save
fig.savefig("fig3_fitted_mu0M.png", dpi=200, bbox_inches="tight")
fig.savefig("fig3_fitted_mu0M.pdf", bbox_inches="tight")
 
 
###EXTRA PRINTS FOR THE REPORT###
print("n_sigma cylinder:", np.round(n_cyl, 1))
print("n_sigma dipole:", np.round(n_dip, 1))
print("fitted mu0M cylinder:", np.round(mu0M_cylinder, 3))
print("fitted mu0M dipole:", np.round(mu0M_dipole, 3))
 
 
###CHECK OF THE 2.8 % MODEL UNCERTAINTY###
# shift R and L by their uncertainty, see how much B changes, combine with 1.9 % from mu0M
for model in [B_cylinder, B_dipole]:
    B_normal = model(z_measured)
    R = 5.05
    change_R = model(z_measured) / B_normal - 1
    R = 5.0
    L = 20.1
    change_L = model(z_measured) / B_normal - 1
    L = 20.0
    total = np.sqrt(0.019**2 + change_R**2 + change_L**2) * 100
    print(model.__name__, "total uncertainty (%):", np.round(total, 1))
