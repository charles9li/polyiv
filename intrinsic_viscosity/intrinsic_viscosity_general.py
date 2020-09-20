from numba import njit
from scipy import integrate
import matplotlib.pyplot as plt
import numpy as np


def compute_viscosity(N, rho, sigma, a, b, vm):
    xi = compute_drag_function(rho, sigma, a)
    kappa = compute_drainage_function(rho, xi, sigma, b)
    integrand1 = lambda r: rho(r)*kappa(r)**2*r**4
    integrand2 = lambda r: (1 - vm*rho(r))*xi(r)*4*np.pi*r**2
    eta = 4*np.pi**2*sigma/N*integrate.quad(integrand1, 0, np.inf) + 2.5*(vm + 1/N*integrate.quad(integrand2, 0, np.inf))


# ================= #
# DRAINAGE FUNCTION #
# ================= #

# @njit
def sphere_to_cart(r, phi, theta):
    x = r * np.cos(phi) * np.sin(theta)
    y = r * np.sin(phi) * np.sin(theta)
    z = r * np.cos(theta)
    return np.array([x, y, z])


def cart_to_sphere(x, y, z):
    r = np.sqrt(x**2 + y**2 + z**2)
    try:
        phi = np.arctan(y / x)
    except ZeroDivisionError:
        if y == 0:
            phi = np.arctan(np.inf)
        else:
            phi = 0
    theta = np.arccos(z / r)
    return np.array([r, phi, theta])


def hydro_tensor(r_vec, r0_vec, rho, b):
    rho_sum = rho(np.linalg.norm(r0_vec)) + rho(np.linalg.norm(r_vec))
    inv_lambda_h = (np.pi*b**2*rho_sum) / 4.
    dist = np.linalg.norm(np.array(r0_vec) - np.array(r_vec))
    return np.exp(-dist * inv_lambda_h) / (6*np.pi*dist)


def integrand_drainage(r_vec, r0_vec, rho, sigma, b):
    return 6*np.pi*sigma * rho(r_vec)*hydro_tensor(r_vec, r0_vec, rho, b)


def compute_drainage_function(rho, xi, sigma, b, truncate=np.inf):

    def integrand2(r0_vec):
        x0, y0, z0 = r0_vec
        r0, phi0, theta0 = cart_to_sphere(*r0_vec)

        cartesian = False

        if cartesian:

            @njit
            def heaviside(x):
                if x < 0:
                    return 0
                return 1

            @njit
            def hydro_tensor(r_vec):
                rho_sum = rho(np.linalg.norm(r0_vec)) + rho(np.linalg.norm(r_vec))
                inv_lambda_h = (np.pi*b**2*rho_sum) / 4.
                dist = np.linalg.norm(r0_vec - r_vec)
                if dist < 2*sigma:
                    return 0
                return heaviside(dist - 2*sigma)*np.exp(-dist * inv_lambda_h) / (6*np.pi*dist)

            @njit
            def integrand(x, y, z):
                r_vec = np.array([x, y, z])
                return 6*np.pi*sigma * rho(np.linalg.norm(r_vec))*hydro_tensor(r_vec)

            def integrate_z(x, y):
                points = np.linspace(z0 - 10*sigma, z0 + 10*sigma, 3)
                return integrate.quad(lambda z: integrand(x, y, z), z0 - truncate, z0 + truncate, points=points)[0]

            def integrate_y(x):
                points = np.linspace(y0 - 10*sigma, y0 + 10*sigma, 3)
                return integrate.quad(lambda y: integrate_z(x, y), y0 - truncate, y0 + truncate, points=points)[0]

            scheme = 1

            if scheme == 0:
                points = np.linspace(x0 - 10*sigma, x0 + 10*sigma, 3)
                return integrate.quad(integrate_y, x0 - truncate, x0 + truncate, points=points)[0]

            elif scheme == 1:
                bounds_x = (x0 - truncate, x0 + truncate)
                bounds_y = (y0 - truncate, y0 + truncate)
                bounds_z = (z0 - truncate, z0 + truncate)
                opts = [{'points': sigma*np.arange()},
                        {'points': [y0 - 5*sigma, y0 - 2*sigma, y0 + 2*sigma, y0 + 5*sigma]},
                        {'points': [z0 - 5*sigma, z0 - 2*sigma, z0 + 2*sigma, z0 + 5*sigma]}]
                return integrate.nquad(integrand, [bounds_x, bounds_y, bounds_z], opts=opts)[0]

            # # boundaries for each octant
            # bound_dict = {0: (0, truncate, y_lower, truncate, z_lower, truncate),
            #               1: (-truncate, 0, y_lower, truncate, z_lower, truncate),
            #               2: (-truncate, 0, -truncate, y_upper, z_lower, truncate),
            #               3: (0, truncate, -truncate, y_upper, z_lower, truncate),
            #               4: (0, truncate, -truncate, y_upper, -truncate, z_upper),
            #               5: (-truncate, 0, -truncate, y_upper, -truncate, z_upper),
            #               6: (-truncate, 0, y_lower, truncate, -truncate, z_upper),
            #               7: (0, truncate, y_lower, truncate, -truncate, z_upper)}

            # # compute integral
            # total = 0.0
            # for bounds in bound_dict.values():
            #     total += integrate.tplquad(integrand, *bounds)[0]
            # return total

        else:

            @njit
            def rho_offset(r_vec):
                return rho(np.linalg.norm(r_vec + r0_vec))

            @njit
            def hydro_tensor(r_vec):
                rho_sum = rho_offset(np.zeros(3)) + rho_offset(np.linalg.norm(r_vec))
                inv_lambda_h = (np.pi*b**2*rho_sum) / 4.
                dist = np.linalg.norm(r_vec)
                return np.exp(-dist * inv_lambda_h) / (6*np.pi*dist)

            @njit
            def integrand(x, y, z):
                r_vec = np.array([x, y, z])
                return 6*np.pi*sigma * rho_offset(r_vec)*hydro_tensor(r_vec)
                # if value < 0:
                #     print("f({}, {}, {}) = {}".format(x, y, z, value))

            scheme = 2

            def integrand_spherical(r, phi, theta):
                r_vec = sphere_to_cart(r, phi, theta)
                return r**2 * np.sin(theta) * integrand(*r_vec)

            if scheme == 0:
                bounds = sigma*np.arange(2, truncate, 2)
                total = 0.0
                for i in range(len(bounds) - 1):
                    r_inner = bounds[i]
                    r_outer = bounds[i + 1]
                    sum_part, err = integrate.tplquad(integrand_spherical, r_inner, r_outer, 0, np.pi, 0, 2*np.pi)
                    # print(sum_part, err)
                    total += sum_part
                    if sum_part < 0:
                        # break
                        print("{} = {}".format(sum_part, total))
                    else:
                        print("+{} = {}".format(sum_part, total))
                return total
            elif scheme == 1:
                r_inner = 2*sigma
                dr = 2*sigma
                total = 0.0
                min_iter = 50
                curr_iter = 0
                while True:
                    r_outer = r_inner + dr
                    sum_part, err = integrate.tplquad(integrand_spherical, r_inner, r_outer, 0, 2*np.pi, 0, np.pi)
                    total += sum_part
                    if sum_part < 0:
                        print("{} = {} (eps={})".format(sum_part, total, np.abs(sum_part / total)))
                    else:
                        print("+{} = {} (eps={})".format(sum_part, total, np.abs(sum_part / total)))
                    if np.abs(sum_part / total) < 1e-8 and curr_iter > min_iter:
                        return total
                    r_inner = r_outer
                    curr_iter += 1

            elif scheme == 2:
                def integrate_theta(r, phi):
                    if False and r > 5:
                        theta_list = np.linspace(0, np.pi)
                        int_list = np.empty(len(theta_list))
                        for i in range(len(int_list)):
                            int_list[i] = integrand_spherical(r, phi, theta_list[i])
                        plt.figure()
                        plt.plot(theta_list, int_list)
                        plt.title("r={}, phi={}".format(r, phi))
                        plt.show()
                    value = integrate.quad(lambda theta: integrand_spherical(r, phi, theta), 0, np.pi)[0]
                    return value

                def integrate_phi(r):
                    if False and r > 100:
                        phi_list = np.linspace(0, 2*np.pi)
                        int_list = np.empty(len(phi_list))
                        for i in range(len(int_list)):
                            int_list[i] = integrate_theta(r, phi_list[i])
                        plt.figure()
                        plt.plot(phi_list, int_list)
                        plt.title("r={}".format(r))
                        plt.show()
                    value = integrate.quad(lambda phi: integrate_theta(r, phi), 0, 2*np.pi)[0]
                    # print("{} (phi)".format(value))
                    return value

                bounds = sigma*np.arange(2, truncate, 50)
                total = 0.0
                for i in range(len(bounds) - 1):
                    r_inner = bounds[i]
                    r_outer = bounds[i+1]
                    if True:
                        r_list = np.linspace(r_inner, r_outer)
                        int_list = np.empty(len(r_list))
                        for j in range(len(int_list)):
                            int_list[j] = integrate_phi(r_list[j])
                        plt.figure()
                        plt.plot(r_list, int_list)
                        plt.title("{} to {}".format(r_inner, r_outer))
                        plt.show()
                    sum_part = integrate.quad(integrate_phi, r_inner, r_outer)[0]
                    total += sum_part
                    print("+{} = {} ({}, {})".format(sum_part, total, r_inner/sigma, r_outer/sigma))
                return total

            elif scheme == 3:
                def integrate_theta(r, phi):
                    if False and r > 5:
                        theta_list = np.linspace(0, np.pi)
                        int_list = np.empty(len(theta_list))
                        for i in range(len(int_list)):
                            int_list[i] = integrand_spherical(r, phi, theta_list[i])
                        plt.figure()
                        plt.plot(theta_list, int_list)
                        plt.title("r={}, phi={}".format(r, phi))
                        plt.show()
                    theta_list = np.linspace(0, np.pi, 51)
                    int_list = np.empty(len(theta_list))
                    for i, theta in enumerate(theta_list):
                        int_list[i] = integrand_spherical(r, phi, theta)
                    value = integrate.simps(int_list, theta_list)
                    return value

                def integrate_phi(r):
                    if False and r > 100:
                        phi_list = np.linspace(0, 2*np.pi)
                        int_list = np.empty(len(phi_list))
                        for i in range(len(int_list)):
                            int_list[i] = integrate_theta(r, phi_list[i])
                        plt.figure()
                        plt.plot(phi_list, int_list)
                        plt.title("r={}".format(r))
                        plt.show()
                    phi_list = np.linspace(0, 2*np.pi, 51)
                    int_list = np.empty(len(phi_list))
                    for i in range(len(int_list)):
                        int_list[i] = integrate_theta(r, phi_list[i])
                    value = integrate.simps(int_list, phi_list)
                    # print("{} (phi)".format(value))
                    return value

                bounds = sigma*np.arange(2, truncate, 100)
                total = 0.0
                for i in range(len(bounds) - 1):
                    r_inner = bounds[i]
                    r_outer = bounds[i+1]
                    r_list = np.linspace(r_inner, r_outer, 51)
                    int_list = np.empty(len(r_list))
                    for j, r in enumerate(r_list):
                        int_list[j] = integrate_phi(r)
                    # plt.figure()
                    # plt.plot(r_list, int_list)
                    # plt.show()
                    sum_part = integrate.simps(int_list, r_list)
                    total += sum_part
                    print("+{} = {} ({}, {})".format(sum_part, total, r_inner/sigma, r_outer/sigma))
                return total

            else:
                def integrate_phi_theta(r):
                    return integrate.dblquad(lambda phi, theta: integrand_spherical(r, phi, theta), 0, 2*np.pi, 0, np.pi)[0]

                bounds = sigma*np.arange(2, truncate, 50)
                total = 0.0
                for i in range(len(bounds) - 1):
                    r_inner = bounds[i]
                    r_outer = bounds[i+1]
                    if True:
                        r_list = np.linspace(r_inner, r_outer)
                        int_list = np.empty(len(r_list))
                        for j, r in enumerate(r_list):
                            int_list[j] = integrate_phi_theta(r)
                        plt.figure()
                        plt.plot(r_list, int_list)
                        plt.show()
                    sum_part = integrate.quad(integrate_phi_theta, r_inner, r_outer)[0]
                    total += sum_part
                    print("+{} = {} ({}, {})".format(sum_part, total, r_inner/sigma, r_outer/sigma))
                return total


    def drainage_function(r):

        # return (1 - xi(r)) / \
        #        (1 - integrate.dblquad(lambda phi, theta: integrand2(sphere_to_cart(r, phi, theta))*np.sin(theta),
        #         0, 2*np.pi, lambda phi: 0, lambda phi: np.pi)[0])
        drag = xi(r)
        integral = integrand2(np.array([0, 0, r]))
        print("drag={}, integral={}".format(drag, integral))
        return (1 - drag) / \
               (1 + 4*np.pi*integral)

    return drainage_function


def compute_drag_function(rho, sigma, a):
    s = np.pi*(sigma + a)**2

    def drag_function(r):
        return 1 - np.exp(-s*integrate.quad(rho, r, np.inf)[0])

    return drag_function


def rho_gaussian(N, Rg2):
    @njit
    def function(r):
        if r < 10 * np.sqrt(Rg2):
            return N * (3/(2*np.pi*Rg2))**(3./2.) * np.exp(-3*r**2/(2*Rg2))
        return 0
    return function
