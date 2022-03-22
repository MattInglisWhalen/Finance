
# built-in libraries
import typing

import numpy
import numpy as np
import matplotlib.pyplot as plt
from math import log, sqrt, exp, ceil
from scipy.stats import norm as normal_dist

# internal classes
from MIW_Finance.src.market import Market
from MIW_Finance.src.timeSlice import TimeSlice
from MIW_Finance.src.asset import Asset
from MIW_Finance.src.option import Option

from time import perf_counter


def true_black_scholes(price:float, market: Market, option: Option):

    S0 = price
    K = option.K
    T = option.T
    sigma = option.underlying.sigma
    r = market.r

    if S0 < 0.001 < K :
        return 0
    d1 = ( log(S0/K)+(r+sigma**2/2)*T ) / ( sigma*sqrt(T) )
    d2 = ( log(S0/K)+(r-sigma**2/2)*T ) / ( sigma*sqrt(T) )
    term1 = S0*normal_dist.cdf(d1)
    term2 = -K*exp( -r*T )*normal_dist.cdf(d2)
    return term1+term2


def geom_grid_everywhere(K, num_points_above:int = 60,
                         lower_limit_absolute:float = 1, upper_limit_multiple:float = 10) -> list:


    above_points = np.geomspace(K + K/100, upper_limit_multiple*K, num_points_above )
    below_points = []
    for Si in reversed(above_points) :
        new_point = K - (Si-K)
        if new_point < lower_limit_absolute :
            continue
        below_points.append(new_point)

    all_points = [0.]
    all_points.extend(below_points)
    all_points.extend(above_points)
    return all_points


def dense_grid_around(K, num_core_points:int = 20, num_points_above:int = 50, core_size_in_tenths:float = 1,
                      lower_limit_absolute:float = 1, upper_limit_multiple:float = 10) -> list:

    M=core_size_in_tenths
    core_points = np.linspace(K-M*K/10,K+M*K/10,num_core_points)
    above_points = np.geomspace(K + M*K/10 + 2*K/(10*num_core_points), upper_limit_multiple*K, num_points_above )
    below_points = []
    for Si in reversed(above_points) :
        new_point = K - (Si-K)
        if new_point < lower_limit_absolute :
            continue
        below_points.append(new_point)

    all_points = [0.]
    all_points.extend(below_points)
    all_points.extend(core_points)
    all_points.extend(above_points)
    return all_points


def init_cond_call(S,K):
    return S-K if S > K else 0


"""
Explicit Euler methods
"""


def black_scholes_Vdot(S, V, Vprime, Vprimeprime, r, sigma) -> float:
    Vdot = 0.
    Vdot += (sigma*sigma*S*S/2)*Vprimeprime
    Vdot += r*S*Vprime
    Vdot += (-r*V)
    return Vdot


def next_timeslice_from_BS_explicit_Euler(tslice: TimeSlice,
                                          market: Market,
                                          option: Option,
                                          dt:float) -> TimeSlice:

    new_values = tslice.values
    for idx, datum in enumerate( tslice.data[1:-1] ):  # don't update at pos=0 or pos[-1] since these are fixed BCs
        n = idx+1  # the [1:] messes up idx start value -- or at least it should???
        new_values[n] += dt*black_scholes_Vdot(S=tslice.position_n(n),
                                                 V=tslice.value_n(n),
                                                 Vprime=tslice.slope_at_point_n(n),
                                                 Vprimeprime=tslice.acceleration_at_point_n(n),
                                                 r=market.r,
                                                 sigma=option.underlying.sigma
                                                 )
    # fix the S->infinity BC to be S-K*exp(-r*tau)
    # from https://www.math.cuhk.edu.hk/~rchan/teaching/math4210/chap08.pdf eq 14
    new_values[-1] = tslice.position_n(-1) - option.K*exp(-market.r*tslice.tau)


    new_slice = TimeSlice( tslice.tau + dt, tslice.positions, new_values )
    return new_slice


"""
Explicit RK4 methods
"""


def black_scholes_Vdot_of_tslice_idx(tslice: TimeSlice, idx: int, market: Market, option:Option) -> float:

    S = tslice.position_n(idx)
    r = market.r
    sigma = option.underlying.sigma

    delta1 = tslice.position_n(idx+1) - tslice.position_n(idx)
    delta2 = tslice.position_n(idx) - tslice.position_n(idx-1)
    norm = (delta2+delta1)/2
    V = tslice.value_n(idx)

    Vprime = (tslice.value_n(idx+1) - tslice.value_n(idx-1)) / (delta1+delta2)
    Vprimeprime = ( tslice.value_n(idx+1)/delta1
                        + tslice.value_n(idx-1)/delta2
                        - (delta1+delta2)*tslice.value_n(idx)/(delta1*delta2) ) / norm
    Vdot = 0.
    Vdot += (sigma * sigma * S * S / 2) * Vprimeprime
    Vdot += r * S * Vprime
    Vdot += (-r * V)

    # print(f"{delta1=} {delta2=} {S=} {tslice.value_n(idx)=} {Vprime=} {Vprimeprime=}")

    return Vdot


def black_scholes_Vdot_of_tslice(tslice: TimeSlice,
                                 market: Market,
                                 option:Option) -> np.ndarray:

    Vdot_list = [0.]
    for idx, datum in enumerate(tslice.data[1:-1]) :

        n = idx+1  # the [1:] messes up the index
        new_vdot = black_scholes_Vdot_of_tslice_idx(tslice=tslice,
                                                    idx=n,      # chopping off the start changes the implicit index
                                                    market=market,
                                                    option=option)
        Vdot_list.append(new_vdot)
    Vdot_list.append(option.K*market.r*exp(-market.r*tslice.tau) )
    return np.array(Vdot_list)


def next_timeslice_from_BS_explicit_RK4(tslice: TimeSlice,
                                        market: Market,
                                        option: Option,
                                        dt:float) -> TimeSlice:

    input_slice_K1 = TimeSlice(tau = tslice.tau,
                               positions_list = tslice.positions,
                               values_list = tslice.values)
    K1_list = black_scholes_Vdot_of_tslice( input_slice_K1, market, option)

    input_slice_K2 = TimeSlice(tau = tslice.tau+dt/2,
                               positions_list = tslice.positions,
                               values_list = tslice.values + K1_list*dt/2)
    K2_list = black_scholes_Vdot_of_tslice( input_slice_K2, market, option)

    input_slice_K3 = TimeSlice(tau = tslice.tau+dt/2,
                               positions_list = tslice.positions,
                               values_list = tslice.values + K2_list*dt/2)
    K3_list = black_scholes_Vdot_of_tslice( input_slice_K3, market, option)

    input_slice_K4 = TimeSlice(tau = tslice.tau+dt,
                               positions_list = tslice.positions,
                               values_list = tslice.values + K3_list*dt)
    K4_list = black_scholes_Vdot_of_tslice( input_slice_K4, market, option)

    new_values = tslice.values + (dt/6)*(K1_list + 2*K2_list + 2*K3_list + K4_list)
    new_tslice = TimeSlice(tau=tslice.tau+dt, positions_list=tslice.positions, values_list=new_values)
    new_tslice.validate_monotonic_increasing()
    return new_tslice


"""
Implicit Euler methods
"""


def create_IE_transfer_matrix(tslice: TimeSlice,
                              market: Market,
                              option:Option,
                              dt: float):

    # Need to think about making the transfer function affine to account for the last BC
    # e.g. input vector is (v0 , ... , v_N-1 , 1) and transfer matrix is (N+1)x(N+1)

    # ... or just do x = Ainv b + c

    N = len(tslice.data)
    matrix = np.zeros( (N,N) )

    r = market.r
    sigma = option.underlying.sigma

    for irow in range(N):
        if irow == 0 :   # put these ifs outside the loop for speedup
            matrix[0][0] = 0
            continue
        elif irow == N-1 :
            matrix[N-1][N-1] = 0
            continue

        S = tslice.position_n(irow)
        delta1 = tslice.position_n(irow+1) - tslice.position_n(irow)
        delta2 = tslice.position_n(irow) - tslice.position_n(irow-1)
        norm = delta1*delta2*(delta2+delta1)/2

        # should shadow the (negative) coefficients of value(n) in "black_scholes_Vdot_of_tslice_idx"

        matrix[irow][irow-1] += (sigma*sigma*S*S/2) * (delta1/norm)
        matrix[irow][irow-1] += r*S*(-1/(delta1+delta2))

        matrix[irow][irow] += (sigma*sigma*S*S/2) * ( -(delta1+delta2)/norm)
        matrix[irow][irow] += -r

        matrix[irow][irow+1] += (sigma*sigma*S*S/2) * (delta2/norm)
        matrix[irow][irow+1] += r*S*(1/(delta1+delta2))

    return (-dt)*matrix


def next_timeslice_from_BS_implicit_Euler(tslice: TimeSlice, market: Market, option: Option, dt:float) -> TimeSlice:

    N = len(tslice.data)

    b = tslice.values
    b[-1] += dt*option.K*market.r*exp(-market.r*tslice.tau)
    A = numpy.identity(N) + create_IE_transfer_matrix(tslice, market, option, dt)

    new_values = np.linalg.solve(A,b)
    new_tslice = TimeSlice(tau=tslice.tau+dt,positions_list=tslice.positions,values_list=new_values)

    return new_tslice

# It seems there's fundamental about this PDE that gives the wrong answer near the kink


"""
Crank Nicholson methods

Can duplicate the work of the implicit Euler method by using the "create_IE_transfer_matrix" with dt=dt/2, dt=-dt/2
"""


def next_timeslice_from_BS_Crank_Nicholson(tslice: TimeSlice, market: Market, option: Option, dt: float) -> TimeSlice:
    N = len(tslice.data)

    explicit_transfer = create_IE_transfer_matrix(tslice, market, option, -dt/2)
    b = tslice.values + np.matmul(explicit_transfer,tslice.values)
    b[-1] += dt*option.K*market.r*exp(-market.r*tslice.tau)
    A = np.identity(N) + create_IE_transfer_matrix(tslice, market, option, dt/2)

    new_values = np.linalg.solve(A, b)
    new_tslice = TimeSlice(tau=tslice.tau + dt, positions_list=tslice.positions, values_list=new_values)

    return new_tslice




"""
Method of lines
"""

"""
Helper function
"""


def get_history_with_method(method: str, market:Market, option:Option,
                            init_slice:TimeSlice, num_timesteps:int) -> list[TimeSlice] :

    timestep = option.T/num_timesteps
    primed_history = [init_slice]
    time_stepping_method: typing.Callable = None

    if method == "Explicit Euler" :
        time_stepping_method = next_timeslice_from_BS_explicit_Euler
    elif method == "Explicit RK4" :
        time_stepping_method = next_timeslice_from_BS_explicit_RK4
    elif method == "Implicit Euler" :
        time_stepping_method = next_timeslice_from_BS_implicit_Euler
    elif method == "Crank-Nicholson" :
        time_stepping_method = next_timeslice_from_BS_Crank_Nicholson
    else :
        print(f"get_history_with_method: No such solution method {method}")
        raise NotImplementedError

    priming_steps = 100
    priming_dt = timestep/priming_steps
    i = 0
    # prime the long-time evolution using smaller time-steps to get away from the "sharp" initial conditions
    while i < priming_steps:
        new_slice = time_stepping_method(primed_history[i],market=market,option=option,dt=priming_dt)
        primed_history.append(new_slice)
        valid = new_slice.validate_monotonic_increasing()
        if not valid:
            print(f"Halving priming time step from {priming_dt} to {priming_dt/2}")
            priming_dt /= 2
            priming_steps *= 2
            primed_history = [init_slice]
            i = -1
        i += 1

    time_left = option.T - timestep
    timestep = time_left/num_timesteps
    history = primed_history.copy()
    j = 0
    while j < num_timesteps:
        i = j + priming_steps
        new_slice = time_stepping_method(history[i],market=market,option=option,dt=timestep)
        history.append(new_slice)
        valid = new_slice.validate_monotonic_increasing()
        if not valid:
            print(f"At time {new_slice.tau}, halving time step from {timestep} to {timestep/2}")
            timestep /= 2
            num_timesteps *= 2
            history = primed_history.copy()
            j = -1
        j += 1

    print(history[-1].tau)
    return history


"""
Main
"""


def solve_black_scholes(market: Market, option: Option, num_timesteps: int) -> float:
    # take current price S0, time-to-expiry T, strike price K, risk-free interest rate r, and volatility sigma
    # and return the price C of a European call. American call works the same, since it is never optimal to
    # exercise a call before expiry
    """

    V(S(t),t) == price of an option
    t == time from when option is sold
    S(t) == price of the underlying asset

    T == time the option expires
    S0 = S(t=0)

    r = risk-free interest rate
    mu = assumed growth of the underlying asset
    sigma = assumed volatility of the underlying asset

    logarithmic Wiener process (geometric Brownian motion
    d logS = mu dt + sigma dEta, where dEta^2 = dt
    then for some quantity V(S,t)
    dV = [ (dV/dS)dS + (d^2V/dS^2)dS^2/2 ] + (dV/dt)dt
       = [ (dV/dS)dS + (d^2V/dS^2)S^2(sigma^2 dt + ...)/2] + (dV/dt)dt
       = dt[ sigma^2 S^2 (d^2V/dS^2)/2 + (dV/dt) ] + (dV/dS) dS

    define H = S dV/dS - V        ( Legendre transform of V(S) )
    then dH = r H dt               if H is a risk-free arbitrage-free portfolio
    but also
    dH = (dV/dS) dS - dV = -dt[ sigma^2 S^2 (d^2V/dS^2)/2 + (dV/dt)]
    so altogether
    sigma^2 S^2 (d^2V / dS^2)/2 + (dV/dt) = -rH = -r[ S(dV/dS) - V ]
    is the Black-Scholes equation

    At time T of expiry, a call at strike-price K has value V(S,T)=(S-K)theta(S>K)
    In transformed coordinates, with tau := T-t, this IC is V(S,tau=0) = (S-K)theta(S-K)  [ramp function]
    and the Black-Scholes equation becomes
    (dV/dTau) = sigma^2 S^2 (d^2V/dS^2)/2 + r S (dV/dS) - rV
    ^^^ This provides a time-step condition to evolve the boundary condition away from its endpoint in time

    """

    # densely pack the grid near the strike price
    grid = dense_grid_around(K=option.K, num_core_points=200, core_size_in_tenths=4)
    # grid = geom_grid_everywhere(K=option.K, num_points_above=100)
    init_profile = [init_cond_call(Si,option.K) for Si in grid]
    init_slice = TimeSlice(tau=0,positions_list=grid,values_list=init_profile)

    # simulated history
    start = perf_counter()
    history_EE = get_history_with_method("Explicit Euler",market=market,option=option,init_slice=init_slice,
                                         num_timesteps=num_timesteps)

    end = perf_counter()
    print(f"Explicit Euler timing: {end - start}")

    start = perf_counter()
    history_RK4 = get_history_with_method("Explicit RK4",market=market,option=option,init_slice=init_slice,
                                          num_timesteps=num_timesteps)

    end = perf_counter()
    print(f"Explicit RK4 timing: {end - start}")

    start = perf_counter()
    history_IE = get_history_with_method("Implicit Euler",market=market,option=option,init_slice=init_slice,
                                         num_timesteps=num_timesteps)

    end = perf_counter()
    print(f"Implicit Euler timing: {end - start}")

    start = perf_counter()
    history_CN = get_history_with_method("Crank-Nicholson",market=market,option=option,init_slice=init_slice,
                                         num_timesteps=num_timesteps)

    end = perf_counter()
    print(f"Crank-Nicholson timing: {end-start}")

    # exact result at t=0 (tau = T)
    true_prices_on_grid = [ true_black_scholes(price=Si, market=market, option=option)
                            for Si in history_EE[0].positions ]


    """
    Plots
    """


    # overlayed explicit Euler
    plt.plot( history_EE[-1].positions, true_prices_on_grid)
    plt.plot( history_EE[-1].positions, history_EE[-1].values)
    plt.ylim(0, 500)
    plt.show()

    # %diff explicit Euler
    percentdiff = [ 100*(history_EE[-1].value_n(idx) - true_prices_on_grid[idx])/true_prices_on_grid[idx]
                    if true_prices_on_grid[idx] != 0. else 0 for idx, _ in enumerate(true_prices_on_grid) ]
    absdiff = [ history_EE[-1].value_n(idx) - true_prices_on_grid[idx] for idx, _ in enumerate(true_prices_on_grid) ]
    plt.scatter( history_EE[-1].positions, percentdiff)

    plt.ylim(-10,10)
    plt.show()

    # abs diff forward Euler
    plt.scatter( history_EE[-1].positions, absdiff)
    max_mag = 10**ceil( max( np.log10( np.abs(absdiff) + 1e-6 ) ) )
    plt.ylim(-max_mag, max_mag)
    plt.show()

    """"""

    # overlayed RK4
    plt.plot( history_RK4[-1].positions, true_prices_on_grid)
    plt.plot( history_RK4[-1].positions, history_RK4[-1].values)
    plt.ylim(0, 500)
    plt.show()

    # %diff explicit RK4
    percentdiff = [ 100*(history_RK4[-1].value_n(idx) - true_prices_on_grid[idx])/true_prices_on_grid[idx]
                    if true_prices_on_grid[idx] != 0. else 0 for idx, _ in enumerate(true_prices_on_grid) ]
    absdiff = [ history_RK4[-1].value_n(idx) - true_prices_on_grid[idx] for idx, _ in enumerate(true_prices_on_grid) ]
    plt.scatter( history_RK4[-1].positions, percentdiff)
    plt.ylim(-10,10)
    plt.show()

    # abs diff RK4
    plt.scatter( history_RK4[-1].positions, absdiff)
    max_mag = 10**ceil( max( np.log10( np.abs(absdiff) + 1e-6 ) ) )
    plt.ylim(-max_mag, max_mag)
    plt.show()

    """ """

    # overlayed implicit Euler
    plt.plot( history_IE[-1].positions, true_prices_on_grid)
    plt.plot( history_IE[-1].positions, history_IE[-1].values)
    plt.ylim(0, 500)
    plt.show()

    # %diff implicit Euler
    percentdiff = [ 100*(history_IE[-1].value_n(idx) - true_prices_on_grid[idx])/true_prices_on_grid[idx]
                    if true_prices_on_grid[idx] != 0. else 0 for idx, _ in enumerate(true_prices_on_grid) ]
    absdiff = [ history_IE[-1].value_n(idx) - true_prices_on_grid[idx] for idx, _ in enumerate(true_prices_on_grid) ]
    plt.scatter( history_IE[-1].positions, percentdiff)
    plt.ylim(-10,10)
    plt.show()

    # abs diff implicit Euler
    plt.scatter( history_IE[-1].positions, absdiff)
    max_mag = 10**ceil( max( np.log10( np.abs(absdiff) + 1e-6 ) ) )
    plt.ylim(-max_mag, max_mag)
    plt.show()

    """ """

    # overlayed Crank-Nicholson
    plt.plot(history_CN[-1].positions, true_prices_on_grid)
    plt.plot(history_CN[-1].positions, history_CN[-1].values)
    plt.ylim(0, 500)
    plt.show()

    # %diff Crank-Nicholson
    percentdiff = [100 * (history_CN[-1].value_n(idx) - true_prices_on_grid[idx]) / true_prices_on_grid[idx]
                   if true_prices_on_grid[idx] != 0. else 0 for idx, _ in enumerate(true_prices_on_grid)]
    absdiff = [history_CN[-1].value_n(idx) - true_prices_on_grid[idx] for idx, _ in enumerate(true_prices_on_grid)]
    plt.scatter(history_CN[-1].positions, percentdiff)
    plt.ylim(-10, 10)
    plt.show()

    # abs diff Crank-Nicholson
    plt.scatter(history_CN[-1].positions, absdiff)
    max_mag = 10**ceil( max( np.log10( np.abs(absdiff) + 1e-6 ) ) )
    plt.ylim(-max_mag, max_mag)
    plt.show()

    print("\n Slices at endpoint:")
    print(history_RK4[-1].positions[20:30])
    print(history_EE[-1].values[20:30])
    print(history_RK4[-1].values[20:30])
    print(history_IE[-1].values[20:30])
    print(history_CN[-1].values[20:30])
    print(true_prices_on_grid[20:30])

    return history_RK4[-1].get_value_at_position(option.underlying.S0)





if __name__ == "__main__" :

    np.seterr(all="raise")

    # Asset and derivative info
    tsx = Market(name="TSX", risk_free_interest_rate=0.05)
    zdv = Asset(name="ZDV", curr_price=51, curr_volatility=0.1)
    zdv_call_50 = Option(name="ZDV:50", underlying=zdv, strike_price=50, time_to_expiry=2.0)

    # sim info
    timesteps = 200

    # get the price of a call
    call_price = solve_black_scholes(market = tsx,
                                     option = zdv_call_50,
                                     num_timesteps=timesteps)
    print(f"\nEnded with {call_price=}")
    print( true_black_scholes(price=zdv.S0,market=tsx,option=zdv_call_50) )

    raise SystemExit
