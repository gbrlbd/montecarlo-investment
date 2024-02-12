import matplotlib.pyplot as plt
import numpy as np
import statistics as st

def montecarlo_simulation(S0, pac, mu, sigma, n, m):
    # steps
    dt = 1

    # simulation with brownian motion
    St = np.exp(
    (mu - sigma ** 2 / 2) * dt
    + sigma * np.random.normal(0, np.sqrt(dt), size=(m,n)).T
    )

    # add initial value to the simulation
    St = np.vstack([np.full(m, S0), St])
    acc = [S0]

    # portfolio cumulative product and addition of the periodic capital
    for i in range(1, n+1):
        St[i] = St[i-1] * St[i] + pac
        acc.append(acc[i-1]+pac)

    # mean and percentiles portfolio values
    mean_port = np.mean(St, axis=1)
    percentiles_port = np.percentile(St, [0, 10, 25, 50, 75, 90, 100], axis=1).T

    print("The mean value of the simulation is: ", round(np.mean(St[-1])))
    print("The minimum value of the simulation is: ", round(min(St[-1])))
    print("The 10% percentile value of the simulation is: ", round(percentiles_port[-1][1]))
    print("The 25% percentile value of the simulation is: ", round(percentiles_port[-1][2]))
    print("The median value of the simulation is: ", round(np.median(St[-1])))
    print("The 75% percentile value of the simulation is: ", round(percentiles_port[-1][4]))
    print("The 90% percentile value of the simulation is: ", round(percentiles_port[-1][5]))
    print("The maximum value of the simulation is: ", round(max(St[-1])))

    # define time intervals
    time = np.linspace(0,n,n+1)
    acc = np.array(acc)

    tt = np.full(shape=(m,n+1), fill_value=time).T
    ttt = np.full(shape=(len(percentiles_port[0]),n+1), fill_value=time).T

    # plot of the simulation
    plt.plot(tt, St, mean_port, 'k-', linewidth=.5)
    plt.plot(tt, acc, 'r', label='Invested capital')
    plt.grid()
    plt.xlabel("Year")
    plt.ylabel("NW")
    plt.title(
        "Montecarlo Brownian Motion simulation\n $dS_t = \mu S_t dt + \sigma S_t dW_t$\n $Investment = {0}, \mu = {1}, \sigma = {2}, years = {3}$".format(S0, mu, sigma, n)
    )
    plt.show()

    # plot of the percentiles of the simulation
    plt.plot(ttt, percentiles_port, linewidth=1, label=['0%', '10%', '25%', '50%', '75%', '90%', '100%'])
    plt.grid()
    plt.plot(ttt, acc, 'r')
    plt.xlabel("Year")
    plt.ylabel("NW")
    plt.title(
        "Percentile for montecarlo simulation"
    )
    plt.legend()
    plt.show()

    # histogram of the final NW
    plt.hist(St[-1],bins=round(m/5),range=(0,max(St[-1])))
    plt.grid()
    plt.xlabel("NW")
    plt.ylabel("n of scenarios")
    plt.title(
        "Distribution of final NW"
    )
    plt.show()

# parameters
initial_investment = 10000
periodical_investment = 0
interest_rate = 0.05
volatility = 0.13
years = 15
simulations = 1000

montecarlo_simulation(initial_investment, 
                      periodical_investment, 
                      interest_rate, 
                      volatility,
                      years,
                      simulations)