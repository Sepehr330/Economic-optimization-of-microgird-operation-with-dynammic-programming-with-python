import numpy as np
import matplotlib.pyplot as plt

#delta_SOH
def delta_SOH(SOC_old, SOC_new, Z):
    delta_SOC = SOC_old - SOC_new
    if delta_SOC < 0:
        return Z * delta_SOC
    else:
        return 0

#The energy of ESS 
def Ess_Energy(SOC_old, SOC_new, eta_inv, eta_ch):
    delta_SOC = SOC_old - SOC_new

    #charge
    if delta_SOC < 0:
        return delta_SOC  / (eta_inv * eta_ch)
    
    #discharge
    elif delta_SOC >= 0:
        return delta_SOC * eta_inv 

#The cost of ESS
def Ess_cost(D_SOH, SOH_min , Bic):
    cost = Bic * (-D_SOH) / (1 - SOH_min)
    return cost

#The cost of PV
def PV_cost(pv_energy, cost_pv):
    cost = pv_energy * cost_pv
    return cost

#The energy of the grid
def grid_Energy(load_energy, Ess_energy, pv_energy):
    Ess = Ess_energy
    pv = pv_energy
    grid = -Ess - pv + load_energy
    return grid
    
#The cost of the grid
def grid_cost(grid_energy, feed_in_tariff, cost_grid):
    if grid_energy > 0:
        cost = grid_energy * cost_grid
    else:
        cost = grid_energy * feed_in_tariff
    return cost

#Cost function of the problem    
def cost_function(C_Ess, C_grid, C_PV):
    cost = C_Ess + C_grid + C_PV
    return cost

def optimal_policy(J_star , U_star , initial_soc , time_horizon , SOC_levels , pv_forecast , load_forecast , eta_inv , eta_ch):
    actions = []
    grids_opt = []
    Esses_opt = []
    index = np.argmin(np.abs(SOC_levels - initial_soc))
    state = SOC_levels[index]
    J_min = J_star[index, 0] #calculate J_min

    for t in range(time_horizon):
        PV = pv_forecast[t]
        load = load_forecast[t]
        Ess = Ess_Energy(SOC_old=state , SOC_new= state - U_star[index,t] , eta_inv=eta_inv , eta_ch=eta_ch)
        grid = grid_Energy(load , Ess , PV)
        actions.append(U_star[index , t])
        state -= U_star[index , t]
        index = np.argmin(np.abs(SOC_levels - state))

        grids_opt.append(grid)
        Esses_opt.append(Ess)

    return J_min , actions , grids_opt , Esses_opt

def normal_policy(pv_forecast , load_forecast , time_horizon , initial_soc , min_soc , max_soc , eta_inv , eta_ch , Z , ESS_capacity , cost_pv , cost_grid , SOH_min , feed_in_tariff , Bic):
    soc_now = initial_soc
    total_cost = 0
    soc_levels_normal_policy = []
    grids = []
    Esses = []
    for t in range(time_horizon):
        PV =pv_forecast[t]
        load = load_forecast[t]
        energy = load - PV
        Ess_charge_max = Ess_Energy(soc_now, max_soc, eta_inv,  eta_ch)
        Ess_decharge_max = Ess_Energy(soc_now, min_soc, eta_inv,  eta_ch)

        if energy < 0 :
            if energy < Ess_charge_max : #Ess would be completely charged
                Ess = Ess_charge_max
                grid = grid_Energy(load_energy = load_forecast[t], Ess_energy = Ess, pv_energy = PV)
                next_soc = max_soc

            else :
                Ess = energy
                delta_soc = Ess*eta_inv*eta_ch #soc_change
                next_soc = soc_now - delta_soc #soc_change in states
                grid = 0
        else :
            if energy > Ess_decharge_max : #Ess would be completely decharge but still not enough
                Ess = Ess_decharge_max
                grid = grid_Energy(load_energy = load_forecast[t], Ess_energy = Ess, pv_energy = PV)
                next_soc = min_soc

            else :
                Ess = energy
                delta_soc = Ess/eta_inv #soc_change
                next_soc = soc_now - delta_soc #soc_change in states
                grid = 0
        D_SOH = delta_SOH(SOC_old = soc_now/ESS_capacity, SOC_new = next_soc/ESS_capacity, Z = Z)
        c_Ess = Ess_cost(D_SOH, SOH_min , Bic)
        c_pv = PV_cost(PV, cost_pv)
        c_grid = grid_cost(grid, feed_in_tariff, cost_grid[t])
        cost = cost_function(C_Ess = c_Ess, C_grid = c_grid, C_PV = c_pv)

        total_cost += cost
        soc_levels_normal_policy.append(soc_now)
        soc_now = next_soc
        grids.append(grid)
        Esses.append(Ess)
    return total_cost , soc_levels_normal_policy , grids , Esses

# Constants
time_horizon = 24  # 24 hours
SOC_min = 0.4      # Minimum SOC (40%)
SOC_max = 0.9      # Maximum SOC (90%)
ESS_capacity = 30  # ESS capacity in kWh
eta_ch = 0.82      # Charging efficiency
eta_dis = 0.82     # Discharging efficiency
cost_grid_peak = 0.32   # Cost of grid electricity during peak hours (€/kWh)
cost_grid_off_peak = 0.2468  # Cost of grid electricity during off-peak hours (€/kWh)
cost_pv = 0.069    # Cost of PV production (€/kWh)
feed_in_tariff = 0.1085  # Feed-in tariff (€/kWh)
Bic = 130   # ESS investment cost (€/kWh)
PV_eff = 0.1   # PV efficiency
PV_S = 60  # PV surface (m^2)
Z = 2 * (10 ** -4)  # ESS aging coefficient
eta_inv = 0.9 # Inverter efficiency
SOH_min = 0.7 # Minimum ESS state of health (70%)

# Discretize SOC (X) states
SOC_levels = np.arange(SOC_min * ESS_capacity, SOC_max * ESS_capacity , .2)

# Discretize delta_SOC (U) inputs
delta_soc_levels = np.arange(SOC_min * ESS_capacity - SOC_max * ESS_capacity, SOC_max * ESS_capacity - SOC_min * ESS_capacity, .2)

#load and PV forecasts for 24 hours
load_forecast = np.array([0.75,0.75,0.75,0.75,0.75,0.75,0.75,0.75,1.75,1.75,1.5,1.5,1.5,1.5,1.5,1.5,1.2,1,0.75,1.5,1.5,1.5,0.75,0.75])
pv_forecast = np.array([0,0,0,0,0,0,0.5,1.5,3.5,4,5,5,6,5.25,4.5,3.75,2.5,1.5,0.5,0,0,0,0,0])

# Electricity cost profile
cost_grid = np.array([cost_grid_off_peak] * 6 + [cost_grid_peak] * 16 + [cost_grid_off_peak] * 2)

# Initialize J_star and U_star functions
J_star = np.zeros((len(SOC_levels), time_horizon))
U_star = np.zeros((len(SOC_levels), time_horizon))

# Dynamic programming backward iteration
for t in range(time_horizon - 1, -1, -1):
    for i, soc in enumerate(SOC_levels):
        min_cost = float('inf')
        best_action = 0

        # Charging constraints
        max_decharge = Ess_Energy(soc, SOC_min * ESS_capacity, eta_inv,  eta_ch) # > 0
        max_charge = Ess_Energy(soc, SOC_max * ESS_capacity, eta_inv,  eta_ch)  # < 0

        # Calculate max and min of delta_soc (our input value)
        delta_soc_min = max(-pv_forecast[t] * eta_inv * eta_ch, max_charge)   # charge , <0
        delta_soc_max = min(load_forecast[t] / eta_inv, max_decharge)   # discharge , >0
        
        # Rounding up delta_soc_min and delta_soc_max to our valid inputs (U)
        index_min = np.argmin(np.abs(delta_soc_levels - delta_soc_min))
        index_max = np.argmin(np.abs(delta_soc_levels - delta_soc_max))

        for delta_soc in delta_soc_levels[index_min:index_max+1]:

            # Determine the state and assign it to nearest the discrete SOC_levels
            next_soc = soc - delta_soc
            index = np.argmin(np.abs(SOC_levels - next_soc))
            next_soc = SOC_levels[index]

            # Calculate energies with our constraints
            PV = pv_forecast[t]
            Ess = Ess_Energy(soc, next_soc, eta_inv,  eta_ch)
            grid = grid_Energy(load_energy = load_forecast[t], Ess_energy = Ess, pv_energy = PV)

            # Calculate the cost
            D_SOH = delta_SOH(SOC_old = soc/ESS_capacity, SOC_new = next_soc/ESS_capacity, Z = Z)
            c_Ess = Ess_cost(D_SOH, SOH_min , Bic)
            c_pv = PV_cost(PV, cost_pv)
            c_grid = grid_cost(grid, feed_in_tariff, cost_grid[t])
            cost = cost_function(C_Ess = c_Ess, C_grid = c_grid, C_PV = c_pv)
            total_cost = cost + (J_star[index, t + 1] if t < time_horizon - 1 else 0)

            #chosing best action
            if total_cost < min_cost:
                min_cost = total_cost
                best_action = delta_soc


        U_star[i, t] = best_action
        J_star[i, t] = min_cost

initial_soc = 0.5*ESS_capacity
J_min , actions , grids_opt , Esses_opt = optimal_policy(J_star , U_star , initial_soc , time_horizon , SOC_levels , pv_forecast , load_forecast , eta_inv , eta_ch)
normal_cost , normal_policy_actions , grids , Esses= normal_policy(pv_forecast , load_forecast , time_horizon , initial_soc , 0.4*ESS_capacity , 0.9*ESS_capacity , eta_inv , eta_ch , Z , ESS_capacity , cost_pv , cost_grid , SOH_min , feed_in_tariff , Bic)
print("cost in optimal policy = " , J_min)
print("cost in normal policy = " , normal_cost)

#plot change of Ess levels for both optimal and normal policy
time = np.arange(0,24,1)
plt.figure(figsize=(10, 6))
plt.plot(time , - np.array(actions) + initial_soc, marker='o', label='Optimal Policy')
plt.plot(time , normal_policy_actions, marker='x', label='Normal Policy')
plt.title('ESS Levels Over Time')
plt.xlabel('Time (hours)')
plt.ylabel('ESS Levels (kWh)')
plt.legend()
plt.grid()
plt.show()

#plot energy in normal policy
plt.figure(figsize=(10, 6))
plt.plot(time , pv_forecast, marker='o', label='PV Forecast')
plt.plot(time , -load_forecast, marker='x', label='Load Forecast')
plt.plot(time , grids, marker='s', label='Grid Energy')
plt.plot(time , Esses, marker='d', label='ESS Energy')
plt.title('Energy in Normal Policy')
plt.xlabel('Time (hours)')
plt.ylabel('Energy (kWh)')
plt.legend()
plt.grid()
plt.show()

#plot energy in optimal policy
plt.figure(figsize=(10, 6))
plt.plot(time , pv_forecast, marker='o', label='PV Forecast')
plt.plot(time , -load_forecast, marker='x', label='Load Forecast')
plt.plot(time , grids_opt, marker='s', label='Grid Energy')
plt.plot(time , Esses_opt, marker='d', label='ESS Energy')
plt.title('Energy in Optimal Policy')
plt.xlabel('Time (hours)')
plt.ylabel('Energy (kWh)')
plt.legend()
plt.grid()
plt.show()