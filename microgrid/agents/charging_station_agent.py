import datetime
from microgrid.environments.charging_station.charging_station_env import ChargingStationEnv
import pulp

class ChargingStationAgent:
    def __init__(self, env: ChargingStationEnv):
        self.env = env

    def take_decision(self,
                      state,
                      previous_state=None,
                      previous_action=None,
                      previous_reward=None):
        lp = pulp.LpProblem('ve', pulp.LpMinimize)
        lp.setSolver()

        nb_voitures = self.env.nb_evs
        nb_periodes = self.env.nb_pdt
        amende = 5

        lbd = state['manager_signal']

        batterie = {}
        charge = {}
        penalite = {}

        ##création des variables
        for t in range(nb_periodes + 1):
            batterie[t] = {}
            charge[t] = {}
            penalite[t] = {}
            for j in range(nb_voitures):
                var_name = 'batterie_' + str(t) + "_" + str(j)
                batterie[t][j] = pulp.LpVariable(var_name, 0.0, self.env.evs[j].battery.capacity)
                var_name = 'charge_' + str(t) + "_" + str(j)
                charge[t][j] = pulp.LpVariable(var_name, self.env.evs[j].battery.pmin, self.env.evs[j].battery.pmax)
                var_name = 'pénalité_' + str(t) + "_" + str(j)
                penalite[t][j] = pulp.LpVariable(var_name, cat='Binary')

        ##création des contraintes
        for j in range(nb_voitures):
            const_name = 'soc_'+str(j)
            lp += batterie[0][j] == state['soc'][j], const_name
            for t in range(1, nb_periodes + 1):
                lp += batterie[t][j] == state['is_plugged_prevision'][j][t-1] * charge[t][j] * self.env.evs[
                    j].battery.efficiency*delta_t/datetime.timedelta(hours=1) + batterie[t - 1][j]
                if state['is_plugged_prevision'][j][t-1] - state['is_plugged_prevision'][j][
                    t - 2] == 1:  # la voiture vient d'arriver
                    const_name = 'batterie_' + str(t) + '_' + str(j)
                    lp += batterie[t][j] == batterie[t-1][j] - 4, const_name  # pendant la journée le véhicule a perdu 4kwh
                elif state['is_plugged_prevision'][j][t-1] - state['is_plugged_prevision'][j][t - 2] == -1: #la voiture vient de partir
                    lp += batterie[t-1][j] >=4
                    const_name = 'pénalités_' + str(t) + '_' + str(j)
                    lp += batterie[t][j]>=0.25*self.env.evs[j].battery.capacity*(1-penalite[t][j]), const_name
        for t in range(1, nb_periodes + 1):
            const_name = 'charge_station' + '_' + str(t) + '_' + str(j)
            lp += pulp.lpSum(charge[t][j] for j in range(nb_voitures)) <= self.env.pmax_site, const_name  # contrainte de station de charge

        ## fonction objectif
        lp.setObjective(pulp.lpSum(charge[t + 1][j] * lbd[t] for j in range(4) for t in range(48)) +amende * pulp.lpSum(penalite[t][j] for t in range(1, 49) for j in range(4)))

        ##on resout
        lp.solve()

        return self.env.action_space.sample()


if __name__ == "__main__":
    delta_t = datetime.timedelta(minutes=30)
    time_horizon = datetime.timedelta(days=1)
    N = time_horizon // delta_t
    evs_config = [
        {
            'capacity': 40,
            'pmax': 22,
        },
        {
            'capacity': 40,
            'pmax': 22,
        },
        {
            'capacity': 40,
            'pmax': 3,
        },
        {
            'capacity': 40,
            'pmax': 3,
        },
    ]
    station_config = {
        'pmax': 40,
        'evs': evs_config
    }
    env = ChargingStationEnv(station_config=station_config, nb_pdt=N)
    agent = ChargingStationAgent(env)
    cumulative_reward = 0
    now = datetime.datetime.now()
    state = env.reset(now, delta_t)
    for i in range(N*2):
        action = agent.take_decision(state)
        state, reward, done, info = env.step(action)
        cumulative_reward += reward
        if done:
            break
        print(f"action: {action}, reward: {reward}, cumulative reward: {cumulative_reward}")
        print("State: {}".format(state))
        print("Info: {}".format(action.sum(axis=0)))