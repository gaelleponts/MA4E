import datetime
from microgrid.environments.industrial.industrial_env import IndustrialEnv


class IndustrialAgent:
    def __init__(self, env: IndustrialEnv):
        self.env = env

        def take_decision(self,
                      state,
                      previous_state=None,
                      previous_action=None,
                      previous_reward=None):

        #données utiles
        delta_t = datetime.timedelta(minutes=30)  #pas de temps
        T=self.env.nb_pdt #cf industrial_env.py  nb de periode temporelles
        capacity= self.env.battery.capacity  #cf battery.py charge max
        pmax=self.env.battery.pmax #cf battery.py puissance max
        efficiency=self.env.battery.efficiency #cf battery.py rendement de la batterie

        manager_signal= state.get("manager_signal") #prix  cf manager.py l95
        consumption_prevision = state.get("consumption_prevision")  # la demande de consommation
        H = datetime.timedelta(hours=1)

        #problème:
        pb=pulp.LpProblem("Site industriel", pulp.LpMinimize)

        #variabless
        l_bat_pos = pulp.LpVariable.dicts("l_bat_pos", [t for t in range(T)], 0)  #l_bat+
        l_bat_neg = pulp.LpVariable.dicts("l_bat_neg", [t for t in range(T)], 0)  #l_bat-
        l_bat = pulp.LpVariable.dicts("l_bat", [t for t in range(T)])
        li = pulp.LpVariable.dicts("li", [t for t in range(T)])  #demande totale du site industriel
        a = pulp.LpVariable.dicts("tock_batterie", [t for t in range(T)], 0, capacity) #cf formulation mathématique  0 <= a <= C

        #fonction objectifll

        pb += pulp.lpSum([li[t] * manager_signal[t] * delta_t/H  for t in range(T)])


        #contraintes
        pb += a[0]==0
        pb += pulp.lpSum([l_bat_pos[t] for t in range(T)]) <= pmax
        pb += pulp.lpSum([l_bat_neg[t] for t in range(T)]) <= pmax
        pb += pulp.lpSum([li[t] - consumption_prevision[t] - l_bat[t] for t in range(T)])== 0  #li(t)=ldem(t)+lbat(t) cf formulation math
        pb += pulp.lpSum([a[t]-a[t-1]- (efficiency*l_bat_pos[t] - l_bat_neg[t]*1/efficiency)*delta_t/H for t in range(1,T)])==0  #formule de recurrence a(t)


        #Résolution
        pb.solve()
        resultat = self.env.action_space.sample()
        for t in range(T):
            resultat[t] = li[t].value()

        return resultat

if __name__ == "__main__":
    delta_t = datetime.timedelta(minutes=30)
    time_horizon = datetime.timedelta(days=1)
    N = time_horizon // delta_t
    industrial_config = {
        'battery': {
            'capacity': 100,
            'efficiency': 0.95,
            'pmax': 25,
        },
        'building': {
            'site': 1,
        }
    }
    env = IndustrialEnv(industrial_config=industrial_config, nb_pdt=N)
    agent = IndustrialAgent(env)
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
