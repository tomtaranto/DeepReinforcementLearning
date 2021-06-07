from dataclasses import dataclass
import numpy as np
import SingleAgentEnv
@dataclass
class PolicyAndActionValueFunction:
    pi: dict[int, dict[int, float]]
    q: dict[int, dict[int,float]]


def on_policy_first_visit_monte_carlo(env: SingleAgentEnv,epsilon: float, gamma : float, max_iter: int) -> PolicyAndActionValueFunction:
    pi = {}
    q = {}
    returns = {}

    for it in range(max_iter):
        env.reset()
        S = []
        A= []
        R = []
        while not env.is_game_over():
            s = env.state_id()
            S.append(s)
            available_actions = env.available_actions_ids()
            if s not in pi:
                pi[s] = {}
                q[s] = {}
                for a in available_actions:
                    pi[s][a] = 1.0 / available_actions.len()
                    q[s][a] =  0.0

            chosen_action = np.random.choice(pi[s].keys(), 1, False, list(pi[s].values()))
            A.append(chosen_action)
            old_score = env.score()
            env.act_with_action_id(chosen_action)
            reward = env.score() - old_score
            R.append(reward)

            G = 0

            for t in reversed(range(len(S))):
                G = gamma * G + R[t]








    return



















def main():
    return


if __name__ == "__main__":
    main()