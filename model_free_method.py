from dataclasses import dataclass
import numpy as np
from typing import Dict

import SingleAgentEnv
from line_world_single_agent_env import LineWorld


@dataclass
class PolicyAndActionValueFunction:
    pi: Dict[int, Dict[int, float]]
    q: Dict[int, Dict[int, float]]


def on_policy_first_visit_monte_carlo_control(env: SingleAgentEnv, epsilon: float, gamma: float,
                                              max_iter: int) -> PolicyAndActionValueFunction:
    pi = {}
    q = {}
    returns = {}

    for it in range(max_iter):
        env.reset()
        S = []
        A = []
        R = []
        while not env.is_game_over():
            s = env.state_id()
            S.append(s)
            available_actions = env.available_actions_id()
            if s not in pi:
                pi[s] = {}
                q[s] = {}
                returns[s] = {}
                for a in available_actions:
                    pi[s][a] = 1.0 / len(available_actions)
                    q[s][a] = 0.0
                    returns[s][a] = []

            chosen_action = np.random.choice(list(pi[s].keys()), 1, False, list(pi[s].values()))[0]
            A.append(chosen_action)
            old_score = env.score()
            env.act_with_action(chosen_action)
            reward = env.score() - old_score
            R.append(reward)

            G = 0

            for t in reversed(range(len(S))):
                G = gamma * G + R[t]
                s_t = S[t]
                a_t = A[t]
                found = False
                for p_s, p_a in zip(S[:t], A[:t]):
                    if s_t == p_s and a_t == p_a:
                        found = True
                        break
                if found:
                    continue
                returns[s_t][a_t].append(G)
                q[s_t][a_t] = np.mean(returns[s_t][a_t])
                optimal_a_t = list(q[s_t].keys())[np.argmax(list(q[s_t].values()))]
                available_actions_t_count = len(q[s_t])
                for a_key, q_s_a in q[s_t].items():
                    if a_key == optimal_a_t:
                        pi[s_t][a_key] = 1 - epsilon + epsilon / available_actions_t_count
                    else:
                        pi[s_t][a_key] = epsilon / available_actions_t_count

    return PolicyAndActionValueFunction(pi, q)


def main():
    env = LineWorld(5)
    rslt = on_policy_first_visit_monte_carlo_control(env, 0.1, 0.9999, 1000)
    print(rslt.pi)
    print(rslt.q)

    return


if __name__ == "__main__":
    main()
