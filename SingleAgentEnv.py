import numpy as np


class SingleAgentEnv:
    def state_id(self) -> int:
        pass

    def is_game_over(self) -> bool:
        pass

    def act_with_action(self,action_id):
        pass

    def score(self) -> float:
        pass

    def available_actions_id(self) ->np.ndarray:
        pass

    def reset(self):
        pass

