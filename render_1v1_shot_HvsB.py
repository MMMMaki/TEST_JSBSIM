import numpy as np
import torch
from envs.JSBSim.envs import SingleCombatEnv
from algorithms.ppo.ppo_actor import PPOActor
import logging
logging.basicConfig(level=logging.DEBUG)

class Args:
    def __init__(self) -> None:
        self.gain = 0.01
        self.hidden_size = '128 128'
        self.act_hidden_size = '128 128'
        self.activation_id = 1
        self.use_feature_normalization = False
        self.use_recurrent_policy = True
        self.recurrent_hidden_size = 128
        self.recurrent_hidden_layers = 1
        self.tpdv = dict(dtype=torch.float32, device=torch.device('cpu'))
        self.use_prior = True
    
def _t2n(x):
    return x.detach().cpu().numpy()

render = True
ego_policy_index = "latest"
experiment_name = "1v1_Shot_HvsB_latest"

# 你的模型路径保持不变
ego_run_dir = r"D:\JSBSIM\LAG-master\LAG-master\scripts\results\SingleCombat\1v1\ShootMissile\HierarchyVsBaseline\ppo\v4\wandb\run-20260413_191450-gxn17uwp\files"

args = Args()
device = torch.device("cpu")

env = SingleCombatEnv("1v1/ShootMissile/HierarchyVsBaseline")

# 加载蓝方 RL 模型
ego_policy = PPOActor(args, env.observation_space, env.action_space, device)
ego_policy.eval()
ego_policy.load_state_dict(torch.load(ego_run_dir + f"/actor_{ego_policy_index}.pt"))

print("Start render")
obs = env.reset()

ego_rnn_states = np.zeros((1, 1, 128), dtype=np.float32)
masks = np.ones((1, 1))

step_counter = 0
while True:
    if render:
        env.render(mode='txt', filepath=f'{experiment_name}.txt.acmi')
    # 蓝方决策
    ego_actions, _, ego_rnn_states = ego_policy(obs, ego_rnn_states, masks, deterministic=True)
    ego_actions = _t2n(ego_actions)
    
    # 【核心】：老老实实只传 ego_actions (长度为 1 的数组)！红方会在底层被接管。
    obs,  rewards, dones, infos = env.step(ego_actions)
    
    step_counter += 1
    if np.any(dones):
        print(f"Episode finished at step {step_counter}!")
        break

# 保存录像
env.close()