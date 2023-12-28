from gym.spaces import Box, Dict, Discrete
import torch
import numpy as np
from habitat_baselines.rl.ddppo.policy import PointNavResNetPolicy
import habitat
from habitat_baselines.utils.common import batch_obs
from habitat.sims.habitat_simulator.actions import HabitatSimActions
import cv2
import random
import sys
from lavis.models import load_model_and_preprocess
from lavis.processors import load_processor
from PIL import Image

def get_action(action_id):
    action = np.zeros((10,), dtype=np.float32)
    if (action_id) == 1:
        action[8] = 20
    elif (action_id) == 2:
        action[9] = 20
    elif (action_id) == 3:
        action[9] = -20
    return action

class MyPolicy:
    def __init__(self):
        random.seed(10)
        torch.random.manual_seed(10)
        self.actor_critic = PointNavResNetPolicy(
            observation_space = Dict({
                "depth": Box(0.0, 1.0, (224, 224, 1), np.float32),
                "pointgoal_with_gps_compass": Box(np.finfo(np.float32).min, np.finfo(np.float32).max, (2,), np.float32)
            }),
            action_space = Discrete(4),
            hidden_size = 512,
            num_recurrent_layers = 2,
            rnn_type = "LSTM",
            resnet_baseplanes = 32,
            backbone = "resnet18",
            normalize_visual_inputs=False,
            force_blind_policy = False,
            policy_config = None,
            aux_loss_config = {},
            fuse_keys = None
        )
        self.device = torch.device("cpu")
        self.actor_critic.to(self.device)
        model = torch.load("./examples/pointnav_weights.pth", map_location = self.device)
        self.actor_critic.load_state_dict(model)
        self.test_recurrent_hidden_states = torch.zeros(
            1,
            self.actor_critic.net.num_recurrent_layers,
            512,
            device=self.device,
        )
        self.not_done_masks = torch.zeros(
            1, 1, device=self.device, dtype=torch.bool
        )
        self.prev_actions = torch.zeros(
            1, 1, dtype=torch.long, device=self.device
        )
        # self.blip_model, self.vis_processors, self.text_processors = load_model_and_preprocess(
        #     "blip2_image_text_matching",
        #     "pretrain",
        #     device=self.device,
        #     is_eval=True
        # )

    def act(self, obs):
        batch = batch_obs([{
            "pointgoal_with_gps_compass": obs["pointgoal_with_gps_compass"],
            "depth": obs["depth"],
        }], device=self.device)
        # image = self.vis_processors["eval"](
        #     Image.fromarray(obs['head_rgb'])
        # ).unsqueeze(0).to(self.device)
        # txt = self.text_processors["eval"]("banana")
        # itc_score = self.blip_model({"image": image, "text_input": txt}, match_head='itc')
        # print('The image feature and text feature has a cosine similarity of %.4f'%itc_score)
        with torch.no_grad():
            action_data = self.actor_critic.act(
                batch,
                self.test_recurrent_hidden_states,
                self.prev_actions,
                self.not_done_masks,
                deterministic=True
            )
            self.test_recurrent_hidden_states = action_data.rnn_hidden_states
            self.not_done_masks.fill_(True)
            self.prev_actions.copy_(action_data.actions)
        action_id = action_data.env_actions[0][0].item()
        # num_times = [1, 10, 16, 16][action_id]
        num_times = 1
        return get_action(action_id), num_times

def preprocess(obs):
    obs['depth'] = cv2.resize(obs['head_depth'], dsize=(224, 224))
    obs['depth'] = np.expand_dims(obs['depth'], axis=-1)
    obs['pointgoal_with_gps_compass'] = obs['obj_start_sensor'][3:]
    # obs['pointgoal_with_gps_compass'][1] *= -1

if __name__ == "__main__":
    policy = MyPolicy()
    print("Policy loaded")

    cfg = habitat.get_config("benchmark/rearrange/pick.yaml")
    env = habitat.gym.make_gym_from_config(cfg)
    print("Environment loaded")

    obs = env.reset()
    terminal = False
    i = 0
    while not terminal:
        cv2.imwrite("./depth_imgs/env_depth"+str(i)+".png", obs['head_depth']*255)
        cv2.imwrite("./rgb_imgs/env_rgb"+str(i)+".png", obs['head_rgb'])
        preprocess(obs)
        if len(sys.argv) >=2 and sys.argv[1] == "manual":
            action_id, num_times = list(map(int, input("Enter action: ").split()))
            action = get_action(action_id)
        else:
            action, num_times = policy.act(obs)
        for _ in range(num_times):
            obs, reward, terminal, info = env.step(action)
            preprocess(obs)
            # unused_action = policy.act(obs)
        # obs, reward, terminal, info = env.step(action)
        print(obs['obj_start_sensor'])
        i+=num_times
        print("Finished step", i)
        print("Terminal:", terminal)
    env.close()
