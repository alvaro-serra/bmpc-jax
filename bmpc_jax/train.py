import os
from collections import defaultdict
from functools import partial

import flax.linen as nn
import gymnasium as gym
import hydra
import jax
import jax.numpy as jnp
import numpy as np
import optax
import orbax.checkpoint as ocp
# Tensorboard: Prevent tf from allocating full GPU memory
import tensorflow as tf
import tqdm
from flax.metrics import tensorboard
from flax.training.train_state import TrainState

from bmpc_jax import BMPC, WorldModel
from bmpc_jax.common.activations import mish, simnorm
from bmpc_jax.data import SequentialReplayBuffer
from bmpc_jax.envs.dmcontrol import make_dmc_env
from bmpc_jax.envs.humanoid import make_humanoid_env
from bmpc_jax.networks import NormedLinear

import wandb

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
  tf.config.experimental.set_memory_growth(gpu, True)



@hydra.main(config_name='config', config_path='.', version_base=None)
def train(cfg: dict):
  env_config = cfg['env']
  encoder_config = cfg['encoder']
  model_config = cfg['world_model']
  tdmpc_config = cfg['tdmpc2']
  bmpc_config = cfg['bmpc']
  wandb_config = cfg['wandb']

  ##############################
  # Logger setup
  ##############################
  output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
  if wandb_config.log_tensorboard:
    writer = tensorboard.SummaryWriter(os.path.join(output_dir, 'tensorboard'))
    writer.hparams(cfg)
  # Wandb TODO it acts as writer, use it parallely at the same time that we would be logging things with writer
  if wandb_config.log_wandb:
      wandb.init(
          project=wandb_config.project,
          notes="baseline",
          tags=["PPO", "MPPI"],
          config=dict(cfg),
          name=None if wandb_config.name=="unflagged" else wandb_config.name,
          # mode=config["WANDB_MODE"],
      )

  ##############################
  # Environment setup
  ##############################
  def make_env(env_config, seed):
    def make_gym_env(env_id, seed):
      env = gym.make(env_id)
      env = gym.wrappers.RescaleAction(env, min_action=-1, max_action=1)
      env = gym.wrappers.RecordEpisodeStatistics(env)
      env = gym.wrappers.Autoreset(env)
      env.action_space.seed(seed)
      env.observation_space.seed(seed)
      return env

    if env_config.backend == "gymnasium":
      return make_gym_env(env_config.env_id, seed)
    elif env_config.backend == "dmc":
      env = make_dmc_env(env_config.env_id, seed, env_config.dmc.obs_type)
      env = gym.wrappers.RecordEpisodeStatistics(env)
      env = gym.wrappers.AutoResetWrapper(env)
      env.action_space.seed(seed)
      env.observation_space.seed(seed)
      return env
    elif env_config.backend == "humanoid":
      _env = make_humanoid_env(env_config.env_id,cfg)
      env = gym.wrappers.RecordEpisodeStatistics(_env)
      env = gym.wrappers.AutoResetWrapper(env)
      return env
    else:
      raise ValueError("Environment not supported:", env_config)

  if env_config.asynchronous:
    vector_env_cls = gym.vector.AsyncVectorEnv
  else:
    vector_env_cls = gym.vector.SyncVectorEnv
  env = vector_env_cls(
      [
          partial(make_env, env_config, seed)
          for seed in range(cfg.seed, cfg.seed+env_config.num_envs)
      ]
  )
  np.random.seed(cfg.seed)
  rng = jax.random.PRNGKey(cfg.seed)

  # SETUP VECTORIZED EVALUATE ENVIRONMENTS # TODO
  if env_config.eval_env:
    eval_env = vector_env_cls(
      [
        partial(make_env, env_config, seed)
        for seed in range(cfg.seed, cfg.seed + env_config.num_envs)
      ])
    eval_interval_steps = cfg["max_steps"] // env_config.eval_points

  ##############################
  # Agent setup
  ##############################
  dtype = jnp.dtype(model_config.dtype)
  rng, model_key, encoder_key = jax.random.split(rng, 3)
  encoder_module = nn.Sequential(
      [
          NormedLinear(
              encoder_config.encoder_dim, activation=mish, dtype=dtype
          )
          for _ in range(encoder_config.num_encoder_layers-1)
      ] + [
          NormedLinear(model_config.latent_dim, activation=None, dtype=dtype)
      ]
  )

  if encoder_config.tabulate:
    print("Encoder")
    print("--------------")
    print(
        encoder_module.tabulate(
            jax.random.key(0),
            env.observation_space.sample(),
            compute_flops=True
        )
    )

  ##############################
  # Replay buffer setup
  ##############################
  dummy_obs, _ = env.reset()
  dummy_action = env.action_space.sample()
  dummy_next_obs, dummy_reward, dummy_term, dummy_trunc, _ = env.step(
      dummy_action
  )
  replay_buffer = SequentialReplayBuffer(
      capacity=cfg.buffer_size,
      num_envs=env_config.num_envs,
      seed=cfg.seed,
      dummy_input=dict(
          observation=dummy_obs,
          action=dummy_action,
          reward=dummy_reward,
          next_observation=dummy_next_obs,
          terminated=dummy_term,
          truncated=dummy_trunc,
          expert_mean=np.zeros_like(dummy_action),
          expert_std=np.ones_like(dummy_action),
      )
  )

  encoder = TrainState.create(
      apply_fn=encoder_module.apply,
      params=encoder_module.init(encoder_key, dummy_obs)['params'],
      tx=optax.chain(
          optax.zero_nans(),
          optax.clip_by_global_norm(model_config.max_grad_norm),
          optax.adamw(encoder_config.learning_rate),
      )
  )

  model = WorldModel.create(
      action_dim=np.prod(env.single_action_space.shape),
      encoder=encoder,
      **model_config,
      key=model_key
  )
  if model.action_dim >= 20:
    tdmpc_config.mppi_iterations += 2

  agent = BMPC.create(world_model=model, **tdmpc_config)
  global_step = 0

  options = ocp.CheckpointManagerOptions(
      max_to_keep=1, save_interval_steps=cfg['save_interval_steps']
  )
  checkpoint_path = os.path.join(output_dir, 'checkpoint')
  with ocp.CheckpointManager(
      checkpoint_path,
      options=options,
      item_names=('agent', 'global_step', 'buffer_state')
  ) as mngr:
    if mngr.latest_step() is not None:
      print('Checkpoint folder found, restoring from', mngr.latest_step())
      abstract_buffer_state = jax.tree.map(
          ocp.utils.to_shape_dtype_struct, replay_buffer.get_state()
      )
      restored = mngr.restore(
          mngr.latest_step(),
          args=ocp.args.Composite(
              agent=ocp.args.StandardRestore(agent),
              global_step=ocp.args.JsonRestore(),
              buffer_state=ocp.args.StandardRestore(abstract_buffer_state),
          )
      )
      agent, global_step = restored.agent, restored.global_step
      replay_buffer.restore(restored.buffer_state)
    else:
      print('No checkpoint folder found, starting from scratch')
      mngr.save(
          global_step,
          args=ocp.args.Composite(
              agent=ocp.args.StandardSave(agent),
              global_step=ocp.args.JsonSave(global_step),
              buffer_state=ocp.args.StandardSave(replay_buffer.get_state()),
          ),
      )
      mngr.wait_until_finished()

    ##############################
    # Training loop
    ##############################
    ep_count = np.zeros(env_config.num_envs, dtype=int)
    eval_ep_count = np.zeros(env_config.num_envs, dtype=int)
    prev_logged_step = global_step
    prev_eval_step = global_step
    observation, _ = env.reset(seed=cfg.seed)

    T = 500
    seed_steps = int(
        max(5*T, 1000) * env_config.num_envs * env_config.utd_ratio
    )
    total_num_updates = 0
    total_reanalyze_steps = 0
    pbar = tqdm.tqdm(initial=global_step, total=cfg.max_steps)
    done = np.zeros(env_config.num_envs, dtype=bool)
    plan = None
    for global_step in range(global_step, cfg.max_steps, env_config.num_envs):
      if global_step <= seed_steps:
        action = env.action_space.sample()
        expert_mean, expert_std = np.zeros_like(action), np.ones_like(action)
      else:
        rng, action_key = jax.random.split(rng)
        action, plan = agent.act(
            obs=observation,
            prev_plan=plan,
            deterministic=False,
            train=True,
            key=action_key
        )
        expert_mean, expert_std = plan[2][..., 0, :], plan[3][..., 0, :]
        # if log_this_step:
        #   writer.scalar('train/plan_mean', np.mean(plan[0]), global_step)
        #   writer.scalar('train/plan_std', np.mean(plan[1]), global_step)

      next_observation, reward, terminated, truncated, info = env.step(action)

      if np.any(~done):
        replay_buffer.insert(
            dict(
                observation=observation,
                action=action,
                reward=reward,
                next_observation=next_observation,
                terminated=terminated,
                truncated=truncated,
                expert_mean=expert_mean,
                expert_std=expert_std,
            ),
            mask=~done
        )
      observation = next_observation

      # Handle terminations/truncations
      done = np.logical_or(terminated, truncated)
      if np.any(done):
        if plan is not None:
          plan = (
              plan[0].at[done].set(0),
              plan[1].at[done].set(agent.max_plan_std)
          )
        for ienv in range(env_config.num_envs):
          if done[ienv]:
            if wandb_config.log_tensorboard:
              r = info["final_info"][ienv]['final_info']['episode']['r']
              l = info["final_info"][ienv]['final_info']['episode']['l']
              print(
                  f"Episode {ep_count[ienv]}: r = {r:.2f}, l = {l}"
              )
              writer.scalar(f'episode/return', r, global_step + ienv)
              writer.scalar(f'episode/length', l, global_step + ienv)
            ep_count[ienv] += 1
            wandb_log_dict = {"train/env_step": global_step + ienv,
                              "train/episode_return": info["final_info"][ienv]['final_info']['episode']['r'],
                              "train/episode_length": info["final_info"][ienv]['final_info']['episode']['l']
                              }
            if wandb_config.log_wandb: wandb.log(wandb_log_dict)

      if global_step >= seed_steps:
        if global_step == seed_steps:
          print('Pre-training on seed data...')
          num_updates = seed_steps
          pretrain = True
        else:
          num_updates = max(1, int(env_config.num_envs * env_config.utd_ratio))
          pretrain = False

        rng, *update_keys = jax.random.split(rng, num_updates+1)
        log_this_step = global_step >= prev_logged_step + \
            cfg['log_interval_steps']
        if log_this_step:
          all_train_info = defaultdict(list)
          prev_logged_step = global_step
          wandb_log_dict_train = {}

        for iupdate in range(num_updates):
          batch, batch_inds = replay_buffer.sample(
              agent.batch_size, agent.horizon, return_inds=True
          )
          agent, train_info = agent.update_world_model(
              observations=batch['observation'],
              actions=batch['action'],
              rewards=batch['reward'],
              next_observations=batch['next_observation'],
              terminated=batch['terminated'],
              truncated=batch['truncated'],
              key=update_keys[iupdate]
          )
          total_num_updates += 1

          # Reanalyze
          encoder_zs = train_info.pop('encoder_zs')
          latent_zs = train_info.pop('latent_zs')
          finished = train_info.pop('finished')
          if total_num_updates % bmpc_config.reanalyze_interval == 0:
            total_reanalyze_steps += 1
            rng, reanalyze_key = jax.random.split(rng)
            b = bmpc_config.reanalyze_batch_size
            h = bmpc_config.reanalyze_horizon
            _, reanalyzed_plan = agent.plan(
                z=encoder_zs[:, :b, :],
                horizon=h,
                key=reanalyze_key
            )
            reanalyze_mean = reanalyzed_plan[2][..., 0, :]
            reanalyze_std = reanalyzed_plan[3][..., 0, :]
            # Update expert policy in buffer
            # Reshape for buffer: (T, B, A) -> (B, T, A)
            env_inds = batch_inds[0][:b, None]
            seq_inds = batch_inds[1][:b]
            replay_buffer.data['expert_mean'][seq_inds, env_inds] = \
                np.swapaxes(reanalyze_mean, 0, 1)
            replay_buffer.data['expert_std'][seq_inds, env_inds] = \
                np.swapaxes(reanalyze_std, 0, 1)
            batch['expert_mean'][:, :b, :] = reanalyze_mean
            batch['expert_std'][:, :b, :] = reanalyze_std

          # Update policy with reanalyzed samples
          if not pretrain:
            rng, policy_key = jax.random.split(rng)
            agent, policy_info = agent.update_policy(
                zs=latent_zs,
                expert_mean=batch['expert_mean'],
                expert_std=batch['expert_std'],
                finished=finished,
                key=policy_key
            )
            train_info.update(policy_info)

          if log_this_step:
            for k, v in train_info.items():
              all_train_info[k].append(np.array(v))

        if log_this_step:
          for k, v in all_train_info.items():
            if wandb_config.log_tensorboard:
              writer.scalar(f'train/{k}_mean', np.mean(v), global_step)
              writer.scalar(f'train/{k}_std', np.std(v), global_step)
            wandb_log_dict_train[f'updates/{k}_mean'] = np.mean(v)
            wandb_log_dict_train[f'updates/{k}_std'] = np.std(v)
            wandb_log_dict_train[f'updates/global_step'] = global_step
          if wandb_config.log_wandb: wandb.log(wandb_log_dict_train)


        mngr.save(
            global_step,
            args=ocp.args.Composite(
                agent=ocp.args.StandardSave(agent),
                global_step=ocp.args.JsonSave(global_step),
                buffer_state=ocp.args.StandardSave(replay_buffer.get_state()),
            ),
        )

        eval_this_step = global_step >= prev_eval_step + eval_interval_steps

        if eval_this_step:
          prev_eval_step = global_step
          # INIT LOGGER
          wandb_log_dict_eval = {"eval/env_step": 0.,
                                 "eval/episode_return": 0.,
                                 "eval/episode_length": 0.
                                 }
          num_eval_episodes = 0.0
          # INIT RNG
          rng, eval_rng = jax.random.split(rng)
          # INIT PLANNER
          eval_plan = None
          # INIT ENVIRONMENTS
          eval_observation, _ = eval_env.reset(seed=cfg.seed)
          # FOR EVERY TRAINING STEP
          for eval_step in range(1005):
            rng, eval_action_key = jax.random.split(rng)
            eval_action, eval_plan = agent.act(
              obs=eval_observation,
              prev_plan=eval_plan,
              deterministic=True,
              train=False,
              key=eval_action_key)

            eval_next_observation, eval_reward, eval_terminated, eval_truncated, eval_info = eval_env.step(eval_action)
            eval_observation = eval_next_observation

            # Handle terminations/truncations
            eval_done = np.logical_or(eval_terminated, eval_truncated)
            if np.any(eval_done):
              if eval_plan is not None:
                eval_plan = (
                  eval_plan[0].at[eval_done].set(0),
                  eval_plan[1].at[eval_done].set(agent.max_plan_std)
                )
              for ienv in range(env_config.num_envs):
                if eval_done[ienv]:
                  eval_ep_count[ienv] += 1
                  num_eval_episodes+=1
                  wandb_log_dict_eval["eval/env_step"] += global_step
                  wandb_log_dict_eval["eval/episode_length"] += eval_info["final_info"][ienv]['final_info']['episode']['l'].astype(float)
                  wandb_log_dict_eval["eval/episode_return"] += eval_info["final_info"][ienv]['final_info']['episode']['r']
          wandb_log_dict_eval["eval/env_step"] /= num_eval_episodes
          wandb_log_dict_eval["eval/episode_length"] /= num_eval_episodes
          wandb_log_dict_eval["eval/episode_return"] /= num_eval_episodes
          if wandb_config.log_wandb: wandb.log(wandb_log_dict_eval)




      pbar.update(env_config.num_envs)
    pbar.close()


if __name__ == '__main__':
  train()
