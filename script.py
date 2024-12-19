#!/bin/python
import retro
import time

def main():
    env = retro.make(game='MsPacMan-Nes')
    obs = env.reset()
    while True:
        time.sleep(0.06)
        obs, rew, done, info = env.step(env.action_space.sample())
        env.render()
        if done:
            obs = env.reset()
    env.close()


if __name__ == "__main__":
    main()
