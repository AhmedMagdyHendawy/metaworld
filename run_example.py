import metaworld
import random

print(metaworld.ML1.ENV_NAMES)  # Check out the available environments

ml1 = metaworld.ML1('basketball-v2') # Construct the benchmark, sampling tasks

env = ml1.train_classes['basketball-v2']()  # Create an environment with task `pick_place`
task = random.choice(ml1.train_tasks)
env.set_task(task)  # Set task

done = False
while not done:
    obs = env.reset()  # Reset environment
    a = env.action_space.sample() * 10 # Sample an action
    obs, reward, done, info = env.step(a) 
    env.render()