# Copyright 2017 the pycolab Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Extended by Haowen Xu, 06-14-2018

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import curses
import sys
import random
import os
import copy
import numpy as np

root_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.insert(0, root_path)
sys.path.insert(0, os.path.join(root_path, 'lib/pycolab'))

from pycolab import ascii_art
from pycolab import human_ui
from utils import myUi
from pycolab.prefab_parts import sprites as prefab_sprites
from pycolab import things as plab_things


GAME_ART = ['#####################',
            '#     ##     ##     #',
            '#     ##     ##     #',
            '#     ##     ##     #',
            '#                   #',
            '#     ##     ##     #',
            '#     ##     ##     #',
            '#     ##     ##     #',
            '#####################']

def modify_art(art, position, char):
    new_art = art.copy()
    line = new_art[position[0]]
    new_line = line[0:position[1]] + char + line[position[1]+1:]
    new_art[position[0]] = new_line
    return new_art

def make_game(game_art):
    return ascii_art.ascii_art_to_game(
        game_art, what_lies_beneath=' ',
        sprites={'P': PlayerSprite},
        drapes={'$': GoalDrape})


class PlayerSprite(prefab_sprites.MazeWalker):
  # A `Sprite` for our player.
  # This `Sprite` ties actions to going in the four cardinal directions. If we
  # reach a goal location marked by '$', the agent receives a
  # reward of 1 and the epsiode terminates.

    def __init__(self, corner, position, character):
        # Inform superclass that we can't walk through walls.
        super(PlayerSprite, self).__init__(corner, position, character, impassable='#')
        # To enumerate how many times a position has been visited before.
        self.visit = np.zeros((len(GAME_ART), len(GAME_ART[0])))

    def update(self, actions, board, layers, backdrop, things, the_plot):
        del backdrop, things   # Unused.
        row = self.position.row
        col = self.position.col
        self.visit[row, col] += 1

        # Apply motion commands.
        old_position = self.position
        if actions == 0:    # walk upward?
            self._north(board, the_plot)
        elif actions == 1:  # walk downward?
            self._south(board, the_plot)
        elif actions == 2:  # walk leftward?
            self._west(board, the_plot)
        elif actions == 3:  # walk rightward?
            self._east(board, the_plot)
        if actions is None:
            actions = -1
        if actions > -1 and actions < 5 and self.position == old_position:
            the_plot.add_reward(-1.0)
            the_plot.terminate_episode()
            return

        # See if we've found the mystery spot.
        if layers['$'][self.position]:
            the_plot.add_reward(1.0)
            the_plot.terminate_episode()
            return
        elif actions > -1:
            the_plot.add_reward(-0.02)

class GoalDrape(plab_things.Drape):
    # A `Drape` that marks the goal position.
    def __init__(self, curtain, character):
        super(GoalDrape, self).__init__(curtain, character)

    def update(self, actions, board, layers, backdrop, things, the_plot):
        pass


class Env2Rooms():
    def __init__(self, config, default_goal=None, optional_goals=None,
                 default_init=None, optional_inits=None):
        self.config = config
        self.max_time_steps = config.agent.total_steps
        self.game_art = copy.deepcopy(GAME_ART)
        self.default_goal = default_goal
        self.default_init = default_init
        self.optional_goals = optional_goals
        self.optional_inits = optional_inits
        self.time_steps = 0
        self.current_state = None

    def init_episode(self, display_flag):
        self.game = make_game(self.game_art)
        self.display_flag = display_flag
        if display_flag:
            self.ui = myUi.MyUi(self.game)
            observation, reward, discount = self.ui.catch()
        else:
            observation, reward, discount = self.game.its_showtime()
        return observation, reward, discount

    def reset_game_art(self):
        self.game_art = copy.deepcopy(GAME_ART)
        self.time_steps = 0

    def set_goal_position(self, goal=None):
        # Three mode to set the goal position,
        #   1) Specify the goal at run time by using: set_goal_position(goal).
        #   2) Specify a fixed goal when init the env through default_goal.
        #   3) Specify a set of optional goals when init the env through
        #   optional_goals.
        #   4) Randomly sample a feasible position.
        if not goal:
            if self.default_goal:
                goal = self.default_goal
            else:
                goal = self.sample_random_position(self.optional_goals)
        self.game_art = modify_art(self.game_art, goal, '$')

    def set_init_position(self, init=None):
        if not init:
            if self.default_init:
                init = self.default_init
            else:
                init = self.sample_random_position(self.optional_inits)
        self.game_art = modify_art(self.game_art, init, 'P')

    def update(self, action, display=None):
        self.time_steps += 1
        if self.display_flag:
            observation, reward, discount = self.ui.update(action,
                                                           display=display)
        else:
            observation, reward, discount = self.game.play(action)
        # ----Terminate when timestep exceeds a limit
        if self.time_steps > self.max_time_steps:
            discount = 0
        return observation, reward, discount

    def sample_random_position(self, optional_position=None):
        if optional_position is None:
            height = len(self.game_art)
            width = len(self.game_art[0])
            h = random.randint(0, height-1)
            w = random.randint(0, width-1)
            while self.game_art[h][w] != ' ':
                h = random.randint(0, height-1)
                w = random.randint(0, width-1)
            return (h, w)
        else:
            i = random.randint(0, len(optional_position)-1)
            return optional_position[i]



def main(argv=()):
    del argv  # Unused.

    # Build a 2-rooms game environment
    env = Env2Rooms(1)
    env.set_goal_position((3,3))
    env.set_init_position((2,2))

    env.init_episode(True)
    env.update(1)
    env.update(0)
    env.update(1)
    env.update(0)
    env.update(1)
    env.update(0)
    env.update(1)
    env.update(0)
    env.update(1)
    env.update(0)
    env.update(1)
    env.update(0)
    env.update(1)
    env.update(0)
    env.update(1)
    env.update(0)
    env.update(1)


if __name__ == '__main__':
  main(sys.argv)
