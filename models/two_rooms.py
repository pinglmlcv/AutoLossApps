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

root_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.insert(0, root_path)
sys.path.insert(0, os.path.join(root_path, 'lib/pycolab'))

from pycolab import ascii_art
from pycolab import human_ui
from utils import myUi
from pycolab.prefab_parts import sprites as prefab_sprites
from pycolab import things as plab_things


GAME_ART = ['####################',
            '#     ########     #',
            '#     ########     #',
            '#     ########     #',
            '#                  #',
            '#     ########     #',
            '#     ########     #',
            '#     ########     #',
            '####################']

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

    def update(self, actions, board, layers, backdrop, things, the_plot):
        del backdrop, things   # Unused.
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
            the_plot.add_reward(-0.4)

        # See if we've found the mystery spot.
        if layers['$'][self.position]:
            the_plot.add_reward(1.0)
            the_plot.terminate_episode()
        elif actions > -1:
            the_plot.add_reward(-0.1)



class GoalDrape(plab_things.Drape):
    # A `Drape` that marks the goal position.
    def __init__(self, curtain, character):
        super(GoalDrape, self).__init__(curtain, character)

    def update(self, actions, board, layers, backdrop, things, the_plot):
        pass


class Env2Rooms():
    def __init__(self, config, default_goal=None,
                 default_goals=None, default_init=None):
        self.config = config
        self.game_art = copy.deepcopy(GAME_ART)
        self.default_goal = default_goal
        self.default_init = default_init
        self.default_goals = default_goals

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

    def set_goal_position(self, goal=None):
        if not goal:
            if self.default_goal:
                goal = self.default_goal
            else:
                goal = self.sample_random_position(self.default_goals)
        self.game_art = modify_art(self.game_art, goal, '$')

    def set_init_position(self, init=None):
        if not init:
            if self.default_init:
                init = self.default_init
            else:
                init = self.sample_random_position()
        self.game_art = modify_art(self.game_art, init, 'P')


    def update(self, action):
        if self.display_flag:
            observation, reward, discount = self.ui.update(action)
        else:
            observation, reward, discount = self.game.play(action)
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
            i = random.randint(0, len(self.default_goals)-1)
            return self.default_goals[i]



def main(argv=()):
    del argv  # Unused.

    # Build a 2-rooms game environment
    env = Env2Rooms(1)
    env.set_goal_position((3,3))
    env.set_init_position((2,2))

    ob, _, _ = env.init_episode(True)
    print(ob)
    env.update(1)
    env.update(1)
    env.update(1)
    env.update(1)
    env.update(1)
    env.update(1)
    env.update(1)
    env.update(1)
    env.update(1)
    env.update(1)
    env.update(1)
    env.update(1)
    env.update(1)


if __name__ == '__main__':
  main(sys.argv)
