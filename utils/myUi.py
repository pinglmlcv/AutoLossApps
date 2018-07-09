# __author__ = 'Haowen Xu'
# __date__ = '06-14-2018'

""" Show how the agent play the game """
import sys
import os
import curses
import datetime

from pycolab import human_ui
from pycolab.protocols import logging as plab_logging

class MyUi(human_ui.CursesUi):
    def __init__(self, game, delay=None, repainter=None, colour_fg=None,
                 colour_bg=None, croppers=None):
        keys_to_actions={curses.KEY_UP: 0, curses.KEY_DOWN: 1,
                         curses.KEY_LEFT: 2, curses.KEY_RIGHT: 3,
                         -1: 4}
        super(MyUi, self).__init__(keys_to_actions, delay, repainter,
                                   colour_fg, colour_bg, croppers)
        if self._game is not None:
            raise RuntimeError('CursesUi is not at all thread safe')
        self._game = game
        self._start_time = datetime.datetime.now()
        # Inform the croppers which game we're playing.
        for cropper in self._croppers:
            cropper.set_engine(self._game)
        curses.wrapper(self._init_curses)

    def _init_curses(self, screen):
        self._init_colour()
        curses.curs_set(0)
        if self._delay is None:
            screen.timeout(-1)
        else:
            screen.timeout(self._delay)

        # Create the curses window for the log display
        rows, cols = screen.getmaxyx()
        self.console = curses.newwin(rows // 2, cols, rows - (rows // 2), 0)

        # By default, the log display window is hidden
        self.paint_console = False

        def crop_and_repaint(observation):
            observations = [cropper.crop(observation) for cropper in self._croppers]
            if self._repainter:
                if len(observations) == 1:
                    return [self._repainter(observations[0])]
                else:
                    return [copy.deepcopy(self._repainter(obs)) for obs in observations]
            else:
                return observations

        observation, reward, discount = self._game.its_showtime()
        self.observation = observation
        self.reward = reward
        self.discount = discount
        observations = crop_and_repaint(observation)
        self._total_return = reward
        self._display(screen, observations,
                      self._total_return, elapsed=datetime.timedelta())
        self._update_game_console(
            plab_logging.consume(self._game.the_plot), self.console,
            self.paint_console)

        curses.doupdate()
        screen.getkey()

    def update(self, action, display=None):
        self.action = action
        curses.wrapper(self._update)
        return self.observation, self.reward, self.discount

    def _update(self, screen):

        def crop_and_repaint(observation):
            observations = [cropper.crop(observation) for cropper in self._croppers]
            if self._repainter:
                if len(observations) == 1:
                    return [self._repainter(observations[0])]
                else:
                    return [copy.deepcopy(self._repainter(obs)) for obs in observations]
            else:
                return observations

        action = self.action
        observation, reward, discount = self._game.play(action)
        self.observation = observation
        self.reward = reward
        self.discount = discount
        observations = crop_and_repaint(observation)
        if self._total_return is None:
            self._total_return = reward
        elif reward is not None:
            self._total_return += reward

        elapsed = datetime.datetime.now() - self._start_time
        self._display(screen, observations, self._total_return, elapsed)
        #self._display(screen, observations, self.display, elapsed)
        self._update_game_console(
            plab_logging.consume(self._game.the_plot), self.console,
            self.paint_console)

        curses.doupdate()
        screen.getkey()

    def catch(self):
        return self.observation, self.reward, self.discount
