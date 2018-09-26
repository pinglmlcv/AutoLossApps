'''
Early stop checker,
__Author__ == 'Haowen Xu'
__Date__ == '09-26-2018'
'''

class Early_stop_checker():
    def __init__(self, early_stop_rule, early_stop_args):
        self.rule = early_stop_rule
        self.args = early_stop_args
        self.best_performance = self.args['init_best_performance']
        # compare_fn: two args A, B, return True if A is better than B,
        # otherwise return False
        self.compare_fn = self.args['compare_fn']
        self.no_impr_count = 0
        self.step = 0

    def recording(self, step, performance):
        self.step = step
        if self.compare_fn(performance, self.best_performance):
            self.best_performance = performance
            self.no_impr_count = 0
        else:
            self.no_impr_count += 1

    def is_terminated(self):
        pass


