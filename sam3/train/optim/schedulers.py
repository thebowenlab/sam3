# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

# pyre-unsafe

import math


class InverseSquareRootParamScheduler:
    def __init__(
        self,
        base_lr: float,
        warmup_steps: int,
        cooldown_steps: int,
        timescale: int,
    ):
        self.base_lr = base_lr
        self.warmup_steps = warmup_steps
        self.cooldown_steps = cooldown_steps
        self.timescale = timescale

    def __call__(self, step: int, where: float):
        lr = self.base_lr

        if where > 0:
            total_steps = step / where
            progress = (step - self.warmup_steps) / float(
                total_steps - self.warmup_steps
            )
            progress = max(min(progress, 1), 0)
        else:
            progress = 0
            total_steps = 1

        shift = self.timescale - self.warmup_steps
        if self.warmup_steps < step:
            lr = lr / math.sqrt((step + shift) / self.timescale)

        if self.warmup_steps:
            lr = lr * min(1.0, step / self.warmup_steps)
        if self.cooldown_steps:
            lr = lr * min(1.0, (total_steps - step) / self.cooldown_steps)

        return lr
        
        

class WarmupMultiStepParamScheduler:
    def __init__(
        self,
        base_lr: float,
        warmup_steps: int,
        warmup_factor: float,
        steps,
        step_ratio: float,
        frozen_till=-1,
    ):
        self.base_lr = base_lr
        self.warmup_steps = warmup_steps
        self.warmup_factor = warmup_factor
        self.steps = steps
        self.step_ratio = step_ratio
        self.frozen_till = frozen_till

    def __call__(self, step: int, where: float):
        lr = self.base_lr
        if step < self.frozen_till:
                return 0

        if step < self.warmup_steps:
                return self.base_lr*step/self.warmup_steps + (1-step/self.warmup_steps)*self.warmup_factor*self.base_lr
        
        for s in self.steps:
                if step > s:
                        lr *= self.step_ratio
        return lr
 
