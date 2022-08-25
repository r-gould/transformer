class Scheduler:
    def __init__(self, optimizer, d_model, warmup_steps):

        self.optimizer = optimizer
        self.d_model = d_model
        self.warmup_steps = warmup_steps

        self.step_num = 1

    def step(self):

        self.lr_step()
        self.optimizer.step()

    def zero_grad(self):

        self.optimizer.zero_grad()

    def lr_step(self):

        lr = self.calc_lr()
        for group in self.optimizer.param_groups:
            group["lr"] = lr

        self.step_num += 1

    def calc_lr(self):

        lr = min(self.step_num**-0.5, self.step_num*self.warmup_steps**-1.5) * self.d_model ** -0.5
        return lr