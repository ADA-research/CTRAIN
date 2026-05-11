# Much of the following code is adapted from the auto_LiRPA licensed under the BSD 3-Clause License.

import matplotlib.pyplot as plt
import torch

class BaseScheduler():
    
    def __init__(self, num_epochs, eps, mean, std, start_eps=0, start_kappa=1, end_kappa=0, start_beta=0, end_beta=0, eps_schedule_unit='batch', eps_schedule=(0, 20), batches_per_epoch=None, start_epoch=-1):
        """
        Initializes the Base EpsScheduler.
        
        Args:
            num_epochs (int): The number of epochs for training.
            eps (float): The epsilon value for the scheduler.
            mean (float): The mean value for normalization.
            std (float): The standard deviation value for normalization.
            start_eps (float, optional): The starting epsilon value. Defaults to 0.
            start_kappa (float, optional): The starting kappa value. Defaults to 1.
            end_kappa (float, optional): The ending kappa value. Defaults to 0.
            start_beta (float, optional): The starting beta value. Defaults to 0.
            end_beta (float, optional): The ending beta value. Defaults to 0.
            eps_schedule_unit (str, optional): The unit for epsilon scheduling ('batch' or 'epoch'). Defaults to 'batch'.
            eps_schedule (tuple, optional): The schedule for epsilon values. Defaults to (0, 20).
            batches_per_epoch (int, optional): The number of batches per epoch. Defaults to None.
            start_epoch (int, optional): The starting epoch number. Defaults to -1.
        
        Raises:
            AssertionError: If num_epochs is None and eps_schedule_unit is 'epoch'.
            AssertionError: If the length of eps_schedule is not 2 or 3.
            AssertionError: If num_epochs is incompatible with eps_schedule.
        """
        
        if num_epochs is None and eps_schedule_unit=='epoch':
            num_epochs = sum(eps_schedule)
        elif num_epochs is None:
            assert False, "Please provide number of epochs!"
        if eps_schedule_unit=='epoch':
            if len(eps_schedule) == 3:
                assert num_epochs == sum(eps_schedule), "Eps Schedule is incompatible with specified number of epochs. Please adjust!"
            elif len(eps_schedule)==2:
                assert num_epochs >= sum(eps_schedule), "Eps Schedule is incompatible with specified number of epochs. Please adjust!"
            else:
                assert False, "Eps Schedule is incompatible with specified number of epochs. Please adjust!"
        
        self.num_epochs = num_epochs
        if len(eps_schedule) == 2:
            self.warm_up, self.ramp_up = eps_schedule
        elif len(eps_schedule) == 3:
            self.warm_up, self.ramp_up, _ = eps_schedule
        
        print(self.warm_up, self.ramp_up)
        self.cur_eps = start_eps
        self.cur_kappa = self.start_kappa = start_kappa
        self.end_kappa = end_kappa
        self.eps = eps
        self.start_eps = start_eps
        self.batches_per_epoch = batches_per_epoch
        self.start_beta = self.cur_beta = start_beta
        self.end_beta = end_beta
        self.mean = mean
        self.std = std
        
        if eps_schedule_unit == 'epoch':
            self.warm_up *= batches_per_epoch
            self.ramp_up *= batches_per_epoch
        
        self.training_steps = num_epochs * batches_per_epoch
        
        self.no_batches = 0
        
        if start_epoch > 0:
            self.no_batches = self.batches_per_epoch * start_epoch - 1            
        
    def get_cur_eps(self, normalise=True):
        """
        Get the current epsilon value, optionally normalised.

        Args:
            normalise (bool): If True, the returned epsilon value will be normalised by the standard deviation.

        Returns:
            torch.Tensor: The current epsilon value, normalised if specified.

        Notes:
            - The method checks for numerical instabilities and adjusts the current epsilon value if necessary.
        """
        max_eps = self.get_max_eps(normalise=False)
        cur_eps = torch.as_tensor(self.cur_eps)
        # Check needed to mitigate numerical instabilities
        if (torch.as_tensor(max_eps) - cur_eps < 1e-7).all():
            self.cur_eps = max_eps
            cur_eps = torch.as_tensor(self.cur_eps)
        return cur_eps / torch.as_tensor(self.std) if normalise else self.cur_eps
    
    def get_cur_kappa(self):
        return self.cur_kappa
    
    def get_cur_beta(self):
        return self.cur_beta
    
    def get_max_eps(self, normalise=True):
        return torch.as_tensor(self.eps) / torch.as_tensor(self.std) if normalise else self.eps
    
    def batch_step(self, metric=None):
        raise NotImplementedError
    
    

class LinearScheduler(BaseScheduler):
    """
    A scheduler that linearly adjusts epsilon, kappa, and beta values over a specified number of epochs.
    
    Args:
        num_epochs (int): Total number of epochs for training.
        eps (float): The target epsilon value.
        mean (float): The mean value for normalization.
        std (float): The standard deviation value for normalization.
        start_eps (float, optional): The starting epsilon value. Defaults to 0.
        start_kappa (float, optional): The starting kappa value. Defaults to 1.
        end_kappa (float, optional): The ending kappa value. Defaults to 0.
        start_beta (float, optional): The starting beta value. Defaults to 1.
        end_beta (float, optional): The ending beta value. Defaults to 0.
        eps_schedule_unit (str, optional): The unit for epsilon scheduling ('batch' or 'epoch'). Defaults to 'batch'.
        eps_schedule (tuple, optional): The schedule for epsilon adjustment. Defaults to (0, 20).
        batches_per_epoch (int, optional): Number of batches per epoch. Defaults to None.
        start_epoch (int, optional): The epoch to start the scheduler. Defaults to -1.
    
    Methods:
        batch_step():
            Adjusts the current epsilon, kappa, and beta values based on the current batch number.
    """
    def __init__(self, num_epochs, eps, mean, std,start_eps=0, start_kappa=1, end_kappa=0, start_beta=1, end_beta=0, eps_schedule_unit='batch', eps_schedule=(0, 20), batches_per_epoch=None, start_epoch=-1):
        super().__init__(
            num_epochs=num_epochs, 
            eps=eps, 
            mean=mean, 
            std=std,
            start_eps=start_eps, 
            start_kappa=start_kappa, 
            end_kappa=end_kappa, 
            eps_schedule_unit=eps_schedule_unit, 
            eps_schedule=eps_schedule, 
            batches_per_epoch=batches_per_epoch, 
            start_beta=start_beta, 
            end_beta=end_beta
        )
        
        if start_epoch > 0:
            self.batch_step()
    
    def batch_step(self, metric=None):
        if self.warm_up < self.no_batches < (self.warm_up + self.ramp_up):
            self.cur_eps += (self.eps / self.ramp_up)
            kappa_step = (self.start_kappa - self.end_kappa) / self.ramp_up
            self.cur_kappa -= kappa_step
            beta_step = (self.start_beta - self.end_beta) / self.ramp_up
            self.cur_beta -= beta_step
        self.cur_eps = min(self.cur_eps, self.eps)
        self.cur_kappa = max(self.cur_kappa, self.end_kappa)
        self.no_batches += 1
    
class SmoothedScheduler(BaseScheduler):
    """
    A scheduler that smoothly transitions epsilon, kappa, and beta values over the course of training.
    
    Args:
        num_epochs (int): Number of epochs for training.
        eps (float): Final epsilon value.
        mean (float): Mean value for normalization.
        std (float): Standard deviation value for normalization.
        start_eps (float, optional): Initial epsilon value. Default is 0.
        start_kappa (float, optional): Initial kappa value. Default is 1.
        end_kappa (float, optional): Final kappa value. Default is 0.
        start_beta (float, optional): Initial beta value. Default is 1.
        end_beta (float, optional): Final beta value. Default is 0.
        eps_schedule_unit (str, optional): Unit for epsilon scheduling ('batch' or 'epoch'). Default is 'batch'.
        batches_per_epoch (int, optional): Number of batches per epoch. Required if eps_schedule_unit is 'batch'.
        start_epoch (int, optional): Epoch to start the scheduling. Default is -1.
        eps_schedule (tuple, optional): Tuple indicating the start and end of epsilon scheduling. Default is (0, 20).
        midpoint (float, optional): Midpoint for the transition from exponential to linear schedule. Default is 0.25.
        exponent (float, optional): Exponent for the exponential schedule. Default is 4.0.
    
    Methods:
        batch_step():
            Updates the current epsilon, kappa, and beta values based on the current batch number.
    """
    def __init__(self, num_epochs, eps, mean, std, start_eps=0, start_kappa=1, end_kappa=0, start_beta=1, end_beta=0, eps_schedule_unit='batch', batches_per_epoch=None, start_epoch=-1, eps_schedule=(0, 20), midpoint=.25, exponent=4.0):
        super().__init__(
            num_epochs=num_epochs, 
            eps=eps, 
            mean=mean,
            std=std,
            start_eps=start_eps, 
            start_kappa=start_kappa, 
            end_kappa=end_kappa, 
            eps_schedule_unit=eps_schedule_unit, 
            eps_schedule=eps_schedule, 
            batches_per_epoch=batches_per_epoch, 
            start_epoch=start_epoch,
            start_beta=start_beta, 
            end_beta=end_beta)
        self.midpoint = midpoint
        self.exponent = exponent
        if start_epoch > 0:
            self.batch_step()
    
    
    def batch_step(self, metric=None):
        init_value = self.start_eps
        final_value = self.eps
        beta = self.exponent
        step = self.no_batches
        # Batch number for schedule start
        init_step = self.warm_up + 1
        # Batch number for schedule end
        final_step = self.warm_up + self.ramp_up
        # Batch number for switching from exponential to linear schedule
        mid_step = int((final_step - init_step) * self.midpoint) + init_step
        t = (mid_step - init_step) ** (beta - 1.)
        # find coefficient for exponential growth, such that at mid point the gradient is the same as a linear ramp to final value
        alpha = (final_value - init_value) / ((final_step - mid_step) * beta * t + (mid_step - init_step) * t)
        # value at switching point
        mid_value = init_value + alpha * (mid_step - init_step) ** beta
        # return init_value when we have not started
        is_ramp = float(step > init_step)
        # linear schedule after mid step
        is_linear = float(step >= mid_step)
        exp_value = init_value + alpha * float(step - init_step) ** beta
        linear_value = min(mid_value + (final_value - mid_value) * (step - mid_step) / (final_step - mid_step), final_value)
        self.cur_eps = is_ramp * ((1.0 - is_linear) * exp_value + is_linear * linear_value) + (1.0 - is_ramp) * init_value
        self.cur_kappa = self.start_kappa * (1 - (self.cur_eps * (1-self.end_kappa)) / self.eps)
        self.cur_beta = self.start_beta * (1 - (self.cur_eps * (1-self.end_beta)) / self.eps)
        self.cur_eps = min(self.cur_eps, self.eps)
        self.cur_kappa = max(self.cur_kappa, self.end_kappa)
        self.cur_beta = max(self.cur_beta, self.end_beta)
        self.no_batches += 1
        # print(self.no_batches, self.cur_eps, self.cur_kappa)


if __name__ == "__main__":
    num_epochs = 100
    batches_per_epoch = 50
    eps_scheduler = SmoothedScheduler(
        num_epochs=num_epochs,
        eps=.2,
        mean=1,
        std=1,
        start_eps=0,
        start_kappa=1,
        end_kappa=0,
        eps_schedule_unit='epoch',
        eps_schedule=(0, 20, 1),
        batches_per_epoch=32,
        # This parameter controls for how much time the epsilon should increase exponentially
        midpoint=.25 ,
        # exponent
        exponent=4.0
    )
    
    fig = plt.figure(figsize=(20, 10))
    eps_list = []
    kappa_list = []
    beta_list = []
    
    for i in range(num_epochs * batches_per_epoch):
        cur_eps = eps_scheduler.get_cur_eps()
        cur_kappa = eps_scheduler.get_cur_kappa()
        cur_beta = eps_scheduler.get_cur_beta()
        eps_list.append(cur_eps)
        kappa_list.append(cur_kappa)
        beta_list.append(cur_beta)
        eps_scheduler.batch_step()
    
    plt.plot(eps_list)
    plt.plot(kappa_list)
    plt.plot(beta_list)
    plt.savefig('./eps_schedule_test_linear.png')
