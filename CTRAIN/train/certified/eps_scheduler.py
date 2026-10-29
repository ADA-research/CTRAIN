"""Schedule perturbation radii and loss-mixture weights during certified training."""
# Much of the following code is adapted from the auto_LiRPA licensed under the BSD 3-Clause License.

import matplotlib.pyplot as plt
import torch


def linear_schedule_fraction(epoch, batch, batches, warm_up_epochs, ramp_up_epochs,
                             *, at_batch_end=False):
    """Return progress through a linear ramp at the requested batch boundary.

    Warm-up returns zero and completed ramp-up returns one. at_batch_end selects
    whether the current batch contributes to the ramp fraction.
    """
    if epoch < warm_up_epochs:
        return 0.
    elapsed = (epoch - warm_up_epochs + (batch + 1) / batches if at_batch_end
               else epoch + batch / batches - warm_up_epochs)
    return min(1., max(0., elapsed / ramp_up_epochs)) if ramp_up_epochs else 1.


class BaseScheduler():
    
    """Store the perturbation and loss-mixture schedule shared by certified trainers.

    The scheduler tracks raw-input epsilon together with kappa and beta. Epsilon
    can be returned in model coordinates using the supplied normalisation
    statistics. LinearScheduler and SmoothedScheduler implement the updates.
    """
    def __init__(self, num_epochs, eps, mean, std, start_eps=0, start_kappa=1, end_kappa=0, start_beta=0, end_beta=0, eps_schedule_unit='batch', eps_schedule=(0, 20), batches_per_epoch=None, start_epoch=-1):
        """Configure perturbation and mixture schedules with input-normalisation metadata.

        Args:
            num_epochs (int): The number of epochs for training.
            eps (float): The epsilon value for the scheduler.
            mean (float): The mean value for normalisation.
            std (float): The standard deviation value for normalisation.
            start_eps (float, optional): The starting epsilon value.
            start_kappa (float, optional): The starting kappa value.
            end_kappa (float, optional): The ending kappa value.
            start_beta (float, optional): The starting beta value.
            end_beta (float, optional): The ending beta value.
            eps_schedule_unit (str, optional): The unit for epsilon scheduling ('batch' or
                'epoch').
            eps_schedule (tuple, optional): Warm-up, ramp-up and remaining training durations in
                the selected scheduling unit.
            batches_per_epoch (int, optional): Positive number of training batches per epoch,
                required for both scheduling units.
            start_epoch (int, optional): Starting epoch used to initialise the batch counter; -1
                denotes a new schedule.

        Raises:
            ValueError: Invalid schedule durations, epoch budget, batch count, normalisation statistics or mixture endpoints.
        """
        
        if eps_schedule_unit not in ('epoch', 'batch'):
            raise ValueError("eps_schedule_unit must be 'epoch' or 'batch'")
        if len(eps_schedule) not in (2, 3) or any(type(v) is not int or v < 0 for v in eps_schedule):
            raise ValueError('eps_schedule must contain two or three nonnegative integer durations')
        if num_epochs is None and eps_schedule_unit == 'epoch':
            num_epochs = sum(eps_schedule)
        if type(num_epochs) is not int or num_epochs <= 0:
            raise ValueError('num_epochs must be a positive integer')
        if type(batches_per_epoch) is not int or batches_per_epoch <= 0:
            raise ValueError('batches_per_epoch must be a positive integer')
        if eps_schedule_unit == 'epoch' and (sum(eps_schedule) > num_epochs or
                (len(eps_schedule) == 3 and sum(eps_schedule) != num_epochs)):
            raise ValueError('eps_schedule is incompatible with num_epochs')
        if not (0 <= start_eps <= eps < float('inf')):
            raise ValueError('epsilon values must be finite and satisfy 0 <= start_eps <= eps')
        if any(not 0 <= v <= 1 for v in (start_kappa, end_kappa, start_beta, end_beta)):
            raise ValueError('kappa and beta values must be finite and in [0, 1]')
        if not torch.isfinite(torch.as_tensor(std)).all() or (torch.as_tensor(std) <= 0).any():
            raise ValueError('std must be finite and positive')

        self.num_epochs = num_epochs
        if len(eps_schedule) == 2:
            self.warm_up, self.ramp_up = eps_schedule
        elif len(eps_schedule) == 3:
            self.warm_up, self.ramp_up, _ = eps_schedule
        
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
        """Return the current perturbation radius, optionally divided by the input standard deviation."""
        max_eps = self.get_max_eps(normalise=False)
        cur_eps = torch.as_tensor(self.cur_eps)
        # Check needed to mitigate numerical instabilities
        if (torch.as_tensor(max_eps) - cur_eps < 1e-7).all():
            self.cur_eps = max_eps
            cur_eps = torch.as_tensor(self.cur_eps)
        return cur_eps / torch.as_tensor(self.std) if normalise else self.cur_eps
    
    def get_cur_kappa(self):
        """Return the current natural-loss mixture weight."""
        return self.cur_kappa
    
    def get_cur_beta(self):
        """Return the current CROWN-IBP bound-mixture weight."""
        return self.cur_beta
    
    def get_max_eps(self, normalise=True):
        """Return the target perturbation radius, optionally divided by the input standard deviation."""
        return torch.as_tensor(self.eps) / torch.as_tensor(self.std) if normalise else self.eps
    
    def batch_step(self, metric=None):
        """Define the batch-step interface; derived schedulers implement the numerical update."""
        raise NotImplementedError
    
    

class LinearScheduler(BaseScheduler):
    """Increase epsilon linearly while interpolating the loss-mixture weights.

    Args:
        num_epochs (int): Total number of epochs for training.
        eps (float): The target epsilon value.
        mean (float): The mean value for normalisation.
        std (float): The standard deviation value for normalisation.
        start_eps (float, optional): The starting epsilon value.
        start_kappa (float, optional): The starting kappa value.
        end_kappa (float, optional): The ending kappa value.
        start_beta (float, optional): The starting beta value.
        end_beta (float, optional): The ending beta value.
        eps_schedule_unit (str, optional): The unit for epsilon scheduling ('batch' or
            'epoch').
        eps_schedule (tuple, optional): Warm-up, ramp-up and remaining training durations in
            the selected scheduling unit.
        batches_per_epoch (int, optional): Positive number of training batches per epoch,
            required for both scheduling units.
        start_epoch (int, optional): Starting epoch used to initialise the batch counter; -1
            denotes a new schedule.

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
            end_beta=end_beta,
            start_epoch=start_epoch,
        )
        
        if start_epoch > 0 or (self.warm_up == 0 and self.ramp_up == 0):
            self.batch_step()
    
    def batch_step(self, metric=None):
        """Advance epsilon and mixture weights by one training batch."""
        progress = (float(self.no_batches >= self.warm_up) if self.ramp_up == 0 else
                    min(1., max(0., (self.no_batches - self.warm_up) / self.ramp_up)))
        self.cur_eps = self.start_eps + (self.eps - self.start_eps) * progress
        self.cur_kappa = self.start_kappa + (self.end_kappa - self.start_kappa) * progress
        self.cur_beta = self.start_beta + (self.end_beta - self.start_beta) * progress
        self.no_batches += 1

class SmoothedScheduler(BaseScheduler):
    """Increase epsilon through a polynomial segment followed by a matched linear ramp.

    Args:
        num_epochs (int): Number of epochs for training.
        eps (float): Final epsilon value.
        mean (float): Mean value for normalisation.
        std (float): Standard deviation value for normalisation.
        start_eps (float, optional): Initial epsilon value.
        start_kappa (float, optional): Initial kappa value.
        end_kappa (float, optional): Final kappa value.
        start_beta (float, optional): Initial beta value.
        end_beta (float, optional): Final beta value.
        eps_schedule_unit (str, optional): Unit for epsilon scheduling ('batch' or 'epoch').
        batches_per_epoch (int, optional): Positive number of training batches per epoch,
            required for both scheduling units.
        start_epoch (int, optional): Starting epoch used to initialise the batch counter; -1
            denotes a new schedule.
        eps_schedule (tuple, optional): Warm-up, ramp-up and remaining training durations in
            the selected scheduling unit.
        midpoint (float, optional): Ramp fraction where epsilon changes from polynomial
            growth to a matched linear segment.
        exponent (float, optional): Power of the initial polynomial epsilon-growth segment.

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
        if not 0 < midpoint < 1 or not 1 < exponent < float('inf'):
            raise ValueError('midpoint must be in (0, 1) and exponent must be finite and greater than 1')
        self.midpoint = midpoint
        self.exponent = exponent
        if start_epoch > 0 or (self.warm_up == 0 and self.ramp_up == 0):
            self.batch_step()
    
    
    def batch_step(self, metric=None):
        """Advance epsilon and mixture weights by one training batch."""
        step = self.no_batches
        init_step = self.warm_up + 1
        final_step = self.warm_up + self.ramp_up
        mid_step = int((final_step - init_step) * self.midpoint) + init_step

        if self.ramp_up == 0:
            progress = float(step >= self.warm_up)
        elif step <= self.warm_up:
            progress = 0.
        elif step >= final_step:
            progress = 1.
        elif mid_step <= init_step:
            # Short ramps have no exponential segment; use a finite linear ramp.
            progress = (step - self.warm_up) / self.ramp_up
        elif step <= init_step:
            progress = 0.
        else:
            power = (mid_step - init_step) ** (self.exponent - 1)
            alpha = 1 / ((final_step - mid_step) * self.exponent * power +
                         (mid_step - init_step) * power)
            mid_value = alpha * (mid_step - init_step) ** self.exponent
            progress = (alpha * (step - init_step) ** self.exponent if step < mid_step else
                        mid_value + (1 - mid_value) * (step - mid_step) / (final_step - mid_step))

        self.cur_eps = self.start_eps + (self.eps - self.start_eps) * progress
        self.cur_kappa = self.start_kappa + (self.end_kappa - self.start_kappa) * progress
        self.cur_beta = self.start_beta + (self.end_beta - self.start_beta) * progress
        self.no_batches += 1


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
