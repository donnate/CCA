import torch
import torch.nn.functional as F



def sim(z1: torch.Tensor, z2: torch.Tensor):
    z1 = F.normalize(z1)
    z2 = F.normalize(z2)
    return torch.mm(z1, z2.t())


def semG_semi_loss(z1: torch.Tensor, z2: torch.Tensor, indices,
                   tau: float = 0.5, num_per_class=20):
    f = lambda x: torch.exp(x / tau)
    refl_sim = f(sim(z1, z1))
    between_sim = f(sim(z1, z2))
    N = z1.shape[0]
    return -torch.log(
        (1/(2*num_per_class - 1)) * (between_sim[indices].reshape(N,num_per_class).sum(1) + refl_sim[indices].reshape(N,num_per_class).sum(1) - between_sim.diag())
        / (between_sim.sum(1) + refl_sim.sum(1) - refl_sim.diag()))


def selfG_semi_loss(z1: torch.Tensor, z2: torch.Tensor,  indices,
                   tau: float = 0.5, num_per_class=20):
    f = lambda x: torch.exp(x / tau)
    refl_sim = f(sim(z1, z1))
    between_sim = f(sim(z1, z2))
    return -torch.log(between_sim.diag()/ (between_sim.sum(1) +
                                               refl_sim.sum(1) -
                                               refl_sim.diag()))


def cl_loss_fn(z1: torch.Tensor, z2: torch.Tensor, indices,
               mean: bool = True,
               tau: float = 0.5, type='selfG', num_per_class=20):
    if type=='selfG':
        semi_loss = selfG_semi_loss
    else:
        semi_loss = semG_semi_loss
    l1 = semi_loss(z1, z2, indices=indices, tau=tau, num_per_class=num_per_class)
    l2 = semi_loss(z2, z1, indices=indices, tau=tau, num_per_class=num_per_class)
    ret = (l1 + l2) * 0.5
    ret = torch.mean(ret) if mean else torch.sum(ret)
    return ret
