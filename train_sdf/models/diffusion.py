import numpy as np
from tqdm.auto import tqdm
import torch
from torch import nn, einsum
import torch.nn.functional as F
from collections import namedtuple

from diff_utils.helpers import * 


# constants
ModelPrediction =  namedtuple('ModelPrediction', ['pred_noise', 'pred_x_start'])


def make_beta_schedule(schedule, n_timestep, linear_start=1e-4, linear_end=2e-2, cosine_s=8e-3):
    if schedule == "linear":
        betas = (
                torch.linspace(linear_start ** 0.5, linear_end ** 0.5, n_timestep, dtype=torch.float64) ** 2
        )

    elif schedule == "cosine":
        timesteps = (
                torch.arange(n_timestep + 1, dtype=torch.float64) / n_timestep + cosine_s
        )
        alphas = timesteps / (1 + cosine_s) * np.pi / 2
        alphas = torch.cos(alphas).pow(2)
        alphas = alphas / alphas[0]
        betas = 1 - alphas[1:] / alphas[:-1]
        betas = np.clip(betas, a_min=0, a_max=0.999)

    elif schedule == "sqrt_linear":
        betas = torch.linspace(linear_start, linear_end, n_timestep, dtype=torch.float64)
    elif schedule == "sqrt":
        betas = torch.linspace(linear_start, linear_end, n_timestep, dtype=torch.float64) ** 0.5
    else:
        raise ValueError(f"schedule '{schedule}' unknown.")
    return betas


class GaussianDiffusion(nn.Module):
    def __init__(
        self,
        model,
        timesteps = 1000, sampling_timesteps = None, beta_schedule = 'cosine',
        loss_type = 'l2', objective = 'pred_x0', noise_scale = '1.0',
        p2_loss_weight_gamma = 0., # p2 loss weight, from https://arxiv.org/abs/2204.00227 - 0 is equivalent to weight of 1 across time - 1. is recommended
        p2_loss_weight_k = 1,
        ddim_sampling_eta = 1.,
        v_posterior=0.,
        original_elbo_weight=0.,
        l_simple_weight=1.,
        learn_logvar=False,
        logvar_init=0.
    ):
        super().__init__()

        self.model = model
        self.objective = objective
        self.noise_scale = noise_scale

        betas = make_beta_schedule(beta_schedule, timesteps)
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value = 1.)

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)

        self.loss_type = loss_type

        # sampling related parameters
        self.sampling_timesteps = default(sampling_timesteps, timesteps) # default num sampling timesteps to number of timesteps at training
        assert self.sampling_timesteps <= timesteps
        self.ddim_sampling_eta = ddim_sampling_eta
    
        self.v_posterior = v_posterior
        self.original_elbo_weight = original_elbo_weight
        self.l_simple_weight = l_simple_weight

        self.learn_logvar = learn_logvar
        self.logvar = torch.full(fill_value=logvar_init, size=(self.num_timesteps,))
        if self.learn_logvar:
            self.logvar = nn.Parameter(self.logvar, requires_grad=True)

        # helper function to register buffer from float64 to float32
        register_buffer = lambda name, val: self.register_buffer(name, val.to(torch.float32))

        register_buffer('betas', betas)
        register_buffer('alphas_cumprod', alphas_cumprod)
        register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = (1 - self.v_posterior) * betas * (1. - alphas_cumprod_prev) / (
                1. - alphas_cumprod) + self.v_posterior * betas
        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)
        register_buffer('posterior_variance', posterior_variance)
        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        register_buffer('posterior_log_variance_clipped', torch.log(posterior_variance.clamp(min =1e-20)))
        register_buffer('posterior_mean_coef1', betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        register_buffer('posterior_mean_coef2', (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod))
        
        # calculate p2 reweighting
        register_buffer('p2_loss_weight', (p2_loss_weight_k + alphas_cumprod / (1 - alphas_cumprod)) ** -p2_loss_weight_gamma)

        if self.objective == "pred_noise":
            lvlb_weights = self.betas ** 2 / (
                    2 * self.posterior_variance * alphas * (1 - self.alphas_cumprod))
        elif self.objective == "pred_x0":
            lvlb_weights = 0.5 * np.sqrt(torch.Tensor(alphas_cumprod)) / (2. * 1 - torch.Tensor(alphas_cumprod))
        elif self.objective == "pred_v":
            lvlb_weights = torch.ones_like(self.betas ** 2 / (
                    2 * self.posterior_variance * alphas * (1 - self.alphas_cumprod)))
        else:
            raise NotImplementedError("mu not supported")
        lvlb_weights[0] = lvlb_weights[1]
        self.register_buffer('lvlb_weights', lvlb_weights, persistent=False)
        assert not torch.isnan(self.lvlb_weights).all()

    def predict_start_from_noise(self, x_t, t, noise):
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )
    
    def predict_start_from_noise_masked(self, x_t, t, noise, latent_init, latent_mask):
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * (x_t * (1 - latent_mask) + latent_init * latent_mask) -
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def predict_noise_from_start(self, x_t, t, x0):
        return (
            (x0 - extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t) / \
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
        )
    
    def predict_start_from_z_and_v(self, x_t, t, v):
        return (
                extract(self.sqrt_alphas_cumprod, t, x_t.shape) * x_t -
                extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape) * v
        )
    
    def predict_eps_from_z_and_v(self, x_t, t, v):
        return (
                extract(self.sqrt_alphas_cumprod, t, x_t.shape) * v +
                extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape) * x_t
        )

    @torch.no_grad()
    def ddim_sample(self, dim, batch_size, noise=None, clip_denoised = True, traj=False, cond=None):
        batch, device, total_timesteps, sampling_timesteps, eta, objective = batch_size, self.betas.device, self.num_timesteps, self.sampling_timesteps, self.ddim_sampling_eta, self.objective
        times = torch.linspace(0., total_timesteps, steps = sampling_timesteps + 2)[:-1]
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:]))

        traj = []

        x_T = default(noise, torch.randn(batch, dim, device = device)) 

        for time, time_next in tqdm(time_pairs, desc = 'sampling loop time step'):
            alpha = self.alphas_cumprod_prev[time]
            alpha_next = self.alphas_cumprod_prev[time_next]

            time_cond = torch.full((batch,), time, device = device, dtype = torch.long)

            model_input = (x_T, cond) if cond is not None else x_T
            pred_noise, x_start, *_ = self.model_predictions(model_input, time_cond)

            if clip_denoised:
                x_start.clamp_(-1., 1.)

            sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            c = ((1 - alpha_next) - sigma ** 2).sqrt()

            noise = torch.randn_like(x_T) if time_next > 0 else 0.

            x_T = x_start * alpha_next.sqrt() + \
                  c * pred_noise + \
                  sigma * noise
            
            traj.append(x_T.clone())
    
        if traj:
            return x_T, traj
        else:
            return x_T

    @torch.no_grad()
    def sample(self, dim, batch_size, noise=None, clip_denoised = True, traj=False, cond=None):

        batch, device, objective = batch_size, self.betas.device, self.objective

        traj = []

        x_T = default(noise, torch.randn(batch, dim, device = device))

        for t in reversed(range(0, self.num_timesteps)):
            
            time_cond = torch.full((batch,), t, device = device, dtype = torch.long)

            model_input = (x_T, cond) if cond is not None else x_T
            pred_noise, x_start, *_ = self.model_predictions(model_input, time_cond)
            if clip_denoised:
                x_start.clamp_(-1., 1.)

            model_mean, _, model_log_variance = self.q_posterior(x_start = x_start, x_t = x_T, t = time_cond)

            noise = torch.randn_like(x_T) if t > 0 else 0. # no noise if t == 0

            x_T = model_mean + (0.5 * model_log_variance).exp() * noise
            
            traj.append(x_T.clone())
    
        if traj:
            return x_T, traj
        else:
            return x_T
        

    @torch.no_grad()
    def sample_unet(self, batch_size, noise=None, clip_denoised=True, traj=False, cond=None, mode='semantic'):

        print('Start sampling')

        batch, device, objective = batch_size, self.betas.device, self.objective

        traj = []

        if mode == 'semantic':
            x_T = default(noise, torch.randn(batch, 1, 16, 16, 16, device=device)) # semantic
        elif mode == 'geometry':
            x_T = default(noise, torch.randn(batch, 1, 32, 16, 32, device=device)) # geometry
        elif mode == 'superres':
            x_T = default(noise, torch.randn(batch, 1, 32, 32, 32, device=device)) # superres

        for t in tqdm(reversed(range(0, self.num_timesteps))):
            
            time_cond = torch.full((batch,), t, device = device, dtype = torch.long)

            model_input = x_T
            pred_noise, x_start, attn, *_ = self.model_predictions(model_input, time_cond, cond=cond)
            if clip_denoised:
                x_start.clamp_(-1., 1.)

            model_mean, _, model_log_variance = self.q_posterior(x_start = x_start, x_t = x_T, t = time_cond)

            noise = torch.randn_like(x_T) if t > 0 else 0. # no noise if t == 0

            x_T = model_mean + (0.5 * model_log_variance).exp() * noise
            
            traj.append(x_T.clone())
    
        if traj:
            return x_T, traj, attn
        else:
            return x_T, None, attn
        

    @torch.no_grad()
    def sample_unet_concat(self, batch_size, noise=None, clip_denoised=True, traj=False, cond=None):

        print('Start sampling')

        batch, device, objective = batch_size, self.betas.device, self.objective

        traj = []

        # geo
        x_T = default(noise, torch.randn(batch, 1, 64, 32, 64, device=device))

        for t in tqdm(reversed(range(0, self.num_timesteps))):
            
            time_cond = torch.full((batch,), t, device = device, dtype = torch.long)

            model_input = x_T
            pred_noise, x_start, *_ = self.model_predictions_concat(model_input, time_cond, condition=cond)
            if clip_denoised:
                x_start.clamp_(-1., 1.)

            model_mean, _, model_log_variance = self.q_posterior(x_start = x_start, x_t = x_T, t = time_cond)

            noise = torch.randn_like(x_T) if t > 0 else 0. # no noise if t == 0

            x_T = model_mean + (0.5 * model_log_variance).exp() * noise
            
            traj.append(x_T.clone())
    
        if traj:
            return x_T, traj
        else:
            return x_T, None
        

    @torch.no_grad()
    def sample_unet_room(self, batch_size, noise=None, clip_denoised=True, traj=False, cond=None, latent_init=None, latent_mask=None, mode='semantic'):

        batch, device, objective = batch_size, self.betas.device, self.objective
        U_max = 3

        traj = []

        if mode == 'semantic':
            x_T = default(noise, torch.randn(batch, 1, 16, 16, 16, device=device)) # semantic main
        elif mode == 'geometry':
            x_T = default(noise, torch.randn(batch, 1, 32, 16, 32, device=device)) # geometry
        elif mode == 'superres':
            x_T = default(noise, torch.randn(batch, 1, 32, 32, 32, device=device)) # superres

        for t in tqdm(reversed(range(0, self.num_timesteps))):

            for u in range(U_max):
            
                time_cond = torch.full((batch,), t, device = device, dtype = torch.long)

                model_input = x_T
                pred_noise, x_start, attn, *_ = self.model_predictions(model_input, time_cond, cond=cond)
                                                                            
                if clip_denoised:
                    x_start.clamp_(-1., 1.)

                model_mean, _, model_log_variance = self.q_posterior(x_start=x_start, x_t=x_T, t=time_cond)

                noise = torch.randn_like(x_T) if t > 0 else 0. # no noise if t == 0

                x_T_1 = model_mean + (0.5 * model_log_variance).exp() * noise

                noise_known = torch.randn_like(x_T_1)
                latent_init_noisy = self.q_sample(x_start=latent_init, t=time_cond, noise=noise_known) 
                x_T_1 = x_T_1 * (1 - latent_mask) + latent_init_noisy * latent_mask

                if u < U_max - 1 and t > 1:
                    noise_forward = torch.randn_like(x_T_1)
                    x_T = self.q_step_sample(x_start=x_T_1, t=time_cond - 1, noise=noise_forward)
                else:
                    x_T = x_T_1

            if t % 40 == 0:
                traj.append(x_T.cpu().detach().clone())
    
        if traj:
            return x_T, traj, attn
        else:
            return x_T, None, attn
        

    def q_mean_variance(self, x_start, t):
        """
        Get the distribution q(x_t | x_0).
        :param x_start: the [N x C x ...] tensor of noiseless inputs.
        :param t: the number of diffusion steps (minus 1). Here, 0 means one step.
        :return: A tuple (mean, variance, log_variance), all of x_start's shape.
        """
        mean = (extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start)
        variance = extract(1.0 - self.alphas_cumprod, t, x_start.shape)
        log_variance = extract(self.log_one_minus_alphas_cumprod, t, x_start.shape)
        return mean, variance, log_variance

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
            extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped


    # "nice property": return x_t given x_0, noise, and timestep
    def q_sample(self, x_start, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))
        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )
    
    def q_step_sample(self, x_start, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))
        return (
            extract(torch.sqrt(1. - self.betas), t, x_start.shape) * x_start +
            extract(torch.sqrt(self.betas), t, x_start.shape) * noise
        )
    
    def get_v(self, x, noise, t):
        return (
                extract(self.sqrt_alphas_cumprod, t, x.shape) * noise -
                extract(self.sqrt_one_minus_alphas_cumprod, t, x.shape) * x
        )

    def get_loss(self, pred, target, mean=True):
        if self.loss_type == 'l1':
            loss = (target - pred).abs()
            if mean:
                loss = loss.mean()
        elif self.loss_type == 'l2':
            if mean:
                loss = torch.nn.functional.mse_loss(target, pred)
            else:
                loss = torch.nn.functional.mse_loss(target, pred, reduction='none')
        else:
            raise NotImplementedError("unknown loss type '{loss_type}'")

        return loss

    # main function for calculating loss
    def forward(self, x_start, t, ret_pred_x=False, noise=None, cond=None):

        noise = default(noise, lambda: torch.randn_like(x_start)) * self.noise_scale
        x = self.q_sample(x_start=x_start, t=t, noise=noise) 
        model_in = x
        model_out, _ = self.model(model_in, time_emb=t, context=cond)

        loss_dict = {}
        if self.objective == 'pred_noise':
            target = noise
        elif self.objective == 'pred_x0':
            target = x_start
        elif self.objective == 'pred_v':
            target = self.get_v(x_start, noise, t)
        else:
            raise ValueError(f'unknown objective {self.objective}')

        loss = self.get_loss(model_out, target, mean=False).mean(dim=[1, 2, 3, 4])
        unreduced_loss = loss.detach().clone()

        loss_dict.update({f'loss_simple': loss.mean()})
        loss_simple = loss.mean() * self.l_simple_weight

        loss_vlb = (self.lvlb_weights[t] * loss).mean()
        loss_dict.update({f'loss_vlb': loss_vlb})

        loss = loss_simple + self.original_elbo_weight * loss_vlb

        loss_dict.update({f'loss': loss})
        
        if ret_pred_x:
            return loss.mean(), x, target, model_out, unreduced_loss, loss_dict
        else:
            return loss.mean(), unreduced_loss, loss_dict
        

    # main function for calculating loss
    def forward_concat(self, x_start, t, ret_pred_x=False, noise=None, cond=None):

        noise = default(noise, lambda: torch.randn_like(x_start)) * self.noise_scale
        x = self.q_sample(x_start=x_start, t=t, noise=noise) 
        model_in = torch.cat([x, cond], dim=1)
        model_out, _ = self.model(model_in, time_emb=t, context=None)

        loss_dict = {}
        if self.objective == 'pred_noise':
            target = noise
        elif self.objective == 'pred_x0':
            target = x_start
        elif self.objective == 'pred_v':
            target = self.get_v(x_start, noise, t)
        else:
            raise ValueError(f'unknown objective {self.objective}')

        loss = self.get_loss(model_out, target, mean=False).mean(dim=[1, 2, 3, 4])
        unreduced_loss = loss.detach().clone()

        loss_dict.update({f'loss_simple': loss.mean()})
        loss_simple = loss.mean() * self.l_simple_weight

        loss_vlb = (self.lvlb_weights[t] * loss).mean()
        loss_dict.update({f'loss_vlb': loss_vlb})

        loss = loss_simple + self.original_elbo_weight * loss_vlb

        loss_dict.update({f'loss': loss})
        
        if ret_pred_x:
            return loss.mean(), x, target, model_out, unreduced_loss, loss_dict
        else:
            return loss.mean(), unreduced_loss, loss_dict
        

    def model_predictions(self, model_input, t, cond=None, controller=None):
        
        # model_output1 = self.model(model_input, t, pass_cond=0, context=cond * 0.0 + 0.07)
        # model_output2 = self.model(model_input, t, pass_cond=1, context=cond)
        # model_output = model_output2*5 - model_output1*4

        guidance = 5.0
        model_output1, attn = self.model(model_input, t, pass_cond=0, context=cond * 0.0, controller=None)
        model_output2, attn = self.model(model_input, t, pass_cond=1, context=cond, controller=controller)
        model_output = model_output2 * (1 + guidance) - model_output1 * guidance

        # model_output, attn = self.model(model_input, t, pass_cond=1, context=cond)

        x = model_input[0] if type(model_input) is tuple else model_input

        if self.objective == 'pred_noise':
            pred_noise = model_output
            x_start = self.predict_start_from_noise(x, t, model_output)

        elif self.objective == 'pred_x0':
            pred_noise = self.predict_noise_from_start(x, t, model_output)
            x_start = model_output

        elif self.objective == 'pred_v':
            pred_noise = self.predict_eps_from_z_and_v(x, t, model_output)
            x_start = self.predict_start_from_z_and_v(x, t, model_output)

        return pred_noise, x_start, attn
    

    def model_predictions_concat(self, model_input, t, condition=None):
        
        model_input_concat = torch.cat([model_input, condition], dim=1)
        model_output, _ = self.model(model_input_concat, t, context=None)

        x = model_input[0] if type(model_input) is tuple else model_input

        if self.objective == 'pred_noise':
            pred_noise = model_output
            x_start = self.predict_start_from_noise(x, t, model_output)

        elif self.objective == 'pred_x0':
            pred_noise = self.predict_noise_from_start(x, t, model_output)
            x_start = model_output

        elif self.objective == 'pred_v':
            pred_noise = self.predict_eps_from_z_and_v(x, t, model_output)
            x_start = self.predict_start_from_z_and_v(x, t, model_output)

        return pred_noise, x_start


    def model_predictions_masked(self, model_input, t, cond=None, latent_init=None, latent_mask=None):
        
        model_output, attn = self.model(model_input, t, pass_cond=1, context=cond)

        x = model_input[0] if type(model_input) is tuple else model_input

        pred_noise = model_output
        x_start = self.predict_start_from_noise_masked(x, t, model_output, latent_init=latent_init, latent_mask=latent_mask)

        return pred_noise, x_start, attn


    def diffusion_model_from_latent(self, x_start, cond=None):

        t = torch.randint(0, self.num_timesteps, (x_start.shape[0],), device=x_start.device).long()

        loss, x, target, model_out, unreduced_loss, loss_dict = self(x_start, t, cond=cond, ret_pred_x=True)
        loss_100 = unreduced_loss[t<100].mean().detach()
        loss_1000 = unreduced_loss[t>100].mean().detach()

        return loss, loss_100, loss_1000, model_out, loss_dict


    def generate_unconditional(self, num_samples):
        self.eval()
        with torch.no_grad():
            samp,_ = self.sample_unet(batch_size=num_samples, traj=False, cond=None)

        return samp
    

    def generate_conditional(self, num_samples, cond=None, mode='semantic'):
        self.eval()
        with torch.no_grad():
            samp, _, attn = self.sample_unet(batch_size=num_samples, traj=False, cond=cond, mode=mode)

        return samp, attn
    

    def generate_conditional_concat(self, num_samples, cond=None):
        self.eval()
        with torch.no_grad():
            samp, _ = self.sample_unet_concat(batch_size=num_samples, traj=False, cond=cond)

        return samp
    

    def generate_conditional_room(self, num_samples, cond=None, latent_init=None, latent_mask=None, traj=False, mode='semantic'):
        self.eval()
        with torch.no_grad():
            samp, traj, attn = self.sample_unet_room(batch_size=num_samples, traj=traj, cond=cond, latent_init=latent_init, latent_mask=latent_mask, mode=mode)

        return samp, attn, traj
    
