import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict

torch.set_default_dtype(torch.float64)

class LSJVEngine:
    def __init__(self, S0: float, v0: float, rates: Dict[str, float], 
                 heston_params: Dict[str, float], jump_params: Dict[str, float],
                 sim_params: Dict[str, int], device: torch.device):
        
        self.device = device
        self.S0 = torch.tensor(S0, device=self.device, requires_grad=True)
        self.v0 = torch.tensor(v0, device=self.device, requires_grad=True)
        
        self.r, self.q = rates.get('r', 0.0), rates.get('q', 0.0)
        self.kappa, self.theta = heston_params['kappa'], heston_params['theta']
        self.xi, self.rho = heston_params['xi'], heston_params['rho']
        self.lam, self.mu_J, self.sig_J = jump_params['lambda'], jump_params['mu_J'], jump_params['sig_J']
        
        self.jump_compensator = np.exp(self.mu_J + 0.5 * self.sig_J**2) - 1.0
        
        self.T, self.N_steps, self.N_paths = sim_params['T'], sim_params['steps'], sim_params['paths']
        self.dt = self.T / self.N_steps

    def _market_local_vol(self, t: torch.Tensor, S: torch.Tensor) -> torch.Tensor:
        S_val = S.detach() 
        moneyness = torch.clamp(S_val / self.S0.detach(), min=0.4, max=2.0)
        return 0.25 + 0.20 * (moneyness - 1.0)**2 + 0.05 * torch.exp(-2.0 * t)

    def _apply_gaussian_kernel(self, S: torch.Tensor, v: torch.Tensor, bandwidth: int) -> torch.Tensor:
        sorted_S, indices = torch.sort(S)
        sorted_v = v[indices]
        
        x = torch.arange(-bandwidth, bandwidth + 1, device=self.device, dtype=torch.float64)
        kernel = torch.exp(-0.5 * (x / (bandwidth / 3))**2)
        kernel = kernel / kernel.sum()
        kernel = kernel.view(1, 1, -1)
        
        v_in = sorted_v.view(1, 1, -1)
        smoothed_v = F.conv1d(v_in, kernel, padding=bandwidth).squeeze()
        
        cond_exp_v = torch.empty_like(v)
        cond_exp_v[indices] = smoothed_v
        return torch.clamp(cond_exp_v, min=1e-8)

    def calibrate_and_simulate(self) -> torch.Tensor:
        S = self.S0 * torch.ones(self.N_paths, device=self.device)
        v = self.v0 * torch.ones(self.N_paths, device=self.device)
        
        paths = [S]
        dt_tensor = torch.tensor(self.dt, device=self.device)
        sqrt_dt = torch.sqrt(dt_tensor)
        
        Z1 = torch.randn((self.N_steps, self.N_paths), device=self.device)
        Z2 = torch.randn((self.N_steps, self.N_paths), device=self.device)
        W_v = Z1
        W_S = self.rho * Z1 + torch.sqrt(1.0 - torch.tensor(self.rho**2, device=self.device)) * Z2
        
        poisson_dist = torch.distributions.Poisson(self.lam * self.dt)
        bandwidth = max(5, self.N_paths // 200)

        for i in range(self.N_steps):
            t_tensor = torch.tensor(i * self.dt, device=self.device)
            
            cond_exp_v = self._apply_gaussian_kernel(S.detach(), v.detach(), bandwidth)
            lv_market = self._market_local_vol(t_tensor, S)
            L = lv_market / torch.sqrt(cond_exp_v)
            L = torch.clamp(L, min=0.05, max=5.0) 
            
            dN = poisson_dist.sample((self.N_paths,)).to(self.device)
            J = torch.randn(self.N_paths, device=self.device) * self.sig_J + self.mu_J
            jump_multiplier = torch.exp(J * dN)
            
            v_pos = torch.clamp(v, min=0.0)
            v_next = v + self.kappa * (self.theta - v_pos) * self.dt + \
                     self.xi * torch.sqrt(v_pos) * sqrt_dt * W_v[i]
            
            drift = (self.r - self.q - self.lam * self.jump_compensator - 0.5 * (L**2) * v_pos) * self.dt
            diffusion = L * torch.sqrt(v_pos) * sqrt_dt * W_S[i]
            
            S_next = S * torch.exp(drift + diffusion) * jump_multiplier
            
            S = S_next
            v = v_next
            paths.append(S)
            
        return torch.stack(paths)
