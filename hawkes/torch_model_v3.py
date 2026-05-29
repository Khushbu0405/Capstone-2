
import math
import logging
import numpy as np
import torch

log = logging.getLogger(__name__)

LOG_K_MIN   = math.log(0.5)
LOG_K_MAX   = math.log(3.0)
LOG_LAM_MIN = math.log(0.5)
LOG_LAM_MAX = math.log(5.0)
MU_RAW_MIN = -10.0
MU_RAW_MAX = 10.0


def _inv_softplus(x: float) -> float:
    return math.log(math.expm1(x)) if x > 0 else -20.0


class HawkesTopicModelTorchV3(torch.nn.Module):
    def __init__(self, K, beta, theta, times,
                 citation_weight=1.0,
                 mu_init=0.1,
                 weibull_k=1.5,
                 weibull_lam=2.0,
                 attn_init_std=0.01,
                 context_window=2.0,
                 decay_rate=1.0,
                 max_history=500,
                 neg_samples=50,
                 focal_gamma=2.0,
                 excitation_weight=0.3,
                 random_state=42,
                 device="cuda"):
        super().__init__()

        self.K = int(K)
        self.D = int(theta.shape[0])
        self.V = int(beta.shape[1])
        self.citation_weight = float(citation_weight)
        self._context_window = float(context_window)
        self._decay_rate = float(decay_rate)

        self.max_history = int(max_history)
        self.neg_samples = int(neg_samples)
        self.focal_gamma = float(focal_gamma)

        self.excitation_weight = float(excitation_weight)

        if device == "cuda" and not torch.cuda.is_available():
            device = "cpu"
        self.device = torch.device(device)

        rng = np.random.RandomState(random_state)

        self.beta = torch.as_tensor(beta, dtype=torch.float32, device=self.device)
        self.theta = torch.as_tensor(theta, dtype=torch.float32, device=self.device)
        self.times = torch.as_tensor(times, dtype=torch.float32, device=self.device)

        t_min = float(np.min(times))
        t_range = float(max(np.max(times) - np.min(times), 1.0))
        self._t_min = t_min
        self._t_range = t_range

        mu_raw_init = _inv_softplus(mu_init)
        self.mu_raw = torch.nn.Parameter(
            torch.full((self.K,), mu_raw_init, dtype=torch.float32, device=self.device)
        )

        wq_init = np.eye(self.K) + attn_init_std * rng.randn(self.K, self.K)
        wk_init = np.eye(self.K) + attn_init_std * rng.randn(self.K, self.K)
        self.W_Q = torch.nn.Parameter(
            torch.as_tensor(wq_init, dtype=torch.float32, device=self.device)
        )
        self.W_K = torch.nn.Parameter(
            torch.as_tensor(wk_init, dtype=torch.float32, device=self.device)
        )

        init_log_k = max(LOG_K_MIN, min(LOG_K_MAX, math.log(weibull_k)))
        init_log_lam = max(LOG_LAM_MIN, min(LOG_LAM_MAX, math.log(weibull_lam)))

        self.log_k = torch.nn.Parameter(
            torch.tensor(init_log_k, dtype=torch.float32, device=self.device)
        )
        self.log_lam = torch.nn.Parameter(
            torch.tensor(init_log_lam, dtype=torch.float32, device=self.device)
        )

        log.info(f"HawkesTopicModelTorchV3: K={self.K}, D={self.D}, V={self.V}")
        log.info(f"  Citation weight  : {citation_weight}")
        log.info(f"  Excitation weight: {self.excitation_weight}")
        log.info(f"  Max history      : {self.max_history}")
        log.info(f"  Kernel clamps    : k∈[0.50, 3.00], lam∈[0.50, 5.00]")

    @property
    def mu(self):
        return torch.nn.functional.softplus(self.mu_raw)

    @property
    def kernel_k(self):
        return float(torch.exp(self.log_k).detach().cpu().item())

    @property
    def kernel_lam(self):
        return float(torch.exp(self.log_lam).detach().cpu().item())

    @property
    def mu_mean(self):
        return float(self.mu.mean().detach().cpu().item())

    def _attention_forward(self, theta_d, theta_e):
        q = theta_d @ self.W_Q
        k = theta_e @ self.W_K
        scores = (k @ q) / math.sqrt(self.K)
        weights = torch.softmax(scores, dim=0)
        return weights, scores

    def _kernel_pdf(self, dt):
        dt = torch.clamp(dt, min=1e-8)
        k = torch.exp(self.log_k)
        lam = torch.exp(self.log_lam)
        dt_over_lam = torch.clamp(dt / lam, min=1e-8, max=1e6)
        return (k / lam) * (dt_over_lam ** (k - 1)) * torch.exp(-(dt_over_lam ** k))

    def _kernel_cdf(self, dt):
        dt = torch.clamp(dt, min=1e-8)
        k = torch.exp(self.log_k)
        lam = torch.exp(self.log_lam)
        dt_over_lam = torch.clamp(dt / lam, min=1e-8, max=1e6)
        return 1.0 - torch.exp(-(dt_over_lam ** k))

    def clamp_parameters(self):
        with torch.no_grad():
            self.log_k.clamp_(LOG_K_MIN, LOG_K_MAX)
            self.log_lam.clamp_(LOG_LAM_MIN, LOG_LAM_MAX)
            self.mu_raw.clamp_(MU_RAW_MIN, MU_RAW_MAX)

    def _subsample_history(self, history_indices, cited_set, t_d):
        N = history_indices.numel()
        if N <= self.max_history:
            return history_indices
        hist_list = history_indices.tolist()
        keep = set()
        for e in cited_set:
            if e in hist_list:
                keep.add(e)
        t_hist = self.times[history_indices]
        sorted_idx = torch.argsort(t_hist, descending=True)
        remaining = self.max_history - len(keep)
        for i in sorted_idx.tolist():
            if remaining <= 0:
                break
            doc_id = hist_list[i]
            if doc_id not in keep:
                keep.add(doc_id)
                remaining -= 1
        return torch.tensor(sorted(keep), dtype=torch.long, device=self.device)

    def _focal_bce_loss(self, scores, true_cit):
        probs = torch.sigmoid(scores)
        pos_mask = true_cit > 0.5
        neg_mask = ~pos_mask
        loss = torch.zeros((), device=self.device)
        if pos_mask.any():
            p_pos = probs[pos_mask]
            focal_weight = (1.0 - p_pos) ** self.focal_gamma
            loss = loss - (focal_weight * torch.log(p_pos + 1e-10)).sum()
        if neg_mask.any():
            neg_indices = torch.where(neg_mask)[0]
            n_neg = min(len(neg_indices), self.neg_samples)
            if n_neg > 0:
                perm = torch.randperm(len(neg_indices), device=self.device)[:n_neg]
                sampled_neg = neg_indices[perm]
                p_neg = probs[sampled_neg]
                focal_weight = p_neg ** self.focal_gamma
                loss = loss - (focal_weight * torch.log(1.0 - p_neg + 1e-10)).sum()
        n_total = pos_mask.sum() + min(neg_mask.sum(), self.neg_samples)
        return loss / max(n_total.item(), 1)

    
    def compute_intensity(self, t_query, theta_query, history_indices,
                          return_components=False):
        lambda_k = self.mu
        if history_indices.numel() == 0:
            if return_components:
                return lambda_k, {}
            return lambda_k
        theta_hist = self.theta[history_indices]
        t_hist = self.times[history_indices]
        dt = t_query - t_hist
        attn_weights, scores = self._attention_forward(theta_query, theta_hist)
        f_dt = self._kernel_pdf(dt)
        combined = attn_weights * f_dt
        excitation = theta_hist.t() @ combined
        lambda_k = lambda_k + excitation
        if return_components:
            return lambda_k, {
                "attn_weights": attn_weights,
                "scores": scores,
                "f_dt": f_dt,
                "combined": combined,
                "theta_hist": theta_hist,
                "dt": dt,
                "history_indices": history_indices,
                "excitation": excitation,
            }
        return lambda_k

    def _compute_compensator(self, t_d, history_indices, comp):
        t_norm = (t_d - self._t_min) / self._t_range
        compensator = self.mu * t_norm
        if history_indices.numel() > 0:
            f_dt = self._kernel_cdf(comp["dt"])
            combined = comp["attn_weights"] * f_dt
            compensator = compensator + comp["theta_hist"].t() @ combined
        return compensator, t_norm

    def hawkes_log_likelihood(self, doc_indices, citation_matrix):
        n_docs = len(doc_indices)
        total_loss = torch.zeros((), device=self.device)
        hawkes_ll = torch.zeros((), device=self.device)
        cit_loss_total = torch.zeros((), device=self.device)

        for d in doc_indices:
            d = int(d)
            t_d = self.times[d]
            all_history = torch.where(self.times < t_d)[0]
            if all_history.numel() == 0:
                continue
            cited_set = citation_matrix.get(d, set())
            history_indices = self._subsample_history(all_history, cited_set, t_d)
            theta_d = self.theta[d]
            lambda_k, comp = self.compute_intensity(
                t_d, theta_d, history_indices, return_components=True
            )
            lambda_k_safe = torch.clamp(lambda_k, min=1e-10)
            comp_val, _ = self._compute_compensator(t_d, history_indices, comp)
            ll_d = torch.sum(torch.log(lambda_k_safe)) - torch.sum(comp_val)
            hawkes_ll = hawkes_ll + ll_d

            cit_loss_d = torch.zeros((), device=self.device)
            if self.citation_weight > 0 and cited_set:
                hist_list = history_indices.tolist()
                true_cit = torch.zeros(len(hist_list), device=self.device)
                for i, e in enumerate(hist_list):
                    if e in cited_set:
                        true_cit[i] = 1.0
                if torch.sum(true_cit) > 0:
                    cit_loss_d = self._focal_bce_loss(comp["scores"], true_cit)
            cit_loss_total = cit_loss_total + cit_loss_d
            total_loss = total_loss + (-ll_d + self.citation_weight * cit_loss_d)

        if n_docs == 0:
            return total_loss, 0.0, 0.0
        total_loss = total_loss / n_docs
        return total_loss, (hawkes_ll / n_docs).detach().cpu().item(), \
               (cit_loss_total / n_docs).detach().cpu().item()

    
    def citation_scores(self, theta_query, theta_hist, dt=None, cosine_alpha=0.7):
        theta_q = torch.as_tensor(theta_query, dtype=torch.float32, device=self.device)
        theta_h = torch.as_tensor(theta_hist, dtype=torch.float32, device=self.device)

        #Learned attention score (raw, no softmax)
        q = theta_q @ self.W_Q
        k = theta_h @ self.W_K
        learned_scores = (k @ q) / math.sqrt(self.K)  # [N_hist]

        #Cosine similarity
        norm_q = torch.norm(theta_q) + 1e-10
        norm_h = torch.norm(theta_h, dim=1) + 1e-10
        cosine = (theta_h @ theta_q) / (norm_h * norm_q)  # [N_hist]

        def _z(x):
            return (x - x.mean()) / (x.std() + 1e-10)

        if learned_scores.numel() > 1:
            learned_z = _z(learned_scores)
            cosine_z = _z(cosine)
        else:
            learned_z = learned_scores
            cosine_z = cosine
        combined = cosine_alpha * cosine_z + (1.0 - cosine_alpha) * learned_z
        return combined.detach().cpu().numpy()


    def _compute_context_vector(self, t_star, window=2.0, decay_rate=1.0):
        mask = (self.times >= t_star - window) & (self.times < t_star)
        if torch.any(mask):
            dt = t_star - self.times[mask]
            exp_weights = torch.exp(-decay_rate * dt)
            exp_weights = exp_weights / (exp_weights.sum() + 1e-10)
            c = (exp_weights.unsqueeze(1) * self.theta[mask]).sum(dim=0)
        else:
            c = self.theta.mean(dim=0)
        return c / (c.sum() + 1e-10)

    def predict_topic_intensity(self, T_star, context_window=2.0):
        t_star = torch.tensor(T_star, dtype=torch.float32, device=self.device)
        history_indices = torch.where(self.times < t_star)[0]

        c = self._compute_context_vector(
            t_star, window=context_window, decay_rate=self._decay_rate,
        )

        if history_indices.numel() == 0:
            return c.detach().cpu().numpy(), c.detach().cpu().numpy()

        # Compute excitation only 
        theta_hist = self.theta[history_indices]
        t_hist = self.times[history_indices]
        dt = t_star - t_hist

        attn_weights, _ = self._attention_forward(c, theta_hist)
        f_dt = self._kernel_pdf(dt)
        combined = attn_weights * f_dt
        excitation = theta_hist.t() @ combined  # [K]

        # Normalize excitation to a distribution
        exc_dist = torch.softmax(excitation, dim=0)

        w = self.excitation_weight
        pi_k = (1.0 - w) * c + w * exc_dist
        pi_k = pi_k / (pi_k.sum() + 1e-10)

        lambda_k = self.mu + excitation  # keep for compatibility
        return lambda_k.detach().cpu().numpy(), pi_k.detach().cpu().numpy()

    def predict_words(self, T_star, top_n=50, context_window=2.0):
        _, pi_k = self.predict_topic_intensity(T_star, context_window)
        word_probs = pi_k @ self.beta.detach().cpu().numpy()
        word_probs = word_probs / (word_probs.sum() + 1e-10)
        top_indices = word_probs.argsort()[::-1][:top_n]
        return top_indices, word_probs, pi_k

    def attention_weights(self, theta_query, theta_hist):
        theta_query_t = torch.as_tensor(theta_query, dtype=torch.float32, device=self.device)
        theta_hist_t = torch.as_tensor(theta_hist, dtype=torch.float32, device=self.device)
        weights, _ = self._attention_forward(theta_query_t, theta_hist_t)
        return weights.detach().cpu().numpy()

    def get_topic_words(self, vocab, top_n=10):
        beta = self.beta.detach().cpu().numpy()
        print("\n=== Topic-Word Distributions (from LDA) ===")
        for k in range(self.K):
            top_ids = beta[k].argsort()[-top_n:][::-1]
            top_words = [vocab[i] for i in top_ids]
            print(f"  Topic {k:2d}: {', '.join(top_words)}")

    def save(self, path):
        payload = {
            "state_dict": self.state_dict(),
            "K": self.K,
            "citation_weight": self.citation_weight,
        }
        torch.save(payload, path)
