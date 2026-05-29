import math
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

log = logging.getLogger(__name__)

TINY = 1e-10


class DSNTMBaseline(nn.Module):
    def __init__(self, K, V, T, emb_dim=300, t_hidden=800,
                 eta_hidden=200, eta_nlayers=3, n_heads=10,
                 delta=0.005, enc_drop=0.0,
                 citation=True, citation_weight=1.0,
                 min_citation_count=4,
                 pretrained_embeddings=None, train_embeddings=True,
                 frozen_beta=None,
                 device="cuda"):
        super().__init__()

        self.K = K
        self.V = V
        self.T = T
        self.emb_dim = emb_dim
        self.delta = delta
        self.use_citation = citation
        self.citation_weight = citation_weight
        self.min_citation_count = min_citation_count

        if device == "cuda" and not torch.cuda.is_available():
            device = "cpu"
        self.device = torch.device(device)

        # Frozen β mode
        self.use_frozen_beta = frozen_beta is not None
        if self.use_frozen_beta:
            self.register_buffer(
                "frozen_beta_matrix",
                torch.as_tensor(frozen_beta, dtype=torch.float32, device=torch.device(device))
            )
            log.info(f"  Using FROZEN LDA β: [{K}, {V}]")

        self.train_embeddings = train_embeddings
        if train_embeddings:
            self.rho = nn.Linear(emb_dim, V, bias=False)
        else:
            if pretrained_embeddings is not None:
                emb_tensor = torch.as_tensor(
                    pretrained_embeddings, dtype=torch.float32
                )
                self.rho = emb_tensor.to(self.device)
            else:
                raise ValueError(
                    "pretrained_embeddings required when train_embeddings=False"
                )

        self.mu_0_alpha = nn.Parameter(
            0.01 * torch.randn(1, K, emb_dim)
        )
        self.qkv_proj = nn.Linear(emb_dim, 3 * emb_dim)
        self.multi_attn = nn.MultiheadAttention(
            embed_dim=emb_dim, num_heads=n_heads, batch_first=True
        )
        self.layer_norm = nn.LayerNorm(emb_dim)
        self.mu_q_alpha = nn.Linear(emb_dim, emb_dim, bias=True)
        self.logsigma_q_alpha = nn.Linear(emb_dim, emb_dim, bias=True)

        
        self.q_theta = nn.Sequential(
            nn.Linear(V + K, t_hidden),
            nn.ReLU(),
            nn.Linear(t_hidden, t_hidden),
            nn.ReLU(),
        )
        self.mu_q_theta = nn.Linear(t_hidden, K, bias=True)
        self.logsigma_q_theta = nn.Linear(t_hidden, K, bias=True)
        self.t_drop = nn.Dropout(enc_drop)
        self.enc_drop = enc_drop

        self.q_eta_map = nn.Linear(V, eta_hidden)
        self.q_eta = nn.LSTM(
            eta_hidden, eta_hidden, eta_nlayers,
            dropout=0.0, batch_first=False
        )
        self.eta_nlayers = eta_nlayers
        self.eta_hidden = eta_hidden
        self.mu_q_eta = nn.Linear(eta_hidden + K, K, bias=True)
        self.logsigma_q_eta = nn.Linear(eta_hidden + K, K, bias=True)

        #Citation loss
        self.bce_loss = nn.BCELoss(reduction="sum")

        # Citation data 
        self.bow_articles = None       # dict {t: [D_t, V] tensor}
        self.dict_label_matrix = None  # list of [D_t, cum_D] tensors
        self.num_cum_articles = None   # cumulative doc counts

        log.info(
            f"DSNTMBaseline initialized: K={K}, V={V}, T={T}, "
            f"emb_dim={emb_dim}, citation={citation}, "
            f"citation_weight={citation_weight}, "
            f"frozen_beta={self.use_frozen_beta}"
        )

    def update_citation_data(self, citation_by_time, bow_by_time):
        self.bow_articles = {}
        for t in range(self.T):
            if t in bow_by_time and bow_by_time[t].shape[0] > 0:
                self.bow_articles[t] = bow_by_time[t]
            else:
                self.bow_articles[t] = torch.zeros(0, self.V, device=self.device)

        # Cumulative article counts
        self.num_cum_articles = [0] * (self.T + 1)
        for t in range(self.T):
            self.num_cum_articles[t + 1] = (
                self.num_cum_articles[t] + self.bow_articles[t].shape[0]
            )

        # Build label matrices: for each t, a [D_t, cum_D_up_to_t] matrix
        self.dict_label_matrix = [None] * self.T
        for t in range(1, self.T):
            D_t = self.bow_articles[t].shape[0]
            cum_D = self.num_cum_articles[t]
            if D_t == 0 or cum_D == 0:
                self.dict_label_matrix[t] = torch.zeros(
                    0, 0, device=self.device
                )
                continue

            mat_label = torch.zeros(D_t, cum_D, device=self.device)
            if t in citation_by_time:
                for s in range(t):
                    if s in citation_by_time[t]:
                        cit_mat = citation_by_time[t][s]
                        start = self.num_cum_articles[s]
                        end = self.num_cum_articles[s + 1]
                        if cit_mat.shape[0] == D_t and cit_mat.shape[1] == (end - start):
                            mat_label[:, start:end] = cit_mat

            self.dict_label_matrix[t] = mat_label

        log.info(
            f"Citation data loaded: cum_articles={self.num_cum_articles}"
        )

    @staticmethod
    def _reparameterize(mu, logvar):
        eps = torch.randn_like(mu)
        return mu + eps * torch.exp(0.5 * logvar)

    @staticmethod
    def _kl_divergence(q_mu, q_logsigma, p_mu=None, p_logsigma=None):
        if p_mu is not None and p_logsigma is not None:
            sigma_q_sq = torch.exp(2 * q_logsigma)
            sigma_p_sq = torch.exp(2 * p_logsigma)
            kl = (sigma_q_sq + (q_mu - p_mu) ** 2) / (sigma_p_sq + 1e-6)
            kl = kl - 1 + 2 * (p_logsigma - q_logsigma)
            kl = 0.5 * kl.sum(dim=-1)
        else:
            kl = -0.5 * (1 + 2 * q_logsigma - q_mu.pow(2) - torch.exp(2 * q_logsigma)).sum(dim=-1)
        return kl

    def _init_hidden(self):
        weight = next(self.parameters())
        return (
            weight.new_zeros(self.eta_nlayers, 1, self.eta_hidden),
            weight.new_zeros(self.eta_nlayers, 1, self.eta_hidden),
        )

    def _get_attention(self, predict=False):
        num_times = self.T + 1 if predict else self.T
        alphas = torch.empty(
            num_times, self.K, self.emb_dim, device=self.device
        )

        alpha = self.layer_norm(self.mu_0_alpha)
        alphas[0] = alpha.squeeze(0)
        attentions = {}

        cul_k = cul_v = None
        for t in range(1, num_times):
            qkv = self.qkv_proj(alpha)
            q_t, k_t, v_t = qkv.chunk(3, dim=-1)
            prev_alpha = alpha

            if t == 1:
                cul_k = k_t
                cul_v = v_t
            else:
                cul_k = torch.cat([cul_k, k_t], dim=1)
                cul_v = torch.cat([cul_v, v_t], dim=1)

            alpha, attn_w = self.multi_attn(q_t, cul_k, cul_v)
            attentions[t] = attn_w.squeeze(0)  # [K, K*t]
            alpha = self.layer_norm(alpha + prev_alpha)
            alphas[t] = alpha.squeeze(0)

        return alphas, attentions

    def _get_alpha(self, predict=False):
        outputs, attentions = self._get_attention(predict)
        num_times = outputs.shape[0]

        alphas = torch.empty(
            num_times, self.K, self.emb_dim, device=self.device
        )
        kl_alpha = []

        q_mu_0 = self.mu_q_alpha(outputs[0])
        q_ls_0 = self.logsigma_q_alpha(outputs[0])
        alpha_0 = self._reparameterize(q_mu_0, q_ls_0)
        alphas[0] = alpha_0

        p_mu_0 = torch.zeros(self.K, self.emb_dim, device=self.device)
        p_ls_0 = torch.zeros(self.K, self.emb_dim, device=self.device)
        kl_alpha.append(self._kl_divergence(q_mu_0, q_ls_0, p_mu_0, p_ls_0))

        p_ls_val = 0.5 * math.log(self.delta)

        for t in range(1, num_times):
            q_mu_t = self.mu_q_alpha(outputs[t])
            q_ls_t = self.logsigma_q_alpha(outputs[t])
            alpha_t = self._reparameterize(q_mu_t, q_ls_t)
            alphas[t] = alpha_t

            p_mu_t = alphas[t - 1].detach()
            p_ls_t = torch.full(
                (self.K, self.emb_dim), p_ls_val, device=self.device
            )
            kl_alpha.append(self._kl_divergence(q_mu_t, q_ls_t, p_mu_t, p_ls_t))

        kl_alpha_total = torch.stack(kl_alpha).sum()
        return alphas, attentions, kl_alpha_total

    def _get_beta(self, alpha):
        if self.use_frozen_beta:
            num_times = alpha.shape[0]
            return self.frozen_beta_matrix.unsqueeze(0).expand(num_times, -1, -1)

        leading = alpha.shape[:-1]
        flat = alpha.reshape(-1, self.emb_dim)
        if self.train_embeddings:
            logit = self.rho(flat)
        else:
            logit = flat @ self.rho.T
        logit = logit.reshape(*leading, -1)
        return F.softmax(logit, dim=-1)
    
    def _get_eta(self, rnn_inp):
        inp = self.q_eta_map(rnn_inp).unsqueeze(1)
        hidden = self._init_hidden()
        output, _ = self.q_eta(inp, hidden)
        output = output.squeeze(1)

        etas = torch.zeros(self.T, self.K, device=self.device)
        kl_eta = []

        inp_0 = torch.cat(
            [output[0], torch.zeros(self.K, device=self.device)]
        )
        mu_0 = self.mu_q_eta(inp_0)
        ls_0 = self.logsigma_q_eta(inp_0)
        etas[0] = self._reparameterize(mu_0, ls_0)
        kl_eta.append(
            self._kl_divergence(
                mu_0, ls_0,
                torch.zeros(self.K, device=self.device),
                torch.zeros(self.K, device=self.device),
            )
        )

        p_ls_val = 0.5 * math.log(self.delta)

        for t in range(1, self.T):
            inp_t = torch.cat([output[t], etas[t - 1]], dim=0)
            mu_t = self.mu_q_eta(inp_t)
            ls_t = self.logsigma_q_eta(inp_t)
            etas[t] = self._reparameterize(mu_t, ls_t)
            p_ls_t = torch.full(
                (self.K,), p_ls_val, device=self.device
            )
            kl_eta.append(self._kl_divergence(mu_t, ls_t, etas[t - 1], p_ls_t))

        return etas, torch.stack(kl_eta).sum()

    def _get_theta(self, eta_td, bows):
        inp = torch.cat([bows, eta_td], dim=1)
        q_theta = self.q_theta(inp)
        if self.enc_drop > 0:
            q_theta = self.t_drop(q_theta)
        mu = self.mu_q_theta(q_theta)
        logsigma = self.logsigma_q_theta(q_theta)
        z = self._reparameterize(mu, logsigma)
        theta = F.softmax(z, dim=-1)
        kl = self._kl_divergence(
            mu, logsigma, eta_td,
            torch.zeros(self.K, device=self.device),
        )
        return theta, kl

    def _get_phi(self, theta):
        tmp = theta.t()  # [K, D_t]
        divisor = tmp.sum(dim=1, keepdim=True).clamp(min=TINY)  # [K, 1]
        phi = tmp / divisor
        return phi


    def _get_citation_loss_at_t(self, theta, attention, phi, mat_label):
        paper_to_prev_topics = torch.mm(theta, attention)
        atten_paper = torch.mm(paper_to_prev_topics, phi)
        sum_citation = torch.sum(mat_label, dim=1)
        idx_citation = torch.where(
            sum_citation >= self.min_citation_count
        )[0]

        if len(idx_citation) == 0:
            return torch.tensor(0.0, device=self.device), 0

        labels = mat_label[idx_citation]
        preds = torch.clamp(atten_paper[idx_citation], min=TINY, max=1.0 - TINY)
        loss = self.bce_loss(preds, labels)

        return loss, int(labels.shape[0])

    def _get_citation_loss(self, attentions, etas):
        if self.bow_articles is None or self.dict_label_matrix is None:
            return torch.tensor(0.0, device=self.device), torch.tensor(0.0, device=self.device)

        kl_theta = torch.tensor(0.0, device=self.device)
        total_D = self.num_cum_articles[self.T]
        if total_D == 0:
            return torch.tensor(0.0, device=self.device), kl_theta

        phis = torch.zeros(
            self.T * self.K, total_D, device=self.device
        )
        dict_theta = {}
        dict_phi = {}

        for t in range(self.T):
            bow_t = self.bow_articles[t]
            if bow_t.shape[0] == 0:
                dict_theta[t] = torch.zeros(0, self.K, device=self.device)
                dict_phi[t] = torch.zeros(self.K, 0, device=self.device)
                continue

            sums = bow_t.sum(1, keepdim=True).clamp(min=1)
            norm_bow_t = bow_t / sums

            eta_td = etas[t].unsqueeze(0).expand(bow_t.shape[0], -1)
            theta_t, kl_t = self._get_theta(eta_td, norm_bow_t)
            dict_theta[t] = theta_t
            kl_theta = kl_theta + kl_t.sum()

            phi_t = self._get_phi(theta_t)  # [K, D_t]
            dict_phi[t] = phi_t

        for t in range(1, self.T):
            s = t - 1
            D_s = self.bow_articles[s].shape[0]
            if D_s == 0:
                continue
            k_start = s * self.K
            k_end = (s + 1) * self.K
            d_start = self.num_cum_articles[s]
            d_end = self.num_cum_articles[s + 1]
            phis[k_start:k_end, d_start:d_end] = dict_phi[s]

        total_loss = torch.tensor(0.0, device=self.device)
        for t in range(1, self.T):
            if t not in attentions:
                continue
            D_t = self.bow_articles[t].shape[0]
            if D_t == 0:
                continue

            label = self.dict_label_matrix[t]
            if label is None or label.numel() == 0:
                continue

            att = attentions[t]  # [K, K*t]
            cum_D = self.num_cum_articles[t]
            cum_phi = phis[:t * self.K, :cum_D].detach()

            theta_t = dict_theta[t]
            loss, sz = self._get_citation_loss_at_t(
                theta_t, att, cum_phi, label
            )
            total_loss = total_loss + loss

        return total_loss, kl_theta

    def forward(self, bows, norm_bows, time_ids, rnn_inp, num_docs):
        bsz = bows.shape[0]
        coeff = num_docs / bsz

        alpha, attentions, kl_alpha = self._get_alpha()
        eta, kl_eta = self._get_eta(rnn_inp)

        eta_td = eta[time_ids.long()]
        theta, kl_theta = self._get_theta(eta_td, norm_bows)
        kl_theta = kl_theta.sum() * coeff

        beta = self._get_beta(alpha)
        beta_td = beta[time_ids.long()]

        loglik = torch.bmm(theta.unsqueeze(1), beta_td).squeeze(1)
        loglik = torch.log(loglik + TINY)
        nll = -(loglik * bows).sum(-1).sum() * coeff
        cit_loss = torch.tensor(0.0, device=self.device)
        if self.use_citation and self.citation_weight > 0:
            cit_loss_raw, kl_theta_cit = self._get_citation_loss(
                attentions, eta
            )
            cit_loss = self.citation_weight * cit_loss_raw
            if cit_loss_raw.item() > 0:
                kl_theta = kl_theta_cit

        nelbo = nll + kl_eta + kl_theta + kl_alpha + cit_loss
        return nelbo, nll, kl_eta, kl_theta, kl_alpha, cit_loss

    @torch.no_grad()
    def predict_next_timestep_beta(self):
        self.eval()
        alpha = self._get_alpha_eval(predict=True)
        beta = self._get_beta(alpha)
        return beta[-1].cpu().numpy(), alpha[-1].cpu().numpy()

    @torch.no_grad()
    def predict_word_distribution(self, rnn_inp):
        self.eval()
        eta, _ = self._get_eta(rnn_inp)
        pi_k = F.softmax(eta[-1], dim=0).cpu().numpy()

        if self.use_frozen_beta:
            beta_pred = self.frozen_beta_matrix.cpu().numpy()  # [K, V]
        else:
            beta_pred, _ = self.predict_next_timestep_beta()   # [K, V]

        pi_k_t = torch.as_tensor(pi_k, device=self.device)
        beta_pred_t = torch.as_tensor(beta_pred, device=self.device)
        word_probs = (pi_k_t @ beta_pred_t).cpu().numpy()
        word_probs /= word_probs.sum() + 1e-10
        return word_probs, pi_k

    @torch.no_grad()
    def get_beta_all_times(self):
        self.eval()
        alpha = self._get_alpha_eval(predict=False)
        beta = self._get_beta(alpha)
        return beta.cpu().numpy()

    def _get_alpha_eval(self, predict=False):
        outputs, _ = self._get_attention(predict)
        num_times = outputs.shape[0]
        alphas = torch.empty(
            num_times, self.K, self.emb_dim, device=self.device
        )
        for t in range(num_times):
            alphas[t] = self.mu_q_alpha(outputs[t])
        return alphas

    @torch.no_grad()
    def compute_perplexity(self, tokens_by_time, rnn_inp, bow_norm=True):
        self.eval()
        alpha = self._get_alpha_eval(predict=False)
        beta = self._get_beta(alpha)
        eta, _ = self._get_eta(rnn_inp)

        total_loss = 0.0
        n_batches = 0

        for t in range(self.T):
            if t not in tokens_by_time or tokens_by_time[t].shape[0] == 0:
                continue
            bows = tokens_by_time[t]
            sums = bows.sum(1, keepdim=True).clamp(min=1)
            norm_bows = bows / sums

            eta_td = eta[t].unsqueeze(0).expand(bows.shape[0], -1)
            inp = torch.cat([norm_bows, eta_td], dim=1)
            q_theta = self.q_theta(inp)
            mu_theta = self.mu_q_theta(q_theta)
            theta = F.softmax(mu_theta, dim=-1)

            beta_t = beta[t].unsqueeze(0).expand(bows.shape[0], -1, -1)
            loglik = (theta.unsqueeze(2) * beta_t).sum(1)
            loglik = torch.log(loglik + TINY)
            nll = -(loglik * bows).sum(-1)
            loss = (nll / sums.squeeze()).mean()
            total_loss += loss.item()
            n_batches += 1

        if n_batches == 0:
            return float("inf")
        return round(math.exp(total_loss / n_batches), 1)

    @torch.no_grad()
    def get_topic_diversity(self, top_n=25):
        self.eval()
        beta_all = self.get_beta_all_times()
        diversities = []
        for t in range(self.T):
            unique = set()
            for k in range(self.K):
                top_ids = beta_all[t, k].argsort()[-top_n:]
                unique.update(top_ids.tolist())
            diversities.append(len(unique) / (self.K * top_n))
        return float(np.mean(diversities))

    @torch.no_grad()
    def get_top_words_per_topic(self, vocab, t, top_n=10):
        beta = self.get_beta_all_times()
        results = []
        for k in range(self.K):
            ids = beta[t, k].argsort()[-top_n:][::-1]
            results.append([vocab[i] for i in ids])
        return results
