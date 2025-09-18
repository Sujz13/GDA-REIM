import sys
import torch
from torch import nn
import torch.nn.functional as F
from recbole.model.abstract_recommender import SequentialRecommender
from einops import rearrange
import math

def gelu(x):
    return (
        0.5
        * x
        * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
    )

def lambda_init(lambda_init_value, depth):
    return lambda_init_value - 0.6 * math.exp(-0.3 * (depth - 1))

class GDAREIM(SequentialRecommender):
    def __init__(self, config, dataset):
        super(GDAREIM, self).__init__(config, dataset)
        self.gamma = config["gamma"]
        self.margin = config["margin"]
        self.lambda_init_value = config["lambda_init_value"]
        self.n_layers = config["n_layers"]
        self.n_heads = config["n_heads"]


        self.hidden_size = config["hidden_size"]
        self.inner_size = config[
            "inner_size"
        ]

        self.hidden_dropout_prob = config["hidden_dropout_prob"]
        self.attn_dropout_prob = config["hidden_dropout_prob"]

        self.hidden_act = config["hidden_act"]
        self.layer_norm_eps = config["layer_norm_eps"]
        self.initializer_range = config["initializer_range"]
        self.loss_type = config["loss_type"]
        self.n_facet_all = config["n_facet_all"]
        self.n_facet = config["n_facet"]
        self.n_facet_window = config["n_facet_window"]
        self.n_facet_hidden = min(
            config["n_facet_hidden"], config["n_layers"]
        )
        self.n_facet_MLP = config["n_facet_MLP"]
        self.n_facet_context = config["n_facet_context"]
        self.n_facet_reranker = config[
            "n_facet_reranker"
        ]
        self.n_facet_emb = config["n_facet_emb"]
        self.weight_mode = config["weight_mode"]
        self.context_norm = config["context_norm"]
        self.post_remove_context = config["post_remove_context"]
        self.partition_merging_mode = config["partition_merging_mode"]
        self.reranker_merging_mode = config["reranker_merging_mode"]
        self.reranker_CAN_NUM = [
            int(x) for x in str(config["reranker_CAN_NUM"]).split(",")
        ]
        self.candidates_from_previous_reranker = True
        if self.weight_mode == "max_logits":
            self.n_facet_effective = 1
        else:
            self.n_facet_effective = self.n_facet

        assert (
            self.n_facet
            + self.n_facet_context
            + self.n_facet_reranker * len(self.reranker_CAN_NUM)
            + self.n_facet_emb
            == self.n_facet_all
        )
        assert self.n_facet_emb == 0 or self.n_facet_emb == 2
        assert self.n_facet_MLP <= 0
        assert self.n_facet_window <= 0
        self.n_facet_window = -self.n_facet_window
        self.n_facet_MLP = -self.n_facet_MLP
        self.softmax_nonlinear = "None"
        self.use_proj_bias = config["use_proj_bias"]
        hidden_state_input_ratio = 1 + self.n_facet_MLP
        self.MLP_linear = nn.Linear(
            self.hidden_size * (self.n_facet_hidden * (self.n_facet_window + 1)),
            self.hidden_size * self.n_facet_MLP,
        )
        total_lin_dim = self.hidden_size * hidden_state_input_ratio
        self.project_arr = nn.ModuleList(
            [
                nn.Linear(total_lin_dim, self.hidden_size, bias=self.use_proj_bias)
                for i in range(self.n_facet_all)
            ]
        )

        self.project_emb = nn.Linear(
            self.hidden_size, self.hidden_size, bias=self.use_proj_bias
        )
        if len(self.weight_mode) > 0:
            self.weight_facet_decoder = nn.Linear(
                self.hidden_size * hidden_state_input_ratio, self.n_facet_effective
            )
            self.weight_global = nn.Parameter(torch.ones(self.n_facet_effective))
        self.output_probs = True
        self.item_embedding = nn.Embedding(
            self.n_items, self.hidden_size, padding_idx=0
        )
        self.position_embedding = nn.Embedding(self.max_seq_length, self.hidden_size)
        self.trm_encoder = TransformerEncoder(
            lambda_init_value = self.lambda_init_value,
            n_layers=self.n_layers,
            n_heads=self.n_heads,
            hidden_size=self.hidden_size,
            inner_size=self.inner_size,
            hidden_dropout_prob=self.hidden_dropout_prob,
            attn_dropout_prob=self.attn_dropout_prob,
            hidden_act=self.hidden_act,
            layer_norm_eps=self.layer_norm_eps,
        )
        self.gated_dfattn = GatedDiffAttention(d_model=self.hidden_size, expand=2, d_conv=4,
                           lambda_init_value=self.lambda_init_value,
                                           attn_layers=self.n_layers, num_heads=self.n_heads,
                                           hidden_dropout_prob=self.hidden_dropout_prob,
                                           attn_dropout_prob=self.attn_dropout_prob
                                           )
        self.LayerNorm = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)
        self.dropout = nn.Dropout(self.hidden_dropout_prob)

        if self.loss_type == "BPR":
            print("current softmax-cpr code does not support BPR loss")
            sys.exit(0)
        elif self.loss_type == "CE":
            self.loss_fct = nn.NLLLoss(
                reduction="none", ignore_index=0
            )
        else:
            raise NotImplementedError("Make sure 'loss_type' in ['BPR', 'CE']!")

        self.apply(self._init_weights)

        small_value = 0.0001

    def get_facet_emb(self, input_emb, i):
        return self.project_arr[i](input_emb)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            pass
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def forward(self, item_seq, item_seq_len):
        position_ids = torch.arange(
            item_seq.size(1), dtype=torch.long, device=item_seq.device
        )
        position_ids = position_ids.unsqueeze(0).expand_as(item_seq)
        position_embedding = self.position_embedding(position_ids)

        item_emb = self.item_embedding(item_seq)
        input_emb = item_emb + position_embedding
        input_emb = self.LayerNorm(input_emb)
        input_emb = self.dropout(input_emb)

        trm_output = self.gated_dfattn(input_emb)
        return trm_output

    def calculate_loss_prob(self, interaction, only_compute_prob=False):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        pos_items = interaction[self.ITEM_ID]

        all_hidden_states = self.forward(item_seq, item_seq_len)


        if self.loss_type != "CE":
            print(
                "current softmax-cpr code does not support BPR or the losses other than cross entropy"
            )
            sys.exit(0)
        else:
            test_item_emb = self.item_embedding.weight
            device = all_hidden_states[0].device

            hidden_emb_arr = []
            for i in range(self.n_facet_hidden):
                hidden_states = all_hidden_states[
                    -(i + 1)
                ]
                device = hidden_states.device
                hidden_emb_arr.append(hidden_states)
                for j in range(self.n_facet_window):
                    (
                        bsz,
                        seq_len,
                        hidden_size,
                    ) = (
                        hidden_states.size()
                    )
                    if j + 1 < hidden_states.size(1):
                        shifted_hidden = torch.cat(
                            (
                                torch.zeros((bsz, (j + 1), hidden_size), device=device),
                                hidden_states[:, : -(j + 1), :],
                            ),
                            dim=1,
                        )
                    else:
                        shifted_hidden = torch.zeros(
                            (bsz, hidden_states.size(1), hidden_size), device=device
                        )
                    hidden_emb_arr.append(shifted_hidden)

            if self.n_facet_MLP > 0:
                stacked_hidden_emb_raw_arr = torch.cat(
                    hidden_emb_arr, dim=-1
                )
                hidden_emb_MLP = self.MLP_linear(
                    stacked_hidden_emb_raw_arr
                )
                stacked_hidden_emb_arr_raw = torch.cat(
                    [hidden_emb_arr[0], gelu(hidden_emb_MLP)], dim=-1
                )
            else:
                stacked_hidden_emb_arr_raw = hidden_emb_arr[0]

            stacked_hidden_emb_arr = gather_indexes(stacked_hidden_emb_arr_raw, item_seq_len - 1).unsqueeze(dim=1)
            projected_emb_arr = []
            facet_lm_logits_arr = []
            facet_lm_logits_real_arr = []

            rereanker_candidate_token_ids_arr = []
            for i in range(self.n_facet):
                projected_emb = self.get_facet_emb(
                    stacked_hidden_emb_arr, i
                )
                projected_emb_arr.append(projected_emb)
                lm_logits = F.linear(projected_emb, self.item_embedding.weight, None)
                facet_lm_logits_arr.append(lm_logits)
                if (
                    i < self.n_facet_reranker
                    and not self.candidates_from_previous_reranker
                ):
                    candidate_token_ids = []
                    for j in range(len(self.reranker_CAN_NUM)):
                        _, candidate_token_ids_ = torch.topk(
                            lm_logits, self.reranker_CAN_NUM[j]
                        )
                        candidate_token_ids.append(candidate_token_ids_)
                    rereanker_candidate_token_ids_arr.append(candidate_token_ids)

            for i in range(self.n_facet_reranker):
                for j in range(len(self.reranker_CAN_NUM)):
                    projected_emb = self.get_facet_emb(
                        stacked_hidden_emb_arr,
                        self.n_facet + i * len(self.reranker_CAN_NUM) + j,
                    )
                    projected_emb_arr.append(projected_emb)

            for i in range(self.n_facet_context):
                projected_emb = self.get_facet_emb(
                    stacked_hidden_emb_arr,
                    self.n_facet
                    + self.n_facet_reranker * len(self.reranker_CAN_NUM)
                    + i,
                )
                projected_emb_arr.append(projected_emb)

            for i in range(self.n_facet_emb):
                projected_emb = self.get_facet_emb(
                    stacked_hidden_emb_arr_raw,
                    self.n_facet
                    + self.n_facet_context
                    + self.n_facet_reranker * len(self.reranker_CAN_NUM)
                    + i,
                )
                projected_emb_arr.append(projected_emb)

            for i in range(self.n_facet_reranker):
                bsz, seq_len, hidden_size = projected_emb_arr[i].size()
                for j in range(len(self.reranker_CAN_NUM)):
                    if self.candidates_from_previous_reranker:
                        _, candidate_token_ids = torch.topk(
                            facet_lm_logits_arr[i], self.reranker_CAN_NUM[j]
                        )
                    else:
                        candidate_token_ids = rereanker_candidate_token_ids_arr[i][j]
                    logit_hidden_reranker_topn = (
                        projected_emb_arr[
                            self.n_facet + i * len(self.reranker_CAN_NUM) + j
                        ]
                        .unsqueeze(dim=2)
                        .expand(bsz, seq_len, self.reranker_CAN_NUM[j], hidden_size)
                        * self.item_embedding.weight[candidate_token_ids, :]
                    ).sum(
                        dim=-1
                    )
                    if self.reranker_merging_mode == "add":
                        facet_lm_logits_arr[i].scatter_add_(
                            2, candidate_token_ids, logit_hidden_reranker_topn
                        )
                    else:
                        facet_lm_logits_arr[i].scatter_(
                            2, candidate_token_ids, logit_hidden_reranker_topn
                        )

            for i in range(self.n_facet_context):
                bsz, seq_len_1, hidden_size = projected_emb_arr[i].size()
                bsz, seq_len_2 = item_seq.size()
                logit_hidden_context = (
                    projected_emb_arr[
                        self.n_facet
                        + self.n_facet_reranker * len(self.reranker_CAN_NUM)
                        + i
                    ]
                    .unsqueeze(dim=2)
                    .expand(-1, -1, seq_len_2, -1)
                    * self.item_embedding.weight[item_seq, :]
                    .unsqueeze(dim=1)
                    .expand(-1, seq_len_1, -1, -1)
                ).sum(dim=-1)
                logit_hidden_pointer = 0
                if self.n_facet_emb == 2:
                    pointer = gather_indexes(projected_emb_arr[-2], item_seq_len - 1)
                    logit_hidden_pointer = (
                        pointer.unsqueeze(dim=1).unsqueeze(dim=1).expand(-1, seq_len_1, seq_len_2, -1)
                        * projected_emb_arr[-1]
                        .unsqueeze(dim=1)
                        .expand(-1, seq_len_1, -1, -1)
                    ).sum(dim=-1)
                item_seq_expand = item_seq.unsqueeze(dim=1).expand(-1, seq_len_1, -1)
                only_new_logits = torch.zeros_like(facet_lm_logits_arr[i])
                if self.context_norm:
                    only_new_logits.scatter_add_(
                        dim=2,
                        index=item_seq_expand,
                        src=logit_hidden_context + logit_hidden_pointer,
                    )
                    item_count = torch.zeros_like(only_new_logits) + 1e-15
                    item_count.scatter_add_(
                        dim=2,
                        index=item_seq_expand,
                        src=torch.ones_like(item_seq_expand).to(dtype=item_count.dtype),
                    )
                    only_new_logits = only_new_logits / item_count
                else:
                    only_new_logits.scatter_add_(
                        dim=2, index=item_seq_expand, src=logit_hidden_context
                    )
                    item_count = torch.zeros_like(only_new_logits) + 1e-15
                    item_count.scatter_add_(
                        dim=2,
                        index=item_seq_expand,
                        src=torch.ones_like(item_seq_expand).to(dtype=item_count.dtype),
                    )
                    only_new_logits = only_new_logits / item_count
                    only_new_logits.scatter_add_(
                        dim=2, index=item_seq_expand, src=logit_hidden_pointer
                    )


                if self.partition_merging_mode == "replace":
                    facet_lm_logits_arr[i].scatter_(
                        dim=2,
                        index=item_seq_expand,
                        src=torch.zeros_like(item_seq_expand).to(
                            dtype=facet_lm_logits_arr[i].dtype
                        ),
                    )
                facet_lm_logits_arr[i] = facet_lm_logits_arr[i] + only_new_logits


            weight = None
            if self.weight_mode == "dynamic":
                weight = self.weight_facet_decoder(stacked_hidden_emb_arr).softmax(
                    dim=-1
                )
            elif self.weight_mode == "static":
                weight = self.weight_global.softmax(
                    dim=-1
                )
            elif self.weight_mode == "max_logits":
                stacked_facet_lm_logits = torch.stack(facet_lm_logits_arr, dim=0)
                facet_lm_logits_arr = [stacked_facet_lm_logits.amax(dim=0)]

            prediction_prob = 0
            copy_loss = 0
            for i in range(self.n_facet_effective):
                facet_lm_logits = facet_lm_logits_arr[i]
                copy_loss += CP_copy_loss(facet_lm_logits, self.item_embedding.weight[pos_items], pos_items, item_seq,
                                          item_seq_len, self.item_embedding.weight,
                                          margin1=self.margin, margin2=self.margin) / self.n_facet_effective

                if self.softmax_nonlinear == "sigsoftmax":
                    facet_lm_logits_sig = torch.exp(
                        facet_lm_logits - facet_lm_logits.max(dim=-1, keepdim=True)[0]
                    ) * (1e-20 + torch.sigmoid(facet_lm_logits))
                    facet_lm_logits_softmax = (
                        facet_lm_logits_sig
                        / facet_lm_logits_sig.sum(dim=-1, keepdim=True)
                    )
                elif self.softmax_nonlinear == "None":
                    facet_lm_logits_softmax = facet_lm_logits.softmax(
                        dim=-1
                    )
                if self.weight_mode == "dynamic":
                    prediction_prob += facet_lm_logits_softmax * weight[
                        :, :, i
                    ].unsqueeze(-1)
                elif self.weight_mode == "static":
                    prediction_prob += facet_lm_logits_softmax * weight[i]
                else:
                    prediction_prob += (
                        facet_lm_logits_softmax / self.n_facet_effective
                    )
            if not only_compute_prob:
                inp = torch.log(prediction_prob.view(-1, self.n_items) + 1e-8)
                pos_items = interaction[self.POS_ITEM_ID]
                loss_raw = self.loss_fct(inp, pos_items.view(-1))
                loss = loss_raw.mean()
            else:
                loss = None
            return loss,  copy_loss, prediction_prob.squeeze(dim=1)

    def calculate_loss(self, interaction):
        loss, copy_loss, prediction_prob = self.calculate_loss_prob(interaction)
        return loss + self.gamma * copy_loss

    def predict(self, interaction):
        print(
            "Current softmax cpr code does not support negative sampling in an efficient way just like RepeatNet.",
            file=sys.stderr,
        )
        loss, copy_loss, prediction_prob = self.calculate_loss_prob(
            interaction, only_compute_prob=True
        )
        if self.post_remove_context:
            item_seq = interaction[self.ITEM_SEQ]
            prediction_prob.scatter_(1, item_seq, 0)
        test_item = interaction[self.ITEM_ID]
        prediction_prob = prediction_prob.unsqueeze(-1)
        scores = self.gather_indexes(prediction_prob, test_item).squeeze(-1)

        return scores

    def full_sort_predict(self, interaction):
        loss, copy_loss, prediction_prob = self.calculate_loss_prob(interaction)
        if self.post_remove_context:
            item_seq = interaction[self.ITEM_SEQ]
            prediction_prob.scatter_(1, item_seq, 0)
        return prediction_prob


class MultiHeadDiffAttention(nn.Module):
    def __init__(self, lambda_init_value, n_embd, n_head, hidden_dropout, attn_dropout, layer_idx):
        super().__init__()
        assert n_embd % n_head == 0
        self.n_head = n_head
        self.head_size = n_embd // n_head
        self.lambda_init = lambda_init(lambda_init_value, layer_idx)

        self.q1_proj = nn.Linear(n_embd, n_embd, bias=False)
        self.q2_proj = nn.Linear(n_embd, n_embd, bias=False)
        self.k1_proj = nn.Linear(n_embd, n_embd, bias=False)
        self.k2_proj = nn.Linear(n_embd, n_embd, bias=False)
        self.v_proj = nn.Linear(n_embd, 2 * n_embd, bias=False)

        self.c_proj = nn.Linear(2 * n_embd, n_embd, bias=False)
        self.attn_dropout = nn.Dropout(attn_dropout)
        self.resid_dropout = nn.Dropout(hidden_dropout)
        self.subln = nn.LayerNorm(2 * self.head_size, elementwise_affine=False)

        self.lambda_q1 = nn.Parameter(torch.randn(n_head, self.head_size) * 0.1)
        self.lambda_k1 = nn.Parameter(torch.randn(n_head, self.head_size) * 0.1)
        self.lambda_q2 = nn.Parameter(torch.randn(n_head, self.head_size) * 0.1)
        self.lambda_k2 = nn.Parameter(torch.randn(n_head, self.head_size) * 0.1)

    def forward(self, x):
        B, T, C = x.shape

        q1 = self.q1_proj(x).view(B, T, self.n_head, self.head_size).transpose(1, 2)
        q2 = self.q2_proj(x).view(B, T, self.n_head, self.head_size).transpose(1, 2)
        k1 = self.k1_proj(x).view(B, T, self.n_head, self.head_size).transpose(1, 2)
        k2 = self.k2_proj(x).view(B, T, self.n_head, self.head_size).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_head, 2 * self.head_size).transpose(1, 2)

        scale = 1.0 / math.sqrt(self.head_size)
        att1 = torch.matmul(q1, k1.transpose(-2, -1)) * scale
        att2 = torch.matmul(q2, k2.transpose(-2, -1)) * scale

        attn_mask = torch.tril(torch.ones(T, T, device=x.device)).unsqueeze(0).unsqueeze(0)
        att1 = att1.masked_fill(attn_mask == 0, float('-inf'))
        att2 = att2.masked_fill(attn_mask == 0, float('-inf'))

        att1 = F.softmax(att1, dim=-1)
        att2 = F.softmax(att2, dim=-1)

        lambda_1 = torch.exp(torch.sum(self.lambda_q1 * self.lambda_k1, dim=-1)).unsqueeze(-1).unsqueeze(-1)
        lambda_2 = torch.exp(torch.sum(self.lambda_q2 * self.lambda_k2, dim=-1)).unsqueeze(-1).unsqueeze(-1)
        lambda_full = lambda_1 - lambda_2 + self.lambda_init

        att = att1 - lambda_full * att2
        att = self.attn_dropout(att)

        y = torch.matmul(att, v)
        y = self.subln(y)
        y = y * (1 - self.lambda_init)

        y = y.transpose(1, 2).contiguous().view(B, T, 2 * C)
        y = self.resid_dropout(self.c_proj(y))
        return y


class MLP(nn.Module):
    def __init__(self, n_embd, dropout):
        super().__init__()
        self.c_fc = nn.Linear(n_embd, 4 * n_embd, bias=False)
        self.c_proj = nn.Linear(4 * n_embd, n_embd, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class Block(nn.Module):
    def __init__(self, lambda_init_value, n_embd, n_head, attn_dropout_prob, hidden_dropout_prob, attention_class, layer_idx):
        super().__init__()
        self.ln_1 = nn.LayerNorm(n_embd)
        self.attn = attention_class(lambda_init_value, n_embd, n_head, hidden_dropout_prob, attn_dropout_prob, layer_idx)

    def forward(self, x):
        x = self.ln_1(x + self.attn(x))
        return x

class TransformerEncoder(nn.Module):
    def __init__(
        self,
        lambda_init_value,
        n_layers=2,
        n_heads=2,
        hidden_size=64,
        inner_size=256,
        hidden_dropout_prob=0.5,
        attn_dropout_prob=0.5,
        hidden_act="gelu",
        layer_norm_eps=1e-12,
    ):
        super(TransformerEncoder, self).__init__()
        self.layer = nn.ModuleList([Block(lambda_init_value, hidden_size, n_heads, attn_dropout_prob, hidden_dropout_prob, MultiHeadDiffAttention, layer_idx=i + 1) for i in range(n_layers)])

    def forward(self, hidden_states,  output_all_encoded_layers=True):
        all_encoder_layers = []
        for layer_module in self.layer:
            hidden_states = layer_module(hidden_states)
            if output_all_encoded_layers:
                all_encoder_layers.append(hidden_states)
        if not output_all_encoded_layers:
            all_encoder_layers.append(hidden_states)
        return all_encoder_layers[-1]

class GatedDiffAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        expand: int = 2,
        d_conv: int = 4,
        lambda_init_value: float = 0.5,
        attn_layers: int = 2,
        num_heads: int = 4,
        conv_bias: bool = True,
        hidden_dropout_prob: float = 0.1,
        attn_dropout_prob: float = 0.1
    ):
        super().__init__()
        self.d_model = d_model
        self.expand = expand
        self.d_inner = expand * d_model

        self.in_proj = nn.Linear(d_model, 2 * self.d_inner, bias=False)
        self.conv1d = nn.Conv1d(self.d_inner, self.d_inner, kernel_size=d_conv,
                                 padding=d_conv-1, groups=self.d_inner, bias=conv_bias)
        self.act = nn.SiLU()
        self.attn_encoder = TransformerEncoder(
            lambda_init_value=lambda_init_value,
            n_layers=attn_layers,
            n_heads=num_heads,
            hidden_size=self.d_inner,
            hidden_dropout_prob=hidden_dropout_prob,
            attn_dropout_prob=attn_dropout_prob
        )

        self.dt_in = nn.Linear(self.d_inner, self.d_inner, bias=False)
        self.dt_proj = nn.Linear(self.d_inner, self.d_inner, bias=True)
        dt = torch.exp(torch.rand(self.d_inner) * (math.log(0.1)-math.log(0.001)) + math.log(0.001))
        inv_sp = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_sp)
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

    def forward(self, hidden_states):
        xz = self.in_proj(hidden_states)
        x, z = xz.chunk(2, dim=-1)
        x_conv = rearrange(x, 'b l d -> b d l')
        x_conv = self.act(self.conv1d(x_conv)[..., :hidden_states.size(1)])
        x = rearrange(x_conv, 'b d l -> b l d')
        x_attn = self.attn_encoder(x)
        dt_raw = self.dt_in(x_attn)
        dt = F.linear(dt_raw, self.dt_proj.weight) + self.dt_proj.bias
        dt = F.softplus(dt)
        gated = x_attn * F.silu(z) * dt
        out = self.out_proj(gated)
        return [out]


def CP_copy_loss(logits, pos_item_emb, pos_item_ids, item_seq, item_seq_len, all_item_embeddings,
                          margin1=0.1, margin2=0.1):
    logits = logits.squeeze(dim=1)

    history_emb = logits.gather(dim=1, index=item_seq)
    valid_mask = (item_seq != 0)
    history_emb = history_emb * valid_mask
    history_emb = history_emb.sum(-1) / item_seq_len

    pos_item_emb = logits.gather(dim=1, index=pos_item_ids.unsqueeze(1))

    loss1 = F.relu(margin1 + history_emb.unsqueeze(dim=1) - pos_item_emb)
    loss1 = loss1.mean()

    num_items = logits.size(1)
    rd = torch.randint(low=1, high=num_items - 2, size=(item_seq.size(0), item_seq.size(1)), device=logits.device)
    pos_item_ids_expanded = pos_item_ids.unsqueeze(1).expand(item_seq.size(0), item_seq.size(1))
    neg_ids = rd + (rd >= pos_item_ids_expanded).long()
    neg_emb = logits.gather(dim=1, index=neg_ids)
    s_neg = torch.mean(neg_emb, dim=-1)

    loss2 = F.relu(margin2 + s_neg - history_emb)
    loss2 = loss2.mean()

    total_loss = loss1 + loss2
    return total_loss / 2

def gather_indexes(output, gather_index):
    gather_index = gather_index.view(-1, 1, 1).expand(-1, -1, output.shape[-1])
    output_tensor = output.gather(dim=1, index=gather_index)
    return output_tensor.squeeze(1)