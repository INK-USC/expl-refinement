import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from attributionpriors.pytorch_ops import EmbeddingInputAttributionPriorExplainer



class GradientExplainer:
    def __init__(self, model, configs, tokenizer):
        self.model = model
        self.configs = configs
        self.tokenizer = tokenizer
        self.batch_size = configs.train_batch_size
        # background dataset (baseline inputs / references) to compute expected gradients, require separate init
        self.bg_dataset = None
        self.explainer = None
        self.use_softmax = configs.gradshap_softmax

    def wrap_model(self, model, label, *extra_inputs, **extra_kw_inputs):
        outer = self
        class Wrapper(nn.Module):
            def forward(self, x):
                output = model(x, *extra_inputs, **extra_kw_inputs)
                if outer.use_softmax:
                    output = F.softmax(output, -1)
                return output[:, label].unsqueeze(-1)

            def get_embedding(self, x):
                return model.get_embedding(x)

        wrapped_model = Wrapper()
        return wrapped_model

    def init_bg_dataset(self, ref_dataset):
        self.bg_dataset = ref_dataset
        self.explainer = EmbeddingInputAttributionPriorExplainer(self.bg_dataset, 1, k=1)

    def compute_attribution_single(self, input_ids, input_mask, segment_ids, *shap_params, **shap_kw_params):
        input_ids, input_mask, segment_ids = input_ids.unsqueeze(0), input_mask.unsqueeze(0), segment_ids.unsqueeze(0)
        attribution = self.compute_attribution(input_ids, input_mask, segment_ids, *shap_params, **shap_kw_params)
        return attribution.squeeze(0)

    def compute_attribution(self, input_ids_batch, input_mask_batch, segment_ids_batch, *shap_params, **shap_kw_params):
        # note that segment ids come first
        wrapped_model_pos = self.wrap_model(self.model, 1, segment_ids_batch, input_mask_batch)
        attribution_pos = self.explainer.shap_values(wrapped_model_pos, input_ids_batch, *shap_params, **shap_kw_params)
        wrapped_model_neg = self.wrap_model(self.model, 0, segment_ids_batch, input_mask_batch)
        attribution_neg = self.explainer.shap_values(wrapped_model_neg, input_ids_batch, *shap_params, **shap_kw_params)
        attribution = torch.stack([attribution_neg, attribution_pos], 1)
        attribution = attribution.sum(-1) # sum over embedding dimensions
        return attribution

    def get_interaction_given_region_single(self, input_ids, input_mask, segment_ids, span_i: tuple, span_j: tuple):
        #input_ids, input_mask, segment_ids = input_ids.unsqueeze(0), input_mask.unsqueeze(0), segment_ids.unsqueeze(0)
        #  marginal contributions of i when j exists (exclude j for interpolation)
        mask_indices = [_ for _ in range(span_j[0], span_j[1] + 1)]
        attribution_with_j = self.compute_attribution_single(input_ids, input_mask, segment_ids, mask_indices)
        # marginal contributions of i when j does not exist

        input_mask_new = input_mask.clone()
        input_mask_new[span_j[0]:span_j[1] + 1] = 0

        attribution_wo_j = self.compute_attribution_single(input_ids, input_mask_new, segment_ids)

        interaction = attribution_with_j - attribution_wo_j # [L, T]
        interaction_ij = interaction[:, span_i[0]: span_i[1] + 1].sum(-1) # [L]
        return interaction_ij.unsqueeze(0) # keep consistent with soc output dim

    def get_phrase_importance(self, attribution, start_pos, stop_pos):
        score = attribution[:,start_pos: stop_pos + 1].sum(-1) # [L]
        return score.unsqueeze(0) # keep consistent with soc output dim

    def compute_expl_loss_with_advice(self, input_ids_batch, input_mask_batch, segment_ids_batch, label_ids_batch,
                                      importances_batch, interactions_batch, confidences_batch=None,
                                      do_backprop=False):
        importance_scores = []
        interaction_scores = []

        batch_size = input_ids_batch.size(0)
        for b in range(batch_size):
            input_ids, input_mask, segment_ids, label_ids, \
            importances, interactions = input_ids_batch[b], \
                                        input_mask_batch[b], \
                                        segment_ids_batch[b], \
                                        label_ids_batch[b], \
                                        importances_batch[b], \
                                        interactions_batch[b]

            if self.configs.confidence:
                confidences = confidences_batch[b]
            else:
                confidences = 1

            length = len(input_ids)

            attribution = self.compute_attribution_single(input_ids, input_mask, segment_ids)

            """
            Regularize attribution scores
            """
            if not self.configs.only_interaction:
                reg_loss = 0
                for label in range(importances.shape[0]):
                    start = -1

                    # print(importances[label, :])
                    for i in range(length):

                        if importances[label, i] == 0:
                            start = -1
                            continue

                        if start == -1:
                            start = i

                        if i == length - 1 or importances[label, i + 1] != importances[label, start]:
                            x_region = (start, i)

                            # print(x_region)

                            # compute attribution score using expected gradients
                            scores = self.get_phrase_importance(attribution, start, i)
                            #scores = self.algo.do_attribution(input_ids, input_mask, segment_ids, x_region, label_ids,
                            #                                  return_variable=True, have_target=True)

                            if importances[label, start] > 0:
                                target_score = 1
                            else:
                                assert importances[label, start] < 0, 'Importance extraction error'
                                target_score = 0

                            score = self.configs.reg_strength * confidences * ((scores[0, label] - target_score) ** 2)
                            reg_loss += score
                            importance_scores.append(score.item())
                            start = i + 1
                if do_backprop:
                    if torch.is_tensor(reg_loss):
                        reg_loss.backward()

            """
            Regularize interaction scores
            """
            if self.configs.reg_interaction or self.configs.only_interaction:
                for label in range(interactions.shape[0]):
                    i = 0
                    while i < length and interactions[label, i] != 0:
                        direction, st1, ed1, st2, ed2 = interactions[label, i:i + 5]
                        i += 5
                        pair = (min(st1, st2), max(ed1, ed2))
                        region_i = (st1, ed1)
                        region_j = (st2, ed2)

                        scores = self.get_interaction_given_region_single(input_ids, input_mask, segment_ids,
                                                                          region_i, region_j)
                        #scores = score_i_j - score_i - score_j

                        score = 0
                        target_score = ((direction + 1) / 2).float()
                        score += self.configs.reg_strength * confidences * ((scores[0, label] - target_score) ** 2)

                        if do_backprop:
                            score.backward()

                        interaction_scores.append(score.item())

        if importance_scores and interaction_scores:
            return sum(importance_scores) + sum(interaction_scores), \
                   len(importance_scores) + len(interaction_scores)
        else:
            return 0., 0
