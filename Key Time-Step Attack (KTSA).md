# Key Time-Step Attack (KTSA)

## 1. Method Overview

`Key Time-Step Attack (KTSA)` is a time-series adversarial attack framework designed for sequence forecasting models such as `LSTM`, `BiLSTM`, and `BiLSTM + Attention`.

Unlike standard attacks such as `FGSM` or `BIM`, which perturb all time steps of the input sequence uniformly, `KTSA` first estimates the contribution of each time step to the final prediction and then applies perturbations only to the most influential time steps. The goal is to increase prediction error under a smaller effective perturbation range, thereby improving attack sparsity, concealment, and temporal targeting. In practice, this framework can be instantiated as `KTSA-FGSM` and `KTSA-BIM`.

For course-project use, the current method focuses on:

- multivariate time-series forecasting
- untargeted adversarial attacks on the input sequence
- key-time-step selection based on gradient saliency or attention weights

## 2. Problem Setting

Let the input sequence be:

`X = [x_1, x_2, ..., x_T] ∈ R^(T×D)`

where:

- `T` is the number of time steps
- `D` is the feature dimension at each time step
- `x_t ∈ R^D` is the feature vector at time step `t`

Let the forecasting model be:

`ŷ = f(X; θ)`

where:

- `f(·)` is the trained forecasting model
- `θ` denotes the model parameters
- `ŷ` is the predicted value

Let the ground-truth target be `y`, and let the loss function be:

`L(f(X; θ), y)`

The objective of the attacker is to construct an adversarial example `X_adv` such that:

- the perturbation magnitude remains bounded
- the prediction error is increased as much as possible

## 3. Time-Step Contribution Estimation

The core idea of `KTSA` is to compute an importance score for each time step and select only the highest-scoring time steps for perturbation.

Define a contribution score function:

`S(t; X, f) -> R`

which measures the influence of time step `t` on the final prediction result.

After computing all scores `{S(t)}_(t=1)^T`, select the top `k` or top `p%` time steps:

`K = TopK({S(t)}_(t=1)^T, k)`

or

`K = { t | S(t) belongs to the top p% of all time-step scores }`

In this project, a practical default choice is:

- `p = 10%`

That is, the top `10%` highest-scoring time steps are treated as key time steps.

Then construct a binary temporal mask:

`M ∈ {0,1}^(T×D)`

such that:

- `M_(t,d) = 1`, if `t ∈ K`
- `M_(t,d) = 0`, if `t ∉ K`

This mask ensures that perturbations are applied only at the selected key time steps.

## 4. Attack Definition

### 4.1 One-Step Attack Instance: KTSA-FGSM

Under the `KTSA` framework, a one-step attack instance can be constructed as a temporally masked variant of `FGSM`, denoted by `KTSA-FGSM`:

`X_adv = X + ε · M ⊙ sign(∇_X L(f(X; θ), y))`

where:

- `ε` is the perturbation budget
- `⊙` denotes element-wise multiplication
- `M` is the key-time-step mask
- `sign(·)` is the sign operator

This means that the perturbation direction still follows the input gradient, but perturbations are only retained on the selected key time steps.

### 4.2 Iterative Attack Instance: KTSA-BIM

Under the `KTSA` framework, an iterative attack instance can also be defined in the style of `BIM`, denoted by `KTSA-BIM`:

`X_adv^(n+1) = Π_B(X, ε) [ X_adv^(n) + α · M ⊙ sign(∇_X L(f(X_adv^(n); θ), y)) ]`

where:

- `α` is the step size
- `Π_B(X, ε)` projects the perturbed sample back into the allowed `ε`-bounded neighborhood of `X`

This iterative form is more flexible and usually stronger than the one-step version, but it is also more computationally expensive.

## 5. Key Time-Step Selection Strategies

The following two strategies are recommended as the main practical variants of `KTSA` in this project.

### 5.1 Gradient-Saliency-Based Selection

This strategy uses the input gradient of the loss with respect to each time step to estimate contribution.

First compute:

`G = ∇_X L(f(X; θ), y)`

where `G ∈ R^(T×D)`.

Then aggregate the gradient magnitude at each time step:

`S_grad(t) = Σ_(d=1)^D | ∂L / ∂X_(t,d) |`

or equivalently:

`S_grad(t) = || ∇_(x_t) L(f(X; θ), y) ||_1`

Interpretation:

- if `S_grad(t)` is large, a small change at time step `t` can cause a larger change in the loss
- therefore, time step `t` is more sensitive and more suitable for adversarial perturbation

Selection rule:

- compute `S_grad(t)` for all `t = 1, 2, ..., T`
- sort time steps by score in descending order
- select the top `10%` as the key time-step set `K`

Advantages:

- simple to implement
- model-agnostic
- directly reflects local adversarial sensitivity
- easy to visualize as a temporal saliency map

### 5.2 Attention-Weight-Based Selection

If the forecasting model contains a temporal attention mechanism, the attention weights can be used to estimate the relative importance of each time step.

Let the temporal attention weights be:

`A = [a_1, a_2, ..., a_T]`

where:

- `a_t >= 0`
- typically `Σ_(t=1)^T a_t = 1`

Define the contribution score as:

`S_attn(t) = a_t`

Selection rule:

- extract the temporal attention weights produced by the model
- sort all time steps by `a_t`
- select the top `10%` as the key time-step set `K`

Interpretation:

- a larger attention weight indicates that the model places more emphasis on that time step during prediction
- perturbing those time steps is more likely to disrupt the model's decision process

Advantages:

- highly interpretable
- naturally compatible with `BiLSTM + Attention`
- well aligned with the idea of attacking temporally critical positions

### 5.3 Optional Hybrid Variant

For a stronger variant, gradient saliency and attention weights can be combined:

`S_hybrid(t) = λ · Normalize(S_grad(t)) + (1 - λ) · Normalize(S_attn(t))`

where:

- `λ ∈ [0,1]` controls the balance between gradient sensitivity and model attention

This hybrid design is optional and can be treated as an extension rather than the minimum implementation target.

## 6. Perturbation Constraint

To ensure comparability with standard adversarial attacks, the attack instances under the `KTSA` framework should use a bounded perturbation budget.

Typical constraints include:

### 6.1 `L∞` Constraint

`||X_adv - X||_∞ <= ε`

This is the most common setting and is directly compatible with `FGSM` and `BIM`.

### 6.2 Temporal Sparsity Constraint

In addition to the magnitude bound, the `KTSA` framework explicitly constrains the perturbation to selected time steps only:

- only time steps in `K` may be changed
- all non-selected time steps remain unchanged

This temporal sparsity is one of the defining properties of the method.

## 7. Evaluation Metrics

For time-series regression and forecasting tasks, the following metrics are recommended.

### 7.1 Root Mean Squared Error (`RMSE`)

`RMSE = sqrt((1/N) Σ_(i=1)^N (ŷ_i - y_i)^2 )`

`RMSE` measures the overall prediction deviation and penalizes larger errors more strongly.

### 7.2 Mean Absolute Error (`MAE`)

`MAE = (1/N) Σ_(i=1)^N |ŷ_i - y_i|`

`MAE` reflects the average absolute prediction error and is less sensitive to outliers than `RMSE`.

Recommended evaluation protocol:

\- evaluate the model on clean inputs

\- evaluate the model under `FGSM`

\- evaluate the model under `BIM`

\- evaluate the model under `KTSA-FGSM` and `KTSA-BIM`

\- compare the degradation in `RMSE` and`MAE`



To highlight the characteristics of the `KTSA` framework and its attack instances, it is also recommended to report:

\- perturbation ratio: proportion of perturbed time steps

\- attack gain under equal `ε`

\- error increase per perturbed time step

## 8. Recommended Experimental Comparison

To validate the effectiveness of `KTSA`, the following comparisons are suggested:

1. `Clean` vs `FGSM` vs `BIM` vs `KTSA-FGSM` vs `KTSA-BIM`
2. `LSTM` vs `BiLSTM` vs `BiLSTM + Attention`
3. gradient-saliency selection vs attention-weight selection
4. different key-step ratios such as `5%`, `10%`, and `20%`

These comparisons help answer:

- whether key-time-step attacks are more effective than uniform attacks under the same perturbation budget
- whether temporally selective attacks are more concealed
- whether attention-based models expose more interpretable attack surfaces

## 9. Advantages of KTSA

Compared with standard full-sequence perturbation methods, the `KTSA` framework and its attack instances have the following expected advantages:

- stronger temporal targeting
- better interpretability
- sparser perturbations
- potentially higher concealment
- closer alignment with the inherent structure of sequence forecasting tasks

For sequence models, not all time steps contribute equally to the final output. By concentrating the adversarial budget on the most influential time steps, the `KTSA` framework is more suitable for time-series data than uniform perturbation strategies.

## 10. Limitations and Notes

The following issues should be made explicit when using the `KTSA` framework:

- if the contribution estimator is inaccurate, the selected key time steps may not be truly influential
- pure attention weights do not always guarantee causal importance
- sparse perturbations may be less effective than full-sequence perturbations on some datasets
- iterative variants increase computational overhead
- the effectiveness of the method may depend on the temporal dynamics of the dataset

Therefore, this framework should be presented as a time-series-oriented attack enhancement rather than as a universal replacement for all baseline attacks.

## 11. Implementable Method

1. input a sequence `X` into the trained model
2. compute loss `L(f(X; θ), y)`
3. backpropagate to obtain `∇_X L`
4. compute `S_grad(t)` for each time step
5. select the top `10%` time steps as `K`
6. build the temporal mask `M`
7. generate the adversarial example using the masked `FGSM` formula
8. evaluate forecasting performance on `X_adv`

This version is simple, explainable, and sufficiently complete for reproduction and extension in a course setting.
