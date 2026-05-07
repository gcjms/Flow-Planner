# CMU 10-799 Diffusion & Flow Matching 课程讲义解释

这份文档是对 `CMU_10-799_Spring_2026_Slides` 里 lecture PDF 的中文解释版。风格按“核心一句话 + 主线 + 关键概念直觉”整理，方便直接读。

> 当前已完整整理：Lecture 0, 1, 2, 4, 5, 6, 7, 10。  
> Lecture 9, 12, 13 的 PDF 本地文本解析失败，先保留待补章节，后续如果能换一种提取方式或手动打开逐页读，再继续补全。

## Lecture 0: Overview

核心一句话：

**这门课要教你理解、实现、训练、改进 diffusion 和 flow matching 这类生成模型，它们是近几年图像/视频生成爆发背后的核心技术。**

Lecture 0 基本不是技术课，而是课程导览。开头用 2025 年各种 AI meme、Sora、Veo、图像生成效果来引入：现在生成式 AI 很火，而这些进展背后有一类共同技术，就是这门课要讲的 `Diffusion & Flow Matching`。

课程目标不是只看公式，而是让你做到：

- 理解算法直觉和数学。
- 用 Python 实现基础 diffusion / flow matching。
- 在 GPU 上训练一个图像生成模型。
- 学会改进 vanilla model。
- 理解怎么扩展到离散数据。
- 最后能展示自己的实验和想法。

这节课还提出了“好的图像生成模型”的三个评价维度。

**Fidelity，真实感**：生成图像像不像真的。比如人有没有 6 根手指、画面有没有奇怪 artifact、颜色是否过曝/欠曝。

**Controllability，可控性**：你能不能控制生成内容。比如用文字描述、用参考图约束、生成“我的猫在打篮球”、个性化模型、让别的模型和生成模型交互。

**Speed，速度**：生成是不是快。扩散模型传统上采样步数多，所以速度很重要。后面课程会讲怎么在不明显掉质量的情况下加速。

老师把学习过程比成 RPG 游戏：

```text
新手村 -> 选择技能树 -> 打小 boss -> 打最终 boss
```

对应作业就是：

- `HW1`：搭环境，做第一个 DDPM。
- `HW2`：做第一个 flow matching，并选择自己的方向。
- `HW3`：在选定方向上实现 baseline。
- `HW4`：提出改进，尝试 beat baseline。

这门课很特别：AI-friendly, open everything。可以用 AI assistant、开源代码、预训练模型、paper/tutorial/book，也可以和别人讨论。但要求是：作业必须自己完成，所有资源要引用，包括 AI；不能抄同学，也不能把别人或 AI 的工作说成自己的。

这节课你要带走的东西：

**生成模型这门课不只是学一个公式，而是学从数据分布、训练目标、采样算法，到工程实现和实验展示的一整条链路。**

## Lecture 1: Basics of Probabilistic & Generative Modeling

核心一句话：

**生成模型的目标是学一个数据分布 `p_theta(x)`，让我们既能给真实数据高概率，也能从这个分布里采样出新的“像真的”数据。**

这节课先讲 probabilistic modeling。概率建模就是用概率描述不确定性，比如：

```text
P(snow tomorrow) = 70%
```

里面有两个基本东西：

- **随机变量**：我们要描述的对象，比如天气、图片、文本 token。
- **概率分布**：这些对象出现的概率，比如 `p(X)`。

常见概念：

```text
Joint:       p(X, Y)      X 和 Y 同时发生
Marginal:    p(X)         只看 X
Conditional: p(X | Y)     已知 Y 后 X 的概率
Prior:       p(X)         看到证据前的信念
Posterior:   p(X | Y)     看到证据后的信念
```

然后进入 generative modeling。这里对比了 discriminative model 和 generative model。

**Discriminative model** 学的是：

```text
p(Y | X)
```

比如给一张卧室图片 `X`，判断它是不是豪华卧室 `Y`。

**Generative model** 学的是：

```text
p(X) 或 p(X, Y)
```

也就是说，它不只是判断图片，而是要学会“什么样的图片像真实卧室”，甚至能自己生成一张新的卧室图片。

生成模型有三个目标：

- **Generation**：能采样出真实感强的新样本。
- **Density estimation**：能给已有样本分配合理概率。
- **Unsupervised learning**：只看数据本身，不依赖标签。

第一条路线是最大化似然。真实数据应该在模型下有高概率，所以训练目标是：

```text
maximize log p_theta(x)
```

如果一个样本 `x` 是由很多小元素组成的，比如一句话由 token 组成、一张图由 pixel 组成，可以用 chain rule 拆开：

```text
p_theta(x) = p_theta(x1) p_theta(x2 | x1) ... p_theta(xK | x_<K)
```

这就是 autoregressive modeling。LLM 本质上就是这个：给定前面的 token，预测下一个 token。训练 loss 就是 cross entropy。

第二条路线是 latent variable 和 VAE。真实数据背后可能有一些看不见的隐藏变量。比如人的脸由基因影响，但我们看不到基因。对应到模型里，就是：

```text
z -> x
```

VAE 有两个部分：

```text
Encoder: x -> z
Decoder: z -> x
```

它要同时做两件事：

- 让 decoder 能从 `z` 重建回原始 `x`。
- 让 latent space `z` 服从一个好采样的分布，比如 Gaussian。

所以 VAE 的目标是 ELBO：

```text
E_q(z|x)[log p_theta(x|z)] - D_KL(q_phi(z|x) || p_theta(z))
```

直觉上：

- 第一项：重建得好不好。
- 第二项：编码出来的 `z` 是否接近标准先验分布。

也就是：

```text
reconstruction loss + KL regularization
```

第三条路线是 GAN。它不直接算 likelihood，而是让两个网络对抗：

```text
Generator:     生成假样本，努力骗过 discriminator
Discriminator: 判断样本是真的还是假的
```

Generator 从一个简单分布采样，比如 Gaussian，然后把它变成复杂数据：

```text
z ~ N(0, I)
G(z) -> fake image
```

GAN 的优点是生成效果可以很好；缺点是训练不稳定，容易 mode collapse。

这节课的主线是：

**生成模型就是学数据分布。不同模型的区别在于：怎么表示这个分布、怎么训练、怎么采样。**

## Lecture 2: Denoising Diffusion Models

核心一句话：

**Diffusion model 把生成问题改写成“先把数据逐步加噪到纯噪声，再训练模型一步步去噪回来”。DDPM 的关键简化是：固定前向加噪过程，只训练反向去噪模型预测噪声。**

这节课前半继续讲 VAE，因为 DDPM 的训练推导和 VAE/ELBO 很有关系。

VAE 的关键点是：

- 我们可以设计 `p_theta(z)`、`q_phi(z|x)`、`p_theta(x|z)`。
- 但 `p_theta(x)` 和真实 posterior `p_theta(z|x)` 不好直接算。
- 所以用 ELBO 作为 `log p_theta(x)` 的 lower bound。

VAE 训练时会用 reparameterization trick。如果：

```text
q_phi(z | x) = N(mu_phi(x), sigma_phi(x)^2 I)
```

那么采样可以写成：

```text
epsilon ~ N(0, I)
z = mu_phi(x) + sigma_phi(x) epsilon
```

这样随机性被挪到 `epsilon`，网络输出的 `mu` 和 `sigma` 还能正常反传。

然后 PPT 总结旧模型的问题：

- Autoregressive：要一个一个生成。文本还可以，图像按 pixel/patch 生成会非常慢。
- VAE：容易 blurry，因为 latent 把很多样本压到相近区域时，decoder 会生成“平均脸”。
- GAN：训练不稳定，容易 mode collapse。

于是引出 diffusion 的直觉：

```text
TURN NOISE INTO DATA
```

Diffusion 有两个过程：

**Forward process**：从真实数据开始，一步步加噪。

```text
x0 -> x1 -> x2 -> ... -> xT
```

最后 `xT` 接近标准高斯噪声。

**Reverse process**：从噪声开始，一步步去噪。

```text
xT -> x_{T-1} -> ... -> x0
```

训练目标本来可以像 VAE 一样从 ELBO 推导出来：把中间所有 `x1:T` 看成 latent variables，然后最大化 `log p_theta(x0)` 的下界。但完整 ELBO 有 reconstruction loss、很多 KL matching terms、prior matching term，看起来复杂。

DDPM 的核心简化是：

1. **固定前向过程**，不学习加噪方式。
2. **固定反向过程的 variance**，只学习 mean。
3. 进一步把 mean 的学习改写成 **预测噪声 epsilon**。

前向加噪可以直接写成：

```text
x_t = sqrt(alpha_bar_t) x0 + sqrt(1 - alpha_bar_t) epsilon
epsilon ~ N(0, I)
```

这意味着你不需要真的从 `x0` 一步步加到 `xt`，可以一次性采样任意时间步的 noisy sample。

训练时：

```text
sample x0 from data
sample t
sample epsilon ~ N(0, I)
x_t = sqrt(alpha_bar_t) x0 + sqrt(1 - alpha_bar_t) epsilon
train epsilon_theta(x_t, t) to predict epsilon
```

loss 就是：

```text
|| epsilon - epsilon_theta(x_t, t) ||^2
```

采样时，从纯噪声开始：

```text
x_T ~ N(0, I)
```

然后不断用模型预测噪声，再根据公式往前走一步：

```text
x_t -> x_{t-1}
```

最后得到 `x0`。

这节课还强调：

**Diffusion models are lowkey VAEs.**

因为 diffusion 也可以看成一个特殊的 latent variable model：前向加噪相当于固定 encoder，反向去噪相当于 learned decoder。只是它的 latent 不是一个 `z`，而是一整条 noisy trajectory。

架构上，DDPM 早期主要用 U-Net，因为：

- downsampling 能建粗粒度特征。
- upsampling 能恢复细节。
- skip connection 保留空间信息。
- 输入输出 spatial shape 一致，适合预测同尺寸噪声。

这节课你要记住：

**DDPM 的训练不是让模型直接生成图像，而是让模型在任意噪声水平下预测被加进去的噪声。学会预测噪声，就等价于学会去噪生成。**

## Lecture 4: Score-based Models

核心一句话：

**Score-based model 不直接学概率密度 `p(x)`，而是学“往哪里走会更像真实数据”的方向，也就是 `grad_x log p_data(x)`。然后从噪声出发，沿着这个方向一步步走回数据分布。**

这节先复习 DDPM：前向过程是不断给数据加噪声，反向过程是训练模型去噪，从纯噪声一步步生成数据。

然后它问：既然很多生成模型直接算 likelihood 很麻烦，比如 VAE 有 ELBO、flow 要可逆结构、EBM 有 partition function，那能不能不直接建 `p(x)`？

答案是：可以学 score function：

```text
s_theta(x) ~= grad_x log p_data(x)
```

它表示：在当前位置 `x`，往哪个方向移动，数据概率密度会上升。

训练看起来需要真实的 `grad_x log p_data(x)`，但真实数据分布我们不知道。所以 PPT 先讲经典 score matching：通过积分分部，把 unknown true score 消掉，得到一个不需要真实 score 的 loss。

但这个 loss 里有 `tr(J_s_theta(x))`，也就是 score 网络对输入的 Jacobian trace，计算很贵。

于是引出更实用的 Denoising Score Matching。

给真实数据加高斯噪声：

```text
x_tilde = x + epsilon
```

这时加噪分布的 score 是知道的：

```text
grad_{x_tilde} log q(x_tilde | x) = (x - x_tilde) / sigma^2
```

所以训练目标就变成：让网络看到 noisy sample 后，预测“往原始干净数据方向走”的向量。

训练好 score model 后，用 Langevin Dynamics 采样：

```text
x_{t+1} = x_t + eta s_theta(x_t) + sqrt(2 eta) z_t
```

直觉是：

- `s_theta(x_t)`：把样本往高密度/更像真实数据的方向推。
- `z_t`：加一点随机噪声，避免死板地爬梯度。
- 从高斯噪声开始，反复迭代，最后得到数据样本。

普通 score model 有几个坑：

- 真实数据通常在低维 manifold 上，score 可能不稳定。
- 初始噪声点在低密度区域，score 预测不准。
- 单一噪声尺度不够稳。

解决方法是：先加大噪声，再逐渐降低噪声。这就是 Annealed Langevin Dynamics 和 NCSN：

```text
s_theta(x, sigma)
```

模型不只看 `x`，还看当前噪声强度 `sigma`。采样时从大 `sigma` 到小 `sigma`，逐步把样本从粗糙结构修到细节。

PPT 后半重点说：DDPM 和 score-based model 本质上是同一件事的两种参数化。统一写法：

```text
x_t = lambda_t x0 + sigma_t epsilon
```

DDPM 通常训练模型预测噪声：

```text
epsilon_theta(x_t, t) ~= epsilon
```

NCSN/score model 训练模型预测 score：

```text
s_theta(x_t, sigma_t) ~= -epsilon / sigma_t
```

所以一个预测“噪声是什么”，一个预测“往哪里去噪”。方向相反但信息等价。

最后进入 SDE：当噪声 level 从离散很多步变成连续无限步，就得到连续时间扩散过程：

```text
dx = f(x,t) dt + g(t) dw
```

DDPM 对应 VP SDE，variance preserving。NCSN 对应 VE SDE，variance exploding。反向生成过程就是解 reverse-time SDE，而这个反向 SDE 里需要的核心量，还是 score：

```text
grad_x log p_t(x)
```

这节课真正想建立的桥是：

```text
Diffusion / DDPM ~= Score-based Models ~= SDE formulation
```

## Lecture 5: Flow Matching

核心一句话：

**Flow matching 不像 diffusion 那样随机地一步步去噪，而是直接学习一个 velocity field，让样本沿着连续轨迹从噪声“流”到数据。**

前面几节的 diffusion / score model 都是在讲：从噪声开始，借助 score 或去噪模型一步步生成数据。Lecture 5 问了一个更简单的问题：

如果我们已经有一个噪声点和一个数据点，最简单的从噪声到数据的路径是什么？

答案是：线性插值。

设：

```text
x0 ~ N(0, I)        # noise
x1 ~ p_data         # data
```

中间时刻：

```text
x_t = t x1 + (1 - t) x0
```

这条路径的速度是常数：

```text
dx_t / dt = x1 - x0
```

所以训练非常直观：

```text
sample noise x0
sample data x1
sample t ~ U(0, 1)
x_t = t x1 + (1 - t) x0
v = x1 - x0
train v_theta(x_t, t) to predict v
```

loss 是：

```text
L = E[ || v_theta(x_t, t) - v ||^2 ]
```

采样时，从噪声出发，不断沿着模型预测的速度往前走：

```text
x_{t + dt} = x_t + v_theta(x_t, t) dt
```

一直走到 `t=1`，得到数据样本。

这看起来像简单插值，但它其实和 continuous normalizing flow 有关系。CNF 用 ODE 把一个简单分布 `p0` transport 到目标分布 `p1`：

```text
dx_t / dt = v(x_t, t)
```

只要 ODE 的流线不交叉，这个 transformation 就是 invertible，所以它能作为 normalizing flow。

为了让概率分布被正确 transport，需要两个物理直觉：

- **Conservation of mass**：概率质量不会凭空产生或消失，总和还是 1。
- **Continuity equation**：概率不能瞬移，只能连续流动。

用数学写就是：

```text
partial_t p_t(x) = - div(p_t(x) v(x,t))
```

这里的 `p_t(x) v(x,t)` 是 probability flux，表示概率以多快、多大密度流过某个位置。

一个关键难点是：真实的 marginal velocity `u_t(x_t)` 不好直接算，因为在同一个中间点 `x_t`，可能有很多 data point 的条件路径经过。Flow matching 的漂亮结果是：

**只要匹配 conditional velocity，就等价于匹配 marginal velocity。**

所以训练时不用显式知道复杂的 marginal velocity，只需要对采样到的 `(x0, x1, t)` 监督：

```text
v_theta(x_t, t) ~= x1 - x0
```

这就是 conditional flow matching / rectified flow / stochastic interpolant 这几条线的共同核心。

最后这节课比较 diffusion 和 flow matching：

```text
Diffusion:      像拿着指南针在森林里走，边走边修正方向
Flow matching:  像坐在河流里的船，沿着速度场被带到目标分布
```

它们最终都在做“从简单分布到复杂数据分布”的 transport，只是路径和训练目标不同。

## Lecture 6: The Design Space of Diffusion Models & Solvers for Fast Sampling

核心一句话：

**Vanilla DDPM 太慢，所以这节课讲两类改进：一类是用更好的 sampler / ODE solver 减少采样步数，另一类是从 noise schedule、loss weighting、parameterization、preconditioning 等设计空间提升模型本身。**

Vanilla DDPM 的问题很直接：

- 通常要 1000 步才能生成一张图。
- 步数少了质量会明显下降。
- 分辨率和模型规模一上去，采样成本非常高。

这对实时应用、视频生成、GPU 成本都很不友好。

第一部分讲 DDIM。DDPM 每一步是：

```text
x_t -> x_{t-1}
```

但如果模型能从 `x_t` 预测出干净图像 `x0`，那我们不一定非要走到 `t-1`，可以直接跳到更早的时间点。

已知：

```text
x_t = sqrt(alpha_bar_t) x0 + sqrt(1 - alpha_bar_t) epsilon
```

模型预测 `epsilon_theta(x_t, t)` 后，可以反推出：

```text
x0_hat = (x_t - sqrt(1 - alpha_bar_t) epsilon_theta(x_t,t)) / sqrt(alpha_bar_t)
```

然后直接构造更早的 `x_s`：

```text
x_s ~= sqrt(alpha_bar_s) x0_hat + sqrt(1 - alpha_bar_s) epsilon_theta
```

如果 `s = t - 10`，就相当于一次跳 10 步。这样可以获得 10x 左右加速。DDIM 的 deterministic 版本就是这种思路；也可以加回随机性，让采样带噪声。

第二部分讲 ODE solver。既然 diffusion / flow matching 都可以写成连续时间 ODE，就可以用数值积分方法：

- **Euler solver**：最简单，用当前位置速度走一步。
- **Midpoint solver**：先走半步，在中点估计速度，再走完整一步。
- **Heun solver**：先预测下一点，再在下一点重新估计速度，对两次速度做校正。
- **DPM-Solver**：利用 diffusion ODE 的特殊结构，用更高效的解析近似和 Taylor 展开，在十几步内取得较好效果。

第三部分讲 diffusion model 的 design space，主要来自 EDM 的观点：很多“扩散模型效果差异”其实来自一堆独立设计选择，而不是单一算法名。

主要 knob 包括：

- 前向 noise schedule。
- 训练时如何采样 time/noise level。
- 不同时间步的 loss weighting。
- 预测目标 parameterization：预测 `epsilon`、预测 `x0`、预测 `v`。
- 输入输出 scaling / preconditioning。
- 时间或噪声 level conditioning。
- 采样 solver。
- 采样时间步 schedule。
- 采样步数。

一个核心概念是 SNR：

```text
SNR = signal / noise = alpha_t^2 / sigma_t^2
```

高 SNR 表示信号多、噪声少；低 SNR 表示噪声多、信号少。中间某些噪声水平最难学，所以可以：

- 用 cosine schedule 替代 linear schedule。
- 训练时更多采样困难 noise level。
- 采样时在困难区间放更多步。
- 根据 SNR 给 loss 加权。

parameterization 也很重要：

- 预测 `epsilon`：高噪声时容易，低噪声时难。
- 预测 `x0`：低噪声时容易，高噪声时难。
- 预测 `v`：折中，通常更平衡。

这里的 velocity prediction 是：

```text
v = alpha_t epsilon - sigma_t x0
```

最后讲 EDM 的 preconditioning。模型输入 `x_t` 本身是噪声和数据的混合，所以网络输出可以设计成：

```text
output = c_skip(sigma) * x_t + c_out(sigma) * F_theta(c_in(sigma) * x_t, c_noise(sigma))
```

直觉是：

- 噪声很小时，输入已经接近干净图，应该多走 skip connection。
- 噪声很大时，输入几乎全是噪声，应该更多依赖网络预测。

这节课你要记住：

**扩散模型不是只有“训练一个 U-Net 预测噪声”这么简单。实际性能来自采样器、噪声日程、loss 权重、预测目标、输入输出缩放等一整套设计。**

## Lecture 7: Text-to-Image Generation & SOTA Models

核心一句话：

**现代 text-to-image 模型可以看成三块组合：先把图像压到 latent space，再用 diffusion/flow 模型在 latent 上生成，最后用 text encoder 和 cross-attention/MM-DiT 把文字条件注入生成过程。**

这份 PDF 文件名是 `Lecture7_Guidance.pdf`，但解析出的标题是 `Lecture 7: Text-to-Image Generation & SOTA Models`。它开头先回顾了上一节“conditional diffusion / guidance”，然后把重点放到 text 这种 condition 上。

Text-to-image 的 design space 大概分三块：

```text
Training / Model / Text Encoding
```

第一块是高分辨率图像怎么处理。图像维度非常高：

- `64x64x3 = 12,288`
- `256x256x3 = 196,608`
- `1080p` 已经是几百万维
- `4K` 是几千万维

直接在 pixel space 训练 diffusion 很贵，所以现代模型常用 latent diffusion。

Latent diffusion 的流程：

1. 先训练一个 autoencoder。
2. 用 encoder 把图像压到 latent `z`。
3. 冻结 autoencoder。
4. 在 latent space 训练 diffusion / flow model。
5. 生成时先生成 latent，再用 decoder 解码回图像。

```text
image x -> Encoder -> latent z
latent diffusion: z_T -> ... -> z_0
z_0 -> Decoder -> image
```

这比 pixel space 便宜很多，因为 latent 的空间尺寸和维度更小。

但 autoencoder 本身也有设计选择：

- 普通 VAE 可能 blurry，语义不够清楚，甚至 posterior collapse。
- VQ-VAE / VQGAN 用离散 codebook，有利于压缩和 transformer。
- 新一代模型会用更强的 continuous VAE，并做 representation alignment、更多 latent channels、spatial packing 等。

第二块是 model architecture。

早期 diffusion 主要用 U-Net，因为它适合图像的多尺度结构，也容易让输入输出维度一致。后来 DiT 把 diffusion backbone 换成 Transformer：

- 把 2D latent patchify 成 token。
- 加 2D positional embedding。
- 在每层注入 time/noise conditioning。

这条路线更容易 scale。

第三块是 text encoding。把 text 输入图像生成模型要做两件事：

```text
1. 把文字编码成 feature vectors
2. 把这些 text features 注入 diffusion / flow backbone
```

CLIP 是早期核心方案。它用大量 image-text pairs 做 contrastive learning：匹配的图文是 positive examples，不匹配的是 negative examples。这样训练出来的 text encoder 能抓住视觉语义。

但 CLIP 有缺点：

- 空间关系不强。
- 否定关系不强。
- 数量关系不强。
- token 长度限制，比如 77 tokens。
- 有时会忽略 prompt 细节。

所以后来的模型会加更强的文本编码器，比如 T5，甚至直接用 LLM/VLM/MLLM 做文本理解。

Text conditioning 的注入方式也在演化：

- Stable Diffusion 1/2：U-Net + CLIP + cross-attention。
- Stable Diffusion 3 / Flux 1：flow matching + DiT/MM-DiT + CLIP/T5。
- Flux 2、Z-Image、Qwen-Image 等：更强 VAE + DiT + 更强 text encoder。
- Transfusion、Hunyuan 3.0 等：更接近 native multimodal model，把文本和图像生成放进同一个多模态框架。

PPT 还猜测 Nano Banana / GPT-4o Image 这类系统可能是：

```text
Multimodal LLM 做 reasoning / prompt planning
MM-DiT 或类似 diffusion/flow backbone 在 latent 里生成图像
```

这节课你要记住：

**现代文生图不是一个单独模型，而是 autoencoder、生成 backbone、text encoder、conditioning mechanism 的组合系统。**

## Lecture 9: SOTA

当前状态：

`Lecture9_SOTA.pdf` 在本地 PDF 文本解析时失败，`ReadFile` 返回 `Invalid PDF structure`。文件本身大小约 3 MB，不一定坏，可能是当前文本提取器不兼容。

待补方式：

- 用浏览器或 PDF 阅读器手动打开后逐页总结。
- 或安装/使用更强的 PDF 文本提取工具后重新提取。

先保留本章节，后续补齐。

## Lecture 10: Distillation, Consistency Models & Flow Maps

核心一句话：

**前面用更好的 solver 减少采样步数；这节课进一步问：能不能训练一个新模型，让它天生就能一步或少步采样？答案包括 progressive distillation、consistency models、consistency trajectory models 和 flow maps。**

这节课从一个问题开始：

之前已经有 DDIM、DPM-Solver 等方法让采样更快，但它们主要是在 inference 阶段改采样器。那能不能训练一个模型，本身就用更少步生成？

第一种方法是 progressive distillation。

假设原始 teacher model 用 DDIM 两步从：

```text
x_t -> x_{t-dt} -> x_{t-2dt}
```

student model 学会用一步完成同样效果：

```text
x_t -> x_{t-2dt}
```

这样 student 一步等于 teacher 两步，采样速度 2x。再用这个 student 当新 teacher，继续训练下一个 student，一步匹配两步，就能逐渐把步数压低：

```text
1000 -> 500 -> 250 -> ... -> 1 or few steps
```

为什么不直接让 student 一步匹配 teacher 的很多步？因为训练时 teacher 要跑很多 forward passes，成本高且难并行。所以 progressive distillation 采用逐步折半。

第二种方法是 consistency models。

它换了一个角度：不要求模型模拟每一步，而是要求模型在同一条 probability flow ODE trajectory 上的任意点，都预测同一个最终 clean output。

也就是：

```text
f_theta(x_t, t) = f_theta(x_s, s)
```

只要 `x_t` 和 `x_s` 在同一条轨迹上，它们都应该映射到同一个最终 `x0`。

还需要 boundary condition：

```text
f_theta(x0, 0) = x0
```

直觉是：

**模型不再学“下一小步怎么走”，而是学“无论我在轨迹哪里，都能直接告诉你终点在哪里”。**

训练 consistency model 可以用 pretrained diffusion teacher：从 `x_t` 用 teacher solver 走一步得到 `x_{t-dt}`，然后让：

```text
f_theta(x_t, t)
```

和：

```text
stopgrad(f_theta(x_{t-dt}, t-dt))
```

一致。

Consistency model 的问题：

- 它主要训练预测 `x0`，天然不适合多步采样。
- 很难在 inference 时灵活做 speed/quality tradeoff。
- 会丢掉 exact log-likelihood 的能力。

第三种是 consistency trajectory model。它不只是学“从任意点跳到终点”，而是学“从任意时间跳到任意时间”：

```text
x_t -> x_s
```

这样模型更像学习整条 trajectory，而不是只学最终映射。为了保证局部行为正确，还要加 tangent condition，也就是当跳跃时间无限小时，它应该退化成普通 score / velocity matching。

PPT 后半引入 flow map。

在数学/物理里，flow map 表示：给定当前点和起止时间，直接告诉你这个点沿流走到目标时间后在哪里。

如果 `v(x,t)` 是瞬时速度，flow map 学的是从 `t` 到 `s` 的“大跳跃”：

```text
Phi(x, t, s) ~= x_s
```

一个合法 flow map 至少要满足 tangent condition：

```text
u(x, t, t) = v(x, t)
```

意思是：当起点时间和终点时间一样近时，大跳跃模型必须退化为瞬时速度。

还可以满足下面三类条件之一：

- **Lagrangian condition**：跟着同一只“橡皮鸭”沿河流走。
- **Eulerian condition**：从同一条流线上的不同起点跳，结果一致。
- **Semigroup condition**：两次小跳等于一次大跳。

比喻是：河里有一只橡皮鸭，普通 ODE solver 是让鸭子一点点顺流漂；flow map 是学会“传送鸭子”，直接把它送到如果正常漂流会到达的位置。

最后讲到 flow maps 的限制：它们主要加速 sampling，但 likelihood evaluation 仍然慢，因为 likelihood 还要积分 divergence ODE。于是 F2D2 这类方法尝试同时为 sampling 和 likelihood 学 joint flow maps。

这节课你要记住：

**fast sampling 的最高层思路是：不要每次都小步积分，而是训练模型学会沿生成轨迹做大跳跃。**

## Lecture 12: Discrete Diffusion

当前状态：

`Lecture12_Discrete_Diffusion.pdf` 在本地 PDF 文本解析时失败，`ReadFile` 返回 `Invalid PDF structure`。文件大小约 90 KB，可能是压缩/导出格式导致当前解析器读不了。

待补方式：

- 用 PDF 阅读器手动打开后逐页总结。
- 或换用更强的 PDF 文本提取工具。

先保留本章节，后续补齐。

## Lecture 13: Discrete Flow

当前状态：

`Lecture13_Discrete_Flow.pdf` 在本地 PDF 文本解析时失败，`ReadFile` 返回 `Invalid PDF structure`。文件大小约 45 KB，可能是压缩/导出格式导致当前解析器读不了。

待补方式：

- 用 PDF 阅读器手动打开后逐页总结。
- 或换用更强的 PDF 文本提取工具。

先保留本章节，后续补齐。

## 总结主线

这套课的前半段主线可以压缩成这样：

```text
Lecture 0: 课程目标和生成模型评价维度
Lecture 1: 概率建模、生成模型、AR/VAE/GAN
Lecture 2: 从 VAE/ELBO 到 DDPM
Lecture 4: 从 DDPM 到 score matching 和 SDE
Lecture 5: 从 score/diffusion 到 flow matching
Lecture 6: diffusion 设计空间和快速采样 solver
Lecture 7: text-to-image 系统设计和 SOTA 模型组件
Lecture 10: distillation、consistency、flow map，一步/少步生成
```

整体逻辑是：

```text
学数据分布
-> 从简单噪声采样
-> 逐步去噪或沿流移动到数据
-> 提高质量、速度、可控性
-> 扩展到现代文生图和少步生成
```
