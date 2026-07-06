# Bernini 前缀 token v2v · 因果蒸馏交接文档

> 给"无记忆"的新 agent 重新接手时快速建立上下文。范围限定：Bernini-R-1.3B 的 v2v 编辑能力，用**前缀 token + source_id 旋转编码**范式，蒸馏进 Self-Forcing 的 4 步因果流式架构。代码根目录 `/apdcephfs/private_huitinglu/Self-Forcing`。
>
> **另有一条已放弃的路线（双通道 inpainting 式，扩 16→32）**，其设计、失败复盘与遗留产物已独立存档于 [ROUTE_A.md](ROUTE_A.md)，本文不再展开；仅在需要对照时引用。本文全部内容围绕**前缀 token 前缀 v2v** 展开。

---

## 前置背景：v2v 条件注入的三种主流范式（2026）

先建立行业坐标系。v2v / 视频编辑"源视频怎么进模型"目前是**两到三种范式并存**，各有侧重：

**① Token / in-context 拼接（= Bernini 这一派，本项目采用）**
- 源视频 patchify 成 token，**沿序列维**拼接到噪声 token 前，用**位置编码偏移**区分来源，靠自注意力统一处理。
- Bernini 的 `source_id rotary`（源=id1 / 目标=id0）本质等同 2026 的 **Tele-Omni** RoPE 偏移 `R_θ(Δ=(0,w_tar,0))`（把源沿宽度平移区分目标），思路一致。同派：IC-LoRA、Omni-Transfer、FLUX Kontext(图像)、FullDiT/ICC。
- **优点**：t2v/v2v 生成质量更优、多模态/变长/非对齐条件最灵活。**代价**：序列变长 → 注意力二次方开销暴涨（FullDiT2/ICC 专门优化此点）。

**② 通道拼接 / inpainting 式（= 已放弃路线，见 [ROUTE_A.md](ROUTE_A.md)）**
- 源 / mask latent **沿通道维**与噪声 latent 堆叠，把编辑当 inpainting。同派：SkyReels-V4、Wan-VACE 等。
- **优点**：对空间对齐编辑很主流、计算便宜（不加 token）。**缺点**：刚性，要求严格空间对齐，变长/非对齐参考不好处理。

**③ Cross-attention adapter**
- 条件作为 K/V 经 cross-attn 注入（Omni-Video2、IP-Adapter 式）。SOTA 常**混用** ①②（如 MiVE、SkyReels-V4：通道拼接做空间锚点 + token 注意力做指令跟随）。

**对本项目的意义**：
- 前缀 token（①）是**质量更优、更主流**的范式，复用 Bernini 现成能力方向正确。其唯一硬伤"序列变长二次方算力"，**恰是因果流式 + KV cache 把源前缀一次性 prefill 进 sink 区的价值所在**（源只编码一次永久缓存，免每帧重算长序列）。
- 通道拼接（②）更省算力但更刚性；它在本项目里失败**不是因为范式不主流**，而是 Bernini 权重里没有通道拼接对应的能力、短训没学起来（详见 [ROUTE_A.md](ROUTE_A.md)）。

---

## 0. 一句话现状

- **前缀 token 方案（复用 Bernini 原生前缀 token v2v，适配进因果流式）已选定并实现**。**直接 DMD 到 step 700 实测仍源脱离**，复盘后改为 **Stage-1 初始化 + DMD** 正式路线。
- **Stage-1 初始化现支持三种方案（config 切换，DMD 阶段共用）**：
  - **`ode`**：双向 Bernini teacher 离线 ODE 采样 → ODE 回归（原路线，仅需 src）
  - **`causal_ode`**：因果 AR v2v teacher 离线 ODE 采样 → ODE 回归（Causal Forcing，需 Stage-0 + src）
  - **`causal_cd`**：在线 Causal Consistency Distillation，**免 ODE 采样**（Causal Forcing++，需 Stage-0 + src+tar 配对）
- **Stage-2 DMD 三方案共用**：`real_score` 始终为双向 Bernini（符合 Causal Forcing 观点），仅改 `generator_ckpt` 指向对应 Stage-1 产出。
- 另有一条**双向（不因果）并行蒸馏路线**（§13），靠 config 开关 `generator_causal=false` 切换，与因果路线隔离、可同机并行。

---

## 1. 背景：Bernini 原生 v2v 怎么做的

Bernini 的 v2v 用 **前缀 token + source_id 旋转编码** 注入源条件，**不扩通道**：

- 源视频 VAE latent 被 patchify 成 token，作为**前缀**拼在噪声目标 token 序列**前面**。
- 用一组独立的 **source_id 旋转频率**（`visual_id_freqs`）给不同来源 token 打标记：`0=目标`、`1=源视频`、`2+=参考图`。同一空间位置的"源 token"与"目标 token"靠旋转编码区分。
- 注意力是**双向全注意力**（源↔目标互看），最后只取目标 token 的输出。
- 关键事实：**`visual_id_freqs` 是确定性计算的、无学习权重**（`get_1d_rotary_pos_embed`）。所以 Bernini 的 v2v 能力**全在标准 transformer 权重里**，只要复刻 source_id 旋转逻辑即可复用，无需迁移额外权重。

---

## 2. 前缀 token 范式的核心洞察

**为什么前缀 token 能蒸进因果流式（本方案成立的前提）**：源视频在编辑时是**完整已知、且不随生成变化**的，所以可把源 token 当作一段固定前缀，预填进因果 student 的 KV cache 最前部；之后逐帧因果生成天然能"往前看"到源前缀。当初 Self-Forcing 选双通道只是早期实现简单，不是前缀 token 做不到流式。

（与已放弃的双通道方案的逐项对比表见 [ROUTE_A.md](ROUTE_A.md) §2。）

---

## 3. 方案设计要点

- student/teacher 都沿用 Bernini 前缀 token 范式：`in_channels=16` 不扩通道，源作前缀 token（带 source_id rotary）注入。
- **teacher（real_score，冻结）直接用 Bernini 原生双向 v2v 权重** → v2v 能力现成、**零短训**，DMD 监督信号天生正确。
- **student（generator）用同一权重初始化** → 与 teacher 接口天然一致，DMD 对齐。
- 把源前缀 token 预填进因果 student 的 **KV cache sink 保护区**，适配流式。
- 代价：Bernini 是双向训练的，迁到因果（KV-cache + blockwise causal mask）存在**双向→因果行为失配**，可能需要一个轻量"因果适配 ODE"阶段稳住，再 DMD。

> 相较之下，已放弃的双通道方案需从零短训新增通道、且没学起来，详见 [ROUTE_A.md](ROUTE_A.md)。

---

## 4. 实验记录与失败复盘（驱动"改成 ODE-first"的关键证据）

### 4.1 已验证的失败
- **直接 DMD（未做 ODE 采样 / 初始化）到 step 700**（`logs/bernini_v2v_dmd_routeb/checkpoint_model_000700`）：推理测试发现**仍然与源视频完全脱离**。
- （对照：已放弃的双通道方案 DMD 到 step 800 也源脱离，但根因不同，见 [ROUTE_A.md](ROUTE_A.md) §3。）

### 4.2 复盘：直接 DMD"源脱离"的根因
直接 DMD 的根因**不在监督信号**（teacher 是 Bernini 原生双向 v2v，信号正确），而在 **student 初始化 + DMD 监督形式**：
1. **双向→因果失配**：Bernini 权重是在双向全注意力下学会"读前缀源"的；迁到因果（KV-cache + blockwise mask + 源 prefill 进 sink 区、统一 t=0）后，前缀的可见性与位置编码语义都变了，**student 初始状态并不会在因果约定下正确使用源前缀**。
2. **DMD 是分布级弱监督**：DMD 只通过 real/fake score 的分数差提供梯度，约束的是"生成分布像 teacher 边缘分布"，**不逐帧监督"这一帧相对源该长什么样"**。当 student 初始就严重忽略源时，DMD 很容易收敛到"画面合理但与源无关"的解（mode covering 即可降 loss）；`condition_dropout` 进一步鼓励无条件生成，雪上加霜。
3. **DMD-only 缺纠偏能力**：700 step 仍未把"忽略源"的坏初始拉回来，佐证仅靠 DMD 信号不足以从该初始点纠偏。

### 4.3 决策：先 ODE 采样 + 因果适配初始化，再 DMD
- ODE 回归是 **(源, teacher 去噪轨迹) 的逐帧配对强监督**，直接强制 student 在**因果 t=0 约定下复刻 teacher 的源跟随行为**，先把模型"教会在流式架构里看源"。
- 这正是原版 Self-Forcing 用 ODE 初始化的目的：**先把模型适配到因果生成行为、稳住源跟随，再交给 DMD 提质**。
- 因此从"直接 DMD"改为 **"ODE-first"**：ODE 采样 → build LMDB → ODE 回归（因果适配）→ 产出 checkpoint → 续 DMD。后续 DMD 的 `generator_ckpt` 应指向 ODE 回归产出的 checkpoint，而非裸 `bernini_v2v_prefix_init.pt`。

### 4.4 为什么"一开始判断不需要 ODE，失败后才发现需要"
不是分析错了，而是初始分析里有一个**被乐观处理的赌点**，失败只是把赌点结果揭晓。四层原因：

1. **把"监督信号对"等价于"学得到"**。初始推理链：双通道失败归因为监督信号错 → 换 Bernini 原生双向 v2v 信号就对了 → 信号对了 DMD 就该学到源跟随，ODE 可省。这一步**混淆了两个独立问题**：(a) teacher 端信号对不对（已解决）；(b) student 端能否从其初始点被 DMD 拉到正确解（没解决，当时也没认真评估）。双通道的失败把注意力全吸到 (a)，导致 (b) 被忽略。
2. **低估了 ODE 的真正作用**。当时把 ODE 理解成"喂正确去噪轨迹"（信号问题），既然 teacher 信号已正确就觉得可省。但 ODE 的核心其实是**把双向权重适配到因果生成行为**（KV-cache / blockwise mask / 统一 t=0 约定下重新学会用上下文）+ **逐帧配对强监督做 DMD 初始化**——这是原版 Self-Forcing 的 Stage 1、对 T2V 也必需，并非 v2v 特有。初始分析没把"因果适配"从"信号正确性"里拆出来。
3. **两个"程度问题"无法靠静态分析定量**，只能跑出来才知道：① 双向→因果失配到底有多大（student 虽与 teacher 接口一致、从 Bernini 权重初始化，但 prefill 进 sink、t=0 约定下前缀语义漂移多严重，纸面估不准）；② DMD 的纠偏能力有多强（分布级弱监督 + condition_dropout 偏向 mode covering，能否从"忽略源"的坏初始爬回来）。
4. **这本就是一次有意识的低成本快速试错**。初始分析并未断言"一定不需要 ODE"，而是明确标为赌点（"赌 DMD 直接吸收失配；若不行补 ODE 是候选"）。为省掉 ODE 的几十 GPU-hours 先赌直接 DMD，step-700 实测源脱离即判赌点"输"，回退 ODE-first。

> 一句话：修好了"teacher 信号"却误以为问题只在 teacher；而 student 的"因果适配 + 强初始化"是另一个独立缺口，无法靠静态分析定论，只能用一次低成本直接-DMD 实验证伪后才被坐实。

### 4.5 ODE-first 之后仍可能踩的坑（前瞻风险，按优先级）
即便补了 ODE，下列问题仍可能让源跟随不达标，按"最可能/最该先查"排序：

1. **ODE 数据规模/多样性不足**：当前只采到 ~2000、且 `build` 按 prompt 去重后可能更少；ReCo replace 任务单一，覆盖的编辑类型/源视频分布窄。表现为"训练集内源跟随 OK、泛化弱"。对策：必要时扩采样量与任务多样性。
2. **teacher 采样配置 ↔ 学生推理配置不一致**：ODE 数据用 `num_steps=24, guidance_scale=6.0` 双向采样，而学生是 **4 步因果**、DMD `guidance_scale=3.0`。轨迹关键帧虽对齐 `[1000,750,500,250,0]`，但 teacher 用的 CFG 强度和步数与学生最终工况不同，可能引入分布偏移。对策：观察 ODE loss 分时段（`loss_at_time_*`）是否某段异常。
3. **ODE 只对目标算 loss，未直接约束"源-目标一致性"**：`_forward_train` 剥掉前缀只对目标 token 回归 teacher 轨迹。若 teacher 本身在某些样本就欠编辑/过编辑，student 会忠实复刻其缺陷。对策：抽检 teacher 轨迹质量再入库。（§16 的背景保持项即针对此缺口。）
4. **ODE→DMD 的"再退化"**：ODE 把源跟随学起来后，DMD 阶段 `condition_dropout=0.15` + 弱监督仍可能把模型往"无条件生成"拉回，出现训练后期源跟随回落。
   - **关键认知：`condition_dropout` 是"欠编辑 ↔ 源脱离"的旋钮，方向与直觉相反。** 它按概率整批把源前缀置零（`distillation.py:345-348`，源在 cond/uncond 两侧都保留，CFG 只作用于文本），本意是防 4 步少步生成直接拷贝源（欠编辑）。**调高 → 更多无源训练步 → 更不依赖源 → 加重源脱离（即加重 #4）；调低 → 更贴源。** 所以 **#4 出现时要调低/甚至先设 0，绝不能调高**。`0.15` 已偏高，只在确认源跟随稳住、出现过度拷贝/欠编辑时才小幅上调。
   - **#4 真正的解法（按效力排序）**：
     1. **靠 ODE 强初始化兜底**（ODE-first 的核心）：DMD 从"已会跟源"的 checkpoint 起步，比任何 dropout 调参都重要，是防退化第一闸。
     2. **监控源跟随 + 选点**：DMD 期间定期 `inference --v2v` 抽测，取**回落前**的 checkpoint；配合 `ema_weight=0.99` 的 EMA 权重评估更稳（建议手动多存几个里程碑，默认只留最近 3 个）。
     3. **`condition_dropout` 小火慢调**：先 0 或 0.05 起，确认源跟随稳住后，再视是否欠编辑小幅上调（它解决的是另一头，不是 #4）。
     4. **加配对锚定（可选，若上面仍不够）**：DMD 里混入极小比例 ODE/回归配对监督（对源-条件样本算一项轻量 L2），给"别漂离源"一个硬约束（DMD2 regression 项思路，周期性混 batch，不改主损失）。
5. **双向→因果失配的残差**：ODE 是"缓解"而非"消除"失配。sink 区只有固定容量，长视频/多 block 下源前缀可能被相对稀释；blockwise causal + local window 下远处帧对源的注意力衰减。表现为"前几帧跟源好、后段漂移"。对策：检查 sink 容量与 local window 设置。
6. **timestep 约定的连锁**：源固定 t=0 与所有干净上下文一致（已确认勿改），但若 ODE 后源跟随仍系统性偏弱，t=0 是次优先排查项（非首选）。
7. **续 DMD 的 checkpoint 衔接**：ODE 产出的是纯 `generator`，DMD 首训需要正确把它喂进 generator 初始化、同时 real/fake score 仍各自从 `bernini_v2v_prefix_init.pt` 载入（别误用 ODE ckpt 覆盖 teacher）。
8. **ODE 用双向 teacher 的理论隐患**：默认 `init_method=ode` 违反 injectivity；若效果不达标，切换 `causal_ode` 或 `causal_cd`（§6.4，代码已落地）。

---

## 5. Self-Forcing 因果 student / DMD 地基（已吃透）

- 推理：`pipeline/causal_inference.py` 的 `_forward_inference` 逐 latent 帧用 `kv_cache + current_start/cache_start` 生成。
- **KV cache 有 `sink_tokens`**（永不被驱逐的前部保护区）—— 这正是放源前缀 token 的天然位置。源前缀 prefill 进 sink 区，后续每帧因果生成都能 attend 到它。
- DMD 三模型统一接口 `model(noisy, conditional_dict, timestep)` 调用，源条件藏在 `conditional_dict` 里：
  - `generator`（student）= **因果**，可训练；
  - `real_score`（teacher）= **双向**，冻结 ← 直接用 Bernini 双向 v2v；
  - `fake_score` = **双向**，可训练。
- real/fake score 本来就是双向、对整段视频单次前向 → 与 Bernini 原生双向 v2v 完全契合。只需把三者源条件从"通道拼接"改成"前缀 token + source_id rotary"。

### 关于 timestep（已确认，勿乱改）
- Self-Forcing 流式框架里**所有"干净上下文"统一用 t=0 缓存**（`self_forcing_training.py` Step 3.3 "rerun with timestep zero to update the cache"；i2v 的 initial_latent 也是 `timestep*0` 写入）。
- 源前缀也是干净上下文，所以 **源用 t=0 与框架约定一致，不是 bug**。
- Bernini 的"源用 t"是在双向、每步重算整段下成立的；流式因果是源 prefill 一次永久缓存，无法复刻"源随 t 变化"，只能定固定 t。强行改成 t（如固定 1000）反而破坏与其它干净上下文的一致性。
- 判断：源跟随失效的主因大概率是**没跑 ODE 适配**（student 没被教会在 t=0 约定下使用源前缀），而非 t=0。**先跑因果适配 ODE，再看是否需要动 timestep。**

---

## 6. in-context v2v 蒸进因果流式：主流适配 + Causal Forcing 纠偏

> 行业坐标系：把 ① in-context token 范式蒸进因果流式，主流分"条件怎么进流式"和"双向→因果怎么蒸"两块。本节用于后续 agent 判断"ODE 效果不好时往哪走"。

### 6.1 条件 token 怎么塞进因果流式（本项目已做对）
in-context 范式靠"源 token 拼序列 + 全注意力互看"，与因果流式（只看过去、边checkpoint_model_000600生成边出帧、省算力）有矛盾。主流三招：
1. **条件 token 当持久前缀 prefill 进 KV cache（attention sink）**：源完整已知、不随生成变化 → 只编码一次、prefill 进 cache 最前部、标永不驱逐；后续因果 chunk attend 它而不重算。**化解 in-context "序列变长→二次方算力"的硬伤。** ← 本项目已做。
2. **位置/timestep 静止处理**：条件 token 用固定位置偏移（`source_id rotary` / Tele-Omni RoPE Δ）+ 固定 `t=0`（MiVE "stationary tokens use fixed time"）。流式 prefill 一次永久缓存，**无法复刻双向"源随 t 变化"**，只能锚定固定 t。← 本项目的"源 t=0、勿改"由此而来。
3. **长视频滚动 KV cache + sink 保留**：rolling cache 驱逐旧帧，但条件前缀放 sink 永不驱逐，保证全程源跟随（LongLive / Rolling Forcing 类扩展同款）。

### 6.2 双向→因果的蒸馏（SOTA 争议点，关系本项目隐患）
主流框架：**CausVid / Self-Forcing 的非对称 DMD** —— bidirectional in-context teacher → few-step causal student，中间 **ODE 初始化**做因果适配，再 DMD 提质。

**Causal Forcing（thu-ml, arXiv 2602.02214, 2026）的关键纠偏**（出处：论文摘要 + §3.2「Current ODE initialization in Self Forcing violates frame-level injectivity」+ Fig.3 / Lemma 3.2 / Prop 3.3）：
- **痛点**：用 **bidirectional teacher** 采 ODE 轨迹初始化因果 student，**违反 frame-level injectivity**（双向 PF-ODE 只在 video-level 单射，不满足逐帧单射）→ 同一 noisy frame 对应多个 clean frame → student 学到 conditional-expectation 解、recover 不了 teacher flow map → 模糊 / 不一致。直觉：双向模型去噪第 i 帧时用了所有帧，固定 x_t^i 但不同 x_t^{>i} 仍得到不同 x_0^i；AR student 监督时看不到 x_t^{>i}，信息丢失。
- **正确做法**：先训一个 **自回归(因果) teacher**（teacher forcing，论文证明优于 diffusion forcing），用**它**的 PF-ODE 做 **causal ODE 初始化**（天然满足 injectivity）。
- **⚠️ 限定（论文 FAQ 明确）：批评仅针对 ODE/CD「初始化」阶段**（ODE/CD 要求师生轨迹对齐、结构必须匹配，AR student 无法和双向 teacher 轨迹对齐）。**DMD 阶段用双向 teacher 是对的**——DMD 只需匹配 teacher 的最终分布、不需轨迹对齐，且双向模型通常更强、是更好的 teacher。
- **Causal Forcing++（2605.15141）**：用 **causal Consistency Distillation 替代 ODE**，免 ODE 数据采集。

### 6.3 本项目对照与隐患
- **6.1 三招本项目都做对了**（源前缀 prefill sink、source_id rotary、固定 t=0），与主流一致。
- **6.2 有隐患（仅限双向 ODE 阶段）**：本项目默认 **`init_method=ode`** 的 ODE 采样用 **Bernini 双向前缀 teacher**，正是 Causal Forcing 批评的 injectivity 问题情形。
  - 影响：ODE 适配可能不够干净，蒸出来或偏模糊 / 源跟随不够锐。
  - **已实现升级路径**（代码已落地，config 切换）：
    - **`init_method=causal_ode`**：`generate_v2v_ode_lmdb.py --causal_teacher` + `self_forcing_v2v_causal_ode_routeb.yaml`
    - **`init_method=causal_cd`**：`self_forcing_v2v_causal_cd_routeb.yaml`（免采样）
  - 二者均需 **Stage-0** 因果 AR v2v teacher（`tools/train_causal_v2v_teacher.py` → `bernini_causal_v2v_teacher.pt`）+ ReCo **src+tar 配对**数据。
- **DMD 阶段不在此隐患内**：`real_score` 用双向 Bernini **符合 Causal Forcing 观点，三方案 DMD 相同**。

### 6.4 Stage-1 三种初始化方案（已实现，config 切换）

> Stage-2 一律用 `configs/self_forcing_v2v_dmd.yaml`；仅 Stage-1 的 `init_method` / config 不同。

| init_method | Config | Stage-1 trainer | 采样 teacher | 训练数据 | 解决的核心问题 |
|---|---|---|---|---|---|
| **`ode`** | `self_forcing_v2v_ode_routeb.yaml` | `ode` | 双向 Bernini | 离线 ODE LMDB（仅需 **src**） | 双向→因果前缀读法适配 |
| **`causal_ode`** | `self_forcing_v2v_causal_ode_routeb.yaml` | `ode` | **因果 AR** v2v teacher | 离线 ODE LMDB（仅需 **src**） | injectivity + 前缀适配 |
| **`causal_cd`** | `self_forcing_v2v_causal_cd_routeb.yaml` | `consistency_distillation` | **因果 AR**（在线） | 在线 ReCo **src+tar** 配对 | injectivity + 前缀适配，**免 ODE 采样** |

**Stage-0（仅 `causal_ode` / `causal_cd` 需要）**：
```bash
torchrun --nproc_per_node 8 tools/train_causal_v2v_teacher.py \
  --init_ckpt checkpoints/bernini_v2v_prefix_init.pt \
  --out checkpoints/bernini_causal_v2v_teacher.pt
```
- 数据：`V2VPairedVideoDataset`（ReCo `src_video` + `tar_video` + `instruction_final_refine`）
- 产出：`checkpoints/bernini_causal_v2v_teacher.pt`
- **当前阻塞**：tar 视频大量缺失（仅 src 的 ~74000 条不够 Stage-0）

**Stage-2 DMD（三方案共用）**：
- Config：`configs/self_forcing_v2v_dmd.yaml`
- `generator_ckpt` → 对应 Stage-1 产出 checkpoint
- `real_score_v2v_ckpt` → 始终 `checkpoints/bernini_v2v_prefix_init.pt`（双向 Bernini，不改）

---

## 7. 已落地的代码改动

### 7.1 权重转换
- `tools/convert_bernini_to_wanmodel.py --no_expand`：保留 16 通道，产出 Bernini 原生 v2v 权重。
- 产物：`checkpoints/bernini_v2v_prefix_init.pt`（5.6 GB，825 keys 映射完成）。
- 同一权重用于：teacher(real_score, 冻结) / fake_score / causal generator 初始化。

### 7.2 source_id rotary
- 在 `wan/modules/model.py`(双向) 和 `wan/modules/causal_model.py`(因果) 的 rope 里复刻 Bernini `visual_id_freqs` 调制（源=id1，目标=id0）。

### 7.3 因果 student 前缀注入（`wan/modules/causal_model.py` + `utils/wan_wrapper.py`）
1. **`_forward_train` 源前缀注入**：源 latent patchify 成前缀 token 拼到目标 token 前；用 `build_rope_freqs([src_grid, tgt_grid], source_ids=[1,0])` 给源打 source_id=1、目标=0；源帧时间调制用 **t=0**（clean）；前向后 `x = x[:, prefix_len:]` 剥掉前缀只对目标算 loss。位置编码与可见性与流式推理（prefill 进 KV cache）完全一致。
2. **新因果 mask `_prepare_prefix_blockwise_causal_attn_mask`**：源前缀内部全注意力 ＋ 目标全见源前缀（sink 式）＋ 目标内部 blockwise-causal（兼容 local window），按 (源帧,目标帧,frame_seqlen,block) 缓存。
3. **self-attention / Block 透传 `rope_freqs`**：训练态（非 teacher-forcing）走 `rope_apply_flat` 整段应用预计算复数 rope；`rope_freqs=None` 时行为与原来完全一致，不影响非 v2v 训练。
4. **`wan_wrapper.forward`**：因果 generator 的并行训练路径（无 kv_cache、无 clean_x）也把 `cond_latents` 传进 `_forward_train`；流式推理（kv_cache 分支）仍由 `prefill_source` 注入，两者互不冲突。
5. 约束：与 teacher-forcing(`clean_x`) / `independent_first_frame` **暂不支持同时用**（已加 assert）。
- 自检：因果训练前缀 forward 输出 `(1,16,4,30,52)` + 反向梯度有效，全部自检通过。

### 7.4 DMD trainer（`trainer/distillation.py`）
- `_load_prefix_v2v` 把 `real_score_v2v_ckpt` / `fake_score_v2v_ckpt` 加载进 real/fake score（均用 `bernini_v2v_prefix_init.pt`）。
- `condition_dropout=0.15`：以一定概率丢源前缀做 CFG，防 copy 捷径/欠编辑。

### 7.5 ODE trainer（`trainer/ode.py`）
- `v2v=true` 时 `cond_latent` 经 `conditional_dict` 在 `train_one_step` 注入。
- 因果 generator 的 ODE 前缀注入已实现（`_forward_train`）。
- **`ode` 与 `causal_ode` 共用同一 trainer**，区别仅在 LMDB 数据来源（双向 vs 因果 teacher 采样）。

### 7.6 Stage-1 三方案代码（2026-06-23 新增）
| 组件 | 文件 | 说明 |
|---|---|---|
| 双向 ODE 采样 | `tools/generate_v2v_ode_lmdb.py sample --prefix_v2v` | 默认双向 teacher |
| 因果 ODE 采样 | 同上 + `--causal_teacher --num_frame_per_block 3` | 因果 AR teacher 采样 |
| Stage-0 因果 teacher | `tools/train_causal_v2v_teacher.py` | 产出 `bernini_causal_v2v_teacher.pt` |
| ODE 回归 | `trainer/ode.py` + `model/ode_regression.py` | `ode` / `causal_ode` 共用 |
| Causal CD | `trainer/consistency_distillation.py` + `model/naive_consistency.py` | `causal_cd` 专用 |
| 训练入口 | `train.py` | `trainer: ode \| consistency_distillation \| score_distillation` |
| 编排脚本 | `tools/run_routeb_ode_pipeline.sh` | 双向 ODE 自动流水线 |
| 编排脚本 | `tools/run_routeb_causal_ode_pipeline.sh` | 因果 ODE 自动流水线 |

---

## 8. 关键文件与产物清单

> 已放弃的双通道方案的遗留配置/权重/数据清单见 [ROUTE_A.md](ROUTE_A.md) §4，本文不再列出，避免误用。

### 配置
| 文件 | init_method | 用途 |
|---|---|---|
| `configs/self_forcing_v2v_dmd.yaml` | — | **Stage-2 DMD**（三方案共用） |
| `configs/self_forcing_v2v_ode_routeb.yaml` | `ode` | Stage-1：双向 teacher ODE 回归 |
| `configs/self_forcing_v2v_causal_ode_routeb.yaml` | `causal_ode` | Stage-1：因果 teacher ODE 回归 |
| `configs/self_forcing_v2v_causal_cd_routeb.yaml` | `causal_cd` | Stage-1：Causal CD（免采样） |

### 权重 / checkpoint（`checkpoints/`）
| 文件 | 说明 | 用于 |
|---|---|---|
| `bernini_v2v_prefix_init.pt` (5.6G) | Bernini 原生前缀 v2v 16 通道 | student 初始化、DMD real/fake score |
| `bernini_causal_v2v_teacher.pt` | **待训**：Stage-0 因果 AR v2v teacher | `causal_ode` 采样、`causal_cd` teacher |

### 训练产出
- DMD：`logs/bernini_v2v_dmd_routeb/checkpoint_model_{000600,000650,000700}`。
- ODE 双向：`logs/bernini_v2v_ode_routeb/`（`init_method=ode`）。
- ODE 因果：`logs/bernini_v2v_causal_ode_routeb/`（`init_method=causal_ode`）。
- Causal CD：`logs/bernini_v2v_causal_cd_routeb/`（`init_method=causal_cd`）。

### ODE 数据（`ode_data/`）
| 路径 | 方案 | 说明 | 条目数 | 磁盘占用 |
|---|---|---|---|---|
| `v2v_routeb_shards/` | `ode` | 双向 teacher 采样分片 | - | 47G |
| `bernini_v2v_ode_routeb_lmdb` | `ode` | 双向 ODE LMDB（`self_forcing_v2v_ode_routeb.yaml` 实际训练数据, `latents_shape[0]`） | **2000** | 47G |
| `v2v_routeb_causal_shards/` | `causal_ode` | 因果 teacher 采样分片 | - | - |
| `bernini_v2v_causal_ode_routeb_lmdb` | `causal_ode` | 因果 ODE LMDB | - | - |

> `causal_cd` **不需要** ODE LMDB。
> `ode` 阶段实测只有 **2000 条**样本（源视频 + Bernini 双向 teacher 24 步离线采样轨迹, `cond_latent_shape` 同为 2000），量偏小，与 §11 「ODE 数据规模不足」问题对应。

---

## 9. 流程图：Stage-1 三方案 + 共用 DMD

```
                    ┌─────────────────────────────────────────────────────────┐
                    │  Stage-0（仅 causal_ode / causal_cd）                     │
                    │  train_causal_v2v_teacher.py → bernini_causal_v2v_teacher│
                    │  数据: ReCo src + tar + 指令（tar 当前大量缺失）           │
                    └──────────────────────────┬──────────────────────────────┘
                                               │
         ┌─────────────────────────────────────┼─────────────────────────────────────┐
         │                                     │                                     │
         ▼                                     ▼                                     ▼
  init_method=ode                    init_method=causal_ode              init_method=causal_cd
  (双向 ODE)                          (因果 ODE, Causal Forcing)          (Causal CD, CF++)
         │                                     │                                     │
         │ generate_v2v_ode_lmdb.py            │ generate_v2v_ode_lmdb.py            │ 无采样
         │   sample --prefix_v2v               │   sample --prefix_v2v               │
         │   teacher=双向 Bernini              │         --causal_teacher            │
         │                                     │   teacher=因果 AR                   │
         │ build LMDB                          │ build LMDB                          │
         │ self_forcing_v2v_ode_routeb.yaml    │ self_forcing_v2v_causal_ode_...yaml │ self_forcing_v2v_causal_cd_...yaml
         │ trainer=ode                         │ trainer=ode                         │ trainer=consistency_distillation
         │ 数据: 仅需 src                      │ 数据: 仅需 src                      │ 数据: src+tar 在线
         └─────────────────────────────────────┴─────────────────────────────────────┘
                                               │
                                               ▼
                    ┌─────────────────────────────────────────────────────────┐
                    │  Stage-2 DMD（三方案共用）                                │
                    │  self_forcing_v2v_dmd.yaml                                │
                    │  generator_ckpt ← Stage-1 产出                          │
                    │  real_score ← 双向 Bernini（不变）                        │
                    └──────────────────────────┬──────────────────────────────┘
                                               ▼
                                    inference.py --v2v 验证源跟随
```

**已证伪路径（勿再赌）**：`bernini_v2v_prefix_init.pt` 直接 DMD → step 700 源脱离。

**数据需求速查**（实测数字, 2026-07-03）：
| 阶段 | `ode` | `causal_ode` | `causal_cd` |
|---|---|---|---|
| Stage-0 | 不需要 | src+tar | src+tar |
| Stage-1 | 预采样 LMDB **2000 条**(47G, `ode_data/bernini_v2v_ode_routeb_lmdb`) | src（~74000） | src+tar |
| Stage-2 DMD | src+tar 配对 **11293 条** | src+tar 配对 **11293 条** | src+tar 配对 **11293 条** |

> Stage-2 DMD 三方案共用 `self_forcing_v2v_dmd.yaml`，`data_path=ReCo-Data/replace/replace_data_configs.json` 标注共 **156678 条**（`instruction_final_refine`+`src_video`+`tar_video`）；因 `background_preservation_weight=0.5>0` 会切到 `V2VPairedVideoDataset`，**强制 src+tar 同时存在**：src 已下载 156674 个(153G，基本全量)，tar 仅下载 **11293 个(68G，约7.2%)**，故实际可用训练样本 = **11293 条**，其余 145385 条因 tar 缺失被过滤掉（`tools/sync_reco_tar_videos.sh` 负责补齐 tar）。
> 另需模型权重：`generator_ckpt`（当前 `checkpoint_model_000500/model.pt`, 39.7G）+ teacher/初始化权重 `checkpoints/bernini_v2v_prefix_init.pt`（5.7G）。

---

## 10. 当前工作进度（截至 2026-06-23）

- [x] 前缀 token 方案代码落地 + DMD 跑到 step 700（源脱离，已证伪直接 DMD）
- [x] **Stage-1 三方案代码落地**：`ode` / `causal_ode` / `causal_cd` + config 切换
- [x] `tools/train_causal_v2v_teacher.py`（Stage-0 因果 AR teacher）
- [x] `generate_v2v_ode_lmdb.py --causal_teacher`（因果 ODE 采样）
- [~] **双向 ODE 采样进行中**：`ode_data/v2v_routeb_shards/`，目标 ~2000
- [~] 自动编排 `tools/run_routeb_ode_pipeline.sh`（双向 ODE）
- [ ] 补 ReCo `tar_videos/` → 训 `bernini_causal_v2v_teacher.pt`（阻塞 causal_ode / causal_cd）
- [ ] 三方案之一跑通 Stage-1 → 续 DMD → inference 源跟随验证

---

## 11. 常用命令

> 环境：`cd /apdcephfs/private_huitinglu/Self-Forcing`，`export PATH=/apdcephfs/private_huitinglu/conda_envs/bernini_distill/bin:$PATH`

### 初始化方案 `init_method=ode`（双向 teacher ODE）

**Step 1 — ODE 采样（8 卡）**
```bash
setsid nohup python -m torch.distributed.run --nnodes=1 --nproc_per_node=8 \
  --rdzv_id=7731 --rdzv_backend=c10d --rdzv_endpoint=localhost:29531 \
  tools/generate_v2v_ode_lmdb.py sample --prefix_v2v \
  --teacher_ckpt checkpoints/bernini_v2v_prefix_init.pt \
  --shard_dir ode_data/v2v_routeb_shards \
  --num_steps 24 --max_samples 2000 \
  > logs/ode_gen_routeb_resume.log 2>&1 < /dev/null & disown
```

**Step 2 — build LMDB**
```bash
python tools/generate_v2v_ode_lmdb.py build \
  --shard_dir ode_data/v2v_routeb_shards \
  --lmdb_path ode_data/bernini_v2v_ode_routeb_lmdb
```

**Step 3 — ODE 回归**
```bash
python -m torch.distributed.run --nnodes=1 --nproc_per_node=8 \
  train.py --config_path configs/self_forcing_v2v_ode_routeb.yaml \
  --logdir logs/bernini_v2v_ode_routeb --disable-wandb
```

**自动编排（采样达标 → build → ODE 回归）**
```bash
setsid nohup bash tools/run_routeb_ode_pipeline.sh > /dev/null 2>&1 & disown
cat logs/routeb_ode_pipeline.log
```

---

### 初始化方案 `init_method=causal_ode`（因果 ODE + DMD，Causal Forcing）

**Step 0 — 因果 AR v2v teacher（需 src+tar 配对）**
```bash
python -m torch.distributed.run --nnodes=1 --nproc_per_node=8 \
  tools/train_causal_v2v_teacher.py \
  --init_ckpt checkpoints/bernini_v2v_prefix_init.pt \
  --out checkpoints/bernini_causal_v2v_teacher.pt
```

**Step 1 — 因果 teacher ODE 采样**
```bash
python -m torch.distributed.run --nnodes=1 --nproc_per_node=8 \
  tools/generate_v2v_ode_lmdb.py sample --prefix_v2v --causal_teacher \
  --teacher_ckpt checkpoints/bernini_causal_v2v_teacher.pt \
  --num_frame_per_block 3 \
  --shard_dir ode_data/v2v_routeb_causal_shards \
  --num_steps 24 --max_samples 2000
```

**Step 2 — build LMDB**
```bash
python tools/generate_v2v_ode_lmdb.py build \
  --shard_dir ode_data/v2v_routeb_causal_shards \
  --lmdb_path ode_data/bernini_v2v_causal_ode_routeb_lmdb
```

**Step 3 — ODE 回归**
```bash
python -m torch.distributed.run --nnodes=1 --nproc_per_node=8 \
  train.py --config_path configs/self_forcing_v2v_causal_ode_routeb.yaml \
  --logdir logs/bernini_v2v_causal_ode_routeb --disable-wandb
```

**自动编排**
```bash
setsid nohup bash tools/run_routeb_causal_ode_pipeline.sh > /dev/null 2>&1 & disown
cat logs/routeb_causal_ode_pipeline.log
```

---

### 初始化方案 `init_method=causal_cd`（免采样，Causal Forcing++）

**Step 0 — 同上**，产出 `bernini_causal_v2v_teacher.pt`

**Step 1 — Causal CD（无 ODE 采样/LMDB）**
```bash
python -m torch.distributed.run --nnodes=1 --nproc_per_node=8 \
  train.py --config_path configs/self_forcing_v2v_causal_cd_routeb.yaml \
  --logdir logs/bernini_v2v_causal_cd_routeb --disable-wandb
```
> step ≥ `ema_start_step`(200) 后 checkpoint 含 `generator_ema`，DMD 首训应指向 EMA 权重。

---

### Stage-2 DMD（三方案共用）

```bash
mkdir -p logs/bernini_v2v_dmd_routeb
setsid nohup python -m torch.distributed.run --nnodes=1 --nproc_per_node=8 \
  --rdzv_id=9920 --rdzv_backend=c10d --rdzv_endpoint=localhost:29520 \
  train.py --config_path configs/self_forcing_v2v_dmd.yaml \
  --logdir logs/bernini_v2v_dmd_routeb --disable-wandb \
  > logs/bernini_v2v_dmd_routeb/train.log 2>&1 < /dev/null & disown
```
> 修改 config 中 `generator_ckpt` 指向对应 Stage-1 产出。续训指向最新 `checkpoint_model_XXXXXX/model.pt`。
> `real_score_v2v_ckpt` 始终 `checkpoints/bernini_v2v_prefix_init.pt`，勿用 Stage-1 ckpt 覆盖 teacher。

---

### ODE 回归通用注意
- ODE trainer 强校验 `total_batch_size == batch_size × GPU数`（8 卡 = 8）。
- 停止判据：loss 平稳 **且** `inference --v2v` 出现源跟随（约 1–3k step），不必压 loss 到极低。

### 监控
```bash
grep -avE "FutureWarning|weights_only|warnings.warn" <log> | tail -30
nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader
ls ode_data/v2v_routeb_shards/*.pt | wc -l          # 双向 ODE 进度
ls ode_data/v2v_routeb_causal_shards/*.pt | wc -l   # 因果 ODE 进度
```

---

## 12. 重要参数对齐（勿改）
- `in_channels=16`（不扩通道）、`timestep_shift=3.0`（对齐 Bernini 1.3B config）。
- `denoising_step_list=[1000,750,500,250]`、4 步因果、`num_frame_per_block=3`、`num_training_frames=21`。
- latent 形状 `[1,21,16,60,104]`（480x832，空间/8，时间 4x 下采样）。
- 数据：`ReCo-Data/replace/replace_data_configs.json`，`prompt_key=instruction_final_refine`，`src_key=src_video`。
- ODE 采样 `guidance_scale=6.0`（脚本默认）、`num_steps=24`、关键帧 `[0,6,12,18,-1]`。
- `condition_dropout`（DMD，`self_forcing_v2v_dmd.yaml`）：当前 `0.15`，**方向与直觉相反——调高加重源脱离，调低更贴源**（详见 §4.5 #4）。出现源脱离/再退化时**只能调低或先设 0**，确认源跟随稳后、出现欠编辑才小幅上调。

---

## 13. 双向（不因果）蒸馏路线（2026-06-25 新增，与因果路线并行）

> 需求：在**双向 Bernini teacher 的 2000 步 ODE 采样**基础上，蒸馏一个**双向（不因果）的 4 步 v2v 学生**。与因果路线**完全隔离**，靠一个 config 开关切换，二者可在不同 GPU 同时跑。

### 13.1 核心开关
- 新增 config 字段 **`generator_causal`**（默认 `true`，不影响任何原有因果流程）。
  - `true` → generator 走因果 `CausalWanModel`（原行为）。
  - `false` → generator 走双向 `WanModel`（不因果化）。
- real/fake score **始终双向**（原本如此，符合 Causal Forcing「DMD 阶段用双向 teacher 是对的」）。

### 13.2 为什么双向路线更简单（与 §6.2 对照）
- 学生也不因果化 → **不存在双向→因果失配**（§4.2/§6.2 的核心隐患在此消失）。
- 也不存在 injectivity 问题（§6.2）：teacher 与 student 都双向、轨迹结构天然对齐，双向 ODE 初始化是干净的。
- 所以本路线里 ODE 阶段纯粹是「4 步少步初始化」，理论上比因果版更稳；DMD 阶段沿用双向 teacher 不变。

### 13.3 生成方式差异（与因果 DMD 的唯一实质区别）
| | 因果 DMD（`dmd`） | 双向 DMD（`bidirectional_dmd`） |
|---|---|---|
| 学生生成 | self-forcing 流式 rollout + KV cache（`SelfForcingTrainingPipeline`） | 整段双向多步去噪 rollout（无 KV cache），随机退出步带梯度 |
| 源注入 | 源前缀 prefill 进 KV cache sink 区 | 源前缀 token 整段拼接（与双向 teacher 一致） |
| timestep | blockwise（每 block 不同） | 整段统一 timestep |

### 13.4 已落地代码改动
| 组件 | 文件 | 说明 |
|---|---|---|
| generator 因果可配 | `model/base.py`、`model/ode_regression.py` | 读 `generator_causal`；双向时 ODE 回归用统一 timestep |
| 双向 DMD 模型 | `model/bidirectional_dmd.py`（新，`BidirectionalDMD`） | 复用 DMD 的 real/fake score + KL 梯度，仅替换 `_run_generator` 为双向 rollout |
| 注册 | `model/__init__.py`、`trainer/distillation.py` | `distribution_loss: bidirectional_dmd` |
| 双向 few-step 推理 | `pipeline/bidirectional_inference.py`、`inference.py` | 支持 v2v `cond_latent`；`generator_causal=false` 时自动走双向推理 |

### 13.5 配置
| 文件 | 阶段 | 关键点 |
|---|---|---|
| `configs/self_forcing_v2v_ode_bidir.yaml` | Stage-1 双向 ODE 回归 | `generator_causal=false`，**复用** `ode_data/bernini_v2v_ode_routeb_lmdb`（不重采样） |
| `configs/self_forcing_v2v_dmd_bidir.yaml` | Stage-2 双向 DMD | `distribution_loss=bidirectional_dmd`，`condition_dropout=0`，`real_score_v2v_ckpt` 仍双向 Bernini |

### 13.6 产出目录
- Stage-1：`logs/bernini_v2v_ode_bidir/`
- Stage-2：`logs/bernini_v2v_dmd_bidir/`

### 13.7 启动方式（后台 8 卡）
> 环境：`cd /apdcephfs/private_huitinglu/Self-Forcing && export PATH=/apdcephfs/private_huitinglu/conda_envs/bernini_distill/bin:$PATH`

**Stage-1（双向 ODE 回归，复用现有 LMDB；当前已用此命令启动）**
```bash
mkdir -p logs/bernini_v2v_ode_bidir
setsid nohup python -m torch.distributed.run --nnodes=1 --nproc_per_node=8 \
  --rdzv_id=8841 --rdzv_backend=c10d --rdzv_endpoint=localhost:29541 \
  train.py --config_path configs/self_forcing_v2v_ode_bidir.yaml \
  --logdir logs/bernini_v2v_ode_bidir --disable-wandb \
  > logs/bernini_v2v_ode_bidir/train.log 2>&1 < /dev/null & disown
```
**Stage-2（双向 DMD，先把 config 的 `generator_ckpt` 指向 Stage-1 产出 `logs/bernini_v2v_ode_bidir/checkpoint_model_XXXXXX/model.pt`）**
```bash
mkdir -p logs/bernini_v2v_dmd_bidir
setsid nohup python -m torch.distributed.run --nnodes=1 --nproc_per_node=8 \
  --rdzv_id=9930 --rdzv_backend=c10d --rdzv_endpoint=localhost:29530 \
  train.py --config_path configs/self_forcing_v2v_dmd_bidir.yaml \
  --logdir logs/bernini_v2v_dmd_bidir --disable-wandb \
  > logs/bernini_v2v_dmd_bidir/train.log 2>&1 < /dev/null & disown
```
**验收（双向 4 步源跟随）**
```bash
python inference.py --v2v \
  --config_path configs/self_forcing_v2v_dmd_bidir.yaml \
  --checkpoint_path logs/bernini_v2v_dmd_bidir/checkpoint_model_XXXXXX/model.pt \
  --data_path /apdcephfs/private_huitinglu/ReCo-Data/replace/replace_data_configs.json \
  --output_folder logs/bernini_v2v_dmd_bidir/samples
```

### 13.8 监控
```bash
grep -avE "FutureWarning|weights_only|warnings.warn|OMP_NUM_THREADS" logs/bernini_v2v_ode_bidir/train.log | tail -30
nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader
tensorboard --logdir logs/bernini_v2v_ode_bidir/tensorboard   # Stage-1
tensorboard --logdir logs/bernini_v2v_dmd_bidir/tensorboard   # Stage-2
```
> **启动很慢属正常**：conda 环境在 cephfs(FUSE) 上 `import torch` + 读 11GB T5 / 5.6GB 主权重(×8 进程)，从启动到首个训练步可能要 ~8–12 分钟，期间 stdout 有缓冲、log 可能为空、GPU 显存逐步爬升。判活看 worker 内核栈：`cat /proc/<pid>/stack`（`fuse`/`generic_file_buffered_read` = 正在读权重，非卡死）。

### 13.9 CFG / Tensorboard（两路线均已开）
- **CFG**：双向/因果 DMD 均 `guidance_scale`（双向 3.0），`real_score` 做 cond/uncond CFG；双向 ODE `guidance_scale=6.0`（对齐离线 teacher 采样）。
- **Tensorboard**：`trainer/distillation.py` 已接 `enable_tensorboard` 开关（默认关），双向/因果 DMD config 均已开启并写 `cfg/guidance_scale`、generator/critic loss 等。

> 因果路线（§8/§11）的 config、命令、产物路径全部不变；`generator_causal` 缺省即 `true`，老 config 无需改动。

### 13.10 训练进度（截至 2026-06-25 12:45，已手动停止）
- **Stage-1 双向 ODE 回归**：已用 §13.7 命令在 8 卡跑通并验证健康（全卡 100% 利用率、~37GB/卡），随后**手动停止**（释放 GPU 给其它任务）。
- 停止时进度：**step 44**，`train/generator_loss≈0.363`、`loss/unnormalized_mean≈0.078`、`runtime/per_iteration_time≈46s/step`。
- 已落盘 checkpoint：`logs/bernini_v2v_ode_bidir/checkpoint_model_000000/`（step 0，5.6GB）；`log_iters=200`，下一个本应在 step 200。
- tensorboard 事件：`logs/bernini_v2v_ode_bidir/tensorboard/`（含 `train/generator_loss`、`loss/unnormalized_mean`、`cfg/guidance_scale=6.0` 等）。
- **每步 ~46s 偏慢**：双向整段(源前缀+目标)长序列 + gradient_checkpointing + cephfs，按此速率到 step 200 约需 ~2.5h。续跑时如需提速可考虑（未改）：关 gradient_checkpointing(显存够时)、或减小 prefix 长度评估。
- **续跑方式**：ODE trainer 暂未实现优化器/step 续训（`save` 只存 generator），重启会从 step 0 重新计步；如需接着练，直接重跑 §13.7 Stage-1 命令即可（数据/权重不变）。验收门槛见 §11「ODE 回归通用注意」：loss 平稳 + `inference --v2v` 出现源跟随（约 1–3k step），不必压到极低。
- **下一步**：Stage-1 练到出现源跟随后取一个 checkpoint，改 `configs/self_forcing_v2v_dmd_bidir.yaml` 的 `generator_ckpt` 指向它，再起 §13.7 Stage-2 双向 DMD。

---

## 14. 因果化 ODE（routeb）启动方式 + 显存 OOM 踩坑与修复（2026-06-25 新增）

> 本节针对**因果路线**的 Stage-1 ODE 回归（`configs/self_forcing_v2v_ode_routeb.yaml`，`init_method=ode`、`generator_causal` 缺省 `true` → 因果 `CausalWanModel` 学生）。与 §13 的**双向**路线区分：§13 是 `generator_causal=false`；本节是因果版。两者复用同一份 LMDB（`ode_data/bernini_v2v_ode_routeb_lmdb`），互不影响。

### 14.1 已开启的 CFG / Tensorboard（config 内置，无需命令行额外加）
`configs/self_forcing_v2v_ode_routeb.yaml` 关键项：
- **CFG**：`guidance_scale: 6.0` —— 对齐离线双向 teacher ODE 采样时的 CFG 强度（采样用 `num_steps=24, guidance_scale=6.0`），保证回归目标轨迹与采样轨迹同口径。
- **Tensorboard**：`enable_tensorboard: true` + `tensorboard_dir: logs/bernini_v2v_ode_routeb/tensorboard`。会写 `train/generator_loss`、`loss/unnormalized_mean`、`loss/loss_at_time_*`（分时段）、`cfg/guidance_scale`、`runtime/per_iteration_time` 等。
- 其它：`max_iter=3000`、`log_iters=200`、`gradient_checkpointing=true`、`lr=5e-6`、`total_batch_size=8`（8 卡，ODE trainer 强校验 `==batch_size×GPU数`、不支持梯度累积）。

### 14.2 后台 8 卡启动命令（当前在用）
> 环境：`cd /apdcephfs/private_huitinglu/Self-Forcing && export PATH=/apdcephfs/private_huitinglu/conda_envs/bernini_distill/bin:$PATH`
```bash
# 备份旧日志后再起，避免覆盖丢失现场
mv -f logs/bernini_v2v_ode_routeb_train.log logs/bernini_v2v_ode_routeb_train.log.$(date +%H%M).bak 2>/dev/null
setsid nohup python -m torch.distributed.run --nnodes=1 --nproc_per_node=8 \
  --rdzv_id=8842 --rdzv_backend=c10d --rdzv_endpoint=localhost:29542 \
  train.py --config_path configs/self_forcing_v2v_ode_routeb.yaml \
  --logdir logs/bernini_v2v_ode_routeb --disable-wandb \
  > logs/bernini_v2v_ode_routeb_train.log 2>&1 < /dev/null & disown
```
> 端口/`rdzv_id` 与双向路线（§13.7 用 29541/8841）错开，便于两路线同机并行。

### 14.3 ⚠️ 显存/主机内存 OOM 踩坑（根因 + 现场）
- **现象**：训练起来后崩溃，备份日志栈停在 `trainer/ode.py:238 → generator_loss.backward()`（rank0/rank2 等多卡同时报错）。表面像"反向时显存炸"，**实为主机内存(RAM)被吃爆触发的连锁 CUDA 错误**。
- **根因**：ODE trainer 的 dataloader **`num_workers` 写死 8** → 8 卡 × 8 = **64 个 dataloader 子进程**；LMDB 样本是已编码 latent + cond_latent，单条不小，叠加多训练任务并发，64 个 worker 的预取队列把 ~2.2T 主机内存吃爆 → OOM killer / 进程异常 → GPU 侧表现为 backward 处崩。

### 14.4 修复（已落地 `trainer/ode.py:116-123`）
把 `num_workers` 从写死 8 改为**可配置、默认降到 2**，并限制预取、关闭常驻 worker：
```python
_num_workers = getattr(config, "num_workers", 2)          # 默认 2（原写死 8）
_dl_kwargs = dict(batch_size=config.batch_size, sampler=sampler, num_workers=_num_workers)
if _num_workers > 0:
    _dl_kwargs["prefetch_factor"] = getattr(config, "prefetch_factor", 2)  # 限制每 worker 预取队列
    _dl_kwargs["persistent_workers"] = False               # 不常驻，避免 anon 内存累积
```
- 效果：8 卡下子进程从 64 → 16，主机内存占用显著下降，不再 OOM。
- 如需进一步压内存：config 里加 `num_workers: 1`（或 `0`）、`prefetch_factor: 1`。当前 config 未设 → 走默认 2。
- 同类 `num_workers=8` 写死仍存在于 `trainer/distillation.py` / `gan.py` / `diffusion.py` / `consistency_distillation.py` 等；**若这些 trainer 也遇 OOM，按相同思路改**（本次只修了 ODE）。

### 14.5 启动慢 / 判活（与 §13.8 同理，因果版再记一次）
- **从启动到首个训练步可能 ~8–12 分钟**：8 个 rank **各自** `torch.load` `checkpoints/bernini_v2v_prefix_init.pt`（5.3GB，×8 ≈ 42GB）从 cephfs(FUSE) 读，外加 import torch / T5。期间 **stdout 有缓冲、log 可能只有几行 launcher warning、GPU 显存停在 ~3GB/卡、util=0、主机内存缓慢上升**——**均属正常加载，非卡死**。
- **判活手段**：
  - `ps -eo pid,stat,wchan:24,cmd | grep train.py`：worker 处于 **`D` + `generic_file_buffered_read`** = 正在读权重。
  - `py-spy dump --pid <rank0_pid>`：栈停在 `torch/serialization.py: load_tensor` ← `model/ode_regression.py:28 (torch.load generator_ckpt)` = 在反序列化 ckpt。
  - `cat /proc/<pid>/io | grep read_bytes` 两次采样：`read_bytes` 在涨 = IO 在推进。
- **判定进入训练**：GPU util>0 且显存从 ~3GB 爬升到几十 GB、log 开始刷 `train/...` 行。

### 14.6 续跑说明
- 与 §13.10 一致：ODE trainer 的 `save` 只存 generator，**未实现 optimizer/step 续训**，重启从 step 0 重新计步；要接着练直接重跑 §14.2 命令即可（数据/权重不变）。
- 产出：`logs/bernini_v2v_ode_routeb/`（checkpoint）+ `logs/bernini_v2v_ode_routeb/tensorboard/`。
- 验收门槛见 §11「ODE 回归通用注意」：loss 平稳 + `inference --v2v` 出现源跟随（约 1–3k step），不必压 loss 到极低。

### 14.7 ⚠️ 第二处显存 OOM：step0 `backward()` GPU 显存越界（根因 + 修复 + 已验证）
> 注意与 §14.3 区分：§14.3 是**主机 RAM**被 dataloader worker 吃爆；本节是**GPU 显存**在反向时越界。两者都表现为栈停在 `generator_loss.backward()`，需用错误类型区分。

- **现象**：§14.3 修好 RAM-OOM 后，训练能加载完 ckpt 进入 step0，但**首个 `backward()` 崩溃**。日志栈：
  ```
  generator_loss.backward() → torch/autograd → fsdp/_runtime_utils.py:755 _post_backward_hook
  → torch.distributed.DistBackendError: NCCL error ... unhandled cuda error (NCCL 2.21.5)
  ```
  全卡同时报 NCCL `unhandled cuda error`（伴随 `CUDA calloc 24 bytes` 之类小分配失败）。**这不是通信 bug，是 GPU 显存在 backward 的 reduce-scatter 处耗尽**——FSDP post-backward 要 all-gather 全量参数 + 算梯度 + 开 NCCL/reduce-scatter buffer，叠加激活峰值，把卡顶爆，NCCL 内部小分配随即失败报错。
- **根因**：因果版（`causal: true`，FlexAttention + 21 帧）反向激活峰值比双向版高；config 是 ~48GB A100（双向版本已 ~37–41GB 濒临上限），generator 的 **params + grads + AdamW 优化器状态全在 GPU**，再加激活峰值 → 越界。FlexAttention 用 `create_block_mask`（稀疏 BlockMask，**非** `[L,L]` 稠密），所以**不是 mask 物化导致**，已排除。

- **修复（已落地，两处）**：
  1. **`trainer/ode.py:85-93`** 给 generator 的 `fsdp_wrap` 接入 `cpu_offload`（`fsdp_wrap` 本就支持 `CPUOffload(offload_params=...)`，原先只有 text_encoder 用了）：
     ```python
     self.model.generator = fsdp_wrap(
         self.model.generator,
         sharding_strategy=config.sharding_strategy,
         mixed_precision=config.mixed_precision,
         wrap_strategy=config.generator_fsdp_wrap_strategy,
         cpu_offload=getattr(config, "generator_cpu_offload", False)  # 新增
     )
     ```
  2. **`configs/self_forcing_v2v_ode_routeb.yaml`** 开启开关 + 启动加显存碎片整理 env：
     ```yaml
     generator_cpu_offload: true   # params/grads/AdamW 优化器状态常驻 CPU，削减常驻显存
     ```
     ```bash
     export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True   # 消碎片，濒临上限场景再省几 GB
     ```
  - 原理：`cpu_offload` 把 generator 的参数/梯度/优化器状态搬到 CPU，**常驻 GPU 显存大幅下降**，给 backward 激活峰值腾出空间；`expandable_segments` 消除分配碎片。代价是 CPU↔GPU 搬运，速度变慢（见下）。

- **✅ 已验证结果（2026-06-25 14:27）**：用上面两招重启后**成功越过 step0 backward，稳定训练**：
  - 8 卡 100% util，`nvidia-smi` 显存 ~41.8GB/卡（其中 PyTorch `reserved≈32.2GB`，余量为 CUDA 上下文 + FlexAttention triton workspace + NCCL buffer + offload pinned staging）。
  - `train/generator_loss` step9 ≈ 0.042、`loss/unnormalized_mean` ≈ 0.062，tensorboard 正常写入。
  - `per_iteration_time ≈ 30 s/iter`（cpu_offload 的速度代价；3000 iter 约 25h）。
  - step0 已存 `checkpoint_model_000000/model.pt`。
  - step0 会有一次 **FlexAttention autotune（~20s，日志刷 `triton_flex_attention_backward_* ... AUTOTUNE`）**，属正常、仅首步。

- **调优备注**：
  - 若卡更大/显存有富余且想提速：可把 `generator_cpu_offload` 设回 `false`（去掉 offload 搬运，iter time 可降数倍），但需确认 backward 峰值不越界。
  - 仍紧张时再降激活：减 `num_training_frames`（21→更小）或保持 `gradient_checkpointing: true`（已开）。
  - `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` 建议常备，对濒临上限场景近乎零成本。

---

## 15. ODE 训练 TensorBoard 标量解读（共 19 个）

TensorBoard 服务：`tensorboard --logdir logs/bernini_v2v_ode_routeb/tensorboard --host 0.0.0.0 --port 6006`，浏览器开 `http://localhost:6006` 看 SCALARS 面板。数值示例取自 2026-06-25 step≈101。

### 15.1 训练核心指标（最该盯的两个）
| 标签 | 示例值 | 含义与解读 |
|---|---|---|
| **train/generator_loss** | 0.022（min 0.011） | **ODE 回归主损失**：student 4 步因果输出 vs teacher 去噪轨迹的逐帧 flow 回归误差。0.25→0.02 持续下降 = 学进去了。**判断"学没学到"的第一指标**，平稳低位即达标，不必压到极低。 |
| **train/generator_grad_norm** | 2.36（首步 16.1） | **梯度范数**。16→2 量级回落 = 健康收敛；突然飙升/NaN = 训练发散，需警惕。 |

### 15.2 loss 分布 / 分时段细化（诊断用）
| 标签 | 示例值 | 含义 |
|---|---|---|
| loss/unnormalized_mean | 0.037 | 未归一化 loss 的**批内均值**（贴近真实误差量级，generator_loss 可能经缩放） |
| loss/unnormalized_min / max | 0.012 / 0.13 | 批内**最易/最难**样本 loss，看离散度 |
| loss/unnormalized_std | 0.037 | 批内 loss **标准差**，越小越稳 |
| **loss/by_timestep_bucket_500** | 0.044 | **t≈500 噪声档**的 loss |
| **loss/by_timestep_bucket_750** | 0.036 | **t≈750 噪声档**的 loss |

> ⚠️ **分时段桶是关键诊断工具（对应 §11 #2）**：teacher 用 24 步 CFG=6 采样、student 是 4 步因果，若**某 timestep 桶 loss 系统性偏高且降不下去**，说明该噪声段 teacher↔student 工况失配——这是排查"源跟随不达标"的重要线索。

### 15.3 timestep 采样监控
| 标签 | 示例值 | 含义 |
|---|---|---|
| timestep/mean | 816 | 每步随机采样训练 **timestep 均值** |
| timestep/min / max | 736 / 914 | 采样 timestep 范围 |

> ODE 每步从轨迹关键点 `[1000,750,500,250]`（经 `timestep_shift=3.0` warp）随机采样去噪时刻，覆盖各档即正常。

### 15.4 优化器状态
| 标签 | 示例值 | 含义 |
|---|---|---|
| **optim/lr** | 5e-6（TB 显示 0 是精度截断，非真为 0） | **学习率**，对齐 CausVid ODE init，恒定无 warmup/decay |
| optim/weight_decay | 0.01 | AdamW 权重衰减 |

### 15.5 运行时 / 资源（环境健康度）
| 标签 | 示例值 | 含义 |
|---|---|---|
| **runtime/per_iteration_time** | 29.85 s | **每步耗时**（效率指标），极稳 |
| runtime/cuda_memory_reserved_gb | 32.2（**首步峰值 74.9**） | PyTorch 保留显存。⚠️ **first=74.9GB 是 step0 瞬时峰值**（FlexAttention autotune + FSDP all-gather），稳定后降 32GB——正印证 §14.7 为何必须 cpu_offload 才不越界 |
| runtime/cuda_memory_allocated_gb | 2.99 | 实际激活分配（与 reserved 差值是缓存池） |
| runtime/batch_size_per_gpu | 1 | 单卡 batch |
| runtime/world_size | 8 | 8 卡并行 |

### 15.6 配置快照
| 标签 | 示例值 | 含义 |
|---|---|---|
| cfg/guidance_scale | 6.0 | CFG 强度，对齐离线 teacher 采样设定 |

### 15.7 一句话解读法则
1. 看 **`train/generator_loss`** → 平稳低位 = 在学；
2. 看 **`train/generator_grad_norm`** → 不爆 = 没发散；
3. 看 **`loss/by_timestep_bucket_*`** → 哪档降不下 = teacher/student 工况失配（诊断源跟随的钥匙）；
4. 其余（lr / 显存 / iter_time / timestep / world_size）是**环境健康度**，恒定即正常。

> ⚠️ **最终是否达标不能只看 loss**——必须配合 `inference.py --v2v` 抽测肉眼确认源跟随（§11/§前文反复强调，这是之前直接 DMD 翻车的根本教训）。建议 step≈1000 起定期抽测，源跟随一出现即可转 Stage-2 DMD，不必非跑满 3000。

---

## 16. ODE loss 升级：叠加背景保持项 + 6 卡续训启动（2026-06-30 新增）

> 针对**因果路线** Stage-1 ODE 回归（`configs/self_forcing_v2v_ode_routeb.yaml`）。本节记录：① loss 函数的实际构成与本轮改动；② 背景保持项的设计方案与推荐参数；③ 从 ckpt800 起的 6 卡续训启动方式。

### 16.1 ODE 训练 loss 函数当前构成

**主回归 loss（`model/ode_regression.py:148-151`，始终生效）**：
```python
mask = timestep != 0
loss = F.mse_loss(pred_image_or_video[mask], target_latent[mask], reduction="mean")
```
- student 4 步因果输出 `pred` vs teacher ODE 轨迹末端 clean latent `target_latent = ode_latent[:, -1]` 的**逐帧 flow MSE 回归**；
- **`timestep==0` 的帧被 mask 掉不计入 loss**（clean 帧不监督）；
- `denoising_loss_type: flow`。

**背景保持附加项（`ode_regression.py:160-169`，本轮新启用）**：
```python
if self.background_preservation_weight > 0 and "cond_latent" in conditional_dict:
    loss = loss + self.background_preservation_weight * bg_loss
```
- 改动前：`background_preservation_weight` 默认 `0.0` 且 config 未配置 → 该项**不生效**，loss = 纯 ODE 回归 MSE；
- 改动后：config 显式配置 `background_preservation_weight: 0.5` → loss = **ODE 回归 MSE + 0.5 × 背景保持 MSE**。

### 16.2 背景保持项的设计方案（`utils/loss.py` `background_preservation_loss`）

补的是 §4.5 #3 的缺口（「ODE 只约束目标、不约束源-目标一致性」），机制：
1. **定位编辑区**：`diff = |target − source|`（沿通道取均值），teacher 改动处 diff 大；
2. **分位阈值 + 绝对下限**：按 `mask_quantile` 取分位阈值（`mask_threshold` 兜底下限），`diff > 阈值` 判为编辑区；`dilation` 把编辑区**膨胀**一圈防边界缝隙；
3. **背景贴源约束**：在**非编辑区（背景）** 上算 `MSE(pred, source)`，强制学生在背景处贴回源。

> 它是抑制源脱离/背景漂移的**硬约束**，与 §4.5 #4 的 DMD `condition_dropout`（方向相反的旋钮）是两回事，别混调。本项属 ODE(Stage-1) 阶段、teacher 目标已含正确编辑，与主回归同向、风险低。

### 16.3 参数语义与推荐值（本轮已落地）

| 参数（config 字段） | 默认 | 本轮设置 | 作用 / 调参方向 |
|---|---|---|---|
| `background_preservation_weight` | 0.0(关) | **0.5** | 附加项权重。区间 `0.25~1.0`；背景仍漂→升 `0.75~1.0`；出现该改没改/边界重影→降 `0.25` |
| `background_preservation_mask_quantile` | 0.85 | **0.85** | 多少比例判为背景（0.85=top15% 算编辑）。编辑面积大→降 `0.7~0.8`；面积小且精→升 `0.9` |
| `background_preservation_mask_threshold` | 0.0 | **0.0** | 编辑判定的**绝对下限**，防几乎无编辑样本把噪声当编辑。看日志再定，弱编辑样本 edit_ratio 恒~15% 时加 `0.02~0.05` |
| `background_preservation_dilation` | 3 | **3** | 编辑区膨胀核（latent 空间）护边界。出现边缘缝隙/重影→`5` |

config 落地（`self_forcing_v2v_ode_routeb.yaml`，加在 `v2v: true` 附近）：
```yaml
# ---- 背景保持(抑制源脱离/背景漂移, §4.5 #3): 非编辑区强制贴回源 ----
background_preservation_weight: 0.5
background_preservation_mask_quantile: 0.85
background_preservation_mask_threshold: 0.0
background_preservation_dilation: 3
```

### 16.4 新增 TensorBoard 标量（在 §15 的 19 个基础上 +3）

`trainer/ode.py:288-293`，仅当背景保持项生效时写入：
| 标签 | 期望值 | 含义 |
|---|---|---|
| `loss/background_preservation` | 与主 loss 量级相近 | 背景区 `MSE(pred, source)` |
| `mask/background_preservation_bg_ratio` | ≈ `mask_quantile`（0.85） | 判为背景的像素占比 |
| `mask/background_preservation_edit_ratio` | ≈ `1−quantile`（0.15） | 判为编辑区的像素占比 |

> 调参对照：若肉眼编辑区明显 > 15% 而 edit_ratio 恒 ~0.15，把 `mask_quantile` 调低（0.7~0.8）；若弱编辑样本 edit_ratio 仍恒 ~15%（纯靠分位把背景噪声当编辑），给 `mask_threshold` 加小正值。

### 16.5 CFG / TensorBoard 确认（本轮核对）

- **CFG**：训练前向**不跑实时 CFG**（ODE 回归是 student 输出回归 teacher clean latent，不做 cond/uncond 两路加权）。`guidance_scale: 6.0` 仅为**记录性超参**——对齐离线双向 teacher 采样时的 CFG 强度（采样用 `num_steps=24, guidance_scale=6.0`），由 `trainer/ode.py:64-66`（step0）+ `:281`（每步）写入 TensorBoard `cfg/guidance_scale`，**不参与 loss 计算**。
- **TensorBoard**：已开启（`enable_tensorboard: true`、`tensorboard_dir: logs/bernini_v2v_ode_routeb/tensorboard`）。`trainer/ode.py:58-62` 创建 `SummaryWriter`（仅 rank0），每步 `_write_tensorboard_scalars` 写标量并 flush。

### 16.6 本轮续训启动方式（从 ckpt800、GPU 0-5 六卡）

config 改动（除 §16.3 背景保持四行外）：
```yaml
resume_ckpt: logs/bernini_v2v_ode_routeb/checkpoint_model_000800/model.pt
ckpt_step: 800
total_batch_size: 6          # 必须 == batch_size × GPU数(6 卡=6); 由 8 改 6
```
> ⚠️ ODE trainer 强校验 `total_batch_size == batch_size × GPU数`（`trainer/ode.py:131`，不支持梯度累积）。**改 GPU 数必须同步改 `total_batch_size`**：6 卡 → 6。

启动命令（GPU 0-5，端口/`rdzv_id` 用 29543/8843，与双向 29541、因果旧 29542 错开）：
```bash
cd /apdcephfs/private_huitinglu/Self-Forcing
export PATH=/apdcephfs/private_huitinglu/conda_envs/bernini_distill/bin:$PATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mv -f logs/bernini_v2v_ode_routeb_train.log logs/bernini_v2v_ode_routeb_train.log.$(date +%H%M).bak 2>/dev/null
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 setsid nohup python -m torch.distributed.run \
  --nnodes=1 --nproc_per_node=6 \
  --rdzv_id=8843 --rdzv_backend=c10d --rdzv_endpoint=localhost:29543 \
  train.py --config_path configs/self_forcing_v2v_ode_routeb.yaml \
  --logdir logs/bernini_v2v_ode_routeb --disable-wandb \
  > logs/bernini_v2v_ode_routeb_train.log 2>&1 < /dev/null & disown
```
- `CUDA_VISIBLE_DEVICES=0,1,2,3,4,5` + `--nproc_per_node=6` → **只占 GPU 0-5，GPU6/7 空闲**。
- **续训说明（同 §14.6）**：ODE trainer `save` 只存 generator，`resume_ckpt` 仅暖启动 generator 权重 + `ckpt_step` 接续步数，**优化器状态不恢复**（ODE init 阶段可接受）。
- **旧 origin-loss ckpt 隔离**：为避免当前背景保持 loss 续训覆盖/混淆旧 ckpt，已把旧版无背景保持 loss 的 checkpoint 重命名为：
  - `logs/bernini_v2v_ode_routeb/checkpoint_model_001000_origin_loss/model.pt`
  - `logs/bernini_v2v_ode_routeb/checkpoint_model_001200_origin_loss/model.pt`
  - `logs/bernini_v2v_ode_routeb/checkpoint_model_001400_origin_loss/model.pt`
  后续当前训练新生成的背景保持 loss ckpt 会使用原始目录名：`checkpoint_model_001000/001200/001400/...`。
- **启动慢属正常**（§14.5）：各 rank 从 cephfs 读 5.6GB 权重，~8-12 分钟到首步，期间 GPU 仅 CUDA context、util=0 非卡死。

### 16.7 验收（同 §11 / §15.7）

- 盯 TensorBoard：`train/generator_loss` 平稳低位 + 新增 `loss/background_preservation`、`mask/*_bg_ratio`(≈0.85)、`*_edit_ratio`(≈0.15)；
- **必须** `inference.py --v2v` 抽测肉眼确认：背景稳贴源 + 编辑区该改有改；
  - 背景仍漂 → `background_preservation_weight` 升 `0.75~1.0`；
  - 该改没改/边界重影 → 权重降 `0.25`，或 `mask_quantile` 调低、`dilation` 调 `5`。

### 16.8 背景保持 mask 增强开关（已实现，重启训练后生效）

实现位置：
- `utils/loss.py`：`background_preservation_loss` 支持 temporal smoothing、soft mask、edit ratio clamp、可选返回 mask 细节；
- `model/ode_regression.py`：ODE 训练接入新增开关；
- `model/dmd.py`：DMD 阶段同步接入新增开关；
- `trainer/ode.py`：`background_preservation_visualize_masks=true` 时按间隔保存 mask 可视化。

基础 mask / 增强 mask 都仍基于 `src-target latent diff`：
```text
diff = abs(target_latent - source_latent).mean(channel)
threshold = quantile(diff, background_preservation_mask_quantile)
```
即 `src-target latent diff mask` 仍保留在 `background_preservation_loss` 里；增强版只是对 diff mask 做 temporal smoothing、soft weight、edit_ratio clamp。

#### 如何打开基础 mask（旧 hard latent-diff 行为）

适合做 baseline / 对齐旧实验：
```yaml
background_preservation_weight: 0.5
background_preservation_mask_quantile: 0.85
background_preservation_mask_threshold: 0.0
background_preservation_dilation: 3
background_preservation_temporal_smoothing: none
background_preservation_temporal_kernel: 3
background_preservation_soft_mask: false
background_preservation_soft_temperature: 0.01
background_preservation_edit_ratio_min: 0.0
background_preservation_edit_ratio_max: 1.0
background_preservation_visualize_masks: false
background_preservation_mask_visualize_interval: 200
```

#### 如何打开增强 mask（当前 `configs/self_forcing_v2v_ode_routeb.yaml` 已设置）

当前推荐配置：
```yaml
background_preservation_weight: 0.5
background_preservation_mask_quantile: 0.85
background_preservation_mask_threshold: 0.0
background_preservation_dilation: 3
background_preservation_temporal_smoothing: max
background_preservation_temporal_kernel: 3
background_preservation_soft_mask: true
background_preservation_soft_temperature: 0.01
background_preservation_edit_ratio_min: 0.15
background_preservation_edit_ratio_max: 0.35
background_preservation_visualize_masks: true
background_preservation_mask_visualize_interval: 200
# background_preservation_mask_visualize_dir: logs/bernini_v2v_ode_routeb/background_preservation_masks
```

建议说明：
- `temporal_smoothing: max`：沿时间维扩张编辑区，减少真实编辑区域漏判；
- `soft_mask: true`：从 hard bg_mask 改为连续 `bg_weight`，边界更平滑；
- `edit_ratio_min/max: 0.15/0.35`：约束编辑区比例，避免 mask 过小或过大；
- `visualize_masks: true` + interval `200`：每 200 step 只保存 rank0 第一个 batch 的中间帧 mask 图，存储开销很小。

> 注意：当前正在运行的 ODE 进程不会自动加载这些代码/配置变更；需要重启训练才生效。

---

## 方案②本地缓存启动器（消除重启时对 cephfs 的全量并发读）

### 背景 / 问题
每次训练重启时，**每个 rank 都独立从 cephfs 全量读权重**，8 卡并发放大：
- DMD：`generator_ckpt` 39GB（generator+critic+ema+双优化器动量）+ `real_score_v2v_ckpt` 5.67GB×2（real/fake），单 rank ~51GB，×8 ≈ **~408GB cephfs 并发读**；
- ODE：`resume_ckpt` 5.68GB，×8 ≈ **~45GB**。

这是启动慢的根因（纯 IO，与"续训"无关）。方案②把权重先同步到**节点本地盘 `/dockerdata`（xfs，9TB）**，首次重启读一次 cephfs，之后每次重启从本地盘读（GB/s 级），**cephfs 零并发读**。

> 步数解析依赖目录名 `checkpoint_model_XXXXXX`（`distillation.py:256` 正则提取），本地缓存路径保留了该目录名结构，续训步数不受影响。ODE 用 config 里显式 `ckpt_step` 计步，本地命名更自由。

### 新增启动器：`tools/run_dmd_local_cache.sh`（DMD 用）
起训前把 `generator_ckpt`(39GB) + `real_score_v2v_ckpt`(5.67GB) 增量同步到 `/dockerdata/ckpt_cache`，生成把这两处路径改指本地缓存的 runtime config（`/dockerdata/ckpt_cache/self_forcing_v2v_dmd.runtime.yaml`），再用它拉起 8 卡 torchrun。按文件大小判命中，命中则跳过同步（重启零 IO）。

用法：
```bash
cd /apdcephfs/private_huitinglu/Self-Forcing
# 同步(增量, 命中秒过) + 8 卡起训
bash tools/run_dmd_local_cache.sh
# 仅预热本地缓存, 不起训
CACHE_ONLY=1 bash tools/run_dmd_local_cache.sh
```
换 checkpoint 步数：改 `configs/self_forcing_v2v_dmd.yaml` 的 `generator_ckpt` 后直接跑脚本，会自动同步新那份。

### watcher 改造：`tools/watch_restart_800.sh` / `tools/watch_restart_1400.sh`（ODE 用）
在原重启流程（等 ckpt 写完 → 停旧训练 → 改 config）后新增步骤 4.5：把 `resume_ckpt`（ODE ckpt 5.68GB）增量同步到 `/dockerdata/ckpt_cache`，生成 `resume_ckpt` 指向本地盘的 runtime config（`/dockerdata/ckpt_cache/self_forcing_v2v_ode_routeb.runtime.yaml`），步骤 5 的 torchrun 改用 `--config_path $RT` 从本地盘加载。

同时放宽停旧进程的匹配：`configs/self_forcing_v2v_ode_routeb.yaml` → `.*self_forcing_v2v_ode_routeb`，兼容用 runtime config 路径启动的进程（否则链式 watcher 无法正确 kill 上一次用 runtime config 启动的训练）。

### 注意事项
- `/dockerdata` 是节点本地盘，**容器/节点重建后缓存丢失**，届时首次重启会自动重新同步一次。
- `/dev/shm`（tmpfs，RAM）也可作缓存但会吃内存，默认用本地磁盘 `/dockerdata` 更稳。
- runtime config 只改权重路径，其余（tensorboard/logdir/超参）全部原样继承基础 config。
- 本方案只在**下一次重启**生效，不影响当前正在运行的训练。

---

## 17. DMD 阶段脸部漂移问题排查 + background_preservation 调参修复（2026-07-03 新增）

> 针对 **Stage-2 DMD**（`configs/self_forcing_v2v_dmd.yaml`）checkpoint 600 相比 400 脸部改动明显更多的问题。§16 的背景保持项设计是给 **ODE(Stage-1)** 阶段用的，DMD 阶段接入方式见 `model/dmd.py:256-280`（`dmd_loss = dmd_loss + weight * bg_loss`），本节记录 DMD 阶段该问题的复盘与修复。

### 17.1 现象

用同一份测试数据（`Data/data0` + `prompt1.txt`，眼镜/发型编辑，seed=0）对比 `checkpoint_model_000400` / `000500` / `000600` 的推理结果（`logs/bernini_v2v_dmd_routeb/eval_ckpt{400,500,600}_data0/compare_src_vs_ckpt*.png`）：**脸部本不应被编辑，但 ckpt600 比 ckpt400 出现更明显的脸型/五官改动**。三个 checkpoint 用的是完全相同的推理脚本、config、seed、测试数据，差异只在训练 step 数（500→600，共约 100~200 步续训）。

### 17.2 根因分析

1. **`background_preservation_loss` 的 mask 完全基于 src/tar latent diff 自动推断（`utils/loss.py:181-234`），不是人工标注的固定脸部保护区**：`diff = |target-source|.mean(channel)`，按 `mask_quantile` 分位阈值判编辑区。删眼镜/改发型这类编辑天然紧贴脸部边界，mask 容易连带把脸部边缘一起判成"编辑区"。
2. **两个放大泄漏的参数在 DMD config 里偏松**：
   - `background_preservation_dilation: 3` → 编辑区**空间膨胀**一圈，主动把边界往脸部扩散；
   - `background_preservation_temporal_smoothing: max` → 编辑区**跨帧取并集**，只会越滚越大、不会缩小，训练越久越容易把脸部并进编辑区。
3. **约束力度固定、不随训练步数增强**：`background_preservation_weight` 是 DMD 主 loss 上的固定加法项（`model/dmd.py:275`：`dmd_loss = dmd_loss + weight * bg_loss`），DMD 本身是分布匹配的弱监督（不逐帧监督"这里不该变"），当 mask 泄漏 + 约束权重跟不上时，泄漏区域会被主蒸馏 loss 自由改动。
4. **样本量提醒**：目前只用了单个测试视频/单 seed 观察，不能 100% 排除是这一个样本的训练震荡（DMD 属对抗式蒸馏，checkpoint 间细节非单调收敛），但从 mask 机制本身看，dilation+max-smoothing 的组合确实存在"越训越容易泄漏"的结构性风险，值得先修。

### 17.3 修复：收紧 DMD 阶段 background_preservation 参数（已落地 `configs/self_forcing_v2v_dmd.yaml`）

```yaml
# 改前 → 改后
background_preservation_weight: 0.5          → 0.8     # 约束力度不足, 上调(注释里本就允许 0.3~1.0)
background_preservation_mask_quantile: 0.85  → 0.92    # 编辑区判定收紧(原 top15% 判编辑 → top8%)
background_preservation_dilation: 3          → 1       # 关闭空间膨胀, 不再主动把编辑区边界扩散到脸部
background_preservation_temporal_smoothing: max → median  # 原 max 跨帧取并集只会越滚越大; median 不再单调扩张
background_preservation_edit_ratio_max: 0.5  → 0.35    # 编辑区占比上限收紧, 防大面积误判
```
- `background_preservation_soft_mask: true`、`soft_temperature: 0.01`、`edit_ratio_min`、`mask_threshold` 未改。
- **该修改只影响下一次续训的行为，不能修正 400/500/600 这几个已训出的 checkpoint**——它们的脸部漂移已固化在权重里，无法靠改 inference 侧参数挽回。

### 17.4 下一次训练起点

`generator_ckpt` 已改为指向 **`checkpoint_model_000800`**（原为 000500）：
```yaml
generator_ckpt: logs/bernini_v2v_dmd_routeb/checkpoint_model_000800/model.pt
```
即下一次 DMD 续训 **从 ckpt600 出发**，套用本节收紧后的 background_preservation 参数，观察续训若干步后新 checkpoint 是否脸部更稳（用 §11 的 `inference.py --v2v` 抽测流程，多测几个不同来源的视频而非只用 `data0`，避免单样本误判）。

### 17.5 更根治但未做的方向（需新增依赖，未擅自加）

当前 mask 完全依赖 src/tar latent diff 的通用启发式，没有真正的人脸感知。更根治的方案是引入人脸检测/关键点，对脸部区域施加独立于 diff-mask 的硬保护下限（即使 diff-mask 误判，脸部 `bg_weight` 也不低于某个 floor）。这需要新增人脸检测依赖（如 mediapipe/insightface），按项目规则未经确认不擅自加依赖，**留待后续需要时再评估**。
