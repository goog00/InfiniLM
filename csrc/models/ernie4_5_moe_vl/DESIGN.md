# ERNIE-4.5-VL-28B-A3B 适配设计文档

> 赛题 T2-1-1：在 InfiniCore / InfiniLM 框架内适配文心一言 4.5VL（ERNIE-4.5-VL-28B-A3B）多模态 MoE 模型。
> 代码状态：骨架 commit `9797838` + **P0 增量（2026-06）**。
> 验证环境：**沐曦 MetaX C500 ×2**（赛题必选 NVIDIA，其后端算子齐全；沐曦缺口已补，见 §3.10）。
>
> 状态图例：✅ 已实现并自洽 ｜ 🔶 已实现、待 C500 编译 + HF 数值验证 ｜ ⚠️ 可跑但需改进/核对 ｜ ❌ 未实现

---

## 1. 概述

### 1.1 赛题要求
- 指定模型：ERNIE-4.5-VL-28B-A3B-Thinking（多模态 MoE）。
- 必须支持平台：NVIDIA（必选）；可选天数 / 摩尔 / 沐曦。本仓库当前在 **沐曦 C500 ×2** 上验证。
- 核心模块：(1) Vision Tower (2) MoE (3) 多模态适配器 (4) 单卡与分布式推理。
- 正确性验证：覆盖 **文本 / 图像 / 视频** 三种模态，与 HuggingFace transformers 参照对比 token 序列一致性。
- 约束（评分项）：不修改框架公共代码、复用已有模块、不遗留调试代码、与主分支无冲突。

### 1.2 模型架构（来自 HF config.json）

| 部分 | 关键参数 |
|---|---|
| 文本 backbone | 28 层，hidden 2560，GQA 20/4 头，head_dim 128，无 QK-Norm，`use_bias=false`，`rope_theta=500000`，`tie_word_embeddings=true` |
| MoE | `moe_num_experts=[64 文本, 64 视觉]`，`moe_intermediate_size=[1536, 512]`，`moe_k=6`，2 个 shared experts，`moe_use_aux_free=true`，**第 0 层 dense、1~27 层 MoE** |
| 路由 | DeepSeek-V3 风格（**无 group routing**）：sigmoid 打分 + `e_score_correction_bias` 选 top-k，combine 权重取原始分数 |
| 3D RoPE | `rope_3d=true`，`mrope_section=[22, 22, 20]`（time/height/width，和=64=head_dim/2） |
| Vision Tower | DFNRope ViT，depth 32，embed 1280，16 头（head_dim 80），patch 14，`spatial_merge_size=2`，`quick_gelu`，2D RoPE，`attn_sep=true`（块对角注意力） |
| Adapter | spatial_conv 2 + temporal_conv 2，pixel_hidden 1280 → text_hidden 2560 |
| 特殊 token | `im_patch_id=100295`，image_start/end=101304/101305，video_start/end=101306/101307 |

### 1.3 模块与文件对照

| 模块 | 文件 |
|---|---|
| 顶层模型 | [ernie4_5_moe_vl_for_conditional_generation.{hpp,cpp}](ernie4_5_moe_vl_for_conditional_generation.cpp) |
| 文本 backbone / Decoder Layer | [ernie4_5_moe_vl_text_model](ernie4_5_moe_vl_text_model.cpp) / [..._decoder_layer](ernie4_5_moe_vl_decoder_layer.cpp) |
| 文本注意力（3D mrope） | [ernie4_5_moe_vl_attention.{hpp,cpp}](ernie4_5_moe_vl_attention.cpp) |
| MoE（aux-free 路由） | [ernie4_5_moe_vl_moe.{hpp,cpp}](ernie4_5_moe_vl_moe.cpp) |
| Vision Tower（2D rope + 块对角） | [ernie4_5_moe_vl_vision.{hpp,cpp}](ernie4_5_moe_vl_vision.cpp) |
| Adapter / Resampler | [ernie4_5_moe_vl_resampler.{hpp,cpp}](ernie4_5_moe_vl_resampler.cpp) |
| 权重映射 | [python/infinilm/modeling_utils.py](../../../python/infinilm/modeling_utils.py) |
| Processor（多模态链路） | [ernie4_5_moe_vl_processor.py](../../../python/infinilm/processors/ernie4_5_moe_vl_processor.py) |
| 正确性测试 | [test_correctness.py](../../../test/models/ernie4_5_moe_vl/test_correctness.py) |
| **InfiniCore 沐曦算子** | `InfiniCore/src/infiniop/ops/{softmax,quickgelu}/metax/*.{h,maca}` + 各 `operator.cc` |

---

## 2. 整体架构与数据流

### 2.1 设计决策（非显然）
- **token_type_ids 在模型内部从 `input_ids` 推导**（`im_patch_id` → vision），不改框架级 `Input` 结构。
- 因层异构（dense + MoE）且需透传 `token_type_ids`，**自定义** `Ernie4_5_VLMoeModel` / `DecoderLayer`。
- Vision Tower + Adapter 整体注册为 `visual`，`visual.merger.*` 对齐 HF checkpoint 前缀。
- **3D mrope 不改底层 op**：`nn::RoPE` 不支持 mrope，转而自构 sin/cos table 复用 `op::rope`（§3.3）。
- **多模态数据流不改框架公共代码**：在 ERNIE processor override `build_model_inputs` 注入多模态张量（§3.7），而非改 `BasicLLMProcessor`。

### 2.2 多模态 prefill 数据流（Python → C++）
```
chat → apply_chat_template(图项→哨兵) → resolve_multimodal_inputs(PIL)
     → processor.__call__: _preprocess_images + token级组装(image_start+im_patch×N+image_end)
     → processed_inputs{input_ids, pixel_values, grid_thw}  （存于 request）
     → build_model_inputs override: 注入 pixel_values / tgt_sizes / 3D position_ids
     → engine.forward(**dict) → C++ Input
─────────────────────────── C++ ───────────────────────────
input_ids ─┬─> embed_tokens ───────────────┐
pixel_values,grid_thw ─> Vision Tower(2D rope+块对角) ─> Adapter ─> vision_embeds
           │            merge_vision_embeddings (按 im_patch_id 1:1 散射)
           └─> derive_token_type_ids ─> [merged_embeds, 3D position_ids, token_type_ids]
                          28 × DecoderLayer (layer0 dense, 1~27 MoE; mrope; aux-free 路由)
                          → final RMSNorm → lm_head → logits
```
> Decode：视觉已在 KV cache，走纯文本分支；position_ids 取该 token 的 mrope 三元组（文本段三轴相等）。

---

## 3. 模块设计与实现状态

### 3.1 顶层模型 `Ernie4_5_VLMoeForConditionalGeneration` ⚠️
- ✅ 模块组装、注册名对齐 HF、`tie_word_embeddings` 共享、KV cache 分配、`derive_token_type_ids`。
- ⚠️ `merge_vision_embeddings` 仅支持 **batch==1**，CPU 逐 token 散射（P2 优化）。

### 3.2 文本 backbone + Decoder Layer ✅
- ✅ `embed_tokens` + 28 层异构 decoder + 末端 RMSNorm；`compute_is_moe_layer`（layer0 dense，1~27 MoE）；`forward`/`forward_embeds` 双入口。

### 3.3 文本注意力 + 3D mrope ✅（文本路径已验证）
- ✅ QKV 融合、GQA（20/4）、`use_bias=false`、o_proj、TP 切分、static/paged 两路径、无 QK-Norm。L0 stream 与 HF 完全一致。
- ✅ **algo = GPT-J / interleaved**（`get_rope(..., GPT_J)`）。⚠️ 原默认 GPT_NEOX 是 bug，见 §8.1：ERNIE 用相邻配对 + `sin_pos=[θ0,θ0,θ1,θ1]`。
- 🔶 **3D mrope**（`build_mrope_`）：自构 `[seq, 64]` sin/cos table，三轴相等时退化为标准 GPT-J 1D（纯文本即走此退化，已验证）。**注意 §8.1：HF `apply_rotary_3d` 的轴↔频率分配为 `freq_allocation=20`（j<44 偶→height 奇→width，j≥44→time），`build_mrope_` 的 `[22,22,20]` 顺序分配在 image 路径需按此核对/改写**。
- ⚠️ mrope table 每层重建（28×）+ H2D，正确但浪费 → P2。
- **VERIFY(C500/image)**：3D 分配轴序（见上）、`rope_theta=5e5`。

### 3.4 MoE 模块 ✅（文本路径已验证）
- ✅ 模态专家（文本 `[0,64)`/视觉 `[64,128)`，intermediate 1536/512）、双门 `gate.weight`/`weight_1`、2 shared experts、`set_alpha` 折叠 combine、`e_score_correction_bias` [2,64]。
- ✅ **aux-free 路由**（gate forward）：CPU `softmax(logits)+correction_bias` 选 top-k，combine 取**未加 bias 的原始 softmax prob、不归一化**。⚠️ 原 sigmoid + 归一化是 bug，见 §8.2。tok0 路由与 HF 一字不差。
- ✅ **token_type_ids 在 CPU 显式构造**（text_model）——MetaX `zeros(I64,device)` 不清零是 bug，会让全部 token 误判 vision，见 §8.4。
- ✅ **gate 权重经 remapper `tensor.t().contiguous()` 转置**（checkpoint `[hidden,E]`）——`.t()` 非连续视图未材料化曾导致路由乱选，见 §8.3。
- 🔶 **放弃 infiniop `topkrouter`**：硬编码 DeepSeek 256 专家 + 8-group（`width!=256`→`BAD_PARAM`），对 ERNIE 64 专家无 group 不适用。
- ⚠️ dispatch 逐 token×逐 expert 调 `down_proj`（RowParallel），TP=2 每次 all-reduce，正确但慢 → P2。gate 用 bf16（HF fp32），L1 量级 414 vs 488，不影响 argmax。

### 3.5 Vision Tower（DFNRope ViT）🔶
- ✅ 线性 patch embed、32 ViT block、融合 QKV、`quick_gelu` MLP、末端 LayerNorm `visual.norm1`。
- 🔶 **2D RoPE 已实现**（`build_rope_`）：head_dim 80 → 表 `[N,40]`，前 20 维 height、后 20 维 width；patch (h,w) 坐标按 processor 的 merge-block patchify 序生成（与 Qwen2-VL 一致）；NEOX rotate_half。
- 🔶 **块对角注意力已实现**（`build_cu_seqlens_`）：按帧分段、每段内独立 scaled-dot-product，替代缺失的 mask；单图=单段=全注意力。
- **VERIFY(C500)**：vision `theta=1e4`、(h,w) 序与 patchify 对齐。

### 3.6 Adapter / Resampler ⚠️
- ✅ spatial_linear + temporal_linear（视频）+ mlp 投影 + after_norm；视频检测（任一 `t>1`）。
- ⚠️ 激活用 `gelu` 为猜测，需核对是否 `quick_gelu`；`after_norm` 无 bias 靠 remap 合成。

### 3.7 Processor（Python，多模态链路）🔶
- ✅ 纯文本复用 `BasicLLMProcessor`；图像预处理（CLIP 归一化 + patchify）。
- 🔶 **多模态 chat template 已实现**：图项 → 哨兵标记（角色结构仍由 tokenizer 模板渲染），`__call__` 按哨兵做 **token 级组装** `image_start + im_patch×N + image_end`（N=`h*w/spatial_conv²`，用已知 token id，**不依赖未知占位符字符串**）。
- 🔶 **3D position_ids 已实现**（`_build_3d_position_ids`）：Qwen2-VL get_rope_index——文本顺序、图像 2D 网格 + 段后偏移。
- 🔶 **`build_model_inputs` override**：从 `req.processed_inputs` 注入 `pixel_values`（prefill）、`tgt_sizes`、`[3,seq]` mrope `position_ids`，打通 pixel_values → C++ forward。
- ❌ 视频仍 `NotImplementedError`。
- **VERIFY(C500/HF)**：`image_start/end` 是否真包裹 patch run、段边界 tokenization、`from_list` 建 bf16 pixel_values、归一化常数、get_rope_index 轴序。

### 3.8 权重映射 `_remap_ernie4_5_vl` 🔶
- ✅ 已注册到 `_WEIGHT_REMAPPER`；gate 转置已修为 **`tensor.t().contiguous()`**（见 §8.3，非连续视图曾打乱权重）。
- ⚠️ 文本相关前缀已实测有效；vision/resampler 前缀映射仍为**推测**，含 no-op 替换 → image 路径待真实核对 + 清理（P1）。

### 3.9 正确性测试 🔶
- 🔶 **`run_reference` 已实现**：transformers `AutoModelForCausalLM`+`AutoProcessor` greedy 参照，返回 token ids + 文本；`compare` 文本对比。
- ✅ 已清理 `run_infinilm` 全部 `[DEBUG]` 打印。
- 🔶 text / image case 可跑到 forward（占位符链路已通）；video 仍 NotImplementedError。
- ⚠️ `_disable_maca_device_heap`（C500 显存）保留；HF 参照 message 格式 / token 对比口径需 C500 校。

### 3.10 InfiniCore 沐曦（metax）算子 🔶（新增）
- 背景：沐曦 metax 后端缺 vision 必需的 `softmax`（非 causal）与 `quick_gelu`（NVIDIA 齐全），运行时会报 "No implementation"。
- 🔶 **metax `quick_gelu`**：复用 elementwise 框架 + 共享 `QuickGeluOp`（`quickgelu/metax/*.{h,maca}` + operator.cc 注册 4 处）。VERIFY：bf16 走 `__nv_bfloat16` 分支是否命中（不中改 `cuda_bfloat16`）。
- 🔶 **metax `softmax`**：镜像 nvidia block/warp 启发式 + `causal_softmax` 的 metax 习语（`INFINIOP_METAX_KERNEL`/`hcStream_t`/`__hpcc_bfloat16`/`METAX_BLOCK_SIZE_*`）。
- 构建自动纳入（`xmake/metax.lua` glob `ops/*/metax/*.maca`），**模型侧零改动**经 infiniop 分发生效。
- 其余依赖算子（rope/gemm→matmul,linear/layer_norm/rms_norm/add/swiglu/silu/embedding）metax 已有。

---

## 4. 实现状态总表

### P0 实现状态
- ✅ **文本路径已在 C500×2 编译 + HF 逐层/路由数值验证通过**（修复 4 个 bug，见 §8）。
- 🔶 image/video 数值核心已实现但**未在真实输入上验证**。
- **数值核心**：MoE aux-free 路由（CPU **softmax**+bias+topk）、文本 3D mrope（**GPT-J**）、视觉 2D RoPE + 块对角 attention。
- **多模态链路**：多模态 chat template + token 级占位符组装 + `build_model_inputs` 注入 pixel_values/tgt_sizes/3D pos + get_rope_index。
- **沐曦算子**：metax `softmax` + `quick_gelu`。
- **正确性**：`run_reference`（HF 参照）+ 清理调试。
- （骨架已有）顶层组装、文本 backbone、异构 decoder、Adapter 结构、图像预处理、weight remap 注册。

### ❌ 未实现
- 视频预处理与链路（`_preprocess_videos`；三模态里 text/image 通、video 待做）。

### ⚠️ 待优化 / 待核对
1. **`csrc/engine/rank_worker.cpp` 的 `[LOGIT DEBUG]` 调试代码仍未回退**（改公共文件 + 留调试，评分硬性扣分项，P1 优先）。
2. 权重 remap 前缀/张量名用真实 checkpoint 核对、清 no-op。
3. resampler 激活（gelu vs quick_gelu）、processor 归一化常数对齐真实配置。
4. 性能 P2：MoE 逐 token all-reduce、mrope 表每层重建、`merge`/`pixel_values` 的 CPU/from_list、`get_rope_index` 每步重算、batch>1。

---

## 5. 风险与依赖（更新）

| # | 风险 | 状态 |
|---|---|---|
| R1 | `nn::RoPE` 不支持 mrope/3D | ✅ 解决：自构 sin/cos table 走 `op::rope` |
| R2 | 缺带 additive bias 的 topk 选择 op | ✅ 解决：CPU sigmoid+bias+topk（`topkrouter` 硬编码 256 不适用） |
| R4 | 视觉 2D RoPE 表构造（变分辨率） | ✅ 解决：DFNRope/Qwen2-VL 式构表 |
| R6 | metax 缺 softmax/quickgelu | 🔶 已补 kernel，待 C500 编译验证（尤其 quickgelu bf16 分支） |
| R7 | mrope/路由/vision 数值约定（rope style、段序、theta、sigmoid） | 🔶 需 HF 源码 + C500 逐层比对（代码内 `VERIFY` 注记） |
| R8 | 占位符包裹、`from_list` bf16 pixel_values、get_rope_index 轴序 | 🔶 待 C500 链路验证 |
| R5 | 分布式 TP=2 | 🔶 实现支持、功能正确；MoE all-reduce 慢，待验证 |
| R3 | 真实 checkpoint state_dict 命名 | ⚠️ 仍待核对（P1） |

## 6. 路线图

**P0 — 数值核心 + 链路打通 ✅（已实现，待 C500 验证）**
- run_reference、文本 mrope、视觉 2D rope + 块对角、MoE aux-free 路由、多模态 chat template + 数据流、metax softmax/quickgelu。

**P1 — 补全 + 合规（待办）**
1. 回退 `rank_worker.cpp` 的 `[LOGIT DEBUG]`（评分硬性项）。
2. 视频预处理与链路。
3. weight remap 清 no-op + 真实 checkpoint 核对命名。

**P2 — 性能（加分）**
1. MoE GPU 化：去逐 token all-reduce、grouped gemm。
2. mrope 表上提到 text_model 一次构建；`pixel_values`/位置 ID 高效化；batch>1。
3. TP=2 性能与正确性实测。

---

## 7. C500 ×2 构建 / 验证

```bash
# InfiniCore 带沐曦后端：C500 是 MC stack，必须 --use-mc；多卡 TP 必须 --ccl（见 §8.6）
cd InfiniCore && xmake f --metax-gpu=true --use-mc=true --ccl=true -cv && xmake && xmake install
cd InfiniLM && pip install -e . --no-build-isolation
# 依赖（见 §8.6）：pip install "huggingface-hub<1.0" sentencepiece decord accelerate
# 三模态：--device cuda（METAX 经 device.py 映射）；59G 权重单卡 OOM，必须 --tp 2
python test/models/ernie4_5_moe_vl/test_correctness.py --model <path> --device cuda --tp 2 --cases text
python test/models/ernie4_5_moe_vl/test_correctness.py --model <path> --device cuda --tp 2 --cases image --image <img>
```
**验证顺序**：① metax 算子编过 → ② **text case 与 HF 逐层/路由对齐 ✅（已完成，见 §8）** → ③ image case 链路跑通（占位符数==视觉 token 数、无 shape 报错）再逐层比对 HF → ④ video。

---

## 8. 调试记录：文本路径数值 bug 定位与修复（2026-06-14，C500×2）

**背景**：C500×2（TP=2）编译通过后，文本 case 输出全是垃圾（重复单一 token，先是数字"7"/"0"，后是换行）。`[LOGIT DEBUG]` 显示 logits 有限、无 NaN/Inf，但 top 全是低 id token——典型「前向数值活着但语义全错」。逐层 + HF 对照，定位到 **4 个数值 bug，全部不在最初 §3 的设计假设里**。

### 8.0 验证方法（关键，纯静态分析多次误判）
- **HF 参照在 MetaX 上跑通**：MetaX torch 的 `F.scaled_dot_product_attention` 无可用 kernel（`No available kernel`）→ monkeypatch 每层 `self_attn.attn_func = self_attn.core_attn` 走手写 eager attention。
- **HF forward 入参**：`rope_3d=True` 强制要 `position_ids`；纯文本传 `[1,seq,3]` 三轴 arange；`token_type_ids` 长度须为 **seq+1**（HF 怪癖）、全 0。
- **逐层对照**：用 `forward_hook` 抓 HF 每层 hidden 统计 + gate logits + 各专家输出，与模型侧 `ERNIE_DBG` 打点（embed / Lk attn_out / Lk moe_out / **Lk stream=hidden+residual** / final_norm）逐层比对，锁定**首个分歧层**。教训：静态分析数次定位错（一度误判 gate 未转置、误改 gate 声明导致 load shape mismatch），**必须实测对照**。

### 8.1 bug① rope 配对方式（GPT-J vs GPT-NEOX）
- **现象**：L0 起就偏；输出重复低 id token。
- **根因**：ERNIE `RopeEmbedding.apply_rotary` 用 **GPT-J / interleaved**（相邻配对 (q0,q1)(q2,q3)，`sin_pos=[θ0,θ0,θ1,θ1,…]`，`rotate_half=[-q1,q0,-q3,q2,…]`）；框架 `get_rope` 默认 **GPT_NEOX**（半分配对）。两者结果不同 → attention 全错。
- **修**：`get_rope(model_config, device, GPT_J)`。HF 的 `apply_rotary_3d` 对纯文本三轴相等时退化为 GPT-J 1D rope；`freq_allocation=20`=`mrope_section` 末段（freq j<44：偶→height 奇→width；j≥44→time）。
- **验证**：修后 **L0 stream 与 HF 完全一致**（absmax 6.25 vs 6.28，head 0.1104/0.1006 vs 0.1104/0.1016）。

### 8.2 bug② MoE gating 激活（softmax vs sigmoid）
- **根因**：HF `gate.act = softmax`（不是我假设的 DeepSeek sigmoid）。选择 = `topk(softmax(logits) + e_score_correction_bias[0])`；组合权重 = 选中处的**原始 softmax prob**（不含 bias）；**MOEAllGatherLayerV2 fused 路径不归一化**。
- **修**：gate forward 改数值稳定 softmax，删 `norm_topk_prob` 归一化（连同删除该成员/构造读取）。

### 8.3 bug③ gate 权重转置未材料化（决定性）
- **现象**：MoE 输入 absmax=15.375、shared=4.72 **均与 HF 一致**，但**选错专家**（我们 [16,0,9,…] vs HF [5,37,56,18,35,1]）。
- **根因**：checkpoint `mlp.gate.weight` 存为 `[hidden,E]=[2560,64]`；remapper `_remap_ernie4_5_vl` 用 `tensor.t()` 转置到 `[E,hidden]`——但 **`.t()` 是非连续 stride 视图**，且 checkpoint 已是 bf16（dtype 转换 no-op，不材料化），`infinicore.from_torch` 按**连续内存**读取 → gate 权重被打乱 → 路由乱选。
- **修**：`tensor.t().contiguous()`（纯 Python 改动，无需重编 C++）。
- **验证**：修后 tok0 选中专家 [5,37,56,18,35,1]、权重 [0.2604,0.1735,0.1416,…] **与 HF 一字不差**。

### 8.4 bug④ token_type_ids 全成 vision（MetaX zeros 不可靠）
- **现象**：3456 次 MoE dispatch **全是 `mod=1`（vision），无一 `mod=0`** → 文本 token 全走 vision gate/专家。
- **根因**：`infinicore::Tensor::zeros(I64, device=metax)` **不可靠清零**（device 端整型 fill 似未实现，留垃圾）。
- **修**：CPU 显式 `std::vector<int64_t>(numel,0)` + `from_blob(Device::cpu())` + `->to(device)`。

### 8.5 结果
- 文本逐层 + 路由 + 输出**与 HF 对齐**：`L1 stream` absmax 414 vs HF 488（差异源于 gate 我们 bf16、HF fp32，可接受，不改 argmax）；tok0 路由完全一致；输出连贯中文（`用户要求用一句话介绍自己，我需要简洁明了地介绍ERNIE…` vs HF `用户让我用一句话介绍自己…`）。

### 8.6 构建 / 运行环境坑（C500）
| 问题 | 解决 |
|---|---|
| 编译 `fatal error: hcblas/hcblas.h` | `xmake f` 加 **`--use-mc=true`**（C500 是 MC stack=mcblas，否则走 HPCC 找 hcblas） |
| TP=2 `infinicclCommInitAll` Error 5（DEVICE_TYPE_NOT_SUPPORTED） | `xmake f` 加 **`--ccl=true`**（否则 metax CCL 编成 NOOP 桩） |
| `ImportError: huggingface-hub>=0.34,<1.0` | `pip install "huggingface-hub<1.0"` |
| tokenizer remote code 需 `decord`,`sentencepiece` | `pip install sentencepiece decord` |
| HF 参照 `device_map=auto` 需 `accelerate` | `pip install accelerate` |
| 59G 权重单卡 64G **OOM** | 必须 **TP=2**（`--tp 2`）；`--device cuda`（METAX 经 device.py 映射 cuda） |

### 8.7 遗留（提交前必做）
- **删除全部调试代码**（评分硬性）：`ernie_debug.hpp` + 三处 `ERNIE_DBG` 打点（text_model / decoder_layer / moe）+ `csrc/engine/rank_worker.cpp` 的 `[LOGIT DEBUG]`。
- image / video 路径尚未端到端验证（vision tower + metax softmax/quick_gelu 未在真实图像上跑过）。

---
*本文档随实现推进增量更新。当前：**文本 case 已在 C500×2 上数值打通并与 HF 对齐**（§8）；image/video 与调试代码清理待办。*
