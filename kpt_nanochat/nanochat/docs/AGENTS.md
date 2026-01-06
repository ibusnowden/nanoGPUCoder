# R1-Style Post-Training Pipeline (nanochat)

This repo now supports a three-stage “cold start → GRPO → rejection sampling SFT” pipeline aligned with modern R1-style reasoning training.

## Repo Map (Relevant Files)

- `scripts/chat_sft.py`: SFT entrypoint (now supports recipes via `--sft_recipe`)
- `scripts/chat_grpo.py`: mixed-task GRPO with KL regularization (Phase 2)
- `scripts/chat_rl.py`: GSM8K RL entrypoint (single-task baseline)
- `scripts/rs_generate.py`: rejection-sampling generator (RS-CoT → JSONL)
- `nanochat/data_recipes.py`: dataset recipes for SFT (default, R1 cold-start, R1 mixed, R1 RS-SFT)
- `tasks/openthoughts.py`: OpenThoughts loader (CoT or answer-only)
- `tasks/jsonl_chat.py`: local JSONL chat loader (for RS-CoT SFT)
- `tasks/mmlu.py`: supports `subjects="science"` (MMLU science slice)
- `tasks/smoltalk.py`: SmolTalk chat SFT dataset
- `slurm/itiger_r1_ot_sft.sh`: Phase 1 cold-start SFT
- `slurm/itiger_r1_grpo.sh`: Phase 2 mixed-task GRPO
- `slurm/itiger_r1_full_run.sh`: submit the full pipeline with dependencies
- `slurm/itiger_r1_rs_generate.sh`: Phase 3a RS-CoT generation
- `slurm/itiger_r1_rs_sft.sh`: Phase 3b RS-CoT + Chat-SFT mix
- `slurm/itiger_r1_eval.sh`: Phase 4 downstream evaluation
- `scripts/prepare_dolci_think.py`: Dolci-Think preprocessing script
- `tasks/dolci_think.py`: Dolci-Think dataset loader

## Using Dolci-Think (Alternative to OpenThoughts)

Dolci-Think-SFT-7B can replace OpenThoughts as the cold-start SFT data source. It provides
high-quality reasoning traces with `<think>` blocks, suitable for training 1.8B reasoning models.

### Step 1: Preprocess Dolci-Think (Run Once)

Filter the 2.2M dataset down to ~800k English-only examples with stratified sampling:

```bash
python scripts/prepare_dolci_think.py \
    --output ~/.cache/nanochat/dolci_think_800k.jsonl \
    --target_rows 800000 \
    --max_tokens 1500
```

Target distribution (for 1.8B reasoning model):
- 40% structured reasoning (math/logic/science)
- 20% coding/algorithmic
- 15% general instruction/chat
- 10% safety + refusal
- 15% short reasoning (1-3 sentence explanation)

### Step 2: Run Dolci-Think SFT (Phase 1)

Use the existing SFT script with `SFT_RECIPE=dolci_mid`:

```bash
sbatch --export=SFT_RECIPE=dolci_mid,SFT_SOURCE=chatsft,DOLCI_STOP=800000 slurm/itiger_r1_ot_sft.sh
```

This starts from the d32 checkpoint (`SFT_SOURCE=chatsft`) and trains on Dolci-Think.

### Step 3: Continue with GRPO + RS (Phases 2-3)

The remaining pipeline stays the same:
```bash
sbatch slurm/itiger_r1_grpo.sh        # Phase 2: GRPO RL
sbatch slurm/itiger_r1_rs_generate.sh # Phase 3a: RS generation
sbatch slurm/itiger_r1_rs_sft.sh      # Phase 3b: RS-SFT 
```

## Phase 4: Downstream Evaluation

```bash
sbatch slurm/itiger_r1_eval.sh
```

## Using OpenThoughts Mixed (r1_ot_mixed)

Train a mixed recipe so the model learns when to think (OpenThoughts) and when to answer directly (SmolTalk):

```bash
torchrun --standalone --nproc_per_node=8 -m scripts.chat_sft -- \
    --source=sft \
    --model_tag=d32 \
    --step=20832 \
    --sft_recipe=r1_ot_mixed \
    --ot_stop=1000000 \
    --chat_ratio=0.30
```

Slurm equivalent:

```bash
sbatch --export=SFT_SOURCE=sft,SFT_RECIPE=r1_ot_mixed,OT_STOP=1000000,CHAT_RATIO=0.30 slurm/itiger_r1_ot_sft.sh
```
