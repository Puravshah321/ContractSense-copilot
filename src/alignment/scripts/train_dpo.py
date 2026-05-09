"""
train_dpo.py (entry point script)
Runs the full DPO pipeline in sequence:
1. Dataset builder → creates raw DPO pairs from generation outputs
2. Dataset cleaner → validates and filters pairs
3. Dataset augmenter → scales to target size
4. DPO trainer → trains the model (GPU required)
5. Training monitor → logs and plots results
"""

import sys
import json
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from alignment.config import (
    GENERATION_TRAIN_PATH, GENERATION_EVAL_PATH,
    RAW_DPO_DATASET_PATH, VALIDATED_DATASET_PATH, AUGMENTED_DATASET_PATH,
    LORA_MODEL_PATH, DPO_OUTPUT_DIR, TRAINING_LOG_PATH,
    TARGET_DATASET_SIZE, MAX_SEQ_LENGTH, DPO_BETA,
    NUM_EPOCHS, BATCH_SIZE, LEARNING_RATE,
    GRADIENT_ACCUMULATION_STEPS, LORA_R, LORA_ALPHA, LORA_DROPOUT,
    USE_4BIT_QUANTIZATION,
)
from alignment.dataset.dataset_builder import build_dataset_from_generation_files
from alignment.dataset.dataset_cleaner import clean_dataset
from alignment.dataset.dataset_augmenter import augment_dataset


def parse_args():
    parser = argparse.ArgumentParser(description="Full DPO Training Pipeline")
    parser.add_argument("--skip-build", action="store_true",
                        help="Skip dataset building (use existing dataset)")
    parser.add_argument("--skip-clean", action="store_true",
                        help="Skip dataset cleaning")
    parser.add_argument("--skip-augment", action="store_true",
                        help="Skip dataset augmentation")
    parser.add_argument("--skip-train", action="store_true",
                        help="Skip training (dataset prep only)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Validate setup without training")
    parser.add_argument("--target-size", type=int, default=TARGET_DATASET_SIZE,
                        help="Target dataset size")
    parser.add_argument("--model-path", default=LORA_MODEL_PATH,
                        help="Path to LoRA model")
    parser.add_argument("--output-dir", default=DPO_OUTPUT_DIR,
                        help="Output directory for trained model")
    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 60)
    print("ContractSense DPO Alignment Pipeline")
    print("=" * 60)

    # ── Step 1: Build dataset ──
    if not args.skip_build:
        print("\n📦 STEP 1: Building DPO dataset from generation outputs...")
        build_dataset_from_generation_files(
            GENERATION_TRAIN_PATH,
            GENERATION_EVAL_PATH,
            RAW_DPO_DATASET_PATH,
            target_size=args.target_size,
        )
    else:
        print("\n⏭️  Skipping dataset build")

    # ── Step 2: Clean dataset ──
    if not args.skip_clean:
        print("\n🧹 STEP 2: Cleaning and validating dataset...")
        clean_dataset(
            RAW_DPO_DATASET_PATH,
            VALIDATED_DATASET_PATH,
            max_seq_tokens=MAX_SEQ_LENGTH,
            balance_risk=True,
        )
    else:
        print("\n⏭️  Skipping dataset cleaning")

    # ── Step 3: Augment dataset ──
    if not args.skip_augment:
        print("\n🔄 STEP 3: Augmenting dataset to target size...")
        augment_dataset(
            VALIDATED_DATASET_PATH,
            AUGMENTED_DATASET_PATH,
            target_size=args.target_size,
        )
    else:
        print("\n⏭️  Skipping augmentation")

    # ── Step 4: Train DPO ──
    if not args.skip_train:
        print("\n🚀 STEP 4: Starting DPO training...")
        print("⚠️  This requires GPU. If running locally without GPU, use --skip-train")
        print("   For GPU training, use the Colab notebook in colab/ folder.")

        try:
            from alignment.training.dpo_trainer import train_dpo

            training_dataset = AUGMENTED_DATASET_PATH
            if not Path(training_dataset).exists():
                training_dataset = VALIDATED_DATASET_PATH
            if not Path(training_dataset).exists():
                training_dataset = RAW_DPO_DATASET_PATH

            train_dpo(
                model_path=args.model_path,
                dataset_path=training_dataset,
                output_dir=args.output_dir,
                beta=DPO_BETA,
                use_4bit=USE_4BIT_QUANTIZATION,
                lora_r=LORA_R,
                lora_alpha=LORA_ALPHA,
                lora_dropout=LORA_DROPOUT,
                num_epochs=NUM_EPOCHS,
                batch_size=BATCH_SIZE,
                grad_accum=GRADIENT_ACCUMULATION_STEPS,
                lr=LEARNING_RATE,
                max_length=MAX_SEQ_LENGTH,
                dry_run=args.dry_run,
            )

            # ── Step 5: Monitor ──
            print("\n📊 STEP 5: Generating training logs & plots...")
            trainer_state = Path(args.output_dir) / "trainer_state.json"
            if trainer_state.exists():
                from alignment.training.training_monitor import build_monitor_from_trainer_state
                monitor = build_monitor_from_trainer_state(str(trainer_state), TRAINING_LOG_PATH)
                monitor.plot_loss_curve()
                monitor.plot_epoch_metrics()

        except ImportError as e:
            print(f"\n⚠️  Cannot train locally: {e}")
            print("   Use the Colab notebook at colab/06_dpo_alignment_colab.ipynb")
    else:
        print("\n⏭️  Skipping training")

    print("\n" + "=" * 60)
    print("✅ Pipeline complete!")
    print("=" * 60)

    print("\nGenerated files:")
    for path in [RAW_DPO_DATASET_PATH, VALIDATED_DATASET_PATH, AUGMENTED_DATASET_PATH]:
        p = Path(path)
        if p.exists():
            size_mb = p.stat().st_size / (1024 * 1024)
            with open(p) as f:
                count = len(json.load(f))
            print(f"  ✅ {p.name}: {count} pairs ({size_mb:.1f} MB)")
        else:
            print(f"  ❌ {p.name}: not found")


if __name__ == "__main__":
    main()
