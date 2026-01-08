#!/usr/bin/env python3
"""
FINAL TRAINING: YOLOv8, YOLOv10, YOLOv11
Training dengan PPE-Dataset-Balanced untuk comparison yang proper
"""

import os
import time
from ultralytics import YOLO

def train_models(model_variants, data_yaml):
    """Train multiple YOLO models dengan dataset balanced"""

    results_summary = []

    for idx, variant in enumerate(model_variants, 1):
        print(f'\n{"#"*80}')
        print(f'# MODEL {idx}/{len(model_variants)}: {variant}')
        print(f'{"#"*80}\n')

        try:
            # Initialize model
            model = YOLO(variant)

            # Train dengan konfigurasi optimal
            start_time = time.time()

            model.train(
                data=data_yaml,
                epochs=100,
                imgsz=800,          # Larger untuk small objects
                batch=8,
                device=1,           # GPU device
                workers=8,
                save_period=10,
                project='YOLO-PPE-Final-Comparison',
                name=f'{variant.replace(".pt", "")}_balanced',
                exist_ok=True,

                # Augmentation untuk minority classes
                mosaic=1.0,         # Penting untuk small objects
                copy_paste=0.3,     # Bagus untuk minority classes

                # Optimizer
                optimizer='AdamW',
                lr0=0.001,
                cos_lr=True,

                # Patience untuk early stopping
                patience=50,

                # Save best model
                save=True,
                plots=True,
            )

            training_time = time.time() - start_time

            # Validate
            print(f"\n{'='*80}")
            print(f"VALIDATING: {variant}")
            print(f"{'='*80}")

            metrics = model.val()

            # Summary
            summary = {
                'model': variant,
                'training_time_hours': training_time / 3600,
                'mAP50': float(metrics.box.map50),
                'mAP50-95': float(metrics.box.map),
                'precision': float(metrics.box.mp),
                'recall': float(metrics.box.mr),
            }

            results_summary.append(summary)

            print(f'\n✅ {variant} Training SELESAI!')
            print(f'   mAP50: {summary["mAP50"]:.4f}')
            print(f'   mAP50-95: {summary["mAP50-95"]:.4f}')
            print(f'   Time: {summary["training_time_hours"]:.2f} hours')

        except Exception as e:
            print(f'\n❌ ERROR training {variant}: {str(e)}')
            import traceback
            traceback.print_exc()

    return results_summary

def main():
    print("\n" + "="*80)
    print("FINAL TRAINING: YOLOv8, YOLOv10, YOLOv11 COMPARISON")
    print("Dataset: PPE-Dataset-Balanced (Proper Split)")
    print("="*80)

    # Dataset balanced (proper split)
    DATA_YAML = "PPE-Dataset-Balanced/data.yaml"

    # Check if dataset exists
    if not os.path.exists(DATA_YAML):
        print("\n❌ ERROR: Balanced dataset not found!")
        print(f"   Expected: {DATA_YAML}")
        return

    # Comprehensive model list
    models_yolo = [
        # YOLOv8 variants
        'yolov8n.pt',   # Nano - fastest
        'yolov8s.pt',   # Small
        'yolov8m.pt',   # Medium
        'yolov8l.pt',   # Large
        'yolov8x.pt',   # Extra Large - most accurate

        # YOLOv10 variants 
        'yolov10n.pt',  # Nano
        'yolov10s.pt',  # Small
        'yolov10m.pt',  # Medium
        'yolov10b.pt',  # Balanced
        'yolov10l.pt',  # Large
        'yolov10x.pt',  # Extra Large

        # YOLOv11 variants
        'yolo11n.pt',   # Nano
        'yolo11s.pt',   # Small
        'yolo11m.pt',   # Medium
        'yolo11l.pt',   # Large
        'yolo11x.pt',   # Extra Large
    ]

    print(f"\n📁 Dataset: {DATA_YAML}")
    print(f"🎯 Models to train: {len(models_yolo)}")
    print("\nModel list:")
    for i, model in enumerate(models_yolo, 1):
        version = "v8" if "v8" in model else ("v10" if "v10" in model else "v11")
        print(f"  {i:2d}. {model:<20} (YOLO{version})")

    print(f"\n⏱️  Estimated total time: ~{len(models_yolo) * 2.5} hours")
    print(f"    (assuming 2.5 hours per model average)")

    confirm = input(f"\n⏸️  Start training {len(models_yolo)} models? (yes/no): ").strip().lower()

    if confirm not in ['yes', 'y']:
        print("\n❌ Training cancelled!")
        return

    # Training
    print("\n" + "="*80)
    print("STARTING TRAINING...")
    print("="*80)

    start_time = time.time()
    results = train_models(models_yolo, DATA_YAML)
    total_time = time.time() - start_time

    # Final Summary
    if results:
        print("\n\n" + "="*80)
        print("TRAINING COMPLETED - FINAL COMPARISON")
        print("="*80)
        print(f"✅ Models trained: {len(results)}/{len(models_yolo)}")
        print(f"⏱️  Total time: {total_time/3600:.2f} hours")

        # Sort by mAP50-95
        results.sort(key=lambda x: x['mAP50-95'], reverse=True)

        print("\n" + "="*80)
        print("RESULTS RANKING (by mAP50-95)")
        print("="*80)
        print(f"{'Rank':<6} {'Model':<25} {'mAP50':<10} {'mAP50-95':<10} {'Time(h)':<10}")
        print("-"*70)

        for rank, result in enumerate(results, 1):
            version_emoji = "🔸" if "v8" in result['model'] else ("🔹" if "v10" in result['model'] else "🔺")
            print(f"{rank:<6} {version_emoji} {result['model']:<23} "
                  f"{result['mAP50']:<10.4f} "
                  f"{result['mAP50-95']:<10.4f} "
                  f"{result['training_time_hours']:<10.2f}")

        # Version comparison
        print("\n" + "="*80)
        print("VERSION COMPARISON")
        print("="*80)

        v8_models = [r for r in results if 'v8' in r['model']]
        v10_models = [r for r in results if 'v10' in r['model']]
        v11_models = [r for r in results if 'v11' in r['model']]

        if v8_models:
            avg_v8 = sum(r['mAP50-95'] for r in v8_models) / len(v8_models)
            best_v8 = max(v8_models, key=lambda x: x['mAP50-95'])
            print(f"\n🔸 YOLOv8  (n={len(v8_models)})")
            print(f"   Average mAP50-95: {avg_v8:.4f}")
            print(f"   Best: {best_v8['model']} - {best_v8['mAP50-95']:.4f}")

        if v10_models:
            avg_v10 = sum(r['mAP50-95'] for r in v10_models) / len(v10_models)
            best_v10 = max(v10_models, key=lambda x: x['mAP50-95'])
            print(f"\n🔹 YOLOv10 (n={len(v10_models)})")
            print(f"   Average mAP50-95: {avg_v10:.4f}")
            print(f"   Best: {best_v10['model']} - {best_v10['mAP50-95']:.4f}")

        if v11_models:
            avg_v11 = sum(r['mAP50-95'] for r in v11_models) / len(v11_models)
            best_v11 = max(v11_models, key=lambda x: x['mAP50-95'])
            print(f"\n🔺 YOLOv11 (n={len(v11_models)})")
            print(f"   Average mAP50-95: {avg_v11:.4f}")
            print(f"   Best: {best_v11['model']} - {best_v11['mAP50-95']:.4f}")

        # Best overall
        best_overall = results[0]
        print("\n" + "="*80)
        print("🏆 BEST MODEL OVERALL")
        print("="*80)
        print(f"Model:     {best_overall['model']}")
        print(f"mAP50:     {best_overall['mAP50']:.4f}")
        print(f"mAP50-95:  {best_overall['mAP50-95']:.4f}")
        print(f"Precision: {best_overall['precision']:.4f}")
        print(f"Recall:    {best_overall['recall']:.4f}")
        print(f"Time:      {best_overall['training_time_hours']:.2f} hours")

        print("\n📁 Model path: YOLO-PPE-Final-Comparison/{model}_balanced/weights/best.pt")
        print("="*80 + "\n")

    print("🎉 SELESAI! Sekarang Anda punya comparison lengkap YOLOv8 vs v10 vs v11!")
    print("📊 Data ini bisa dipakai untuk analisis di skripsi Anda.")

if __name__ == "__main__":
    main()
