import sys
import os

# Append current directory to path to allow importing local modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from fineweb_curator import FineWebCurator
    from code_curator import CodeCurator
    from finemath_curator import FineMathCurator
    from ultrachat_curator import UltraChatCurator
    from curator_base import BaseDatasetCurator
except ModuleNotFoundError:
    from tools.data_curation.fineweb_curator import FineWebCurator
    from tools.data_curation.code_curator import CodeCurator
    from tools.data_curation.finemath_curator import FineMathCurator
    from tools.data_curation.ultrachat_curator import UltraChatCurator
    from tools.data_curation.curator_base import BaseDatasetCurator


def run_all_checks():
    print("==================================================")
    print("🎯 STARTING MODULAR DATA CURATION INSPECTIONS 🎯")
    print("==================================================")

    # Instantiate all curators
    curators = [
        FineWebCurator(),
        CodeCurator(),
        FineMathCurator(),
        UltraChatCurator()
    ]

    results = []
    for curator in curators:
        print("\n──────────────────────────────────────────────────")
        print(f"📊 Analyzing Dataset Stream: {curator.alias}")
        print("──────────────────────────────────────────────────")
        
        # Analyze a small batch of 5 samples for fast responsiveness
        stats = curator.analyze_batch(n=5)
        results.append(stats)
        
        print("\n📈 Batch Ingestion Results:")
        for k, v in stats.items():
            if isinstance(v, float):
                print(f"   - {k:22}: {v:.3f}")
            else:
                print(f"   - {k:22}: {v}")

        # Extra checks for special curators
        if isinstance(curator, FineWebCurator):
            print()
            curator.evaluate_score_threshold(threshold=3.0, sample_size=5)
            
        if isinstance(curator, UltraChatCurator):
            print("\n💬 Unpacked multi-turn dialogue preview:")
            samples = curator.fetch_samples(1)
            if samples:
                formatted_dialogue = curator.extract_text(samples[0])
                # Show first 300 chars of dialogue
                preview = formatted_dialogue[:300] + "..." if len(formatted_dialogue) > 300 else formatted_dialogue
                print("   " + "\n   ".join(preview.split("\n")))

    print("\n==================================================")
    print("📦 EXPORTING CONFIGURATION REGISTRY FOR PREFILL 📦")
    print("==================================================")
    for alias, curator in BaseDatasetCurator.REGISTRY.items():
        cfg = curator.get_export_config()
        print(f"\n🔑 Registered Configuration for '{alias}':")
        for k, v in cfg.items():
            print(f"   - {k:10}: {v}")

    print("\n==================================================")
    print("✅ DATA CURATION CHECKS FINISHED SUCCESSFULLY! ✅")
    print("==================================================")

if __name__ == "__main__":
    run_all_checks()
