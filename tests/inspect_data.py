import os
import sys
from start_training import TextDataGenerator, MAX_SEQ_LEN

# Ensure UTF-8 for console output
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

def inspect():
    sources = [
        ("FineWeb-Edu", "HuggingFaceFW/fineweb-edu", None),
        ("Python-Code (Verified)", "iamtarun/python_code_instructions_18k_alpaca", None),
        ("FineMath", "HuggingFaceTB/finemath", "finemath-4plus"),
        ("Cosmopedia-v2", "HuggingFaceTB/cosmopedia-v2", "cosmopedia-v2"),
    ]

    print("🔍 Beginning data source inspection...")
    with open("inspection_results.txt", "w", encoding="utf-8") as out:
        for name, path, config in sources:
            msg = f"\n{'='*20} {name} {'='*20}\nPath: {path} | Config: {config}\n"
            print(msg.strip())
            out.write(msg)
            
            try:
                gen = TextDataGenerator(path, MAX_SEQ_LEN, config_name=config)
                gen._ensure_iterator()
                item = next(gen.iterator)
                
                text_column = None
                for col in ["text", "content", "code", "markdown"]:
                    if col in item:
                        text_column = col
                        break
                if text_column is None:
                    # Fallback to the first string that isn't an ID
                    text_column = next((k for k, v in item.items() 
                                      if isinstance(v, str) and "id" not in k.lower()), None)
                    if text_column is None:
                        text_column = next(k for k, v in item.items() if isinstance(v, str))

                res = f"✅ Fields found: {list(item.keys())}\n🎯 Logic selected column: '{text_column}'\n"
                content = str(item[text_column]).replace('\n', '\n    ') # Indent for readability
                res += f"📄 Preview (2000 chars):\n    {content[:2000]}...\n"
                print(res.strip())
                out.write(res)
                
            except Exception as e:
                err = f"❌ Failed to load {name}: {e}\n"
                print(err.strip())
                out.write(err)

    print("\n✨ Inspection complete. Results saved to inspection_results.txt")

if __name__ == "__main__":
    inspect()
