
try:
    import chronos
    print("chronos installed")
except ImportError:
    print("chronos NOT installed")

try:
    from transformers import PatchTSTConfig, PatchTSTForPrediction
    print("transformers PatchTST available")
except ImportError:
    print("transformers PatchTST NOT available")

try:
    # Checking for TTM might be trickier if it's a custom implementation or specific library
    # but often it's in transformers or a dedicated package
    from transformers import TinyTimeMixerConfig, TinyTimeMixerForPrediction
    print("transformers TinyTimeMixer available")
except ImportError:
    print("transformers TinyTimeMixer NOT available")

try:
    import gluonts
    print("gluonts installed")
except ImportError:
    print("gluonts NOT installed")
