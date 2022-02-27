from pathlib import Path
import sys

path = Path(__file__).parent.parent

sys.path.insert(0, str(path))
