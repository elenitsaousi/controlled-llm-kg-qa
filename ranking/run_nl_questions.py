import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ranking.build_features_domain import main


if __name__ == "__main__":
    main()
