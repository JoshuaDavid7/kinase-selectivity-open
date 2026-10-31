"""Compatibility name: check the generated corrected report against saved JSON."""
import sys
from render_results import main
if __name__ == '__main__':
    sys.argv.append('--check')
    main()
