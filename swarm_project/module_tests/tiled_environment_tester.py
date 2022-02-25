import unittest
import sys

from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

print(sys.path)

from swarm_project.modules.tiled_environment import tiled_environment

class tiled_environment_tester(unittest.TestCase):
    def test_tiled_environment_object_is_created(self):
        test_tiled_environment = tiled_environment()

if __name__ == '__main__':
    unittest.main()