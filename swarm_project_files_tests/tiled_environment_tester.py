import unittest

from swarm_project_files.tiled_environment import tiled_environment_class


class tiled_environment_tester(unittest.TestCase):
    def test_tiled_environment_object_is_created(self):
        test_tiled_environment = tiled_environment_class()

if __name__ == '__main__':
    unittest.main()