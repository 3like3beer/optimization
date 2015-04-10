from unittest import TestCase
from coloring.solver import Graph

__author__ = 'ruizj'


class TestGraph(TestCase):
    def test_init(self):
        g = Graph(5, [[1,2],[1,4]])
        self.assertEqual(len(g.vertices),5)
        self.assertEqual(len(g.sorted_vertices),5)
