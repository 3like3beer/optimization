from unittest import TestCase
from solver import Vertex
__author__ = 'ruizj'


class TestVertex(TestCase):
    def test_add_adjacent(self):
        v1 = Vertex(2)
        v2 = Vertex(3)
        self.assertEqual(v1.degree,0)
        self.assertEqual(v2.degree,0)
        v1.add_adjacent(v2)
        self.assertEqual(v1.degree,1)
        self.assertEqual(v2.degree,1)
