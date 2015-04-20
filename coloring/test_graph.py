from unittest import TestCase
from solver import Graph

__author__ = 'ruizj'


class TestGraph(TestCase):
    def test_init(self):
        g = Graph(5, [[1,2],[1,4]])
        self.assertEqual(len(g.vertices),5)
        self.assertEqual(len(g.sorted_vertices),5)
        self.assertEqual(g.sorted_vertices[0].name,1)

    def test_vertex_to_visit(self):
        g = Graph(5, [[1,2],[1,4]])
        self.assertEqual(g.vertex_to_visit()[0].name,1)
        g.vertices[1].color = 1
        self.assertEqual(g.vertex_to_visit()[0].name,2)
        g.vertices[2].color = 2
        self.assertEqual(g.vertex_to_visit()[0].name,4)
        print(g.sorted_vertices)

