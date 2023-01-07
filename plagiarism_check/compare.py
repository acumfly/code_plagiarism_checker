import numpy as np
import os
import argparse
import ast

cur_dir = os.getcwd()


class Renamer(ast.NodeTransformer):

    def __init__(self):
        self._arg_count = 0
        self._func_count = 0
        self._id_count = 0
        self._ctx_count = 0
        self._attr_count = 0
        self._const_count = 0
        self._str_count = 0
        self._expr_count = 0

    def visit_FunctionDef(self, node):
        node.name = "method_name_{}".format(self._func_count)
        self._func_count += 1
        self.generic_visit(node)
        return node

    def visit_arg(self, node):
        node.arg = "arg_{}".format(self._arg_count)
        self._arg_count += 1
        self.generic_visit(node)
        return node

    def visit_name(self, node):
        node.id = "id_{}".format(self._id_count)
        self._id_count += 1
        node.ctx = "ctx_{}".format(self._ctx_count)
        self._ctx_count += 1
        self.generic_visit(node)
        return node

    def visit_attribute(self, node):
        node.attr = "attr_{}".format(self._attr_count)
        self._attr_count += 1
        node.ctx = "ctx_{}".format(self._ctx_count)
        self._ctx_count += 1
        self.generic_visit(node)
        return node

    def visit_constant(self, node):
        if type(node) == str:
            node.value = "const_{}".format(self._const_count)
            self._const_count += 1
        self.generic_visit(node)

    def visit_str(self, node):
        node.s = "str_{}".format(self._str_count)
        self._str_count += 1
        self.generic_visit(node)
        return node


class Distance:

    def __init__(self, tree1, tree2):
        self.text_length = None
        self.tree1 = tree1
        self.tree2 = tree2

    def to_source(self):
        code1 = ast.unparse(tree1)
        code2 = ast.unparse(tree2)
        setattr(self, 'text_length', (len(code1)+len(code2))/2)
        return code1, code2

    @staticmethod
    def levenstein_distance(seq1, seq2):
        size_x = len(seq1) + 1
        size_y = len(seq2) + 1
        matrix = np.zeros((size_x, size_y))
        for x in range(size_x):
            matrix[x, 0] = x
        for y in range(size_y):
            matrix[0, y] = y

        for x in range(1, size_x):
            for y in range(1, size_y):
                if seq1[x - 1] == seq2[y - 1]:
                    matrix[x, y] = min(
                        matrix[x - 1, y] + 1,
                        matrix[x - 1, y - 1],
                        matrix[x, y - 1] + 1
                    )
                else:
                    matrix[x, y] = min(
                        matrix[x - 1, y] + 1,
                        matrix[x - 1, y - 1] + 1,
                        matrix[x, y - 1] + 1
                    )
        return (matrix[size_x - 1, size_y - 1])

    def compare_codes(self):
        code1, code2 = list(map(lambda x: x.splitlines(), self.to_source()))

        total_dist = 0
        size_code1 = len(code1)
        size_code2 = len(code2)
        n = min(size_code1, size_code2)

        for i in range(n):
            total_dist += self.levenstein_distance(code1[i], code2[i])

        total_dist += abs(size_code1 - size_code2)
        return np.round(total_dist/self.text_length, 3)


def remove_docstrings(tree):
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.ClassDef, ast.AsyncFunctionDef)):
            continue

        if not len(node.body):
            continue

        if not isinstance(node.body[0], ast.Expr):
            continue

        if not hasattr(node.body[0], 'value') or not isinstance(node.body[0].value, ast.Str):
            continue

        node.body = node.body[1:]

    return tree


def read_from_console():
    parser = argparse.ArgumentParser(description='Compared files')
    parser.add_argument('indirs', type=str, help='Input file')
    parser.add_argument('outdir', type=str, help='Output scores')

    args = parser.parse_args()
    compared_files = [[indir.rstrip('\n') for indir in line.split()]
                      for line in open(args.indirs, encoding='utf-8').readlines()]
    return compared_files


def read_codes(pair):
    path1, filename1 = pair[0].split('/')
    path1 = cur_dir + '\\plagiat\\' + path1 + '\\' + filename1
    raw_code1 = '\n'.join([line.rstrip('\n') for line in open(path1, encoding='utf-8').readlines()])

    path2, filename2 = pair[1].split('/')
    path2 = cur_dir + '\\plagiat\\' + path2 + '\\' + filename2
    raw_code2 = '\n'.join([line.rstrip('\n') for line in open(path2, encoding='utf-8').readlines()])

    return raw_code1, raw_code2


def create_output_file(distances):
    os.chdir(cur_dir)
    with open("scores.txt", "w") as file:
        file.write('\n'.join(str(elem) for elem in distances))


compared_files = read_from_console()
distances = []

for pair in compared_files:
    raw_code1, raw_code2 = read_codes(pair)
    tree1 = ast.parse(raw_code1)
    tree2 = ast.parse(raw_code2)

    renamer = Renamer()
    tree1_normalized = remove_docstrings(renamer.visit(tree1))
    tree2_normalized = remove_docstrings(renamer.visit(tree2))

    distance = Distance(tree1_normalized, tree2_normalized)
    distances.append(distance.compare_codes())

create_output_file(distances)
