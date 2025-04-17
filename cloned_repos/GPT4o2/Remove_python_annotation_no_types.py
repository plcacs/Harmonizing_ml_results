import os
import glob
import ast
import astor

def remove_type_annotations(file_path, output_filename):
    stats = {'total_parameters': 0, 'parameters_with_annotations': 0}
    with open(file_path, 'r', encoding='utf-8') as f:
        code = f.read()
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return (code, stats, False)

    class TypeAnnotationRemover(ast.NodeTransformer):

        def visit_FunctionDef(self, node):
            for arg in node.args.args:
                stats['total_parameters'] += 1
                if arg.annotation:
                    stats['parameters_with_annotations'] += 1
                    arg.annotation = None
            if node.returns:
                stats['total_parameters'] += 1
                stats['parameters_with_annotations'] += 1
                node.returns = None
            self.generic_visit(node)
            return node
    remover = TypeAnnotationRemover()
    tree = remover.visit(tree)
    modified_code = ast.unparse(tree)
    with open(output_filename, 'w', encoding='utf-8') as f:
        f.write(modified_code)

def remove_type_hints(filename, output_filename):
    """
    Removes type hints from a Python file and saves the result to a new file.
    """
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            tree = ast.parse(f.read())
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                node.returns = None
                for arg in node.args.args:
                    arg.annotation = None
            elif isinstance(node, ast.AnnAssign):
                if node.value is None:
                    parent = next((p for p in ast.walk(tree) if any((n is node for n in ast.walk(p)))))
                    parent.body.remove(node)
                else:
                    assign_node = ast.Assign(targets=[node.target], value=node.value)
                    parent = next((p for p in ast.walk(tree) if any((n is node for n in ast.walk(p)))))
                    parent.body[parent.body.index(node)] = assign_node
        with open(output_filename, 'w', encoding='utf-8') as f:
            f.write(astunparse.unparse(tree))
    except (FileNotFoundError, SyntaxError, UnicodeDecodeError) as e:
        print(f'Error processing {filename}: {e}')

def generate_safe_filename(file_path):
    """Ensures `_no_types.py` is appended correctly without breaking complex filenames."""
    dir_name, base_name = os.path.split(file_path)
    if base_name.endswith('.py'):
        new_base_name = base_name.rsplit('.py', 1)[0] + '_no_types.py'
    return os.path.join(dir_name, new_base_name)

def generate_no_type_versions(directory='.'):
    """
    Generates no-type versions of all Python files in the given directory.
    """
    for filename in glob.glob(os.path.join(directory, '*.py')):
        if '__init__.py' in filename:
            continue
        output_filename = generate_safe_filename(filename)
        remove_type_annotations(filename, output_filename)
        print(f'Generated {output_filename}')
if __name__ == '__main__':
    generate_no_type_versions()