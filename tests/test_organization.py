"""Regression checks for the refactor's public contracts and preserved training bodies."""
import ast
import contextlib
import importlib
import io
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import warnings

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

class NormalizePaths(ast.NodeTransformer):
    def visit_Constant(self,node):
        if isinstance(node.value,str) and node.value.startswith(('/mnt/nvme2/yyc/clinicalbert/','/mnt/nvme4/Datasets/MIMIC-III/physionet.org/files/mimiciii/1.4/')):
            return ast.Constant('__PATH__')
        return node
    def visit_Call(self,node):
        code=ast.unparse(node)
        if isinstance(node.func,ast.Name) and node.func.id in ('device_name','difficulty_file'):
            return self.visit(node.args[-1])
        if isinstance(node.func,ast.Name) and node.func.id=='data_file':return self.visit(node.args[-1])
        if code.startswith(('str(DATA_DIR /','str(MODEL_DIR /','str(RAW_DIR /')):return ast.Constant('__PATH__')
        if 'TACL_READMISSION_WORD2VEC_MODEL' in code and '.get(' in code:return self.visit(node.args[-1])
        return self.generic_visit(node)
    def visit_BoolOp(self,node):
        if '_TACL_ARGS' in ast.unparse(node):return self.visit(node.values[-1])
        return self.generic_visit(node)

class OrganizationTests(unittest.TestCase):
    def test_readmission_training_function_bodies_preserved(self):
        for name in ('run_readmission.py','run_readmission_CL.py','run_biobert.py','runword2vec.py','runword2vec_CL.py','CNN.py'):
            with self.subTest(name=name):
                old=subprocess.check_output(['git','show','6730635:clinicalBERT-master/'+name],cwd=ROOT).decode('utf-8-sig')
                new=(ROOT/'tasks/readmission/_impl'/name).read_text(encoding='utf-8')
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore',SyntaxWarning)
                    before=ast.parse(old);after=ast.parse(new)
                def methods(tree):
                    return {n.name:ast.dump(NormalizePaths().visit(n),include_attributes=False)
                            for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
                self.assertEqual(methods(before),methods(after))

    def test_readmission_help_is_dependency_free(self):
        code='import sys; from tasks.readmission.run_readmission import build_parser; build_parser().parse_args(["--help"])'
        result=subprocess.run([sys.executable,'-S','-c',code],cwd=ROOT,capture_output=True,text=True)
        self.assertEqual(result.returncode,0,result.stderr)
        self.assertIn('--train_batch_size',result.stdout)

    def test_readmission_path_failures_do_not_create_outputs(self):
        from tasks.readmission.CNN import main
        with tempfile.TemporaryDirectory(prefix='TACL path with spaces ') as d:
            output=Path(d)/'not-created'
            with contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as raised:
                    main(['--train-file',str(Path(d)/'missing.csv'),'--output-dir',str(output),'--check-paths'])
            self.assertEqual(raised.exception.code,2)
            self.assertFalse(output.exists())

    def test_native_readmission_arguments_retained(self):
        from tasks.readmission.run_readmission import build_parser
        p=build_parser();a=p.parse_args(['--task_name','readmission'])
        self.assertEqual(a.train_batch_size,64)
        self.assertEqual(a.learning_rate,5e-5)
        self.assertEqual(a.num_train_epochs,3.0)
        self.assertIsNone(a.readmission_mode)

    def test_template_does_not_claim_valid_resources(self):
        from tasks.readmission.curriculum_learning.PCA import main
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as raised:main(['--check-paths'])
        self.assertEqual(raised.exception.code,2)

    def test_local_single_component_model_path_remains_local(self):
        from tasks.readmission._entry import launch
        from tasks.readmission.run_readmission import build_parser
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);(p/'pretraining').mkdir();(p/'data').mkdir()
            previous=Path.cwd();os.chdir(p)
            try:
                with contextlib.redirect_stdout(io.StringIO()):
                    code=launch(ROOT/'tasks/readmission/run_readmission.py',build_parser(),['--task_name','readmission','--data_dir','data','--bert_model','pretraining','--check-paths'])
                self.assertEqual(code,0)
            finally:os.chdir(previous)

if __name__=='__main__':unittest.main()
