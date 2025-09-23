import argparse
import csv
import io
import json
import os
import pathlib
import shutil
import sys
import tempfile
from typing import Any, Dict, List, Optional, Tuple, Type
import pytest
from allennlp.commands import main
from allennlp.commands.predict import Predict
from allennlp.common.checks import ConfigurationError
from allennlp.common.testing import AllenNlpTestCase
from allennlp.common.util import JsonDict, push_python_path
from allennlp.data.dataset_readers import DatasetReader, TextClassificationJsonReader
from allennlp.models import Model
from allennlp.models.archival import load_archive
from allennlp.predictors import Predictor, TextClassifierPredictor

class TestPredict(AllenNlpTestCase):

    def setup_method(self) -> None:
        super().setup_method()
        self.classifier_model_path: pathlib.Path = self.FIXTURES_ROOT / 'basic_classifier' / 'serialization' / 'model.tar.gz'
        self.classifier_data_path: pathlib.Path = self.FIXTURES_ROOT / 'data' / 'text_classification_json' / 'imdb_corpus.jsonl'
        self.tempdir: pathlib.Path = pathlib.Path(tempfile.mkdtemp())
        self.infile: pathlib.Path = self.tempdir / 'inputs.txt'
        self.outfile: pathlib.Path = self.tempdir / 'outputs.txt'

    def test_add_predict_subparser(self) -> None:
        parser: argparse.ArgumentParser = argparse.ArgumentParser(description='Testing')
        subparsers: Any = parser.add_subparsers(title='Commands', metavar='')
        Predict().add_subparser(subparsers)
        kebab_args: List[str] = ['predict', '/path/to/archive', '/dev/null', '--output-file', '/dev/null', '--batch-size', '10', '--cuda-device', '0', '--silent']
        args: argparse.Namespace = parser.parse_args(kebab_args)
        assert args.func.__name__ == '_predict'
        assert args.archive_file == '/path/to/archive'
        assert args.output_file == '/dev/null'
        assert args.batch_size == 10
        assert args.cuda_device == 0
        assert args.silent

    def test_works_with_known_model(self) -> None:
        with open(self.infile, 'w') as f:
            f.write('{"sentence": "the seahawks won the super bowl in 2016"}\n')
            f.write('{"sentence": "the mariners won the super bowl in 2037"}\n')
        sys.argv = ['__main__.py', 'predict', str(self.classifier_model_path), str(self.infile), '--output-file', str(self.outfile), '--silent']
        main()
        assert os.path.exists(self.outfile)
        with open(self.outfile, 'r') as f:
            results: List[Dict[str, Any]] = [json.loads(line) for line in f]
        assert len(results) == 2
        for result in results:
            assert set(result.keys()) == {'label', 'logits', 'probs', 'tokens', 'token_ids'}
        shutil.rmtree(self.tempdir)

    def test_using_dataset_reader_works_with_known_model(self) -> None:
        sys.argv = ['__main__.py', 'predict', str(self.classifier_model_path), str(self.classifier_data_path), '--output-file', str(self.outfile), '--silent', '--use-dataset-reader']
        main()
        assert os.path.exists(self.outfile)
        with open(self.outfile, 'r') as f:
            results: List[Dict[str, Any]] = [json.loads(line) for line in f]
        assert len(results) == 3
        for result in results:
            assert set(result.keys()) == {'label', 'logits', 'loss', 'probs', 'tokens', 'token_ids'}
        shutil.rmtree(self.tempdir)

    def test_uses_correct_dataset_reader(self) -> None:

        @Predictor.register('test-predictor')
        class _TestPredictor(Predictor):

            def dump_line(self, outputs: JsonDict) -> str:
                data: Dict[str, str] = {'dataset_reader_type': type(self._dataset_reader).__name__}
                return json.dumps(data) + '\n'

            def load_line(self, line: str) -> JsonDict:
                raise NotImplementedError

        @DatasetReader.register('fake-reader')
        class FakeDatasetReader(TextClassificationJsonReader):
            pass
        sys.argv = ['__main__.py', 'predict', str(self.classifier_model_path), str(self.classifier_data_path), '--output-file', str(self.outfile), '--overrides', '{"validation_dataset_reader": {"type": "fake-reader"}}', '--silent', '--predictor', 'test-predictor', '--use-dataset-reader']
        main()
        assert os.path.exists(self.outfile)
        with open(self.outfile, 'r') as f:
            results: List[Dict[str, Any]] = [json.loads(line) for line in f]
            assert results[0]['dataset_reader_type'] == 'FakeDatasetReader'
        sys.argv = ['__main__.py', 'predict', str(self.classifier_model_path), str(self.classifier_data_path), '--output-file', str(self.outfile), '--overrides', '{"validation_dataset_reader": {"type": "fake-reader"}}', '--silent', '--predictor', 'test-predictor', '--use-dataset-reader', '--dataset-reader-choice', 'train']
        main()
        assert os.path.exists(self.outfile)
        with open(self.outfile, 'r') as f:
            results: List[Dict[str, Any]] = [json.loads(line) for line in f]
            assert results[0]['dataset_reader_type'] == 'TextClassificationJsonReader'
        sys.argv = ['__main__.py', 'predict', str(self.classifier_model_path), str(self.classifier_data_path), '--output-file', str(self.outfile), '--overrides', '{"validation_dataset_reader": {"type": "fake-reader"}}', '--silent', '--predictor', 'test-predictor', '--use-dataset-reader', '--dataset-reader-choice', 'validation']
        main()
        assert os.path.exists(self.outfile)
        with open(self.outfile, 'r') as f:
            results: List[Dict[str, Any]] = [json.loads(line) for line in f]
            assert results[0]['dataset_reader_type'] == 'FakeDatasetReader'
        sys.argv = ['__main__.py', 'predict', str(self.classifier_model_path), str(self.classifier_data_path), '--output-file', str(self.outfile), '--overrides', '{"validation_dataset_reader": {"type": "fake-reader"}}', '--silent', '--predictor', 'test-predictor']
        with pytest.raises(NotImplementedError):
            main()

    def test_base_predictor(self) -> None:
        model_path: str = str(self.classifier_model_path)
        archive: Any = load_archive(model_path)
        model_type: Optional[str] = archive.config.get('model').get('type')
        from allennlp.models import Model
        (model_class, _) = Model.resolve_class_name(model_type)
        saved_default_predictor: Optional[str] = model_class.default_predictor
        model_class.default_predictor = None
        try:
            sys.argv = ['__main__.py', 'predict', model_path, str(self.classifier_data_path), '--output-file', str(self.outfile), '--silent', '--use-dataset-reader']
            main()
            assert os.path.exists(self.outfile)
            with open(self.outfile, 'r') as f:
                results: List[Dict[str, Any]] = [json.loads(line) for line in f]
            assert len(results) == 3
            for result in results:
                assert set(result.keys()) == {'logits', 'probs', 'label', 'loss', 'tokens', 'token_ids'}
        finally:
            model_class.default_predictor = saved_default_predictor

    def test_batch_prediction_works_with_known_model(self) -> None:
        with open(self.infile, 'w') as f:
            f.write('{"sentence": "the seahawks won the super bowl in 2016"}\n')
            f.write('{"sentence": "the mariners won the super bowl in 2037"}\n')
        sys.argv = ['__main__.py', 'predict', str(self.classifier_model_path), str(self.infile), '--output-file', str(self.outfile), '--silent', '--batch-size', '2']
        main()
        assert os.path.exists(self.outfile)
        with open(self.outfile, 'r') as f:
            results: List[Dict[str, Any]] = [json.loads(line) for line in f]
        assert len(results) == 2
        for result in results:
            assert set(result.keys()) == {'label', 'logits', 'probs', 'tokens', 'token_ids'}
        shutil.rmtree(self.tempdir)

    def test_fails_without_required_args(self) -> None:
        sys.argv = ['__main__.py', 'predict', '/path/to/archive']
        with pytest.raises(SystemExit) as cm:
            main()
        assert cm.value.code == 2

    def test_can_specify_predictor(self) -> None:

        @Predictor.register('classification-explicit')
        class ExplicitPredictor(TextClassifierPredictor):
            """same as classifier predictor but with an extra field"""

            def predict_json(self, inputs: JsonDict) -> JsonDict:
                result: JsonDict = super().predict_json(inputs)
                result['explicit'] = True
                return result
        with open(self.infile, 'w') as f:
            f.write('{"sentence": "the seahawks won the super bowl in 2016"}\n')
            f.write('{"sentence": "the mariners won the super bowl in 2037"}\n')
        sys.argv = ['__main__.py', 'predict', str(self.classifier_model_path), str(self.infile), '--output-file', str(self.outfile), '--predictor', 'classification-explicit', '--silent']
        main()
        assert os.path.exists(self.outfile)
        with open(self.outfile, 'r') as f:
            results: List[Dict[str, Any]] = [json.loads(line) for line in f]
        assert len(results) == 2
        for result in results:
            assert set(result.keys()) == {'label', 'logits', 'explicit', 'probs', 'tokens', 'token_ids'}
        shutil.rmtree(self.tempdir)

    def test_can_specify_extra_args(self) -> None:

        @Predictor.register('classification-extra-args')
        class ExtraArgsPredictor(TextClassifierPredictor):

            def __init__(self, model: Model, dataset_reader: DatasetReader, frozen: bool = True, tag: str = '') -> None:
                super().__init__(model, dataset_reader, frozen)
                self.tag: str = tag

            def predict_json(self, inputs: JsonDict) -> JsonDict:
                result: JsonDict = super().predict_json(inputs)
                result['tag'] = self.tag
                return result
        with open(self.infile, 'w') as f:
            f.write('{"sentence": "the seahawks won the super bowl in 2016"}\n')
            f.write('{"sentence": "the mariners won the super bowl in 2037"}\n')
        sys.argv = ['__main__.py', 'predict', str(self.classifier_model_path), str(self.infile), '--output-file', str(self.outfile), '--predictor', 'classification-extra-args', '--silent', '--predictor-args', '{"tag": "fish"}']
        main()
        assert os.path.exists(self.outfile)
        with open(self.outfile, 'r') as f:
            results: List[Dict[str, Any]] = [json.loads(line) for line in f]
        assert len(results) == 2
        for result in results:
            assert set(result.keys()) == {'label', 'logits', 'tag', 'probs', 'tokens', 'token_ids'}
            assert result['tag'] == 'fish'
        shutil.rmtree(self.tempdir)

    def test_other_modules(self) -> None:
        packagedir: pathlib.Path = self.TEST_DIR / 'testpackage'
        packagedir.mkdir()
        (packagedir / '__init__.py').touch()
        with push_python_path(self.TEST_DIR):
            from allennlp.predictors import text_classifier
            with open(text_classifier.__file__) as f:
                code: str = f.read().replace('@Predictor.register("text_classifier")', '@Predictor.register("duplicate-test-predictor")')
            with open(os.path.join(packagedir, 'predictor.py'), 'w') as f:
                f.write(code)
            self.infile = os.path.join(self.TEST_DIR, 'inputs.txt')
            self.outfile = os.path.join(self.TEST_DIR, 'outputs.txt')
            with open(self.infile, 'w') as f:
                f.write('{"sentence": "the seahawks won the super bowl in 2016"}\n')
                f.write('{"sentence": "the mariners won the super bowl in 2037"}\n')
            sys.argv = ['__main__.py', 'predict', str(self.classifier_model_path), str(self.infile), '--output-file', str(self.outfile), '--predictor', 'duplicate-test-predictor', '--silent']
            with pytest.raises(ConfigurationError):
                main()
            sys.argv.extend(['--include-package', 'testpackage'])
            main()
            assert os.path.exists(self.outfile)
            with open(self.outfile, 'r') as f:
                results: List[Dict[str, Any]] = [json.loads(line) for line in f]
            assert len(results) == 2
            for result in results:
                assert set(result.keys()) == {'label', 'logits', 'probs', 'tokens', 'token_ids'}

    def test_alternative_file_formats(self) -> None:

        @Predictor.register('classification-csv')
        class CsvPredictor(TextClassifierPredictor):
            """same as classification predictor but using CSV inputs and outputs"""

            def load_line(self, line: str) -> JsonDict:
                reader: csv.reader = csv.reader([line])
                (sentence, label) = next(reader)
                return {'sentence': sentence, 'label': label}

            def dump_line(self, outputs: JsonDict) -> str:
                output: io.StringIO = io.StringIO()
                writer: csv.writer = csv.writer(output)
                row: List[Any] = [outputs['label'], *outputs['probs']]
                writer.writerow(row)
                return output.getvalue()
        with open(self.infile, 'w') as f:
            writer: csv.writer = csv.writer(f)
            writer.writerow(['the seahawks won the super bowl in 2016', 'pos'])
            writer.writerow(['the mariners won the super bowl in 2037', 'neg'])
        sys.argv = ['__main__.py', 'predict', str(self.classifier_model_path), str(self.infile), '--output-file', str(self.outfile), '--predictor', 'classification-csv', '--silent']
        main()
        assert os.path.exists(self.outfile)
        with open(self.outfile) as f:
            reader: csv.reader = csv.reader(f)
            results: List[List[str]] = [row for row in reader]
        assert len(results) == 2
        for row in results:
            assert len(row) == 3
            (label, *probs) = row
            for prob in probs:
                assert 0 <= float(prob) <= 1
            assert label != ''
        shutil.rmtree(self.tempdir)
