"""Lazy command execution for preserved ClinicalBERT experiments."""
import os
from pathlib import Path
import runpy
from ._paths import REPOSITORY

def launch(module_file, parser, argv=None):
    args = parser.parse_args(argv)
    output = Path(getattr(args, 'output_dir', None) or REPOSITORY / 'runs/readmission' / Path(module_file).stem).expanduser().resolve()
    for key, value in vars(args).items():
        if isinstance(value, str) and key in ('data_dir', 'output_dir', 'bert_model', 'model_dir', 'raw_dir', 'train_file', 'val_file', 'test_file', 'word2vec_model', 'output_file', 'config_file'):
            # BERT identifiers can be remote model names; filesystem arguments are absolute.
            if key == 'bert_model' and not Path(value).exists() and (value.startswith(('https://', 'http://')) or value in ('bert-base-uncased', 'bert-large-uncased', 'bert-base-cased', 'bert-base-multilingual', 'bert-base-chinese')):
                continue
            value_path = Path(value).expanduser()
            if key == 'output_file' and not value_path.is_absolute() and value == parser.get_default(key):
                value_path = output / value_path
            setattr(args, key, str(value_path.resolve()))
    for split in ('train', 'val', 'test'):
        value = getattr(args, split + '_file', None)
        if value:
            os.environ['TACL_READMISSION_' + split.upper() + '_FILE'] = value
    for key, env in [('word2vec_model', 'TACL_READMISSION_WORD2VEC_MODEL'),
                     ('difficulty_file', 'TACL_READMISSION_DIFFICULTY_FILE'),
                     ('raw_dir', 'TACL_MIMICIII_RAW_DIR'), ('device', 'TACL_READMISSION_DEVICE')]:
        value = getattr(args, key, None)
        if value:
            os.environ[env] = str(value)
    if getattr(args, 'raw_dir', None):
        from . import _paths
        _paths.RAW_DIR = Path(args.raw_dir)
    os.environ['TACL_READMISSION_OUTPUT_DIR'] = str(output)
    missing = []
    for key in ('data_dir', 'raw_dir', 'train_file', 'val_file', 'test_file', 'word2vec_model', 'config_file'):
        value = getattr(args, key, None)
        if value and not Path(value).exists():
            missing.append(f'{key}: {value}')
    model = getattr(args, 'bert_model', None)
    if model and (Path(model).is_absolute() or '\\' in model) and not Path(model).exists():
        missing.append(f'bert_model: {model}')
    if getattr(args, 'data_dir', None) and Path(args.data_dir).exists():
        names = ['train.csv'] if not hasattr(args, 'do_train') else []
        if getattr(args, 'do_train', False):
            names += [getattr(args, 'difficulty_file', 'train.csv'), 'val.csv']
        if getattr(args, 'do_eval', False):
            names += ['test.csv']
        for name in names:
            target = Path(args.data_dir) / name
            if not target.is_file():
                missing.append(f'input CSV: {target}')
    if getattr(args, 'raw_dir', None):
        for name in ('ADMISSIONS.csv', 'NOTEEVENTS.csv'):
            if not (Path(args.raw_dir) / name).is_file():
                missing.append(f'raw CSV: {Path(args.raw_dir) / name}')
    if missing:
        parser.error('Required external resources are missing:\n  ' + '\n  '.join(missing))
    if args.check_paths:
        if not any(getattr(args, key, None) for key in ('data_dir', 'raw_dir', 'train_file', 'val_file', 'test_file', 'word2vec_model')):
            parser.error('This historic template has no complete resource contract. See README; path validation is unavailable for this entry.')
        print('Paths exist. Dataset schema, model compatibility and training remain unverified.')
        return 0
    # No directory or file is created during import, --help, or failed preflight.
    output.parent.mkdir(parents=True, exist_ok=True)
    if getattr(args, 'output_dir', None) is None:
        output.mkdir(parents=True, exist_ok=True)
    if getattr(args, 'output_file', None):
        Path(args.output_file).parent.mkdir(parents=True, exist_ok=True)
    implementation = Path(module_file).parent / '_impl' / Path(module_file).name
    previous = Path.cwd()
    try:
        # Historic relative plots/checkpoints are contained in the experiment output.
        output.mkdir(parents=True, exist_ok=True)
        os.chdir(output)
        runpy.run_path(str(implementation), run_name='__main__', init_globals={'_TACL_ARGS': args})
    except ModuleNotFoundError as exc:
        parser.exit(2, f'Missing dependency {exc.name!r}. Install this task environment; see tasks/readmission/README.md.\n')
    finally:
        os.chdir(previous)
    return 0
