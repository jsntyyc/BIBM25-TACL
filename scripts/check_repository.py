"""Offline standard-library checks. No ML imports, data downloads, or training."""
import argparse
import ast
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import warnings
import zipfile

ROOT = Path(__file__).resolve().parents[1]

def run_checks():
    errors = []
    sources = list((ROOT/'tasks').rglob('*.py'))
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', SyntaxWarning)
        for p in sources:
            try: ast.parse(p.read_text(encoding='utf-8-sig'), filename=str(p))
            except SyntaxError as exc: errors.append(f'syntax {p.relative_to(ROOT)}:{exc.lineno}: {exc.msg}')
    with (ROOT/'docs/migration.csv').open(encoding='utf-8-sig',newline='') as f:
        records=list(csv.DictReader(f))
    assets=0
    for row in records:
        if row['action']=='omit-generated': continue
        if '::' in row['new_path']:
            archive,member=row['new_path'].split('::',1)
            with zipfile.ZipFile(ROOT/archive) as z:
                data=z.read(member)
            if hashlib.sha256(data).hexdigest()!=row['original_sha256']:
                errors.append('archived member checksum changed: '+row['new_path'])
            assets+=1
            continue
        p=ROOT/row['new_path']
        if not p.is_file():
            errors.append('missing migrated file: '+row['new_path']);continue
        # Non-code data/images/notebooks/archives have no intended content changes.
        if p.suffix.lower() not in ('.py','.yaml','.yml','.md'):
            actual=hashlib.sha256(p.read_bytes()).hexdigest()
            if actual!=row['original_sha256']:errors.append('asset checksum changed: '+row['new_path'])
            assets+=1
    readmission=[]
    for p in (ROOT/'tasks/readmission').rglob('*.py'):
        if '_impl' not in p.parts and p.read_text(encoding='utf-8-sig').startswith('"""Lazy CLI'):
            readmission.append('.'.join(p.relative_to(ROOT).with_suffix('').parts))
    entries=['tasks.icd.main','tasks.tcm.run','tasks.tcm.run_cl_v2']+readmission
    for module in entries:
        result=subprocess.run([sys.executable,'-S','-m',module,'--help'],cwd=ROOT,capture_output=True,text=True,encoding='utf-8',errors='replace',timeout=20)
        if result.returncode!=0 or 'usage' not in result.stdout.lower():
            errors.append(f'help {module}: {(result.stderr or result.stdout)[-700:]}')
    return {'upstream_commit':'67306350a049daf87ef5ac384b367c87d5b349eb',
            'python_source_files':len(sources),'migration_records':len(records),
            'unchanged_assets_verified':assets,'help_entries_verified':len(entries),
            'training_executed':False,'errors':errors,'passed':not errors}

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report',type=Path,help='Optional JSON report destination')
    args=parser.parse_args();report=run_checks()
    text=json.dumps(report,ensure_ascii=False,indent=2)
    print(text)
    if args.report:
        args.report.parent.mkdir(parents=True,exist_ok=True)
        args.report.write_text(text+'\n',encoding='utf-8')
    raise SystemExit(0 if report['passed'] else 1)
