"""Leo management transport with an isolated, self-restoring CPU sshd."""
import argparse
import json
import re
from cloud_proxy import CPU_POD,proxy,run_code

HELPER='/workspace/project/roborender/pipeline/deployment/connectivity/leo_dedicated_sshd.py'


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--helper-sha256',required=True);args=parser.parse_args()
    if not re.fullmatch('[a-f0-9]{64}',args.helper_sha256):raise ValueError('pinned helper required')
    code=("import hashlib,pathlib,runpy,json;"
          f"p=pathlib.Path({HELPER!r});assert hashlib.sha256(p.read_bytes()).hexdigest()=={args.helper_sha256!r};"
          "m=runpy.run_path(str(p));print(json.dumps(m['ensure']()))")
    result=json.loads(run_code(code))
    if result.get('ok') is not True or result.get('port')!=2233:raise RuntimeError('dedicated endpoint unavailable')
    proxy(CPU_POD,2233)


if __name__=='__main__':main()
