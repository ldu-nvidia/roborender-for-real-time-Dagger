"""Pinned long CPU host operations; native tasks remain bounded by cohorts.

This separate schema does not relax old smoke operation deadlines. A deadline
raises for reconciliation; it does not release compute or replay a task.
"""
from pathlib import Path
import hashlib
from pipeline.deployment.beta_operations import CommandOperations,PHASES

class CoverageCommandOperations(CommandOperations):
    def __init__(self,binding,root):
        if set(binding)!={'schema','commands','compute_release_permitted'} or binding.get('schema')!='roborender.full295-campaign-operations.v1' or binding['compute_release_permitted']is not False or set(binding.get('commands',{}))!=PHASES:
            raise ValueError('exact retained full295 operations contract required')
        for name,spec in binding['commands'].items():
            argv=spec.get('argv',[]);entry=Path(spec.get('entrypoint',''))
            if (set(spec)!={'argv','entrypoint','entrypoint_sha256','timeout_seconds'} or not argv
                or any(not isinstance(a,str) or not a for a in argv) or not Path(argv[0]).is_absolute()
                or Path(argv[0]).name in {'sh','bash','zsh'} or not entry.is_absolute() or str(entry) not in argv
                or entry.name!='retained_campaign_site.py'
                or hashlib.sha256(entry.read_bytes()).hexdigest()!=spec['entrypoint_sha256']
                or type(spec['timeout_seconds'])is not int or not 1<=spec['timeout_seconds']<=7*86400):
                raise ValueError('pinned retained provider and bounded CPU operation required: '+name)
        self.binding,self.root,self.children=binding,Path(root).resolve(),{}
