"""Pristine baseline: the unadapted Student evaluated with the shared evaluator (mapsteal.evaluation); no training and no Teacher."""
def evaluate(domain, student, panels, output, benchmarks=None, limit=None, device='cuda:0'):
    """Same as `python -m mapsteal evaluate --checkpoint none ...` (GPU)."""
    from mapsteal import cli
    argv = ['evaluate', '--domain', domain, '--checkpoint', 'none', '--student', student, '--panels', str(panels), '--output', str(output), '--device', device]
    if benchmarks: argv += ['--benchmarks', benchmarks]
    if limit: argv += ['--limit', str(limit)]
    return cli.main(argv)
