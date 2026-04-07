from __future__ import annotations

import json
import logging
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, List, Sequence
from zoneinfo import ZoneInfo

from src.training.action_bandit import ActionBanditStore
from src.training.policy_memory import PolicyMemoryStore
from src.utils.settings import get_setting, load_settings

LOGGER = logging.getLogger(__name__)
JST = ZoneInfo('Asia/Tokyo')


@dataclass(slots=True)
class ParallelSchedulerConfig:
    requested_tasks: int
    logical_jobs: int
    max_workers: int
    episodes_total: int
    use_queue: bool
    scheduler_mode: str = 'bounded-process-pool'


@dataclass(slots=True)
class ParallelJobSpec:
    mode: str
    job_index: int
    episodes: int
    argv: List[str]
    policy_memory_path: str
    action_bandit_path: str
    unknown_pending_path: str
    unknown_overlay_path: str
    runtime_root: str


@dataclass(slots=True)
class ParallelJobResult:
    job_index: int
    exit_code: int
    episodes: int
    policy_memory_path: str
    action_bandit_path: str
    unknown_pending_path: str
    unknown_overlay_path: str
    runtime_root: str



def _now_stamp() -> str:
    return datetime.now(JST).strftime('%Y%m%d%H%M%S')



def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)



def _split_episodes(total: int, jobs: int) -> List[int]:
    total = max(1, int(total))
    jobs = max(1, min(int(jobs), total))
    base = total // jobs
    remainder = total % jobs
    return [base + (1 if index < remainder else 0) for index in range(jobs)]



def _replace_or_add(argv: Sequence[str], flag: str, value: str) -> List[str]:
    result = list(argv)
    if flag in result:
        index = result.index(flag)
        if index + 1 < len(result):
            result[index + 1] = value
        else:
            result.append(value)
        return result
    result.extend([flag, value])
    return result



def _job_runtime_root(base_runtime_dir: Path, job_index: int) -> Path:
    return base_runtime_dir / f'job_{job_index:04d}'



def _build_learning_args(args) -> List[str]:
    argv: List[str] = [
        '--lexicon', args.lexicon,
        '--trace-dir', args.trace_dir,
        '--dataset-dir', args.dataset_dir,
        '--policy-memory', args.policy_memory,
        '--action-bandit', args.action_bandit,
        '--episodes', str(max(1, int(args.episodes))),
        '--target-mode', args.target_mode,
        '--external-mode', args.external_mode,
        '--alpha', str(args.alpha),
        '--beta', str(args.beta),
        '--teacher-target-weight', str(args.teacher_target_weight),
        '--external-fallback', args.external_fallback,
        '--neutral-external-score', str(args.neutral_external_score),
    ]
    if getattr(args, 'no_trace', False):
        argv.append('--no-trace')
    if getattr(args, 'no_policy_memory', False):
        argv.append('--no-policy-memory')
    if getattr(args, 'no_action_bandit', False):
        argv.append('--no-action-bandit')
    if getattr(args, 'no_dataset', False):
        argv.append('--no-dataset')
    if getattr(args, 'debug', False):
        argv.append('--debug')
    if getattr(args, 'words', None):
        argv.append('--words')
        argv.extend(args.words)
    elif getattr(args, 'text', ''):
        argv.extend(['--text', args.text])
    return argv



def _build_auto_learning_args(args) -> List[str]:
    argv: List[str] = [
        '--lexicon', args.lexicon,
        '--trace-dir', args.trace_dir,
        '--dataset-dir', args.dataset_dir,
        '--policy-memory', args.policy_memory,
        '--action-bandit', args.action_bandit,
        '--episodes', str(max(1, int(args.episodes))),
        '--target-mode', args.target_mode,
        '--external-mode', args.external_mode,
        '--input-mode', args.input_mode,
        '--alpha', str(args.alpha),
        '--beta', str(args.beta),
        '--teacher-target-weight', str(args.teacher_target_weight),
        '--external-fallback', args.external_fallback,
        '--neutral-external-score', str(args.neutral_external_score),
        '--seed-topic', args.seed_topic,
        '--seed-text', args.seed_text,
        '--report-every', str(max(1, int(args.report_every))),
    ]
    if getattr(args, 'no_trace', False):
        argv.append('--no-trace')
    if getattr(args, 'no_policy_memory', False):
        argv.append('--no-policy-memory')
    if getattr(args, 'no_action_bandit', False):
        argv.append('--no-action-bandit')
    if getattr(args, 'no_dataset', False):
        argv.append('--no-dataset')
    if getattr(args, 'debug', False):
        argv.append('--debug')
    return argv



def _resolve_scheduler_config(args) -> ParallelSchedulerConfig:
    settings = load_settings()
    total_episodes = max(1, int(getattr(args, 'episodes', 1) or 1))
    requested_tasks = max(1, int(getattr(args, 'learning_tasks', 1) or 1))
    logical_jobs = max(1, min(requested_tasks, total_episodes))
    configured_max_workers = max(
        1,
        int(get_setting(settings, 'learning', 'parallel', 'max_workers', default=50)),
    )
    max_workers = min(logical_jobs, configured_max_workers)
    return ParallelSchedulerConfig(
        requested_tasks=requested_tasks,
        logical_jobs=logical_jobs,
        max_workers=max_workers,
        episodes_total=total_episodes,
        use_queue=logical_jobs > max_workers,
    )



def _build_job_specs(args, scheduler: ParallelSchedulerConfig) -> tuple[List[ParallelJobSpec], Path]:
    mode = str(args.mode)
    total_episodes = scheduler.episodes_total
    if mode == 'learn' and not (getattr(args, 'text', '') or getattr(args, 'words', None)):
        raise ValueError('Parallel learn mode requires --text or --words.')

    stamp = _now_stamp()
    base_runtime_dir = Path('runtime') / 'parallel_learning' / f'{stamp}_{mode}'
    base_runtime_dir.mkdir(parents=True, exist_ok=True)

    if mode == 'learn':
        base_argv = _build_learning_args(args)
    elif mode == 'auto-learn':
        base_argv = _build_auto_learning_args(args)
    else:
        raise ValueError(f'Unsupported parallel mode: {mode}')

    episode_splits = _split_episodes(total_episodes, scheduler.logical_jobs)
    specs: List[ParallelJobSpec] = []
    episode_offset = 0
    for job_index, job_episodes in enumerate(episode_splits, start=1):
        job_root = _job_runtime_root(base_runtime_dir, job_index)
        logs_dir = job_root / 'logs'
        traces_dir = job_root / 'traces'
        datasets_dir = job_root / 'datasets'
        policy_memory_path = job_root / 'policy_memory.json'
        action_bandit_path = job_root / 'action_bandit.json'
        unknown_pending_path = job_root / 'unknown_word_candidates.jsonl'
        unknown_overlay_path = job_root / 'lexicon_overlay.json'

        job_argv = list(base_argv)
        job_argv = _replace_or_add(job_argv, '--episodes', str(job_episodes))
        job_argv = _replace_or_add(job_argv, '--trace-dir', str(traces_dir))
        job_argv = _replace_or_add(job_argv, '--dataset-dir', str(datasets_dir))
        job_argv = _replace_or_add(job_argv, '--policy-memory', str(policy_memory_path))
        job_argv = _replace_or_add(job_argv, '--action-bandit', str(action_bandit_path))
        job_argv = _replace_or_add(job_argv, '--log-dir', str(logs_dir))
        if mode == 'auto-learn':
            job_argv = _replace_or_add(job_argv, '--loop-index-offset', str(episode_offset))

        specs.append(
            ParallelJobSpec(
                mode=mode,
                job_index=job_index,
                episodes=job_episodes,
                argv=job_argv,
                policy_memory_path=str(policy_memory_path),
                action_bandit_path=str(action_bandit_path),
                unknown_pending_path=str(unknown_pending_path),
                unknown_overlay_path=str(unknown_overlay_path),
                runtime_root=str(job_root),
            )
        )
        episode_offset += job_episodes
    return specs, base_runtime_dir



def _run_parallel_job(spec: ParallelJobSpec) -> ParallelJobResult:
    os.environ['LSLM_UNKNOWN_WORD_PENDING_PATH'] = spec.unknown_pending_path
    os.environ['LSLM_UNKNOWN_WORD_OVERLAY_PATH'] = spec.unknown_overlay_path
    try:
        if spec.mode == 'learn':
            from src.apps.chat_learning import main as worker_main
        elif spec.mode == 'auto-learn':
            from src.apps.auto_loop_learning import main as worker_main
        else:
            raise ValueError(f'Unsupported worker mode: {spec.mode}')
        exit_code = int(worker_main(spec.argv))
    finally:
        os.environ.pop('LSLM_UNKNOWN_WORD_PENDING_PATH', None)
        os.environ.pop('LSLM_UNKNOWN_WORD_OVERLAY_PATH', None)
    return ParallelJobResult(
        job_index=spec.job_index,
        exit_code=exit_code,
        episodes=spec.episodes,
        policy_memory_path=spec.policy_memory_path,
        action_bandit_path=spec.action_bandit_path,
        unknown_pending_path=spec.unknown_pending_path,
        unknown_overlay_path=spec.unknown_overlay_path,
        runtime_root=spec.runtime_root,
    )



def _merge_policy_memory(worker_paths: Sequence[str], destination_path: str) -> Path:
    merged = PolicyMemoryStore(destination_path, autoload=False)
    merged.records = []
    for worker_path in worker_paths:
        source = PolicyMemoryStore(worker_path, autoload=True)
        for record in source.records:
            normalized_text = merged._normalize_text(record.text)  # type: ignore[attr-defined]
            stable_slots = merged._stable_slots(record.slots)  # type: ignore[attr-defined]
            existing = merged._find_record(  # type: ignore[attr-defined]
                intent=record.intent,
                slots=stable_slots,
                text=normalized_text,
                source=record.source,
            )
            if existing is None:
                merged.records.append(record)
                continue
            total_count = max(1, existing.count) + max(1, record.count)
            existing.weight = (
                (existing.weight * max(1, existing.count))
                + (record.weight * max(1, record.count))
            ) / float(total_count)
            existing.last_reward_total = max(existing.last_reward_total, record.last_reward_total)
            existing.last_internal = max(existing.last_internal, record.last_internal)
            existing.last_external = max(existing.last_external, record.last_external)
            existing.count = total_count
            existing.created_at = existing.created_at or record.created_at
            existing.updated_at = max(existing.updated_at, record.updated_at)
            if record.template_id:
                existing.template_id = record.template_id
    return merged.save()



def _merge_action_bandit(worker_paths: Sequence[str], destination_path: str) -> Path:
    merged = ActionBanditStore(destination_path, autoload=False)
    merged.contexts = {}
    for worker_path in worker_paths:
        source = ActionBanditStore(worker_path, autoload=True)
        for context_key, arms in source.contexts.items():
            merged_arms = merged.contexts.setdefault(context_key, {})
            for arm_key, arm_state in arms.items():
                existing = merged_arms.get(arm_key)
                if existing is None:
                    merged_arms[arm_key] = arm_state
                    continue
                existing_count = max(0, existing.count)
                incoming_count = max(0, arm_state.count)
                total_count = existing_count + incoming_count
                if total_count <= 0:
                    continue
                existing.mean_reward = (
                    (existing.mean_reward * existing_count)
                    + (arm_state.mean_reward * incoming_count)
                ) / float(total_count)
                existing.ema_reward = (
                    (existing.ema_reward * existing_count)
                    + (arm_state.ema_reward * incoming_count)
                ) / float(total_count)
                existing.count = total_count
                existing.last_reward_total = arm_state.last_reward_total
                existing.last_internal = arm_state.last_internal
                existing.last_external = arm_state.last_external
                if arm_state.template_id:
                    existing.template_id = arm_state.template_id
                if arm_state.text_signature:
                    existing.text_signature = arm_state.text_signature
    return merged.save()



def _merge_unknown_overlay(worker_paths: Sequence[str], destination_path: str) -> Path:
    destination = Path(destination_path)
    payload: dict[str, object] = {'meta': {'version': 'unknown-overlay-v1'}, 'entries': {}}
    entries = payload['entries']
    assert isinstance(entries, dict)
    for worker_path in worker_paths:
        path = Path(worker_path)
        if not path.exists():
            continue
        try:
            data = json.loads(path.read_text(encoding='utf-8'))
        except Exception:
            LOGGER.exception('parallel_learning.unknown_overlay_merge_failed path=%s', path)
            continue
        for word, entry in dict(data.get('entries', {}) or {}).items():
            entries[str(word)] = entry
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
    return destination



def _merge_unknown_pending(worker_paths: Sequence[str], destination_path: str) -> Path:
    destination = Path(destination_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open('w', encoding='utf-8') as out:
        for worker_path in worker_paths:
            path = Path(worker_path)
            if not path.exists():
                continue
            with path.open('r', encoding='utf-8') as src:
                for line in src:
                    if line.strip():
                        out.write(line if line.endswith('\n') else f'{line}\n')
    return destination



def _job_summary(result: ParallelJobResult) -> dict[str, object]:
    return {
        'job_index': result.job_index,
        'episodes': result.episodes,
        'exit_code': result.exit_code,
        'runtime_root': result.runtime_root,
        'policy_memory_path': result.policy_memory_path,
        'action_bandit_path': result.action_bandit_path,
        'unknown_pending_path': result.unknown_pending_path,
        'unknown_overlay_path': result.unknown_overlay_path,
    }



def _write_summary(
    *,
    destination_dir: Path,
    scheduler: ParallelSchedulerConfig,
    results: Sequence[ParallelJobResult],
    merged_policy: Path | None,
    merged_bandit: Path | None,
    merged_pending: Path,
    merged_overlay: Path,
) -> Path:
    reward_paths = [Path(item.action_bandit_path) for item in results]
    summary_path = destination_dir / 'summary.json'
    payload = {
        'scheduler': asdict(scheduler),
        'job_count': len(results),
        'completed_jobs': [
            _job_summary(item)
            for item in results
        ],
        'merged_outputs': {
            'policy_memory': str(merged_policy) if merged_policy else None,
            'action_bandit': str(merged_bandit) if merged_bandit else None,
            'unknown_pending': str(merged_pending),
            'unknown_overlay': str(merged_overlay),
        },
        'job_artifacts_existing': {
            'action_bandit': sum(1 for path in reward_paths if path.exists()),
        },
    }
    summary_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
    return summary_path



def _iter_completed_jobs(executor: ProcessPoolExecutor, specs: Iterable[ParallelJobSpec]):
    future_map = {executor.submit(_run_parallel_job, spec): spec for spec in specs}
    for future in as_completed(future_map):
        yield future.result()



def run_parallel_learning(args) -> int:
    scheduler = _resolve_scheduler_config(args)
    specs, base_runtime_dir = _build_job_specs(args, scheduler)

    LOGGER.info(
        'parallel_learning.start mode=%s requested_tasks=%s logical_jobs=%s max_workers=%s episodes=%s root=%s',
        args.mode,
        scheduler.requested_tasks,
        scheduler.logical_jobs,
        scheduler.max_workers,
        scheduler.episodes_total,
        base_runtime_dir,
    )
    print(
        f'[parallel] mode={args.mode} requested_tasks={scheduler.requested_tasks} '
        f'logical_jobs={scheduler.logical_jobs} max_workers={scheduler.max_workers}'
    )
    print(f'[parallel] runtime_root={base_runtime_dir}')
    if scheduler.use_queue:
        print('[parallel] scheduler=bounded queue mode enabled')

    results: List[ParallelJobResult] = []
    with ProcessPoolExecutor(max_workers=scheduler.max_workers) as executor:
        for result in _iter_completed_jobs(executor, specs):
            results.append(result)
            print(
                f'[parallel] job={result.job_index:04d} '
                f'episodes={result.episodes} exit={result.exit_code}'
            )
            if result.exit_code != 0:
                raise RuntimeError(
                    f'Parallel job {result.job_index} failed with exit code {result.exit_code}.'
                )

    results.sort(key=lambda item: item.job_index)
    policy_paths = [item.policy_memory_path for item in results]
    bandit_paths = [item.action_bandit_path for item in results]
    pending_paths = [item.unknown_pending_path for item in results]
    overlay_paths = [item.unknown_overlay_path for item in results]

    merged_policy = (
        _merge_policy_memory(policy_paths, str(args.policy_memory))
        if not getattr(args, 'no_policy_memory', False)
        else None
    )
    merged_bandit = (
        _merge_action_bandit(bandit_paths, str(args.action_bandit))
        if not getattr(args, 'no_action_bandit', False)
        else None
    )
    merged_pending = _merge_unknown_pending(pending_paths, 'runtime/unknown_word_candidates.jsonl')
    merged_overlay = _merge_unknown_overlay(overlay_paths, 'runtime/lexicon_overlay.json')
    summary_path = _write_summary(
        destination_dir=base_runtime_dir,
        scheduler=scheduler,
        results=results,
        merged_policy=merged_policy,
        merged_bandit=merged_bandit,
        merged_pending=merged_pending,
        merged_overlay=merged_overlay,
    )

    print(
        f'[parallel] merged_policy_memory={merged_policy}'
        if merged_policy
        else '[parallel] merged_policy_memory=disabled'
    )
    print(
        f'[parallel] merged_action_bandit={merged_bandit}'
        if merged_bandit
        else '[parallel] merged_action_bandit=disabled'
    )
    print(f'[parallel] merged_unknown_pending={merged_pending}')
    print(f'[parallel] merged_unknown_overlay={merged_overlay}')
    print(f'[parallel] summary={summary_path}')
    LOGGER.info(
        'parallel_learning.done mode=%s logical_jobs=%s max_workers=%s root=%s summary=%s',
        args.mode,
        len(results),
        scheduler.max_workers,
        base_runtime_dir,
        summary_path,
    )
    return 0
