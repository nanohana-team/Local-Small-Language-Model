from __future__ import annotations

import json
import logging
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Sequence
from zoneinfo import ZoneInfo

from src.training.action_bandit import ActionBanditStore
from src.training.policy_memory import PolicyMemoryStore

LOGGER = logging.getLogger(__name__)
JST = ZoneInfo('Asia/Tokyo')


@dataclass(slots=True)
class ParallelWorkerSpec:
    mode: str
    worker_index: int
    episodes: int
    argv: List[str]
    policy_memory_path: str
    action_bandit_path: str
    unknown_pending_path: str
    unknown_overlay_path: str


@dataclass(slots=True)
class ParallelWorkerResult:
    worker_index: int
    exit_code: int
    episodes: int
    policy_memory_path: str
    action_bandit_path: str
    unknown_pending_path: str
    unknown_overlay_path: str


def _now_stamp() -> str:
    return datetime.now(JST).strftime('%Y%m%d%H%M%S')


def _split_episodes(total: int, tasks: int) -> List[int]:
    total = max(1, int(total))
    tasks = max(1, min(int(tasks), total))
    base = total // tasks
    remainder = total % tasks
    return [base + (1 if index < remainder else 0) for index in range(tasks)]


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


def _worker_runtime_root(base_runtime_dir: Path, worker_index: int) -> Path:
    return base_runtime_dir / f'worker_{worker_index:02d}'


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


def _build_worker_specs(args) -> tuple[List[ParallelWorkerSpec], Path]:
    mode = str(args.mode)
    total_episodes = max(1, int(args.episodes))
    task_count = max(1, min(int(getattr(args, 'learning_tasks', 1) or 1), total_episodes))
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

    episode_splits = _split_episodes(total_episodes, task_count)
    specs: List[ParallelWorkerSpec] = []
    episode_offset = 0
    for worker_index, worker_episodes in enumerate(episode_splits, start=1):
        worker_root = _worker_runtime_root(base_runtime_dir, worker_index)
        logs_dir = worker_root / 'logs'
        traces_dir = worker_root / 'traces'
        datasets_dir = worker_root / 'datasets'
        policy_memory_path = worker_root / 'policy_memory.json'
        action_bandit_path = worker_root / 'action_bandit.json'
        unknown_pending_path = worker_root / 'unknown_word_candidates.jsonl'
        unknown_overlay_path = worker_root / 'lexicon_overlay.json'

        worker_argv = list(base_argv)
        worker_argv = _replace_or_add(worker_argv, '--episodes', str(worker_episodes))
        worker_argv = _replace_or_add(worker_argv, '--trace-dir', str(traces_dir))
        worker_argv = _replace_or_add(worker_argv, '--dataset-dir', str(datasets_dir))
        worker_argv = _replace_or_add(worker_argv, '--policy-memory', str(policy_memory_path))
        worker_argv = _replace_or_add(worker_argv, '--action-bandit', str(action_bandit_path))
        worker_argv = _replace_or_add(worker_argv, '--log-dir', str(logs_dir))
        if mode == 'auto-learn':
            worker_argv = _replace_or_add(worker_argv, '--loop-index-offset', str(episode_offset))

        specs.append(
            ParallelWorkerSpec(
                mode=mode,
                worker_index=worker_index,
                episodes=worker_episodes,
                argv=worker_argv,
                policy_memory_path=str(policy_memory_path),
                action_bandit_path=str(action_bandit_path),
                unknown_pending_path=str(unknown_pending_path),
                unknown_overlay_path=str(unknown_overlay_path),
            )
        )
        episode_offset += worker_episodes
    return specs, base_runtime_dir


def _run_parallel_worker(spec: ParallelWorkerSpec) -> ParallelWorkerResult:
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
    return ParallelWorkerResult(
        worker_index=spec.worker_index,
        exit_code=exit_code,
        episodes=spec.episodes,
        policy_memory_path=spec.policy_memory_path,
        action_bandit_path=spec.action_bandit_path,
        unknown_pending_path=spec.unknown_pending_path,
        unknown_overlay_path=spec.unknown_overlay_path,
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


def run_parallel_learning(args) -> int:
    specs, base_runtime_dir = _build_worker_specs(args)
    worker_count = len(specs)
    LOGGER.info(
        'parallel_learning.start mode=%s workers=%s episodes=%s root=%s',
        args.mode,
        worker_count,
        args.episodes,
        base_runtime_dir,
    )
    print(f'[parallel] mode={args.mode} tasks={worker_count} episodes={args.episodes}')
    print(f'[parallel] runtime_root={base_runtime_dir}')

    results: List[ParallelWorkerResult] = []
    with ProcessPoolExecutor(max_workers=worker_count) as executor:
        future_map = {executor.submit(_run_parallel_worker, spec): spec for spec in specs}
        for future in as_completed(future_map):
            result = future.result()
            results.append(result)
            print(f'[parallel] worker={result.worker_index} episodes={result.episodes} exit={result.exit_code}')
            if result.exit_code != 0:
                raise RuntimeError(f'Parallel worker {result.worker_index} failed with exit code {result.exit_code}.')

    results.sort(key=lambda item: item.worker_index)
    policy_paths = [item.policy_memory_path for item in results]
    bandit_paths = [item.action_bandit_path for item in results]
    pending_paths = [item.unknown_pending_path for item in results]
    overlay_paths = [item.unknown_overlay_path for item in results]

    merged_policy = _merge_policy_memory(policy_paths, str(args.policy_memory)) if not getattr(args, 'no_policy_memory', False) else None
    merged_bandit = _merge_action_bandit(bandit_paths, str(args.action_bandit)) if not getattr(args, 'no_action_bandit', False) else None
    merged_pending = _merge_unknown_pending(pending_paths, 'runtime/unknown_word_candidates.jsonl')
    merged_overlay = _merge_unknown_overlay(overlay_paths, 'runtime/lexicon_overlay.json')

    print(f'[parallel] merged_policy_memory={merged_policy}' if merged_policy else '[parallel] merged_policy_memory=disabled')
    print(f'[parallel] merged_action_bandit={merged_bandit}' if merged_bandit else '[parallel] merged_action_bandit=disabled')
    print(f'[parallel] merged_unknown_pending={merged_pending}')
    print(f'[parallel] merged_unknown_overlay={merged_overlay}')
    LOGGER.info('parallel_learning.done mode=%s workers=%s root=%s', args.mode, len(results), base_runtime_dir)
    return 0
