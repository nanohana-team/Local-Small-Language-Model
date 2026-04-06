from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Optional, Sequence

from src.apps.run_minimal_chat import (
    SurfaceNormalizer,
    parse_args as parse_minimal_args,
    run_pipeline,
)
from src.core.io.lsd_lexicon import load_lexicon_container
from src.core.schema import LexiconContainer
from src.utils.logging import setup_logging
from src.utils.settings import apply_arg_defaults, load_settings

LOGGER = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

    parser.add_argument('--mode', choices=['chat', 'learn', 'auto-learn'], default='chat')
    parser.add_argument('--lexicon', default=None)
    parser.add_argument('--trace-dir', default=None)
    parser.add_argument('--policy-memory', default=None)
    parser.add_argument('--action-bandit', default=None)
    parser.add_argument('--no-policy-memory', action='store_true')
    parser.add_argument('--no-action-bandit', action='store_true')
    parser.add_argument('--no-trace', action='store_true')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--text', default='')
    parser.add_argument('--words', nargs='*', default=None)

    parser.add_argument('--dataset-dir', default=None)
    parser.add_argument('--no-dataset', action='store_true')
    parser.add_argument('--episodes', type=int, default=None)
    parser.add_argument('--target-mode', choices=['llm', 'echo'], default=None)
    parser.add_argument('--external-mode', choices=['llm', 'heuristic', 'none'], default=None)
    parser.add_argument('--alpha', type=float, default=None)
    parser.add_argument('--beta', type=float, default=None)
    parser.add_argument('--teacher-target-weight', type=float, default=None)
    parser.add_argument('--external-fallback', choices=['neutral', 'internal'], default=None)
    parser.add_argument('--neutral-external-score', type=float, default=None)
    parser.add_argument('--input-mode', choices=['llm', 'echo'], default=None)
    parser.add_argument('--seed-topic', default=None)
    parser.add_argument('--seed-text', default=None)
    parser.add_argument('--report-every', type=int, default=None)

    return parser


def resolve_args(args: argparse.Namespace) -> argparse.Namespace:
    settings = load_settings()
    return apply_arg_defaults(
        args,
        settings,
        [
            ('lexicon', ('paths', 'lexicon'), 'libs/dict.lsdx'),
            ('trace_dir', ('paths', 'trace_dir'), 'runtime/traces'),
            ('policy_memory', ('paths', 'policy_memory'), 'runtime/policy_memory.json'),
            ('action_bandit', ('paths', 'action_bandit'), 'runtime/action_bandit.json'),
            ('dataset_dir', ('paths', 'dataset_dir'), 'runtime/datasets'),
            ('episodes', ('learning', 'episodes'), 1),
            ('target_mode', ('learning', 'target_mode'), 'llm'),
            ('external_mode', ('learning', 'external_mode'), 'llm'),
            ('alpha', ('learning', 'reward_aggregator', 'alpha'), 0.8),
            ('beta', ('learning', 'reward_aggregator', 'beta'), 0.2),
            ('teacher_target_weight', ('learning', 'teacher_target_weight'), 0.35),
            ('external_fallback', ('learning', 'reward_aggregator', 'fallback_strategy'), 'neutral'),
            ('neutral_external_score', ('learning', 'reward_aggregator', 'neutral_external_score'), 0.5),
            ('input_mode', ('learning', 'input_mode'), 'llm'),
            ('seed_topic', ('learning', 'seed_topic'), ''),
            ('seed_text', ('learning', 'seed_text'), 'こんにちは。今日はどんな一日になりそうですか。'),
            ('report_every', ('learning', 'report_every'), 1),
        ],
    )


def build_pipeline_args(
    *,
    lexicon: str,
    trace_dir: str,
    no_trace: bool,
    debug: bool,
    text: str,
    words: Optional[Sequence[str]],
    policy_memory: str,
    no_policy_memory: bool,
):
    argv: list[str] = [
        '--lexicon',
        lexicon,
        '--trace-dir',
        trace_dir,
    ]

    if no_trace:
        argv.append('--no-trace')

    if no_policy_memory:
        argv.append('--no-policy-memory')
    else:
        argv.extend(['--policy-memory', policy_memory])

    if debug:
        argv.append('--console-debug')

    if words:
        argv.append('--words')
        argv.extend(words)
    else:
        argv.extend(['--text', text])

    return parse_minimal_args(argv)


def build_learning_args(args: argparse.Namespace) -> list[str]:
    argv: list[str] = [
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

    if args.no_trace:
        argv.append('--no-trace')
    if args.no_policy_memory:
        argv.append('--no-policy-memory')
    if args.no_action_bandit:
        argv.append('--no-action-bandit')
    if args.no_dataset:
        argv.append('--no-dataset')
    if args.debug:
        argv.append('--debug')
    if args.words:
        argv.append('--words')
        argv.extend(args.words)
    elif args.text:
        argv.extend(['--text', args.text])

    return argv


def build_auto_learning_args(args: argparse.Namespace) -> list[str]:
    argv: list[str] = [
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

    if args.no_trace:
        argv.append('--no-trace')
    if args.no_policy_memory:
        argv.append('--no-policy-memory')
    if args.no_action_bandit:
        argv.append('--no-action-bandit')
    if args.no_dataset:
        argv.append('--no-dataset')
    if args.debug:
        argv.append('--debug')

    return argv


def dispatch_auto_learning_mode(args: argparse.Namespace) -> int:
    from src.apps.auto_loop_learning import main as auto_loop_learning_main

    learning_argv = build_auto_learning_args(args)
    return int(auto_loop_learning_main(learning_argv))


def dispatch_learning_mode(args: argparse.Namespace) -> int:
    from src.apps.chat_learning import main as chat_learning_main

    learning_argv = build_learning_args(args)
    return int(chat_learning_main(learning_argv))


def run_chat_mode(args: argparse.Namespace) -> int:
    setup_logging(
        app_name='lslm',
        console_level=logging.DEBUG if args.debug else logging.INFO,
    )

    lexicon_path = Path(args.lexicon)
    LOGGER.info('lexicon.load.start path=%s', lexicon_path)

    container_raw = load_lexicon_container(lexicon_path)
    lexicon = LexiconContainer.from_dict(container_raw)

    LOGGER.info('lexicon.load.done entries=%s', len(lexicon.entries))

    normalizer = SurfaceNormalizer(lexicon)

    if args.text or args.words:
        pipeline_args = build_pipeline_args(
            lexicon=args.lexicon,
            trace_dir=args.trace_dir,
            no_trace=args.no_trace,
            debug=args.debug,
            text=args.text,
            words=args.words,
            policy_memory=args.policy_memory,
            no_policy_memory=args.no_policy_memory,
        )
        response_text, _ = run_pipeline(pipeline_args, lexicon, normalizer)
        print(response_text)
        return 0

    print('LSLM v3 interactive mode')

    while True:
        try:
            raw_text = input('>>> ').strip()
        except KeyboardInterrupt:
            print()
            return 0

        if not raw_text:
            continue

        if raw_text.lower() in ('exit', 'quit'):
            return 0

        try:
            pipeline_args = build_pipeline_args(
                lexicon=args.lexicon,
                trace_dir=args.trace_dir,
                no_trace=args.no_trace,
                debug=args.debug,
                text=raw_text,
                words=None,
                policy_memory=args.policy_memory,
                no_policy_memory=args.no_policy_memory,
            )

            response_text, _ = run_pipeline(
                pipeline_args,
                lexicon,
                normalizer,
            )

            print(response_text)

        except Exception:
            LOGGER.exception('interactive_turn_failed')
            print('[ERROR] 応答生成に失敗しました')


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = resolve_args(parser.parse_args(argv))

    if args.mode == 'learn':
        return dispatch_learning_mode(args)
    if args.mode == 'auto-learn':
        return dispatch_auto_learning_mode(args)

    return run_chat_mode(args)


if __name__ == '__main__':
    raise SystemExit(main())
