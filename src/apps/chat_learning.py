from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Optional, Sequence

from src.apps.run_minimal_chat import SurfaceNormalizer
from src.core.io.lsd_lexicon import load_lexicon_container
from src.core.schema import DialogueState, LexiconContainer, new_session_id
from src.training.action_bandit import ActionBanditConfig, ActionBanditStore
from src.training.external_evaluator import HeuristicExternalEvaluatorConfig, LLMExternalEvaluatorConfig, build_external_evaluator
from src.training.learning_central import LearningRuntimeConfig, run_learning_episode
from src.training.policy_memory import PolicyMemoryConfig, PolicyMemoryStore
from src.training.reward_aggregator import RewardAggregator, RewardAggregatorConfig
from src.training.target_generator import LLMTargetGeneratorConfig, build_target_generator
from src.training.teacher_guidance import TeacherGuidanceConfig, TeacherGuidedReranker
from src.utils.logging import setup_logging
from src.utils.settings import apply_arg_defaults, build_dataclass_config, get_setting, load_settings

LOGGER = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='LSLM v3 learning runner')
    parser.add_argument('--lexicon', default=None)
    parser.add_argument('--trace-dir', default=None)
    parser.add_argument('--dataset-dir', default=None)
    parser.add_argument('--policy-memory', default=None)
    parser.add_argument('--action-bandit', default=None)
    parser.add_argument('--no-policy-memory', action='store_true')
    parser.add_argument('--no-action-bandit', action='store_true')
    parser.add_argument('--no-trace', action='store_true')
    parser.add_argument('--no-dataset', action='store_true')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--text', default='')
    parser.add_argument('--words', nargs='*', default=None)
    parser.add_argument('--episodes', type=int, default=None)
    parser.add_argument('--target-mode', choices=['llm', 'echo'], default=None)
    parser.add_argument('--external-mode', choices=['llm', 'heuristic', 'none'], default=None)
    parser.add_argument('--alpha', type=float, default=None)
    parser.add_argument('--beta', type=float, default=None)
    parser.add_argument('--teacher-target-weight', type=float, default=None)
    parser.add_argument('--external-fallback', choices=['neutral', 'internal'], default=None)
    parser.add_argument('--neutral-external-score', type=float, default=None)
    return parser


def resolve_args(args: argparse.Namespace) -> argparse.Namespace:
    settings = load_settings()
    return apply_arg_defaults(
        args,
        settings,
        [
            ('lexicon', ('paths', 'lexicon'), 'libs/dict.lsdx'),
            ('trace_dir', ('paths', 'trace_dir'), 'runtime/traces'),
            ('dataset_dir', ('paths', 'dataset_dir'), 'runtime/datasets'),
            ('policy_memory', ('paths', 'policy_memory'), 'runtime/policy_memory.json'),
            ('action_bandit', ('paths', 'action_bandit'), 'runtime/action_bandit.json'),
            ('episodes', ('learning', 'episodes'), 1),
            ('target_mode', ('learning', 'target_mode'), 'llm'),
            ('external_mode', ('learning', 'external_mode'), 'llm'),
            ('alpha', ('learning', 'reward_aggregator', 'alpha'), 0.8),
            ('beta', ('learning', 'reward_aggregator', 'beta'), 0.2),
            ('teacher_target_weight', ('learning', 'teacher_target_weight'), 0.35),
            ('external_fallback', ('learning', 'reward_aggregator', 'fallback_strategy'), 'neutral'),
            ('neutral_external_score', ('learning', 'reward_aggregator', 'neutral_external_score'), 0.5),
        ],
    )


def load_lexicon(path: Path) -> LexiconContainer:
    LOGGER.info('lexicon.load.start path=%s', path)
    container_raw = load_lexicon_container(path)
    lexicon = LexiconContainer.from_dict(container_raw)
    LOGGER.info('lexicon.load.done entries=%s version=%s', len(lexicon.entries), lexicon.meta.version)
    return lexicon


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = resolve_args(parser.parse_args(argv))
    settings = load_settings()

    setup_logging(
        app_name='lslm_learning',
        console_level=logging.DEBUG if args.debug else logging.INFO,
    )

    lexicon_path = Path(args.lexicon)
    if not lexicon_path.exists():
        raise FileNotFoundError(f'Lexicon file not found: {lexicon_path}')

    lexicon = load_lexicon(lexicon_path)
    normalizer = SurfaceNormalizer(lexicon)
    runtime_config = LearningRuntimeConfig(
        trace_dir=args.trace_dir,
        dataset_dir=args.dataset_dir,
        save_trace=not args.no_trace,
        save_dataset=not args.no_dataset,
        policy_memory_path=args.policy_memory,
        use_policy_memory=not args.no_policy_memory,
        action_bandit_path=args.action_bandit,
        use_action_bandit=not args.no_action_bandit,
        policy_memory_limit=max(1, int(get_setting(settings, 'learning', 'runtime', 'policy_memory_limit', default=4))),
        teacher_target_weight=float(args.teacher_target_weight),
    )
    target_generator = build_target_generator(
        args.target_mode,
        config=build_dataclass_config(LLMTargetGeneratorConfig, get_setting(settings, 'llm', 'target_generation', default={})),
    )
    external_evaluator = build_external_evaluator(
        args.external_mode,
        llm_config=build_dataclass_config(LLMExternalEvaluatorConfig, get_setting(settings, 'llm', 'external_evaluation', default={})),
        heuristic_config=build_dataclass_config(HeuristicExternalEvaluatorConfig, get_setting(settings, 'llm', 'heuristic_external', default={})),
    )
    reward_aggregator = RewardAggregator(
        RewardAggregatorConfig(
            alpha=float(args.alpha),
            beta=float(args.beta),
            fallback_strategy=str(args.external_fallback),
            neutral_external_score=float(args.neutral_external_score),
            use_power_transform=bool(get_setting(settings, 'learning', 'reward_aggregator', 'use_power_transform', default=False)),
            external_power=float(get_setting(settings, 'learning', 'reward_aggregator', 'external_power', default=1.5)),
        )
    )

    policy_memory = None
    action_bandit = None
    if runtime_config.use_policy_memory:
        policy_memory = PolicyMemoryStore(
            runtime_config.policy_memory_path,
            config=build_dataclass_config(PolicyMemoryConfig, get_setting(settings, 'policy_memory', default={})),
            autoload=True,
        )
    if runtime_config.use_action_bandit:
        action_bandit = ActionBanditStore(
            runtime_config.action_bandit_path,
            config=build_dataclass_config(ActionBanditConfig, get_setting(settings, 'learning', 'action_bandit', default={})),
            autoload=True,
        )
    teacher_reranker = TeacherGuidedReranker(
        build_dataclass_config(TeacherGuidanceConfig, get_setting(settings, 'teacher_guidance', default={}))
    )

    session_id = new_session_id('learnsess')
    dialogue_state = DialogueState()

    def run_once(raw_text: str, explicit_words: Optional[Sequence[str]]) -> None:
        nonlocal dialogue_state
        result = run_learning_episode(
            raw_text=raw_text,
            explicit_words=explicit_words,
            lexicon=lexicon,
            normalizer=normalizer,
            runtime_config=runtime_config,
            target_generator=target_generator,
            external_evaluator=external_evaluator,
            reward_aggregator=reward_aggregator,
            dialogue_state=dialogue_state,
            session_id=session_id,
            policy_memory=policy_memory,
            teacher_reranker=teacher_reranker,
            action_bandit=action_bandit,
        )
        dialogue_state = result.next_dialogue_state or dialogue_state
        print(result.response_text)
        print(f'[target] {result.generated_target.text}')
        print(
            f'[reward] total={result.trace.reward.total:.4f} '
            f'internal={result.trace.reward.internal.total:.4f} '
            f'external={result.trace.reward.external.total:.4f}'
        )

    if args.text or args.words:
        raw_text = args.text if args.text else ' '.join(args.words or [])
        for _ in range(max(1, int(args.episodes))):
            run_once(raw_text=raw_text, explicit_words=args.words)
        return 0

    print('LSLM v3 learning mode')
    while True:
        try:
            raw_text = input('>>> ').strip()
        except KeyboardInterrupt:
            print()
            return 0

        if not raw_text:
            continue
        if raw_text.lower() in {'exit', 'quit'}:
            return 0

        try:
            run_once(raw_text=raw_text, explicit_words=None)
        except Exception:
            LOGGER.exception('learning_interactive_turn_failed')
            print('[ERROR] 学習ターンの実行に失敗しました')


if __name__ == '__main__':
    raise SystemExit(main())
