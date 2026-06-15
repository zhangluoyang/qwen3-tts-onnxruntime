from __future__ import annotations
import argparse
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import onnx
from onnx import ModelProto, NodeProto, TensorProto, numpy_helper
from onnxruntime.quantization.matmul_nbits_quantizer import DefaultWeightOnlyQuantConfig, MatMulNBitsQuantizer
from onnxruntime.quantization.quant_utils import QuantFormat

@dataclass(frozen=True)
class ComponentSpec:
    name: str
    path: Path
    default_exclude_patterns: tuple[str, ...] = ()

COMPONENT_SPECS: tuple[ComponentSpec, ...] = (
    ComponentSpec(
        name="talker_core",
        path=Path("talker") / "talker_core.onnx",
        default_exclude_patterns=("/codec_head/",),
    ),
    ComponentSpec(
        name="sub_talker_sample",
        path=Path("decode") / "sub_talker_sample.onnx",
        default_exclude_patterns=("/lm_head.",),
    ),
    ComponentSpec(
        name="tokenizer_decode",
        path=Path("tokenizer") / "tokenizer12hz_decode_chunk.onnx",
    ),
    ComponentSpec(
        name="tokenizer_encode",
        path=Path("tokenizer") / "tokenizer12hz_encode.onnx",
    ),
    ComponentSpec(
        name="text_project",
        path=Path("text_project") / "text_project.onnx",
    ),
    ComponentSpec(
        name="codec_embed",
        path=Path("codec_embed") / "codec_embed.onnx",
    ),
    ComponentSpec(
        name="speaker_encoder",
        path=Path("speaker_encoder") / "speaker_encoder.onnx",
    ),
)

COMPONENTS_BY_NAME = {spec.name: spec for spec in COMPONENT_SPECS}
DEFAULT_COMPONENTS = ("talker_core")

COMPONENT_ALIASES = {
    "all": "all",
    "talker": "talker_core",
    "talker-core": "talker_core",
    "core": "talker_core",
    "sub-talker-sample": "sub_talker_sample",
    "sub-talker": "sub_talker_sample",
    "sub_talker": "sub_talker_sample",
    "frame-prepare": "sub_talker_sample",
    "frame_prepare": "sub_talker_sample",
    "tokenizer-decode": "tokenizer_decode",
    "tokenizer-encode": "tokenizer_encode",
    "text-project": "text_project",
    "codec-embed": "codec_embed",
    "speaker-encoder": "speaker_encoder",
}

@dataclass
class FoldStats:
    folded: int = 0
    skipped_excluded: int = 0
    skipped_no_transpose: int = 0
    skipped_non_initializer: int = 0
    skipped_non_2d: int = 0
    skipped_unsupported_perm: int = 0

@dataclass
class ComponentStats:
    name: str
    path: Path
    total_matmul: int
    direct_initializer_matmul_before: int
    excluded_matmul: int
    foldable_transposed_matmul: int
    direct_initializer_matmul_after_fold: int
    matmul_nbits_after: int | None = None


def parse_components(value: str) -> tuple[str, ...]:
    raw_items = [item.strip().lower() for item in str(value).split(",") if item.strip()]
    if not raw_items:
        raise argparse.ArgumentTypeError("components cannot be empty")

    selected: list[str] = []
    for item in raw_items:
        normalized = COMPONENT_ALIASES.get(item, item.replace("-", "_"))
        if normalized == "all":
            return tuple(spec.name for spec in COMPONENT_SPECS)
        if normalized not in COMPONENTS_BY_NAME:
            valid = ", ".join(("all",) + tuple(COMPONENTS_BY_NAME))
            raise argparse.ArgumentTypeError(f"unsupported component {item!r}; choose from: {valid}")
        if normalized not in selected:
            selected.append(normalized)
    return tuple(selected)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Convert exported fp16 ONNX files to ORT weight-only n-bit models. "
            "The public runtime inputs stay fp16/int64; only constant MatMul weights are packed."
        )
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("onnx_fp16"),
        help="Source ONNX export directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("onnx_int4"),
        help="Destination directory. The input tree is copied here before quantization.",
    )
    parser.add_argument(
        "--components",
        type=parse_components,
        default=DEFAULT_COMPONENTS,
        help=(
            "Comma-separated components to quantize. Default: talker_core. "
            "Use all, talker_core, sub_talker_sample, tokenizer_decode, tokenizer_encode, "
            "text_project, codec_embed, speaker_encoder."
        ),
    )
    parser.add_argument(
        "--bits",
        type=int,
        default=4,
        choices=(2, 4, 8),
        help="Weight quantization bit width.",
    )
    parser.add_argument(
        "--block-size",
        type=int,
        default=128,
        help="Block size for weight-only quantization.",
    )
    parser.add_argument(
        "--accuracy-level",
        type=int,
        default=None,
        help="Optional MatMulNBits accuracy_level attribute.",
    )
    parser.add_argument(
        "--quant-format",
        choices=("QOperator", "QDQ"),
        default="QOperator",
        help="Use QOperator for MatMulNBits, or QDQ for DeQuantizeLinear + MatMul.",
    )
    parser.add_argument(
        "--symmetric",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use symmetric weight quantization.",
    )
    parser.add_argument(
        "--fold-transposed-weights",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Fold initializer -> Transpose -> MatMul into direct MatMul initializer weights before quantizing.",
    )
    parser.add_argument(
        "--no-default-excludes",
        action="store_true",
        help="Do not apply component-specific default exclusions such as codec_head/lm_head.",
    )
    parser.add_argument(
        "--exclude-pattern",
        action="append",
        default=[],
        help="Additional substring pattern for MatMul node names to exclude. Can be repeated.",
    )
    parser.add_argument(
        "--exclude-node",
        action="append",
        default=[],
        help="Exact MatMul node name to exclude. Can be repeated.",
    )
    parser.add_argument(
        "--include-node",
        action="append",
        default=None,
        help="Optional exact MatMul node names to include. If set, ORT only quantizes these names.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Remove an existing output directory before writing.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the conversion plan without copying or writing files.",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Run onnx.checker.check_model on each saved component.",
    )
    return parser


def print_header(message: str) -> None:
    print(f"\n==> {message}", flush=True)


def resolve_component_specs(names: tuple[str, ...]) -> tuple[ComponentSpec, ...]:
    return tuple(COMPONENTS_BY_NAME[name] for name in names)


def validate_args(args: argparse.Namespace) -> None:
    if args.block_size <= 0:
        raise ValueError("--block-size must be positive")
    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()
    if input_dir == output_dir:
        raise ValueError("--output-dir must be different from --input-dir")
    if not args.dry_run and output_dir.is_relative_to(input_dir):
        raise ValueError("--output-dir must not be inside --input-dir")


def copy_input_tree(input_dir: Path, output_dir: Path, overwrite: bool) -> None:
    if not input_dir.exists():
        raise FileNotFoundError(f"input directory does not exist: {input_dir}")
    if output_dir.exists():
        if not overwrite:
            raise FileExistsError(f"output directory already exists: {output_dir}; pass --overwrite to replace it")
        shutil.rmtree(output_dir)
    shutil.copytree(input_dir, output_dir)


def component_model_path(root: Path, spec: ComponentSpec) -> Path:
    return root / spec.path


def exclude_patterns_for_component(args: argparse.Namespace, spec: ComponentSpec) -> tuple[str, ...]:
    patterns: list[str] = []
    if not args.no_default_excludes:
        patterns.extend(spec.default_exclude_patterns)
    patterns.extend(args.exclude_pattern or [])
    return tuple(patterns)


def matched_node_names(
    model: ModelProto,
    patterns: tuple[str, ...],
    exact_names: list[str],
) -> set[str]:
    exact = set(exact_names or [])
    matches = set()
    for node in model.graph.node:
        if node.op_type != "MatMul":
            continue
        if node.name in exact or any(pattern and pattern in node.name for pattern in patterns):
            matches.add(node.name)
    return matches


def excluded_for_folding(
    model: ModelProto,
    excluded_nodes: set[str],
    include_nodes: list[str] | None,
) -> set[str]:
    if not include_nodes:
        return set(excluded_nodes)
    include_set = set(include_nodes)
    fold_excluded = set(excluded_nodes)
    for node in model.graph.node:
        if node.op_type == "MatMul" and node.name not in include_set:
            fold_excluded.add(node.name)
    return fold_excluded


def initializer_map(model: ModelProto) -> dict[str, TensorProto]:
    return {initializer.name: initializer for initializer in model.graph.initializer}


def producer_map(model: ModelProto) -> dict[str, NodeProto]:
    return {output: node for node in model.graph.node for output in node.output if output}


def consumer_map(model: ModelProto) -> dict[str, list[NodeProto]]:
    consumers: dict[str, list[NodeProto]] = {}
    for node in model.graph.node:
        for input_name in node.input:
            if input_name:
                consumers.setdefault(input_name, []).append(node)
    return consumers


def transpose_perm(node: NodeProto, ndim: int) -> tuple[int, ...]:
    for attr in node.attribute:
        if attr.name == "perm":
            return tuple(int(value) for value in attr.ints)
    return tuple(reversed(range(ndim)))


def unique_initializer_name(model: ModelProto, preferred: str) -> str:
    used = {initializer.name for initializer in model.graph.initializer}
    used.update(output for node in model.graph.node for output in node.output)
    if preferred not in used:
        return preferred
    index = 1
    while f"{preferred}_{index}" in used:
        index += 1
    return f"{preferred}_{index}"


def is_transposed_initializer_matmul(
    node: NodeProto,
    initializers: dict[str, TensorProto],
    producers: dict[str, NodeProto],
) -> bool:
    if node.op_type != "MatMul" or len(node.input) < 2:
        return False
    transpose = producers.get(node.input[1])
    return bool(
        transpose is not None
        and transpose.op_type == "Transpose"
        and transpose.input
        and transpose.input[0] in initializers
    )


def count_matmul_nodes(model: ModelProto) -> int:
    return sum(1 for node in model.graph.node if node.op_type == "MatMul")


def count_direct_initializer_matmuls(model: ModelProto, excluded_nodes: set[str] | None = None) -> int:
    excluded_nodes = excluded_nodes or set()
    initializers = initializer_map(model)
    count = 0
    for node in model.graph.node:
        if node.op_type == "MatMul" and node.name not in excluded_nodes and len(node.input) > 1:
            count += int(node.input[1] in initializers)
    return count


def count_transposed_initializer_matmuls(model: ModelProto, excluded_nodes: set[str] | None = None) -> int:
    excluded_nodes = excluded_nodes or set()
    initializers = initializer_map(model)
    producers = producer_map(model)
    return sum(
        1
        for node in model.graph.node
        if node.name not in excluded_nodes and is_transposed_initializer_matmul(node, initializers, producers)
    )


def count_matmul_nbits(model: ModelProto) -> int:
    return sum(
        1
        for node in model.graph.node
        if node.op_type == "MatMulNBits" and (node.domain == "com.microsoft" or not node.domain)
    )


def fold_transposed_matmul_weights(model: ModelProto, excluded_nodes: set[str]) -> FoldStats:
    stats = FoldStats()
    initializers = initializer_map(model)
    producers = producer_map(model)
    folded_transpose_outputs: set[str] = set()

    for node in model.graph.node:
        if node.op_type != "MatMul" or len(node.input) < 2:
            continue
        if node.name in excluded_nodes:
            stats.skipped_excluded += 1
            continue
        transpose = producers.get(node.input[1])
        if transpose is None or transpose.op_type != "Transpose":
            stats.skipped_no_transpose += 1
            continue
        if not transpose.input or transpose.input[0] not in initializers:
            stats.skipped_non_initializer += 1
            continue

        tensor = initializers[transpose.input[0]]
        array = numpy_helper.to_array(tensor)
        if array.ndim != 2:
            stats.skipped_non_2d += 1
            continue
        perm = transpose_perm(transpose, array.ndim)
        if perm != (1, 0):
            stats.skipped_unsupported_perm += 1
            continue

        folded = np.ascontiguousarray(array.T)
        folded_name = unique_initializer_name(model, f"{tensor.name}.folded_transpose")
        model.graph.initializer.append(numpy_helper.from_array(folded, folded_name))
        node.input[1] = folded_name
        folded_transpose_outputs.update(transpose.output)
        stats.folded += 1

    if folded_transpose_outputs:
        consumers = consumer_map(model)
        remove_nodes = set()
        for node in model.graph.node:
            if node.op_type != "Transpose":
                continue
            if node.output and all(not consumers.get(output) for output in node.output):
                remove_nodes.add(node.name)
        if remove_nodes:
            kept_nodes = [node for node in model.graph.node if node.name not in remove_nodes]
            model.graph.ClearField("node")
            model.graph.node.extend(kept_nodes)

    return stats


def plan_component(
    model: ModelProto,
    spec: ComponentSpec,
    path: Path,
    excluded_nodes: set[str],
    fold_transposed_weights: bool,
    mutate: bool,
) -> tuple[ComponentStats, FoldStats]:
    total_matmul = count_matmul_nodes(model)
    direct_before = count_direct_initializer_matmuls(model, excluded_nodes=excluded_nodes)
    excluded_matmul = sum(1 for node in model.graph.node if node.op_type == "MatMul" and node.name in excluded_nodes)
    foldable = count_transposed_initializer_matmuls(model, excluded_nodes=excluded_nodes)
    fold_stats = FoldStats()
    if fold_transposed_weights and mutate:
        fold_stats = fold_transposed_matmul_weights(model, excluded_nodes=excluded_nodes)
    direct_after = count_direct_initializer_matmuls(model, excluded_nodes=excluded_nodes)
    if fold_transposed_weights and not mutate:
        direct_after += foldable
    stats = ComponentStats(
        name=spec.name,
        path=path,
        total_matmul=total_matmul,
        direct_initializer_matmul_before=direct_before,
        excluded_matmul=excluded_matmul,
        foldable_transposed_matmul=foldable,
        direct_initializer_matmul_after_fold=direct_after,
    )
    return stats, fold_stats


def print_component_plan(stats: ComponentStats, excluded_nodes: set[str], fold_stats: FoldStats | None = None) -> None:
    print(f"{stats.name}: {stats.path}")
    print(f"  MatMul nodes: {stats.total_matmul}")
    print(f"  excluded MatMul nodes: {stats.excluded_matmul}")
    print(f"  direct initializer MatMul before fold: {stats.direct_initializer_matmul_before}")
    print(f"  foldable initializer->Transpose->MatMul: {stats.foldable_transposed_matmul}")
    print(f"  quantizable direct initializer MatMul after fold: {stats.direct_initializer_matmul_after_fold}")
    if stats.matmul_nbits_after is not None:
        print(f"  MatMulNBits after quantization: {stats.matmul_nbits_after}")
    if excluded_nodes:
        sample = sorted(excluded_nodes)[:8]
        suffix = "" if len(excluded_nodes) <= len(sample) else f" ... (+{len(excluded_nodes) - len(sample)} more)"
        print(f"  excluded sample: {sample}{suffix}")
    if fold_stats is not None and fold_stats.folded:
        print(f"  folded transposed weights: {fold_stats.folded}")


def quantize_component(
    model: ModelProto,
    model_path: Path,
    excluded_nodes: set[str],
    include_nodes: list[str] | None,
    args: argparse.Namespace,
) -> ModelProto:
    quant_config = DefaultWeightOnlyQuantConfig(
        block_size=args.block_size,
        is_symmetric=bool(args.symmetric),
        accuracy_level=args.accuracy_level,
        quant_format=QuantFormat[args.quant_format],
        op_types_to_quantize=("MatMul",),
        quant_axes=(("MatMul", 0),),
        bits=args.bits,
    )
    quantizer = MatMulNBitsQuantizer(
        model=model,
        bits=args.bits,
        block_size=args.block_size,
        is_symmetric=bool(args.symmetric),
        accuracy_level=args.accuracy_level,
        nodes_to_exclude=sorted(excluded_nodes),
        nodes_to_include=include_nodes,
        algo_config=quant_config,
    )
    quantizer.process()
    data_path = model_path.with_name(model_path.name + ".data")
    if data_path.exists():
        data_path.unlink()
    quantizer.model.save_model_to_file(str(model_path), use_external_data_format=True)
    return quantizer.model.model


def run(args: argparse.Namespace) -> int:
    validate_args(args)
    specs = resolve_component_specs(args.components)
    input_dir = args.input_dir
    output_dir = args.output_dir

    source_root = input_dir if args.dry_run else output_dir
    if not args.dry_run:
        print_header(f"Copying {input_dir} -> {output_dir}")
        copy_input_tree(input_dir=input_dir, output_dir=output_dir, overwrite=args.overwrite)

    for spec in specs:
        source_path = component_model_path(input_dir if args.dry_run else source_root, spec)
        if not source_path.exists():
            raise FileNotFoundError(f"component {spec.name} does not exist: {source_path}")

        print_header(f"Planning {spec.name}")
        model = onnx.load(source_path, load_external_data=True)
        patterns = exclude_patterns_for_component(args, spec)
        excluded_nodes = matched_node_names(model, patterns=patterns, exact_names=args.exclude_node)
        fold_excluded_nodes = excluded_for_folding(model, excluded_nodes, args.include_node)

        stats, fold_stats = plan_component(
            model=model,
            spec=spec,
            path=source_path,
            excluded_nodes=fold_excluded_nodes,
            fold_transposed_weights=args.fold_transposed_weights,
            mutate=not args.dry_run,
        )
        print_component_plan(stats, excluded_nodes=fold_excluded_nodes, fold_stats=fold_stats)

        if args.dry_run:
            continue

        print_header(f"Quantizing {source_path}")
        quantized_model = quantize_component(
            model=model,
            model_path=source_path,
            excluded_nodes=excluded_nodes,
            include_nodes=args.include_node,
            args=args,
        )
        stats.matmul_nbits_after = count_matmul_nbits(quantized_model)
        print_component_plan(stats, excluded_nodes=excluded_nodes)
        if args.check:
            print_header(f"Checking {source_path}")
            onnx.checker.check_model(str(source_path))

    if args.dry_run:
        print_header("Dry run complete; no files were written")
    else:
        print_header(f"Done: {output_dir}")
    return 0


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    try:
        return run(args)
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
