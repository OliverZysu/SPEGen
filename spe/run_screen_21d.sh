#!/usr/bin/env bash
#
# 21 维输入约束下的改进筛选实验。
#
# 硬约束：所有配置的输入都是 21 维分子描述符（`--no_functional_groups` 恒开），
# 与传统基线完全同输入。溶剂拆解（步骤 4）也恒关 —— 它已被证明无效，
# 且会引入额外的头，留着只会混淆归因。
#
# 每个配置只改一件事，方便把效果归因到具体改动。
# 选模与配置选择一律以**验证集**为依据（train.py 会输出 val_select_score_tuned），
# 测试集指标只用于最终报告，不参与任何选择。
#
# 用法：
#   bash spe/run_screen_21d.sh                 # 全部配置，seed 42
#   SEEDS="42 7 2024" bash spe/run_screen_21d.sh   # 多 seed
#   CONFIGS="R2_count_cls R5_ple" bash spe/run_screen_21d.sh

set -u

OUT_DIR="${OUT_DIR:-output/screen21}"
EPOCHS="${EPOCHS:-60}"
SEEDS="${SEEDS:-42}"
LOG_DIR="$OUT_DIR/logs"
mkdir -p "$LOG_DIR"

# 21 维 + 无溶剂拆解，所有配置共享
BASE_FLAGS="--no_functional_groups --no_decompose_solvent"

ALL_CONFIGS="\
R0_control \
R1_step2 \
R2_count_cls \
R3_count_cls_exp \
R4_count_cls_f1 \
R5_ple \
R6_periodic \
R7_asl_sol \
R8_ratio_bucket \
R9_uncertainty \
R10_sol_w03 \
R11_sol_w01 \
R12_rank_car_up \
R13_count_q \
D1_single_car \
"

# 第二轮：把第一轮里超过噪声底的有效项组合起来。
# 有效项（4 次重复的 2std 为阈值）：
#   R8 ratio 分桶  -> ratio<=0.10 +0.10、Car F1 +0.017、GUS +2.4，无副作用
#   R7 溶剂 ASL    -> 溶剂 F1 +0.054、GUS +2.1
#   R12 排序权重↑  -> Car top-1 +0.048（唯一能撬动 top-1 的），代价 溶剂 F1 -0.016
#   R11 溶剂权重↓  -> Car top-1 +0.020、链路 +0.013
#   R2 计数分类    -> Car 完全匹配 0.040->0.130（超 RandomForest），代价 Car F1 崩
COMBO_CONFIGS="\
C1_r78 \
C2_r78_rank \
C3_r78_rank_solw \
C4_r78_step2 \
C5_r78_cls_q \
C6_rank10 \
C7_full \
"

CONFIGS="${CONFIGS:-$ALL_CONFIGS}"

flags_for() {
  case "$1" in
    # --- 参照组 ---
    # 四步全关，等价于历史 A_control
    R0_control)      echo "--no_cardinality --decode_objective f1" ;;
    # 基数回归 + em_jaccard 目标，等价于历史 C_step2
    R1_step2)        echo "--decode_objective em_jaccard" ;;

    # --- 集合大小：回归 -> 分桶分类 ---
    R2_count_cls)    echo "--count_mode classification --decode_objective em_jaccard" ;;
    # 取期望而不是众数，用来验证「取期望会把分类的好处抵消掉」这个判断
    R3_count_cls_exp) echo "--count_mode classification --count_reduce expectation --decode_objective em_jaccard" ;;
    # 分类 + F1 目标，隔离出「调优目标」和「计数形式」两者的贡献
    R4_count_cls_f1) echo "--count_mode classification --decode_objective f1" ;;
    # 分类 + 在验证集上搜「取分布的哪个分位数」。
    # R2（取众数）完全匹配 0.1301 已超 RandomForest，但集合塌到 1.20 使 F1 崩到 0.265；
    # R3（取期望）F1/GUS 最好但完全匹配只剩 0.018。分位数把两者之间的区间打开。
    R13_count_q)     echo "--count_mode classification --tune_count_quantile --decode_objective em_jaccard" ;;

    # --- 编码器怎么消化那 21 个标量（输入信息不变）---
    R5_ple)          echo "--no_cardinality --numeric_embed ple --decode_objective f1" ;;
    R6_periodic)     echo "--no_cardinality --numeric_embed periodic --decode_objective f1" ;;

    # --- 损失形式 ---
    R7_asl_sol)      echo "--no_cardinality --loss_solvent asl --decode_objective f1" ;;
    R8_ratio_bucket) echo "--no_cardinality --ratio_mode bucket --decode_objective f1" ;;

    # --- 多任务干扰缓解 ---
    R9_uncertainty)  echo "--no_cardinality --uncertainty_weighting --decode_objective f1" ;;
    R10_sol_w03)     echo "--no_cardinality --lambda_solvent 0.3 --decode_objective f1" ;;
    R11_sol_w01)     echo "--no_cardinality --lambda_solvent 0.1 --decode_objective f1" ;;
    # 加大 cartridge 排序损失，直接针对 cartridge top-1
    R12_rank_car_up) echo "--no_cardinality --lambda_rank_cartridge 0.5 --decode_objective f1" ;;

    # --- 诊断：只训 cartridge，用来验证多任务干扰假说 ---
    # 其余任务 lambda 置 0，选模只看 cartridge top-1。
    # 这个配置的 step/solvent/链路指标全部无意义，只读 cartridge top-1。
    D1_single_car)   echo "--no_cardinality --lambda_step 0 --lambda_solvent 0 --lambda_ratio 0 \
                           --lambda_rank_solvent 0 --decode_objective f1 \
                           --sel_w_strict 0 --sel_w_sss 0 --sel_w_hc 0 --sel_w_chain 0 \
                           --sel_w_gus 0 --sel_w_car_top1 1" ;;

    # --- 第二轮：有效项组合 ---
    # 两个无冲突的赢家叠加，作为组合基线
    C1_r78)          echo "--no_cardinality --loss_solvent asl --ratio_mode bucket --decode_objective f1" ;;
    # 再叠 cartridge 排序权重，看 top-1 的 +0.048 能不能保住
    C2_r78_rank)     echo "--no_cardinality --loss_solvent asl --ratio_mode bucket \
                           --lambda_rank_cartridge 0.5 --decode_objective f1" ;;
    # 排序权重↑ 与 溶剂权重↓ 两条撬 top-1 的路一起上；ASL 用来补偿溶剂被降权的损失
    C3_r78_rank_solw) echo "--no_cardinality --loss_solvent asl --ratio_mode bucket \
                           --lambda_rank_cartridge 0.5 --lambda_solvent 0.1 --decode_objective f1" ;;
    # 组合基线 + 基数解码，换取完全匹配/SSS
    C4_r78_step2)    echo "--loss_solvent asl --ratio_mode bucket --decode_objective em_jaccard" ;;
    # 组合基线 + 计数分类 + 分位数搜索，冲 Car 完全匹配
    C5_r78_cls_q)    echo "--loss_solvent asl --ratio_mode bucket --count_mode classification \
                           --tune_count_quantile --decode_objective em_jaccard" ;;
    # 排序权重加到 1.0，测 top-1 还能不能再推
    C6_rank10)       echo "--no_cardinality --lambda_rank_cartridge 1.0 --decode_objective f1" ;;
    # 全家桶
    C7_full)         echo "--loss_solvent asl --ratio_mode bucket --count_mode classification \
                           --tune_count_quantile --lambda_rank_cartridge 0.5 --decode_objective em_jaccard" ;;

    # --- 第三轮：继续推 cartridge top-1 / 链路 ---
    # C6（排序权重 1.0）把 top-1 从 0.5572 推到 0.5703，说明这条路还没到顶
    C8_r78_rank10)   echo "--no_cardinality --loss_solvent asl --ratio_mode bucket \
                           --lambda_rank_cartridge 1.0 --decode_objective f1" ;;
    C9_rank20)       echo "--no_cardinality --lambda_rank_cartridge 2.0 --decode_objective f1" ;;
    C10_r78_rank10_solw) echo "--no_cardinality --loss_solvent asl --ratio_mode bucket \
                           --lambda_rank_cartridge 1.0 --lambda_solvent 0.1 --decode_objective f1" ;;
    # 选模只盯 top-1，看纯粹为这个指标优化能到哪
    C11_r78_rank10_sel) echo "--no_cardinality --loss_solvent asl --ratio_mode bucket \
                           --lambda_rank_cartridge 1.0 --decode_objective f1 \
                           --sel_w_strict 0 --sel_w_sss 0.2 --sel_w_hc 0 --sel_w_chain 1 \
                           --sel_w_gus 0 --sel_w_car_top1 1" ;;

    # --- 第四轮：排序权重继续加大 + 与其他有效项合并 ---
    # top-1 随排序权重单调上升：0.5 -> 0.5520，1.0 -> 0.5563，2.0 -> 0.5817
    C12_r78_rank20_solw) echo "--no_cardinality --loss_solvent asl --ratio_mode bucket \
                           --lambda_rank_cartridge 2.0 --lambda_solvent 0.1 --decode_objective f1" ;;
    C13_r78_rank30_solw) echo "--no_cardinality --loss_solvent asl --ratio_mode bucket \
                           --lambda_rank_cartridge 3.0 --lambda_solvent 0.1 --decode_objective f1" ;;
    C14_rank20_solw) echo "--no_cardinality --lambda_rank_cartridge 2.0 --lambda_solvent 0.1 \
                           --decode_objective f1" ;;
    C15_r78_rank20)  echo "--no_cardinality --loss_solvent asl --ratio_mode bucket \
                           --lambda_rank_cartridge 2.0 --decode_objective f1" ;;

    *) echo "__UNKNOWN__" ;;
  esac
}

for seed in $SEEDS; do
  for cfg in $CONFIGS; do
    flags="$(flags_for "$cfg")"
    if [ "$flags" = "__UNKNOWN__" ]; then
      echo "[SKIP] 未知配置 $cfg"
      continue
    fi
    run_dir="$OUT_DIR/${cfg}_seed${seed}"
    log="$LOG_DIR/${cfg}_seed${seed}.log"
    if [ -f "$run_dir/test_metrics.json" ]; then
      echo "[SKIP] $cfg seed$seed 已完成"
      continue
    fi
    echo "[RUN ] $cfg seed$seed"
    # shellcheck disable=SC2086
    python3 -m spe.train \
      --output_dir "$run_dir" --seed "$seed" --epochs "$EPOCHS" \
      $BASE_FLAGS $flags > "$log" 2>&1
    if [ $? -ne 0 ]; then
      echo "[FAIL] $cfg seed$seed，日志见 $log"
      tail -5 "$log"
    fi
  done
done

echo "全部完成，结果在 $OUT_DIR"
