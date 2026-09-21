#!/usr/bin/env bash
#
# 21 维约束下的第二轮改进实验（阶段 1-4）。
#
# 硬约束不变：输入恒为 21 维分子描述符（`--no_functional_groups` 恒开），
# 溶剂拆解恒关。选模与配置选择一律只看验证集。
#
# 两个基底配置来自上一轮的最优点：
#   C2  均衡型：溶剂 ASL + ratio 分桶 + cartridge 排序权重 0.5
#   C12 链路型：同上，排序权重 2.0、溶剂任务权重 0.1
#
# 用法：
#   STAGE=1 bash spe/run_improve_21d.sh                    # 阶段 1 筛选（8 次训练）
#   STAGE=1 INITS="101 102 103" CONFIGS="..." bash ...      # 阶段 1 确认（多初始化）
#   STAGE=3 bash spe/run_improve_21d.sh                    # 阶段 3 排序损失形式

set -u

OUT_DIR="${OUT_DIR:-output/improve21}"
EPOCHS="${EPOCHS:-60}"
SEED="${SEED:-42}"
INITS="${INITS:-42}"
STAGE="${STAGE:-1}"
LOG_DIR="$OUT_DIR/logs"
mkdir -p "$LOG_DIR"

BASE_FLAGS="--no_functional_groups --no_decompose_solvent"

base_flags_for() {
  case "$1" in
    C2)  echo "--no_cardinality --loss_solvent asl --ratio_mode bucket \
               --lambda_rank_cartridge 0.5 --decode_objective f1" ;;
    C12) echo "--no_cardinality --loss_solvent asl --ratio_mode bucket \
               --lambda_rank_cartridge 2.0 --lambda_solvent 0.1 --decode_objective f1" ;;
    *)   echo "__UNKNOWN__" ;;
  esac
}

# 阶段 1：输入特征的单调变换。假设是 z-score 把极度偏斜的描述符喂给了 MLP，
# 而树模型对单调变换免疫 —— 这可能是 top-1 差距的根因。维度不变、不加特征。
STAGE1_CONFIGS="${CONFIGS:-\
C2_zscore C2_rank_gauss C2_log1p C2_robust \
C12_zscore C12_rank_gauss C12_log1p C12_robust \
}"

# 阶段 3：排序损失的形式（权重已扫到顶，2.0 最优，3.0 回落，说明该换形式）
STAGE3_CONFIGS="${CONFIGS:-\
L0_set_ce L1_any_pos L2_ranknet L3_margin \
}"

flags_for() {
  local cfg="$1"
  case "$STAGE" in
    1)
      local base="${cfg%%_*}"
      local transform="${cfg#*_}"
      local bf
      bf="$(base_flags_for "$base")"
      [ "$bf" = "__UNKNOWN__" ] && { echo "__UNKNOWN__"; return; }
      echo "$bf --feature_transform $transform"
      ;;
    3)
      # 都在阶段 1 选出的最优变换上做，由 BEST_TRANSFORM 传入
      local bf form
      bf="$(base_flags_for "${BASE_CFG:-C12}")"
      form="${cfg#*_}"          # L1_any_pos -> any_pos
      case "$form" in
        set_ce|any_pos|ranknet|margin)
          echo "$bf --feature_transform ${BEST_TRANSFORM:-zscore} --rank_loss $form" ;;
        *) echo "__UNKNOWN__" ;;
      esac
      ;;
    4)
      # bagging：每个成员换 init_seed 的同时对训练集有放回重采样。
      # 验证集与测试集不动，所以评测集与基线完全一致。
      local bf
      bf="$(base_flags_for "${BASE_CFG:-C12}")"
      echo "$bf --feature_transform ${BEST_TRANSFORM:-rank_gauss} \
            --bootstrap --bootstrap_frac ${BOOT_FRAC:-1.0}"
      ;;
    *) echo "__UNKNOWN__" ;;
  esac
}

# 阶段 4 只有一个配置，成员之间靠 INITS 区分
STAGE4_CONFIGS="${CONFIGS:-bag}"

case "$STAGE" in
  1) CFG_LIST="$STAGE1_CONFIGS" ;;
  3) CFG_LIST="$STAGE3_CONFIGS" ;;
  4) CFG_LIST="$STAGE4_CONFIGS" ;;
  *) echo "未知 STAGE=$STAGE"; exit 1 ;;
esac

for init in $INITS; do
  for cfg in $CFG_LIST; do
    flags="$(flags_for "$cfg")"
    if [ "$flags" = "__UNKNOWN__" ]; then
      echo "[SKIP] 未知配置 $cfg"
      continue
    fi
    run_dir="$OUT_DIR/s${STAGE}_${cfg}_init${init}"
    log="$LOG_DIR/s${STAGE}_${cfg}_init${init}.log"
    if [ -f "$run_dir/test_metrics.json" ]; then
      echo "[SKIP] $cfg init$init 已完成"
      continue
    fi
    echo "[RUN ] stage$STAGE $cfg init$init"
    # shellcheck disable=SC2086
    python3 -m spe.train \
      --output_dir "$run_dir" --seed "$SEED" --init_seed "$init" --epochs "$EPOCHS" \
      $BASE_FLAGS $flags > "$log" 2>&1
    if [ $? -ne 0 ]; then
      echo "[FAIL] $cfg init$init，日志见 $log"
      tail -5 "$log"
    fi
  done
done

echo "阶段 $STAGE 完成，结果在 $OUT_DIR"
