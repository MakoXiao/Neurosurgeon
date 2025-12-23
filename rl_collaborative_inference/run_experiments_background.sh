#!/bin/bash
# 后台运行实验的便捷脚本
# 使用方法: ./run_experiments_background.sh [command] [options]

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="${SCRIPT_DIR}/logs"
PYTHON_SCRIPT="${SCRIPT_DIR}/run_training_background.py"

# 默认参数
DATA_DIR="../data/caltech-101"
OUTPUT_DIR="./experiments"
MAX_STEPS=500000
USE_CUDA=""

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 创建日志目录
mkdir -p "${LOG_DIR}"

# 函数：启动对比实验
start_comparison() {
    echo -e "${GREEN}Starting comparison experiment in background...${NC}"
    python3 "${PYTHON_SCRIPT}" start \
        --script train_with_tracking.py \
        --job_name "comparison_experiment" \
        --data_dir "${DATA_DIR}" \
        --output_dir "${OUTPUT_DIR}/comparison" \
        --max_steps "${MAX_STEPS}" \
        --use_cuda ${USE_CUDA}
}

# 函数：启动超参数敏感性实验
start_hyperparameter() {
    local experiment_type="${1:-all}"
    echo -e "${GREEN}Starting hyperparameter sensitivity experiment (${experiment_type}) in background...${NC}"
    
    # 这里需要调用hyperparameter_sensitivity.py，但我们需要修改它以支持后台运行
    # 暂时使用直接调用Python脚本的方式
    nohup python3 "${SCRIPT_DIR}/experiments/hyperparameter_sensitivity.py" \
        --data_dir "${DATA_DIR}" \
        --output_dir "${OUTPUT_DIR}/hyperparameter_sensitivity" \
        --experiment "${experiment_type}" \
        ${USE_CUDA} \
        > "${LOG_DIR}/hyperparameter_${experiment_type}.log" 2>&1 &
    
    echo "Hyperparameter experiment started with PID: $!"
    echo "Log file: ${LOG_DIR}/hyperparameter_${experiment_type}.log"
}

# 函数：查看状态
show_status() {
    python3 "${PYTHON_SCRIPT}" status --log_dir "${LOG_DIR}"
}

# 函数：查看日志
show_log() {
    local job_name="${1:-comparison_experiment}"
    python3 "${PYTHON_SCRIPT}" tail "${job_name}" --log_dir "${LOG_DIR}" --lines 100
}

# 函数：停止任务
stop_job() {
    local job_name="${1}"
    if [ -z "${job_name}" ]; then
        echo -e "${RED}Error: Job name required${NC}"
        echo "Usage: $0 stop <job_name>"
        exit 1
    fi
    python3 "${PYTHON_SCRIPT}" stop "${job_name}" --log_dir "${LOG_DIR}"
}

# 函数：停止所有任务
stop_all() {
    python3 "${PYTHON_SCRIPT}" stop_all --log_dir "${LOG_DIR}"
}

# 函数：实时监控日志
monitor_log() {
    local job_name="${1:-comparison_experiment}"
    local log_file="${LOG_DIR}/${job_name}.log"
    
    if [ ! -f "${log_file}" ]; then
        echo -e "${RED}Error: Log file not found: ${log_file}${NC}"
        exit 1
    fi
    
    echo -e "${GREEN}Monitoring log: ${log_file}${NC}"
    echo "Press Ctrl+C to stop"
    tail -f "${log_file}"
}

# 主函数
main() {
    case "${1}" in
        start_comparison)
            start_comparison
            ;;
        start_hyperparameter)
            start_hyperparameter "${2}"
            ;;
        status)
            show_status
            ;;
        log)
            show_log "${2}"
            ;;
        monitor)
            monitor_log "${2}"
            ;;
        stop)
            stop_job "${2}"
            ;;
        stop_all)
            stop_all
            ;;
        *)
            echo "Usage: $0 {start_comparison|start_hyperparameter|status|log|monitor|stop|stop_all} [options]"
            echo ""
            echo "Commands:"
            echo "  start_comparison              - Start comparison experiment"
            echo "  start_hyperparameter [type]   - Start hyperparameter experiment (lr|reuse_time|memory_size|all)"
            echo "  status                        - Show status of all jobs"
            echo "  log [job_name]                - Show last 100 lines of log"
            echo "  monitor [job_name]            - Monitor log in real-time"
            echo "  stop <job_name>               - Stop a specific job"
            echo "  stop_all                      - Stop all jobs"
            echo ""
            echo "Examples:"
            echo "  $0 start_comparison"
            echo "  $0 start_hyperparameter lr"
            echo "  $0 status"
            echo "  $0 monitor comparison_experiment"
            echo "  $0 stop comparison_experiment"
            exit 1
            ;;
    esac
}

# 检查是否提供了参数
if [ $# -eq 0 ]; then
    main
    exit 1
fi

main "$@"

