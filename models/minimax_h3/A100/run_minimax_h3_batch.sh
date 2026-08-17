#!/usr/bin/env bash
# Batch counterpart of run_minimax_h3_gpu.sh: generates every prompt in
# H3_PROMPTS_FILE instead of the one prompt a benchmark run needs. The model is
# loaded once per GPU group, and H3_GPU_GROUPS decides how many groups run at
# once. Everything else -- container, weights, profile, sampling settings -- is
# identical to the single-prompt path.
set -euo pipefail

: "${OUT_DIR:?OUT_DIR must be set by scripts/launch_config.py}"

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)
RUNTIME_ROOT="${SCRIPT_DIR}"
REPO_ROOT=$(cd "${RUNTIME_ROOT}/../../.." && pwd -P)

export H3_CONTAINER_RUNTIME=${H3_CONTAINER_RUNTIME:-none}
requested_runtime=${H3_CONTAINER_RUNTIME}
export H3_CONTAINER_IMAGE=${H3_CONTAINER_IMAGE:-docker://lmsysorg/sglang:nightly-dev-cu13-20260803-12eadf86}
export H3_MODEL_PATH=${H3_MODEL_PATH:-MiniMaxAI/MiniMax-H3}
export H3_MODEL_REVISION=${H3_MODEL_REVISION:-bfc8ed0353f5a9733be73e6b2c98ec0948195b86}
export H3_PROMPTS_FILE=${H3_PROMPTS_FILE:-${REPO_ROOT}/models/minimax_h3/demo_prompts.json}
export H3_GPU_GROUPS=${H3_GPU_GROUPS:-[0,1,2,3]}
export H3_GPUS_PER_GROUP=${H3_GPUS_PER_GROUP:-4}
export H3_BATCH_WARMUP=${H3_BATCH_WARMUP:-0}
export H3_WARMUP_NUM_STEPS=${H3_WARMUP_NUM_STEPS:-50}
export H3_MEASURED_NUM_STEPS=${H3_MEASURED_NUM_STEPS:-50}
export H3_DURATION_SECONDS=${H3_DURATION_SECONDS:-5.166667}
export H3_SEED=${H3_SEED:-0}
export H3_WARMUP_SEED=${H3_WARMUP_SEED:-10000}
export H3_MASTER_PORT=${H3_MASTER_PORT:-30005}
export H3_MODEL_SUBFOLDER=${H3_MODEL_SUBFOLDER:-}
export H3_EXPECTED_TORCH=${H3_EXPECTED_TORCH:-2.11.0+cu130}
export H3_EXPECTED_TRITON=${H3_EXPECTED_TRITON:-3.6.0}
export H3_STORAGE_ROOT=${H3_STORAGE_ROOT:-${REPO_ROOT}}
export H3_CACHE_ROOT=${H3_CACHE_ROOT:-${H3_STORAGE_ROOT}/cache}
export H3_SGLANG_PYTHON_ROOT=${H3_SGLANG_PYTHON_ROOT:-/sgl-workspace/sglang/python}
export H3_PYTHON_BIN=${H3_PYTHON_BIN:-python3}
export OUT_DIR

if [[ "${H3_PROMPTS_FILE}" != /* ]]; then
  export H3_PROMPTS_FILE=${REPO_ROOT}/${H3_PROMPTS_FILE}
fi

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export HF_HOME=${HF_HOME:-${H3_CACHE_ROOT}/huggingface}
export HUGGINGFACE_HUB_CACHE=${HUGGINGFACE_HUB_CACHE:-${HF_HOME}/hub}
export HF_HUB_DISABLE_XET=${HF_HUB_DISABLE_XET:-1}
export HF_HUB_DOWNLOAD_TIMEOUT=${HF_HUB_DOWNLOAD_TIMEOUT:-600}
export TRITON_CACHE_DIR=${TRITON_CACHE_DIR:-${H3_CACHE_ROOT}/triton}
export TORCH_HOME=${TORCH_HOME:-${H3_CACHE_ROOT}/torch}
export XDG_CACHE_HOME=${XDG_CACHE_HOME:-${H3_CACHE_ROOT}/xdg}
export TMPDIR=${TMPDIR:-/tmp}
export PYTHONPATH=${REPO_ROOT}:${REPO_ROOT}/techniques/sparse_backends:${H3_SGLANG_PYTHON_ROOT}${PYTHONPATH:+:${PYTHONPATH}}

mkdir -p "${OUT_DIR}" "${HF_HOME}" "${TRITON_CACHE_DIR}" "${TORCH_HOME}" "${XDG_CACHE_HOME}" "${TMPDIR}"

if [[ "${H3_CONTAINER_RUNTIME}" == none ]]; then
  "${H3_PYTHON_BIN}" "${RUNTIME_ROOT}/batch_infer.py" 2>&1 | tee "${OUT_DIR}/run.log"
  exit "${PIPESTATUS[0]}"
fi

H3_STORAGE_ROOT=$(cd "${H3_STORAGE_ROOT}" && pwd -P)
host_storage_root=${H3_STORAGE_ROOT}
OUT_DIR=$(cd "${OUT_DIR}" && pwd -P)
H3_CACHE_ROOT=$(cd "${H3_CACHE_ROOT}" && pwd -P)
host_cache_root=${H3_CACHE_ROOT}
prompts_dir=$(cd "$(dirname "${H3_PROMPTS_FILE}")" && pwd -P)
H3_PROMPTS_FILE=${prompts_dir}/$(basename "${H3_PROMPTS_FILE}")

require_below_storage() {
  local label=$1
  local path=$2
  case "${path}" in
    "${H3_STORAGE_ROOT}"|"${H3_STORAGE_ROOT}"/*) ;;
    *)
      echo "${label} must be below H3_STORAGE_ROOT for container runs: ${path}" >&2
      exit 2
      ;;
  esac
}

to_inside_path() {
  local path=$1
  if [[ "${path}" == "${H3_STORAGE_ROOT}" ]]; then
    printf '/h3'
  else
    printf '/h3/%s' "${path#${H3_STORAGE_ROOT}/}"
  fi
}

require_below_storage REPO_ROOT "${REPO_ROOT}"
require_below_storage OUT_DIR "${OUT_DIR}"
require_below_storage H3_CACHE_ROOT "${H3_CACHE_ROOT}"
require_below_storage H3_PROMPTS_FILE "${H3_PROMPTS_FILE}"
inside_repo=$(to_inside_path "${REPO_ROOT}")
inside_output=$(to_inside_path "${OUT_DIR}")
inside_cache=$(to_inside_path "${H3_CACHE_ROOT}")
inside_prompts=$(to_inside_path "${H3_PROMPTS_FILE}")
inside_model=${H3_MODEL_PATH}
if [[ "${H3_MODEL_PATH}" == /* ]]; then
  model_dir=$(cd "${H3_MODEL_PATH}" && pwd -P)
  require_below_storage H3_MODEL_PATH "${model_dir}"
  inside_model=$(to_inside_path "${model_dir}")
fi

export H3_CONTAINER_RUNTIME=none
export H3_STORAGE_ROOT=/h3
export H3_CACHE_ROOT=${inside_cache}
export H3_PROMPTS_FILE=${inside_prompts}
export H3_MODEL_PATH=${inside_model}
export OUT_DIR=${inside_output}
export PYTHONPATH=${inside_repo}:${inside_repo}/techniques/sparse_backends:${H3_SGLANG_PYTHON_ROOT}
export HF_HOME=${inside_cache}/huggingface
export HUGGINGFACE_HUB_CACHE=${HF_HOME}/hub
export TRITON_CACHE_DIR=${inside_cache}/triton
export TORCH_HOME=${inside_cache}/torch
export XDG_CACHE_HOME=${inside_cache}/xdg
export TMPDIR=/tmp

# The variables the run needs on the other side of the container boundary.
# pyxis takes the names, docker takes name=value pairs built from them.
container_env=OUT_DIR,PYTHONPATH,H3_CONTAINER_RUNTIME,H3_STORAGE_ROOT,H3_CACHE_ROOT,H3_SGLANG_PYTHON_ROOT,H3_PYTHON_BIN,H3_MODEL_PATH,H3_MODEL_REVISION,H3_MODEL_SUBFOLDER,H3_PROMPTS_FILE,H3_GPU_GROUPS,H3_GPUS_PER_GROUP,H3_BATCH_WARMUP,H3_FIRST_FRAME_TASK,H3_IMAGE_CONDITION_JSON,H3_WARMUP_NUM_STEPS,H3_MEASURED_NUM_STEPS,H3_DURATION_SECONDS,H3_SEED,H3_WARMUP_SEED,H3_MASTER_PORT,H3_SOL_PROFILE,H3_EXPECTED_TORCH,H3_EXPECTED_TRITON,HF_HOME,HUGGINGFACE_HUB_CACHE,HF_HUB_DISABLE_XET,HF_HUB_DOWNLOAD_TIMEOUT,HF_HUB_OFFLINE,TRITON_CACHE_DIR,TORCH_HOME,XDG_CACHE_HOME,TMPDIR,OMP_NUM_THREADS,OPENBLAS_NUM_THREADS,MKL_NUM_THREADS,NUMEXPR_NUM_THREADS,TOKENIZERS_PARALLELISM,PYTHONUNBUFFERED

case "${requested_runtime}" in
  pyxis)
    exec srun \
      --ntasks=1 \
      --nodes=1 \
      --container-image="${H3_CONTAINER_IMAGE}" \
      --container-mounts="${host_storage_root}:/h3" \
      --container-env="${container_env}" \
      --no-container-mount-home \
      --container-workdir="${inside_output}" \
      --no-container-entrypoint \
      bash "${inside_repo}/models/minimax_h3/A100/run_minimax_h3_batch.sh"
    ;;
  docker)
    # Brev-style bare node: Docker and the NVIDIA runtime are already there, and
    # the pinned image is a Docker image to begin with. All eight devices are
    # exposed -- batch_infer pins each group with CUDA_VISIBLE_DEVICES itself.
    docker_env=()
    IFS=',' read -r -a env_names <<< "${container_env}"
    for name in "${env_names[@]}"; do
      if [[ -n ${!name+x} ]]; then
        docker_env+=(--env "${name}=${!name}")
      fi
    done
    # Running as the host uid leaves it absent from the image's /etc/passwd, and
    # torch resolves its inductor cache through getpass.getuser(), which falls
    # back to pwd.getpwuid() and raises. getuser() reads these first, so setting
    # them keeps the lookup from ever happening; HOME does the same for anything
    # that expands "~".
    mkdir -p "${host_cache_root}/home"
    docker_env+=(--env "USER=$(id -un)")
    docker_env+=(--env "LOGNAME=$(id -un)")
    docker_env+=(--env "HOME=${inside_cache}/home")
    exec docker run --rm \
      --gpus all \
      --ipc=host \
      --ulimit memlock=-1 \
      --ulimit stack=67108864 \
      --user "${H3_DOCKER_USER:-$(id -u):$(id -g)}" \
      --volume "${host_storage_root}:/h3" \
      --workdir "${inside_output}" \
      "${docker_env[@]}" \
      "${H3_CONTAINER_IMAGE#docker://}" \
      bash "${inside_repo}/models/minimax_h3/A100/run_minimax_h3_batch.sh"
    ;;
  apptainer|singularity)
    if ! command -v "${requested_runtime}" >/dev/null 2>&1; then
      for init_script in /etc/profile.d/modules.sh /usr/share/Modules/init/bash; do
        if [[ -f "${init_script}" ]]; then
          # shellcheck disable=SC1090
          source "${init_script}"
          break
        fi
      done
      module load "${H3_CONTAINER_MODULE:-singularity/4.4.1}"
    fi
    exec "${requested_runtime}" exec \
      --nv \
      --bind "${host_storage_root}:/h3" \
      "${H3_CONTAINER_IMAGE}" \
      bash "${inside_repo}/models/minimax_h3/A100/run_minimax_h3_batch.sh"
    ;;
  *)
    echo "H3_CONTAINER_RUNTIME must be none, pyxis, docker, apptainer, or singularity" >&2
    exit 2
    ;;
esac
