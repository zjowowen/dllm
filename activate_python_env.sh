export ASCEND_HOME=/usr/local/Ascend/ascend-toolkit/latest
# Some Ascend env scripts reference ZSH_VERSION; define it to avoid failures in `set -u` shells.
export ZSH_VERSION="${ZSH_VERSION:-}"
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh
export HCCL_NPU_SOCKET_PORT_RANGE=auto 
export LD_LIBRARY_PATH=/usr/local/Ascend/driver/lib64:/usr/local/Ascend/driver/lib64/driver:/usr/local/Ascend/driver/lib64/common:${LD_LIBRARY_PATH:-}

# Keep Ascend's PYTHONPATH (tbe/te/etc). Append repo root for `import dllm`.
_ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="${_ROOT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

# NOTE:
# - Do NOT commit proxy credentials into git.
# - If you need a proxy for downloading (HF / pip / etc), export it in your shell before running:
#     export http_proxy="http://user:pass@host:port/"
#     export https_proxy="$http_proxy"
#     export HTTP_PROXY="$http_proxy"
#     export HTTPS_PROXY="$http_proxy"
