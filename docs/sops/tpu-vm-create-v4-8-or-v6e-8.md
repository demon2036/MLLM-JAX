# TPU VM 创建（v4-8 / v6e-8）SOP

- **标题**：SOP：在“已验证可用”的 zone 创建 TPU VM（优先 `v6e-8 spot`，不行则用 `v4-8 spot`）
  **前置条件**：已安装并登录 `gcloud`；已启用 TPU API；目标 zone 有可用配额/容量
  **步骤**：
  - 确认账号与 project：
    - `gcloud --version`
    - `gcloud auth list --format='table(account,status)'`
    - `gcloud config get-value project`
  - 选择支持 `v4-8`/`v6e-8` 的 zone（示例：`us-central2-b`）：
    - `ZONE=us-central2-b; gcloud compute tpus accelerator-types list --zone="$ZONE" --format='value(name)'`
    - 过滤查看（Linux）：`... | grep -E 'acceleratorTypes/(v4-8|v6e-8)$'`
    - 过滤查看（Windows/PowerShell）：`... | Select-String -Pattern 'acceleratorTypes/(v4-8|v6e-8)$'`
  - 优先尝试创建 `v6e-8 spot`（若 quota/capacity 不足可能失败）：
    - `scripts/create_tpu_vm.sh --type v6e-8 --zone us-central2-b`
  - 若 `v6e-8 spot` 失败，则创建 `v4-8 spot`：
    - `scripts/create_tpu_vm.sh --type v4-8 --zone us-central2-b`
  - 记录创建输出的 `TPU name` 并确认 `READY`：
    - `TPU_NAME=<TPU_NAME>; ZONE=us-central2-b`
    - `gcloud alpha compute tpus tpu-vm describe "$TPU_NAME" --zone="$ZONE" --format='value(state,acceleratorType)'`
  - 验证 root SSH 可用：
    - `scripts/ssh_tpu_vm_root.sh --name "$TPU_NAME" --zone "$ZONE" --command 'whoami; lsb_release -a || cat /etc/os-release; python3 --version || true'`
  - 完成后若需要停止计费可删除（本项目有时会“保留不删”；按需执行）：
    - `scripts/delete_tpu_vm.sh --name "$TPU_NAME" --zone "$ZONE"`
  **预期结果**：TPU VM 进入 `READY`，可 SSH；若选择删除则能成功清理实例
  **故障排查**：
  - `v6e-8 spot` 若报 quota `Limit: 0`（例如 `TPUV6EPreemptiblePerProjectPerZoneForTPUAPI`），需要申请 quota 或改用 `v4-8 spot`。
  - 若长时间卡在 `CREATING`，查看 `describe` 输出并换 zone 重试。
  **参考**：
  - `docs/sops/tpu-vm-lifecycle.md`
  - `docs/sops/tpu-vm-delete-all.md`

