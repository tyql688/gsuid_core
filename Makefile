.PHONY: run install uninstall start stop restart status logs enable disable sync help

SERVICE_NAME = gscore
SERVICE_FILE = $(SERVICE_NAME).service
SYSTEMD_PATH = /etc/systemd/system
INSTALL_DIR = $(shell pwd)
CURRENT_USER = $(shell whoami)
CURRENT_HOME = $(shell echo ~)
UV = $(shell which uv)

# 运行服务
run:
	@test -n "$(UV)" || (echo "错误: 未找到 uv，请先安装 uv"; exit 1)
	$(UV) run core

# 安装 systemd 服务
install:
	@test -n "$(UV)" || (echo "错误: 未找到 uv，请先安装 uv"; exit 1)
	@echo "正在安装 $(SERVICE_NAME) 服务..."
	@sed -e 's|%USER%|$(CURRENT_USER)|g' -e 's|%WORKDIR%|$(INSTALL_DIR)|g' -e 's|%HOME%|$(CURRENT_HOME)|g' -e 's|%UV%|$(UV)|g' $(SERVICE_FILE) > /tmp/$(SERVICE_FILE)
	sudo cp /tmp/$(SERVICE_FILE) $(SYSTEMD_PATH)/$(SERVICE_FILE)
	sudo systemctl daemon-reload
	@echo "服务已安装。"
	@echo "请运行 'make start' 启动服务。"
	@echo "运行 'make enable' 开启开机自启。"

# 卸载服务
uninstall:
	@echo "正在卸载 $(SERVICE_NAME) 服务..."
	sudo systemctl stop $(SERVICE_NAME) 2>/dev/null || true
	sudo systemctl disable $(SERVICE_NAME) 2>/dev/null || true
	sudo rm -f $(SYSTEMD_PATH)/$(SERVICE_FILE)
	sudo systemctl daemon-reload
	@echo "服务已卸载。"

# 启动服务
start:
	sudo systemctl start $(SERVICE_NAME)

# 停止服务
stop:
	sudo systemctl stop $(SERVICE_NAME)

# 重启服务
restart:
	sudo systemctl restart $(SERVICE_NAME)

# 查看状态
status:
	sudo systemctl status $(SERVICE_NAME)

# 查看日志
logs:
	sudo journalctl -u $(SERVICE_NAME) -f

# 开机自启
enable:
	sudo systemctl enable $(SERVICE_NAME)

# 取消开机自启
disable:
	sudo systemctl disable $(SERVICE_NAME)

# 同步依赖
sync:
	$(UV) sync

# 帮助
help:
	@echo "GsCore 服务管理"
	@echo ""
	@echo "用法: make [目标]"
	@echo ""
	@echo "常用:"
	@echo "  run          - 直接运行服务"
	@echo "  sync         - 同步依赖"
	@echo ""
	@echo "Systemd 管理:"
	@echo "  install      - 安装 systemd 服务"
	@echo "  uninstall    - 卸载 systemd 服务"
	@echo "  start        - 启动服务"
	@echo "  stop         - 停止服务"
	@echo "  restart      - 重启服务"
	@echo "  status       - 查看状态"
	@echo "  logs         - 查看日志"
	@echo "  enable       - 开机自启"
	@echo "  disable      - 取消开机自启"
