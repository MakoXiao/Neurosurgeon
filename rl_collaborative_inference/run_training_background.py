"""
后台训练启动脚本
支持长时间训练的后台运行，日志记录，进程管理
"""
import os
import sys
import argparse
import subprocess
import json
import signal
import time
from datetime import datetime
from pathlib import Path


class BackgroundTrainingManager:
    """后台训练管理器"""
    
    def __init__(self, log_dir='./logs'):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self.pid_file = os.path.join(log_dir, 'training_pids.json')
        self.load_pids()
    
    def load_pids(self):
        """加载进程ID记录"""
        if os.path.exists(self.pid_file):
            with open(self.pid_file, 'r') as f:
                self.pids = json.load(f)
        else:
            self.pids = {}
    
    def save_pids(self):
        """保存进程ID记录"""
        with open(self.pid_file, 'w') as f:
            json.dump(self.pids, f, indent=2)
    
    def start_training(self, script_path, args_dict, job_name=None):
        """
        启动后台训练
        :param script_path: 训练脚本路径
        :param args_dict: 参数字典
        :param job_name: 任务名称
        """
        if job_name is None:
            job_name = f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # 创建日志文件
        log_file = os.path.join(self.log_dir, f"{job_name}.log")
        err_file = os.path.join(self.log_dir, f"{job_name}.err")
        
        # 构建命令
        cmd = [sys.executable, script_path]
        for key, value in args_dict.items():
            if value is not None:
                if isinstance(value, bool) and value:
                    cmd.append(f'--{key}')
                elif not isinstance(value, bool):
                    cmd.append(f'--{key}')
                    cmd.append(str(value))
        
        # 使用nohup启动后台进程
        print(f"Starting background training: {job_name}")
        print(f"Command: {' '.join(cmd)}")
        print(f"Log file: {log_file}")
        print(f"Error file: {err_file}")
        
        with open(log_file, 'w') as log, open(err_file, 'w') as err:
            process = subprocess.Popen(
                cmd,
                stdout=log,
                stderr=err,
                preexec_fn=os.setsid,  # 创建新的进程组
                cwd=os.path.dirname(script_path) or '.'
            )
        
        # 记录进程信息
        self.pids[job_name] = {
            'pid': process.pid,
            'pgid': os.getpgid(process.pid),
            'start_time': datetime.now().isoformat(),
            'log_file': log_file,
            'err_file': err_file,
            'command': ' '.join(cmd),
            'status': 'running'
        }
        self.save_pids()
        
        print(f"Training started with PID: {process.pid}")
        print(f"Job name: {job_name}")
        print(f"\nTo view logs: tail -f {log_file}")
        print(f"To check status: python {__file__} --status")
        print(f"To stop: python {__file__} --stop {job_name}")
        
        return process.pid, job_name
    
    def stop_training(self, job_name):
        """停止指定的训练任务"""
        if job_name not in self.pids:
            print(f"Error: Job '{job_name}' not found")
            return False
        
        pid_info = self.pids[job_name]
        pid = pid_info['pid']
        pgid = pid_info.get('pgid', pid)
        
        try:
            # 尝试优雅地终止进程组
            os.killpg(pgid, signal.SIGTERM)
            print(f"Sent SIGTERM to process group {pgid} (job: {job_name})")
            
            # 等待进程结束
            time.sleep(5)
            
            # 检查进程是否还在运行
            try:
                os.killpg(pgid, 0)  # 检查进程是否存在
                # 如果还在运行，强制终止
                os.killpg(pgid, signal.SIGKILL)
                print(f"Process still running, sent SIGKILL")
            except ProcessLookupError:
                print(f"Process terminated successfully")
            
            pid_info['status'] = 'stopped'
            pid_info['stop_time'] = datetime.now().isoformat()
            self.save_pids()
            return True
            
        except ProcessLookupError:
            print(f"Process {pid} not found (may have already terminated)")
            pid_info['status'] = 'stopped'
            self.save_pids()
            return True
        except Exception as e:
            print(f"Error stopping process: {e}")
            return False
    
    def stop_all(self):
        """停止所有训练任务"""
        print("Stopping all training jobs...")
        for job_name in list(self.pids.keys()):
            if self.pids[job_name]['status'] == 'running':
                self.stop_training(job_name)
    
    def get_status(self, job_name=None):
        """获取训练状态"""
        if job_name:
            if job_name not in self.pids:
                print(f"Job '{job_name}' not found")
                return
            
            pid_info = self.pids[job_name]
            pid = pid_info['pid']
            
            # 检查进程是否还在运行
            try:
                os.kill(pid, 0)
                status = 'running'
            except ProcessLookupError:
                status = 'stopped'
            
            pid_info['status'] = status
            self.save_pids()
            
            print(f"\nJob: {job_name}")
            print(f"  PID: {pid}")
            print(f"  Status: {status}")
            print(f"  Start time: {pid_info.get('start_time', 'N/A')}")
            print(f"  Log file: {pid_info.get('log_file', 'N/A')}")
            print(f"  Error file: {pid_info.get('err_file', 'N/A')}")
            print(f"  Command: {pid_info.get('command', 'N/A')}")
            
            if status == 'running':
                # 显示最新的日志
                log_file = pid_info.get('log_file')
                if log_file and os.path.exists(log_file):
                    print(f"\nLast 10 lines of log:")
                    try:
                        with open(log_file, 'r') as f:
                            lines = f.readlines()
                            for line in lines[-10:]:
                                print(f"  {line.rstrip()}")
                    except Exception as e:
                        print(f"  Error reading log: {e}")
        else:
            # 显示所有任务状态
            print("\nAll Training Jobs:")
            print("=" * 80)
            for job_name, pid_info in self.pids.items():
                pid = pid_info['pid']
                try:
                    os.kill(pid, 0)
                    status = 'running'
                except ProcessLookupError:
                    status = 'stopped'
                
                pid_info['status'] = status
                print(f"\nJob: {job_name}")
                print(f"  PID: {pid}")
                print(f"  Status: {status}")
                print(f"  Start time: {pid_info.get('start_time', 'N/A')}")
                print(f"  Log file: {pid_info.get('log_file', 'N/A')}")
            
            self.save_pids()
    
    def tail_log(self, job_name, lines=50):
        """查看日志文件的最后几行"""
        if job_name not in self.pids:
            print(f"Job '{job_name}' not found")
            return
        
        log_file = self.pids[job_name].get('log_file')
        if not log_file or not os.path.exists(log_file):
            print(f"Log file not found: {log_file}")
            return
        
        print(f"\nLast {lines} lines of {log_file}:")
        print("=" * 80)
        try:
            with open(log_file, 'r') as f:
                file_lines = f.readlines()
                for line in file_lines[-lines:]:
                    print(line.rstrip())
        except Exception as e:
            print(f"Error reading log: {e}")


def main():
    parser = argparse.ArgumentParser(description='Background training manager')
    subparsers = parser.add_subparsers(dest='command', help='Command to execute')
    
    # Start command
    start_parser = subparsers.add_parser('start', help='Start background training')
    start_parser.add_argument('--script', type=str, required=True,
                             help='Training script path (e.g., train_with_tracking.py)')
    start_parser.add_argument('--job_name', type=str, default=None,
                             help='Job name (default: auto-generated)')
    start_parser.add_argument('--data_dir', type=str, required=True,
                             help='Path to dataset')
    start_parser.add_argument('--output_dir', type=str, default='./results',
                             help='Output directory')
    start_parser.add_argument('--max_steps', type=int, default=500000,
                             help='Maximum training steps')
    start_parser.add_argument('--lr_actor', type=float, default=0.0001,
                             help='Actor learning rate')
    start_parser.add_argument('--lr_critic', type=float, default=0.0001,
                             help='Critic learning rate')
    start_parser.add_argument('--k_epochs', type=int, default=10,
                             help='Number of update epochs')
    start_parser.add_argument('--batch_size', type=int, default=64,
                             help='Batch size')
    start_parser.add_argument('--network_bandwidth', type=float, default=10.0,
                             help='Network bandwidth (MB/s)')
    start_parser.add_argument('--seed', type=int, default=None,
                             help='Random seed')
    start_parser.add_argument('--use_cuda', action='store_true',
                             help='Use CUDA if available')
    start_parser.add_argument('--log_dir', type=str, default='./logs',
                             help='Log directory')
    
    # Stop command
    stop_parser = subparsers.add_parser('stop', help='Stop training job')
    stop_parser.add_argument('job_name', type=str, help='Job name to stop')
    stop_parser.add_argument('--log_dir', type=str, default='./logs',
                             help='Log directory')
    
    # Status command
    status_parser = subparsers.add_parser('status', help='Check training status')
    status_parser.add_argument('job_name', type=str, nargs='?', default=None,
                              help='Job name (optional, shows all if not specified)')
    status_parser.add_argument('--log_dir', type=str, default='./logs',
                              help='Log directory')
    
    # Tail command
    tail_parser = subparsers.add_parser('tail', help='View log file')
    tail_parser.add_argument('job_name', type=str, help='Job name')
    tail_parser.add_argument('--lines', type=int, default=50,
                            help='Number of lines to show')
    tail_parser.add_argument('--log_dir', type=str, default='./logs',
                            help='Log directory')
    
    # Stop all command
    stop_all_parser = subparsers.add_parser('stop_all', help='Stop all training jobs')
    stop_all_parser.add_argument('--log_dir', type=str, default='./logs',
                                help='Log directory')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    manager = BackgroundTrainingManager(log_dir=args.log_dir)
    
    if args.command == 'start':
        # 构建参数字典
        args_dict = {
            'data_dir': args.data_dir,
            'output_dir': args.output_dir,
            'max_steps': args.max_steps,
            'lr_actor': args.lr_actor,
            'lr_critic': args.lr_critic,
            'k_epochs': args.k_epochs,
            'batch_size': args.batch_size,
            'network_bandwidth': args.network_bandwidth,
            'seed': args.seed,
            'use_cuda': args.use_cuda
        }
        
        script_path = args.script
        if not os.path.isabs(script_path):
            script_path = os.path.join(os.path.dirname(__file__), script_path)
        
        manager.start_training(script_path, args_dict, args.job_name)
    
    elif args.command == 'stop':
        manager.stop_training(args.job_name)
    
    elif args.command == 'status':
        manager.get_status(args.job_name)
    
    elif args.command == 'tail':
        manager.tail_log(args.job_name, args.lines)
    
    elif args.command == 'stop_all':
        manager.stop_all()


if __name__ == "__main__":
    main()

