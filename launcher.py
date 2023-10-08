import os
from typing import List, Tuple, Optional, Dict
import subprocess
import time
import threading
import textwrap

def run_command(
    command: str,
    shell: bool=True,
    verbose=False,
    **kwargs,
) -> str:
    if verbose:
        print(command)
    p = subprocess.Popen(command, shell=shell, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, **kwargs)
    output, err = p.communicate()
    if err is not None:
        raise Exception(err.decode('utf-8'))
    result = output.decode('utf-8')
    if verbose:
        print(result)
    return result

def run_commands_parallel(
        commands: List[str],
        shell: bool=True,
        verbose=False,
        **kwargs,
) -> List[str]:
    results = [None for _ in commands]
    def _run_command(command: str, index: int) -> None:
        results[index] = run_command(
            command,
            shell=shell,
            verbose=verbose,
            **kwargs,
        )
    threads = []
    for i, command in enumerate(commands):
        thread = threading.Thread(
            target=_run_command,
            args=(command, i),
        )
        thread.start()
        threads.append(thread)
    for thread in threads:
        thread.join()
    return results

class TPUPodClient:
    def __init__(
        self,
        tpu_project: str,
        tpu_zone: str,
        user: Optional[str]=None,
        key_path: Optional[str]=None,
    ):
        self.tpu_project = tpu_project
        self.tpu_zone = tpu_zone
        self.user = user
        self.key_path = key_path
    
    def list(self, **kwargs) -> str:
        command = f"gcloud alpha compute tpus tpu-vm list --zone {self.tpu_zone} --project {self.tpu_project}"
        return run_command(command,**kwargs)
    
    def describe(self, tpu_name: str, **kwargs) -> str:
        command = f"gcloud alpha compute tpus tpu-vm describe \"{tpu_name}\" --zone {self.tpu_zone} --project {self.tpu_project}"
        return run_command(command, **kwargs)
    
    def list_ips(
        self,
        tpu_name: str,
        add_user: bool=True,
        **kwargs,
    ) -> List[str]:
        command = f"gcloud alpha compute tpus tpu-vm describe \"{tpu_name}\" --zone {self.tpu_zone} --project {self.tpu_project} | grep -oP 'externalIp: \K(.+)$'"
        ips = run_command(command, **kwargs).strip().split('\n')
        if add_user and self.user is not None:
            ips = [f'{self.user}@{ip}' for ip in ips]
        return ips

    def delete(self, tpu_name: str, **kwargs) -> str:
        command = f"gcloud alpha compute tpus tpu-vm delete \"{tpu_name}\" --zone {self.tpu_zone} --project {self.tpu_project} --quiet"
        return run_command(command, **kwargs)
    
    def maintain(self, tpu_name: str, **kwargs) -> str:
        command = f"gcloud alpha compute tpus tpu-vm simulate-maintenance-event \"{tpu_name}\" --project {self.tpu_project} --zone={self.tpu_zone} --workers=all"
        return run_command(command, **kwargs)

    def create(
        self,
        tpu_name: str,
        accelerator_type: str, # e.g. v3-32
        software_version: str='tpu-vm-base',
        **kwargs,
    ) -> str:
        command = f"gcloud alpha compute tpus tpu-vm create \"{tpu_name}\" --accelerator-type=\"{accelerator_type}\" --version=\"{software_version}\" --zone {self.tpu_zone} --project {self.tpu_project}"
        return run_command(command, **kwargs)
    
    def copy(
        self,
        tpu_name: str,
        local_path: str,
        remote_path: str,
        excludes: Optional[List[str]]=None,
        **kwargs,
    ) -> Dict[str, str]:
        if excludes is None:
            excludes = []
        hosts = self.list_ips(tpu_name)
        excludes_flags = ' '.join([f'--exclude={item}' for item in excludes])
        keypath_str = f'\"ssh -i {self.key_path}\"' if self.key_path is not None else ''
        commands = [f"rsync -avPI -e {keypath_str} {excludes_flags} \"{local_path}\" {host}:\"{remote_path}\"" for host in hosts]
        results = run_commands_parallel(commands, **kwargs)
        return {host: result for host, result in zip(hosts, results)}

    def scp(
        self,
        tpu_name: str,
        local_path: str,
        remote_path: str,
        recursive: bool=True,
        **kwargs,
    ) -> Dict[str, str]:
        hosts = self.list_ips(tpu_name)
        keypath_str = f"-i {self.key_path}" if self.key_path is not None else ""
        recursive_str = "-r" if recursive else ""
        commands = [f"scp {keypath_str} {recursive_str} \"{local_path}\" {host}:\"{remote_path}\"" for host in hosts]
        results = run_commands_parallel(commands, **kwargs)
        return {host: result for host, result in zip(hosts, results)}
    
    def ssh(
        self,
        tpu_name: str,
        command: str,
        **kwargs,
    ) -> Dict[str, str]:
        hosts = self.list_ips(tpu_name)
        keypath_str = f"-i {self.key_path}" if self.key_path is not None else ""
        commands = [f"ssh {keypath_str} {host} \"{command}\"" for host in hosts]
        results = run_commands_parallel(commands, **kwargs)
        return {host: result for host, result in zip(hosts, results)}

class TPUPodProject:
    def __init__(
        self,
        client: TPUPodClient,
        tpu_name: str,
        copy_dirs: List[Tuple[str, str]],
        working_dir: str,
        copy_excludes: Optional[List[str]]=None,
        kill_commands: Optional[List[str]]=None,
    ):
        self.client = client
        self.tpu_name = tpu_name
        self.copy_dirs = copy_dirs
        self.working_dir = working_dir
        self.copy_excludes = copy_excludes
        self.kill_commands = kill_commands
    
    def ssh(
        self,
        command: str,
        **kwargs,
    ) -> Dict[str, str]:
        command = f"cd {self.working_dir}\n{command}"
        return self.client.ssh(self.tpu_name, command, **kwargs)
    
    def scp(
        self,
        local_path: str,
        remote_path: str,
        recursive: bool=True,
        **kwargs,
    ) -> Dict[str, str]:
        return self.client.scp(self.tpu_name, local_path, remote_path, recursive=recursive, **kwargs)
    
    def copy(
        self,
        excludes: Optional[List[str]]=None,
        **kwargs,
    ) -> List[Dict[str, str]]:
        if excludes is None:
            excludes = self.copy_excludes
        results = []
        for local_path, remote_path in self.copy_dirs:
            result = self.client.copy(self.tpu_name, local_path, remote_path, excludes=excludes, **kwargs)
            results.append(result)
        return results
    
    def launch(
        self,
        command: str,
        window_name: str='launch',
        **kwargs,
    ) -> Dict[str, str]:
        command = textwrap.dedent(f"""\
        tmux new -d -s {window_name}
        tmux send \\\"cd {self.working_dir}\n{command}\\\" C-m
        """).strip()
        return self.ssh(command, **kwargs)
    
    def copy_launch(
        self,
        command: str,
        window_name: str='launch',
        copy_retries: int=2,
        **kwargs,
    ) -> Tuple[Dict[str, str], List[Dict[str, str]]]:
        copy_results = []
        for _ in range(copy_retries):
            copy_result = self.copy(**kwargs)
            copy_results.append(copy_result)
            time.sleep(1)
        return self.launch(command, window_name=window_name, **kwargs), copy_results
    
    def check(
        self,
        window_name: str='launch',
        silent: bool=False,
        **kwargs,
    ) -> Dict[str, str]:
        command = f"tmux capture-pane -pt {window_name}"
        results = self.ssh(command, **kwargs)
        if not silent:
            for host, result in results.items():
                print(f"============== Checking host: {host} ==============")
                print(result)
                print(f"============== End of host: {host} ==============")
        return results
    
    def stop(
        self,
        window_name: str='launch',
        kill_commands: Optional[List[str]]=None,
        **kwargs,
    ) -> Dict[str, str]:
        if kill_commands is None:
            kill_commands = self.kill_commands if self.kill_commands is not None else []
        kill_commands = [f"tmux kill-session -t {window_name}"] + kill_commands
        command = '; '.join(kill_commands)
        return self.ssh(command, **kwargs)
