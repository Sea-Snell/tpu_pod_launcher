from typing import List, Tuple, Optional, Dict, Callable
import subprocess
import time
import threading
import textwrap
import shlex
import tyro
import os
import json

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
        strict_host_key_checking: bool=False,
        known_hosts_file: Optional[str]='/dev/null',
    ):
        self.tpu_project = tpu_project
        self.tpu_zone = tpu_zone
        self.user = user
        self.key_path = key_path
        self.strict_host_key_checking = strict_host_key_checking
        self.known_hosts_file = known_hosts_file
    
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
        strict_host_key_checking_str = "-o StrictHostKeyChecking=no" if not self.strict_host_key_checking else ""
        known_hosts_file_str = f"-o UserKnownHostsFile={self.known_hosts_file}" if self.known_hosts_file is not None else ""
        keypath_str = f'-i {self.key_path}' if self.key_path is not None else ''
        ssh_command = f"\"ssh {strict_host_key_checking_str} {known_hosts_file_str} {keypath_str}\""
        if ssh_command.strip().strip('\"') == 'ssh':
            ssh_command = ''
        commands = [f"rsync -avPI -e {ssh_command} {excludes_flags} \"{local_path}\" {host}:\"{remote_path}\"" for host in hosts]
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
        strict_host_key_checking_str = "-o StrictHostKeyChecking=no" if not self.strict_host_key_checking else ""
        known_hosts_file_str = f"-o UserKnownHostsFile={self.known_hosts_file}" if self.known_hosts_file is not None else ""
        commands = [f"scp {strict_host_key_checking_str} {known_hosts_file_str} {keypath_str} {recursive_str} \"{local_path}\" {host}:\"{remote_path}\"" for host in hosts]
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
        strict_host_key_checking_str = "-o StrictHostKeyChecking=no" if not self.strict_host_key_checking else ""
        known_hosts_file_str = f"-o UserKnownHostsFile={self.known_hosts_file}" if self.known_hosts_file is not None else ""
        commands = [f"ssh {strict_host_key_checking_str} {known_hosts_file_str} {keypath_str} {host} {shlex.quote(command)}" for host in hosts]
        results = run_commands_parallel(commands, **kwargs)
        return {host: result for host, result in zip(hosts, results)}
    
    def __str__(self) -> str:
        return textwrap.dedent(f"""\
                TPUPodClient(
                    tpu_project={self.tpu_project},
                    tpu_zone={self.tpu_zone},
                    user={self.user},
                    key_path={self.key_path},
                    strict_host_key_checking={self.strict_host_key_checking},
                    known_hosts_file={self.known_hosts_file},
                )""")

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
        inner_command = 'cd '+self.working_dir+'\n'+command
        command = textwrap.dedent(f"""\
        tmux new -d -s {window_name}
        tmux send {shlex.quote(inner_command)} C-m
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
    
    def __str__(self) -> str:
        return textwrap.dedent(f"""\
                TPUPodProject(
                    client={textwrap.indent(str(self.client), ' '*4*5).strip()},
                    tpu_name={self.tpu_name},
                    copy_dirs={self.copy_dirs},
                    working_dir={self.working_dir},
                    copy_excludes={self.copy_excludes},
                    kill_commands={self.kill_commands},
                )""")

def create_cli(
    projects: Dict[str, TPUPodProject],
    setup: Callable[[TPUPodProject], None],
    custom_commands: Dict[str, Callable[[TPUPodProject], None]],
    launch_config_path: Optional[str]=None,
):
    if (launch_config_path is not None) and os.path.exists(launch_config_path):
        with open(launch_config_path, 'r') as f:
            config = json.load(f)
    else:
        config = dict()
    project_name = config.get('project_name', None)

    def set_project(name: str):
        if launch_config_path is None:
            raise ValueError("Cannot set project name without launch_config_path set.")
        config['project_name'] = name
        with open(launch_config_path, 'w') as f:
            json.dump(config, f)
        print(f"Project set to: {name}")
    
    def list_projects():
        for name in projects:
            print(f"\033[92m{name}\033[0m={projects[name]}")
            print()
    
    no_project_commands = dict(
        set_project=set_project,
        list_projects=list_projects,
    )

    def cli_main(
        settings: List[str],
        /,
        project: Optional[str]=project_name,
        verbose: bool=True,
    ):
        project_name: Optional[str] = project
        project: Optional[TPUPodProject] = projects.get(project_name, None)
        if project is None:
            mode, *settings = settings
            if mode in no_project_commands:
                no_project_commands[mode](*settings)
                return
            else:
                raise ValueError(f"Unknown project: {project_name}")
        print('Using project:', project_name)
        
        def check_forever():
            while True:
                project.check()
                time.sleep(1)
        
        def launch(load_script: str, strip_comments_key: str='#'):
            project.stop()
            if strip_comments_key != '':
                script_lines = []
                with open(load_script, 'r') as f:
                    for line in f:
                        if not line.strip().startswith(strip_comments_key): # remove comments
                            script_lines.append(line)
                script = ''.join(script_lines).strip()
            else:
                with open(load_script, 'r') as f:
                    script = f.read()
            project.copy_launch(script, verbose=verbose)
        
        def custom_command_wrapper(f):
            def wrapper(*args, **kwargs):
                f(project, *args, verbose=verbose, **kwargs)
            return wrapper
        
        commands = dict(
            check=project.check,
            stop=lambda: project.stop(verbose=verbose),
            check_forever=check_forever,
            launch=launch,
            ssh=lambda command: project.ssh(command, verbose=verbose),
            scp=lambda l, r: project.scp(l, r, recursive=True, verbose=verbose),
            copy=lambda: project.copy(verbose=verbose),
            setup=custom_command_wrapper(setup),
            **no_project_commands,
        )
        commands.update({k: custom_command_wrapper(v) for k, v in custom_commands.items()})

        mode, *settings = settings

        if mode in commands:
            commands[mode](*settings)
        else:
            raise ValueError(f"Unknown mode: {mode}")
    
    tyro.cli(cli_main)
