
"""MiniPy Shell

A small interactive shell written in Python with:
- built in commands: cd, pwd, clear, export, unset, alias, unalias, env, set,
which, echo, history, help, exit, quit
- alias support
- environment variable persistence and expansion ($VAR, ${VAR}, $?)
- history file with history command
- tab completion for commands and paths
- colored prompt with user@host:directory format
- exit code tracking
 - cd - support (previous directory)"""

import os
import sys
import shlex
import subprocess
import shutil
import socket
import re
from pathlib import Path


try:
    import readline
except ImportError:
    readline = None


class Colors:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"

    FG_RED = "\033[31m"
    FG_GREEN = "\033[32m"
    FG_YELLOW = "\033[33m"
    FG_BLUE = "\033[34m"
    FG_MAGENTA = "\033[35m"
    FG_CYAN = "\033[36m"
    FG_WHITE = "\033[37m"


class MiniPyShell:
    def __init__(self) -> None:
        self.env = os.environ.copy()
        self.aliases: dict[str, str] = {}
        self.running = True
        self.last_exit_code = 0
        self.prev_dir: str | None = None

        self.history_path = Path.home() / ".minipy_history"
        

        self.command_agent = None


        self.builtins = {
            "cd": self._builtin_cd,
            "pwd": self._builtin_pwd,
            "clear": self._builtin_clear,
            "export": self._builtin_export,
            "alias": self._builtin_alias,
            "unalias": self._builtin_unalias,
            "env": self._builtin_env,
            "set": self._builtin_env,
            "which": self._builtin_which,
            "help": self._builtin_help,
            "exit": self._builtin_exit,
            "quit": self._builtin_exit,
            "history": self._builtin_history,
            "echo": self._builtin_echo,
            "unset": self._builtin_unset,
        }

        self._init_readline()



    def _init_readline(self) -> None:
        if readline is None:
            return


        try:
            readline.read_history_file(str(self.history_path))
        except FileNotFoundError:
            pass

        readline.set_history_length(1000)


        readline.set_completer(self._complete)
        readline.parse_and_bind("tab: complete")

    def _save_history(self) -> None:
        if readline is None:
            return
        try:
            readline.write_history_file(str(self.history_path))
        except OSError:
            pass

    def _complete(self, text: str, state: int) -> str | None:
        """Simple completer:
        - For the first word, completes builtins, aliases and executables
        - For later words, completes filesystem paths"""
        if readline is None:
            return None

        buffer = readline.get_line_buffer()
        begidx = readline.get_begidx()
        endidx = readline.get_endidx()


        try:
            tokens = shlex.split(buffer[:begidx])
        except ValueError:
            tokens = []


        if len(tokens) == 0:

            candidates = self._command_candidates(text)
        elif len(tokens) == 1 and begidx == 0:

            candidates = self._command_candidates(text)
        else:

            candidates = self._path_candidates(text)

        try:
            return candidates[state]
        except IndexError:
            return None

    def _command_candidates(self, prefix: str) -> list[str]:
        result: set[str] = set()


        for name in self.builtins:
            if name.startswith(prefix):
                result.add(name)


        for name in self.aliases:
            if name.startswith(prefix):
                result.add(name)


        path = self.env.get("PATH", os.defpath)
        for directory in path.split(os.pathsep):
            if not directory:
                continue
            try:
                for entry in os.listdir(directory):
                    if entry.startswith(prefix):
                        full = os.path.join(directory, entry)
                        if os.path.isfile(full) and os.access(full, os.X_OK):
                            result.add(entry)

            except OSError:
                continue

        return sorted(result)

    def _path_candidates(self, prefix: str) -> list[str]:

        if not prefix:
            base_dir = "."
            start = ""
            prefix_dir = ""
        else:

            prefix = os.path.expanduser(prefix)
            base_dir, start = os.path.split(prefix)
            if not base_dir:
                base_dir = "."
            prefix_dir = base_dir if base_dir != "." else ""

        candidates: list[str] = []
        try:
            for entry in os.listdir(base_dir):
                if entry.startswith(start):
                    full = os.path.join(base_dir, entry)
                    if os.path.isdir(full):
                        entry += "/"
                    if prefix_dir:
                        candidates.append(os.path.join(prefix_dir, entry))
                    else:
                        candidates.append(entry)
        except OSError:
            pass

        return sorted(candidates)



    def _prompt(self) -> str:
        user = self.env.get("USER", "user")
        try:
            host = socket.gethostname().split(".")[0]
        except (OSError, socket.error):
            host = "localhost"
        
        try:
            cwd = os.getcwd()
            home = str(Path.home())
            if cwd.startswith(home):
                cwd_display = "~" + cwd[len(home) :]
            else:
                cwd_display = cwd
        except OSError:
            cwd_display = "?"

        return (
            f"{Colors.FG_GREEN}{user}@{host}{Colors.RESET}"
            f":{Colors.FG_BLUE}{cwd_display}{Colors.RESET}$ "
        )

    def run(self) -> None:

        self._load_command_agent()
        
        print(f"{Colors.FG_CYAN}MiniPy Shell ready. Type 'Help' for help.{Colors.RESET}")
        print()

        while self.running:
            try:
                line = input(self._prompt())
            except EOFError:
                print()
                break
            except KeyboardInterrupt:
                print()
                continue

            line = line.strip()
            if not line:
                continue

            self._handle_line(line)

        self._save_history()
        print("Good Bye from the best shell in the world!")



    def _handle_line(self, line: str) -> None:

        line = self._translate_natural_language(line)
        

        line = self._expand_variables(line)
        

        line = self._expand_alias(line)


        if self._is_pure_builtin_invocation(line):
            try:
                args = shlex.split(line)
            except ValueError as exc:
                print(f"{Colors.FG_RED}Parse error: {exc}{Colors.RESET}")
                self.last_exit_code = 1
                return

            if not args:
                return

            cmd = args[0]
            handler = self.builtins.get(cmd)
            if handler is not None:
                try:
                    handler(args[1:])
                except Exception as exc:
                    print(f"{Colors.FG_RED}Error in builtin {cmd}: {exc}{Colors.RESET}")
                    self.last_exit_code = 1
                return


        self._run_external(line)

    def _expand_variables(self, line: str) -> str:
        """Expand shell variables like $? and $VAR"""

        line = line.replace("$?", str(self.last_exit_code))
        

        def replace_var(match):
            var_name = match.group(1) or match.group(2)
            if var_name:
                return self.env.get(var_name, "")
            return match.group(0)
        

        line = re.sub(r'(?<!\$)\$(\w+)|(?<!\$)\$\{(\w+)\}', replace_var, line)
        
        return line

    def _expand_alias(self, line: str) -> str:
        try:
            tokens = shlex.split(line, posix=True)
        except ValueError:
            return line

        if not tokens:
            return line

        first = tokens[0]
        if first in self.aliases:
            expansion = self.aliases[first]
            rest = line[line.find(first) + len(first) :].lstrip()
            if rest:
                return f"{expansion} {rest}"
            return expansion

        return line

    def _load_command_agent(self) -> None:
        """Load the command agent at startup.

         shows loading messages and handles errors gracefully."""
        print(f"{Colors.FG_CYAN}Loading command agent...{Colors.RESET}")
        try:
            from src.command_agent import CommandAgent
            self.command_agent = CommandAgent()
            print(f"{Colors.FG_GREEN}Command agent ready!{Colors.RESET}\n")
        except Exception as e:
            print(f"{Colors.FG_YELLOW}Warning: Could not load command agent: {e}{Colors.RESET}")
            print(f"{Colors.FG_YELLOW}Natural language translation disabled.{Colors.RESET}\n")
            self.command_agent = None
    
    def _translate_natural_language(self, line: str) -> str:
        """Detect if input is natural language and translate to command using command agent.
        Takes in:
        line: User input line
        Gives back:
        translated command or original line if not natural language"""

        if not line or not line.strip():
            return line
        

        if self.command_agent is None:
            return line
        
        if self._is_likely_command(line) and not self._is_natural_language(line):
            return line
        
        try:
            translated = self.command_agent.translate(line)
            
            if translated != line and not translated.startswith("echo 'Command not found"):
                print(f"{Colors.DIM}→ {translated}{Colors.RESET}")
            
            return translated
        except Exception as e:
            print(f"{Colors.FG_YELLOW}Warning: Translation error: {e}{Colors.RESET}")
            return line
    
    def _is_likely_command(self, line: str) -> bool:
        """Checks whether line is likely already a direct command.
        Takes in:
        line: Input line
        Gives back:
        true if it looks like a command"""
        line_lower = line.lower().strip()
        

        builtin_commands = list(self.builtins.keys())
        first_word = line_lower.split()[0] if line_lower.split() else ""
        if first_word in builtin_commands:
            return True
        


        command_patterns = [
            r'^[a-z]+\s+[-/]',
            r'^[a-z]+\s+/',
            r'^[a-z]+\s+\./',
            r'^[a-z]+\s+[a-z]+\.[a-z]+',
        ]
        
        import re
        if any(re.match(pattern, line_lower) for pattern in command_patterns):

            natural_markers = [' the ', ' a ', ' an ', ' this ', ' that ', ' i want ', ' please ', ' your ', ' my ', ' me ', ' you ']
            if not any(marker in line_lower for marker in natural_markers):
                return True
        

        if re.match(r'^[a-z]+\s+[a-z]+$', line_lower):

            natural_phrases = ['go back', 'go to', 'go home', 'go previous', 'show me', 'show current', 
                             'list all', 'list files', 'print hello', 'print text', 'navigate back',
                             'enter directory', 'change directory', 'copy files', 'copy all']
            if line_lower not in natural_phrases and 'go ' not in line_lower and 'print ' not in line_lower:
                natural_markers = [' the ', ' a ', ' an ', ' this ', ' that ', ' your ', ' my ', ' me ', ' current ', ' directory ']
                if not any(marker in line_lower for marker in natural_markers):
                    return True
        

        if any(op in line for op in ['|', '&', ';', '>', '<', '&&', '||']):
            return True
        
        return False
    
    def _is_natural_language(self, line: str) -> bool:
        """Checks whether line is natural language (not a direct command).
        Takes care of case-insensitive input and common phrases.
        
        Takes in:
            line: Input line
        Gives back:
        true if it looks like natural language"""

        line_lower = line.lower().strip()
        line_original = line.strip()
        

        natural_indicators = [
            'I want', 'i need', 'Please', 'Can you', 'Could you',
            'How do i', 'How to', 'Show me', 'Help me', 'List',
            'Find', 'Search', 'Copy', 'Move', 'Create', 'Delete',
            'The file', 'The directory', 'All files', 'This directory',
            'Show current', 'Show working', 'Current directory', 'Current folder', 'Current direct',
            'Working directory', 'Working folder', 'Enter', 'Change directory',
            'Go to', 'Navigate to', 'Switch to', 'Copy all', 'Copy files',
            'List all', 'List files', 'Without details', 'With details',
            'Go back', 'Go previous', 'Navigate back', 'Return to',
            'Print', 'Display', 'Output', 'Show', 'Echo',
            'After', 'Then', 'And', 'Before', 'While',
            'Show me current', 'List all files', 'Print all files'
        ]
        

        if any(line_lower.startswith(indicator) for indicator in natural_indicators):
            return True
        

        if any(indicator in line_lower for indicator in natural_indicators):

            if not self._is_likely_command(line):
                return True
        

        if line_original and line_original[0].isupper():
            words = line_original.split()

            first_word_lower = words[0].lower()
            known_commands = ['Cd', 'Ls', 'Pwd', 'Cat', 'Grep', 'Find', 'Cp', 'Mv', 'Rm', 'Mkdir', 'Rmdir', 'Echo', 'Print', 'Export', 'Unset']
            if first_word_lower not in known_commands or len(words) > 2:

                if any(word in line_lower for word in ['The', 'a', 'An', 'This', 'That', 'Your', 'My', 'Me', 'You']):
                    return True

                if first_word_lower in ['Go', 'Show', 'List', 'Print', 'Display', 'Output', 'Navigate', 'Enter', 'Change', 'Copy', 'Move']:
                    return True
        


        words = line.split()
        if len(words) >= 3:
            articles = ['The', 'a', 'An', 'This', 'That', 'These', 'Those', 'Your', 'My', 'Our']
            if any(article in line_lower for article in articles):
                return True

            if any(connector in line_lower for connector in ['After', 'Then', 'And', 'Before', 'While', 'When']):
                return True
        
        return False
    
    def _split_commands(self, line: str) -> list[str]:
        """Split line into multiple commands separated by semicolons.
        preserves quoted strings.
        Takes in:
        line: Input line potentially containing multiple commands
        Gives back:
        list of individual commands"""
        commands = []
        current = ""
        in_quotes = False
        quote_char = None
        
        i = 0
        while i < len(line):
            char = line[i]
            
            if char in ['"', "'"] and (i == 0 or line[i-1] != '\\'):
                if not in_quotes:
                    in_quotes = True
                    quote_char = char
                elif char == quote_char:
                    in_quotes = False
                    quote_char = None
                current += char
            elif char == ';' and not in_quotes:
                if current.strip():
                    commands.append(current.strip())
                current = ""
            else:
                current += char
            
            i += 1
        
        if current.strip():
            commands.append(current.strip())
        
        return commands if commands else [line]

    def _is_pure_builtin_invocation(self, line: str) -> bool:

        for ch in "|&;><":
            if ch in line:
                return False

        try:
            tokens = shlex.split(line, posix=True)
        except ValueError:
            return False

        if not tokens:
            return False

        return tokens[0] in self.builtins

    def _run_external(self, line: str) -> None:
        try:


            commands = self._split_commands(line)
            
            for cmd in commands:
                cmd = cmd.strip()
                if not cmd:
                    continue
                

                cmd_lower = cmd.lower().strip()
                if cmd_lower.startswith('cd '):
                    cd_args = cmd[3:].strip()
                    if cd_args:
                        try:
                            args = shlex.split(cd_args)
                            self._builtin_cd(args)
                            continue
                        except Exception as exc:
                            print(f"{Colors.FG_RED}cd error: {exc}{Colors.RESET}")
                            self.last_exit_code = 1
                            continue
                elif cmd_lower == 'cd':
                    try:
                        self._builtin_cd([])
                        continue
                    except Exception as exc:
                        print(f"{Colors.FG_RED}cd error: {exc}{Colors.RESET}")
                        self.last_exit_code = 1
                        continue
                



                proc = subprocess.Popen(
                    cmd,
                    shell=True,
                    executable='/bin/bash',
                    text=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    env=self.env,
                    cwd=os.getcwd(),
                )

                assert proc.stdout is not None
                for out_line in proc.stdout:
                    print(out_line, end="")

                exit_code = proc.wait()
                self.last_exit_code = exit_code
                



        except FileNotFoundError:
            print(f"{Colors.FG_RED}Command not found{Colors.RESET}")
            self.last_exit_code = 127
        except Exception as exc:
            print(f"{Colors.FG_RED}Execution error: {exc}{Colors.RESET}")
            self.last_exit_code = 1



    def _builtin_exit(self, args: list[str]) -> None:
        self.running = False

    def _builtin_cd(self, args: list[str]) -> None:
        if len(args) == 0:
            target = self.env.get("HOME", str(Path.home()))
        elif args[0] == "-":
            if self.prev_dir is None:
                print(f"{Colors.FG_RED}cd: OLDPWD not set{Colors.RESET}")
                self.last_exit_code = 1
                return
            target = self.prev_dir
        else:
            target = args[0]

        try:
            current_dir = os.getcwd()
            os.chdir(os.path.expanduser(target))
            self.prev_dir = current_dir
            self.last_exit_code = 0
        except FileNotFoundError:
            print(f"{Colors.FG_RED}No such directory: {target}{Colors.RESET}")
            self.last_exit_code = 1
        except NotADirectoryError:
            print(f"{Colors.FG_RED}Not a directory: {target}{Colors.RESET}")
            self.last_exit_code = 1
        except PermissionError:
            print(f"{Colors.FG_RED}Permission denied: {target}{Colors.RESET}")
            self.last_exit_code = 1

    def _builtin_pwd(self, args: list[str]) -> None:
        try:
            print(os.getcwd())
            self.last_exit_code = 0
        except OSError:
            print(f"{Colors.FG_RED}Error: cannot get current directory{Colors.RESET}")
            self.last_exit_code = 1

    def _builtin_clear(self, args: list[str]) -> None:
        if os.name == "nt":
            os.system("cls")
        else:
            os.system("clear")
        self.last_exit_code = 0

    def _builtin_export(self, args: list[str]) -> None:
        """Export without args: list variables

         export NAME=VALUE: set variable"""
        if not args:
            for key, value in sorted(self.env.items()):
                print(f"{key}={value}")
            self.last_exit_code = 0
            return

        error = False
        for item in args:
            if "=" not in item:
                print(f"{Colors.FG_RED}Usage: export NAME=VALUE{Colors.RESET}")
                error = True
                continue
            name, value = item.split("=", 1)
            name = name.strip()
            value = value.strip()
            if not name:
                print(f"{Colors.FG_RED}Invalid variable name{Colors.RESET}")
                error = True
                continue
            self.env[name] = value
            os.environ[name] = value
        
        self.last_exit_code = 1 if error else 0

    def _builtin_alias(self, args: list[str]) -> None:
        """Alias            -> list all aliases

         alias name=value -> set alias"""
        if not args:
            for name, value in sorted(self.aliases.items()):
                print(f"alias {name}='{value}'")
            self.last_exit_code = 0
            return

        error = False
        for item in args:
            if "=" not in item:
                print(
                    f"{Colors.FG_RED}Usage: alias name='Value' or alias name=value{Colors.RESET}"
                )
                error = True
                continue
            name, value = item.split("=", 1)
            name = name.strip()
            value = value.strip()
            if value.startswith(("'", '"')) and value.endswith(("'", '"')):
                value = value[1:-1]
            if not name:
                print(f"{Colors.FG_RED}Invalid alias name{Colors.RESET}")
                error = True
                continue
            self.aliases[name] = value
        
        self.last_exit_code = 1 if error else 0

    def _builtin_unalias(self, args: list[str]) -> None:
        if not args:
            print(f"{Colors.FG_RED}Usage: unalias name{Colors.RESET}")
            self.last_exit_code = 1
            return

        not_found = False
        for name in args:
            if name not in self.aliases:
                print(f"{Colors.FG_YELLOW}Warning: alias {name} not found{Colors.RESET}")
                not_found = True
            else:
                self.aliases.pop(name, None)
        
        self.last_exit_code = 1 if not_found else 0

    def _builtin_env(self, args: list[str]) -> None:
        for key, value in sorted(self.env.items()):
            print(f"{key}={value}")
        self.last_exit_code = 0

    def _builtin_which(self, args: list[str]) -> None:
        if not args:
            print(f"{Colors.FG_RED}Usage: which command ...{Colors.RESET}")
            self.last_exit_code = 1
            return

        for cmd in args:

            if cmd in self.aliases:
                print(f"{cmd}: aliased to {self.aliases[cmd]}")
                continue


            if cmd in self.builtins:
                print(f"{cmd}: shell builtin")
                continue


            path = shutil.which(cmd, path=self.env.get("PATH"))
            if path:
                print(path)
            else:
                print(f"{cmd} not found")
                self.last_exit_code = 1

    def _builtin_history(self, args: list[str]) -> None:
        """Display command history"""
        if readline is None:
            print(f"{Colors.FG_RED}History not available (readline not supported){Colors.RESET}")
            self.last_exit_code = 1
            return
        
        try:

            hist_len = readline.get_current_history_length()
            

            if args and args[0].isdigit():
                n = int(args[0])
                start = max(1, hist_len - n + 1)
            else:
                start = 1
            
            for i in range(start, hist_len + 1):
                line = readline.get_history_item(i)
                if line:
                    print(f"{i:5}  {line}")
        except Exception as exc:
            print(f"{Colors.FG_RED}Error reading history: {exc}{Colors.RESET}")
            self.last_exit_code = 1

    def _builtin_echo(self, args: list[str]) -> None:
        """Echo arguments, with support for -n flag"""
        if not args:
            print()
            return
        

        no_newline = False
        if args[0] == "-n":
            no_newline = True
            args = args[1:]
        

        expanded_args = [self._expand_variables(arg) for arg in args]
        output = " ".join(expanded_args)
        
        if no_newline:
            print(output, end="")
        else:
            print(output)

    def _builtin_unset(self, args: list[str]) -> None:
        """Unset environment variables"""
        if not args:
            print(f"{Colors.FG_RED}Usage: unset VAR ...{Colors.RESET}")
            self.last_exit_code = 1
            return
        
        for var_name in args:
            if var_name in self.env:
                del self.env[var_name]
                if var_name in os.environ:
                    del os.environ[var_name]
            else:
                print(f"{Colors.FG_YELLOW}Warning: {var_name} not set{Colors.RESET}")

    def _builtin_help(self, args: list[str]) -> None:
        print("MiniPy Shell help")
        print()
        print("Built in commands:")
        print("  cd [dir]          change directory (cd - goes to previous dir)")
        print("  pwd               print current directory")
        print("  clear             clear the screen")
        print("  export VAR=VALUE  set environment variable")
        print("  unset VAR         unset environment variable")
        print("  env, set          list environment variables")
        print("  alias [a=b]       define or list aliases")
        print("  unalias name      remove alias")
        print("  which cmd [...]   locate a command")
        print("  echo [-n] text    print text (with variable expansion)")
        print("  history [N]       show command history (last N lines)")
        print("  help              show this help")
        print("  exit, quit        leave the shell")
        print()
        print("Variable expansion:")
        print("  $VAR              expand environment variable")
        print("  ${VAR}            expand environment variable (braced)")
        print("  $?                last command exit code")
        print()
        print("Natural Language Commands:")
        print("  You can type natural language instead of commands!")
        print("  Examples:")
        print("    'I want to list all files in this directory'")
        print("    'Show me the current directory'")
        print("    'Copy file.txt to temp directory'")
        print("  The AI agent will translate your request to the appropriate command.")
        print("  Multi-step commands are supported: 'Do X then do Y'")
        print()
        print("External commands are executed by the system shell with full support")
        print("for pipes, redirection, background jobs and similar features.")
        print()
        print("Features:")
        print("  history saved in ~/.minipy_history")
        print("  tab completion for commands and paths")
        print("  colored prompt with user@host:directory format")
        print("  exit code tracking ($?)")
        print("  AI-powered natural language to command translation")


def main() -> None:
    shell = MiniPyShell()
    shell.run()


if __name__ == "__main__":
    main()
