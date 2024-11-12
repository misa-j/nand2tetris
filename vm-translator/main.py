from enum import Enum
import os
from typing import Iterable


class CommandType(Enum):
    C_ARITHMETIC = "C_ARITHMETIC"
    C_PUSH = "C_PUSH"
    C_POP = "C_POP"
    C_LABEL = "C_LABEL"
    C_GOTO = "C_GOTO"
    C_IF = "C_IF"
    C_FUNCTION = "C_FUNCTION"
    C_RETURN = "C_RETURN"
    C_CALL = "C_CALL"


class ArithmeticLogicalCommand(Enum):
    ADD = "add"
    SUB = "sub"
    NEG = "neg"
    EQ = "eq"
    GT = "gt"
    LT = "lt"
    AND = "and"
    OR = "or"
    NOT = "not"

    @classmethod
    def get_values(cls) -> Iterable[str]:
        return [e.value for e in cls]

    @classmethod
    def get_binary_commands(cls) -> Iterable[str]:
        return [cls.ADD.value, cls.SUB.value, cls.AND.value, cls.OR.value]

    @classmethod
    def get_unary_commands(cls) -> Iterable[str]:
        return [cls.NEG.value, cls.NOT.value]

    @classmethod
    def get_logcal_commands(cls) -> Iterable[str]:
        return [cls.EQ.value, cls.LT.value, cls.GT.value]

    @classmethod
    def get_command(cls, command: str) -> str:
        command_map = {
            cls.ADD.value: "M=D+M",
            cls.SUB.value: "M=M-D",
            cls.NEG.value: "M=-M",
            cls.AND.value: "M=D&M",
            cls.OR.value: "M=D|M",
            cls.NOT.value: "M=!M",
            cls.EQ.value: "M=M-D",
            cls.GT.value: "M=M-D",
            cls.LT.value: "M=M-D",
        }
        return command_map[command]

    @classmethod
    def get_jump_command(cls, command: str) -> str:
        command_map = {
            cls.EQ.value: "JEQ",
            cls.GT.value: "JGT",
            cls.LT.value: "JLT",
        }
        return command_map[command]


class MemoryAccessCommand(Enum):
    POP = "pop"
    PUSH = "push"

    @classmethod
    def get_values(cls) -> Iterable[str]:
        return [e.value for e in cls]


class LableCommand(Enum):
    LABEL = "label"

    @classmethod
    def get_values(cls) -> Iterable[str]:
        return [e.value for e in cls]


class FunctionCommand(Enum):
    FUNCTION = "function"
    CALL = "call"
    RETURN = "return"

    @classmethod
    def get_values(cls) -> Iterable[str]:
        return [e.value for e in cls]


class JumpCommand(Enum):
    GOTO = "goto"
    IF_GOTO = "if-goto"

    @classmethod
    def get_values(cls) -> Iterable[str]:
        return [e.value for e in cls]


class MemorySegment(Enum):
    ARGUMENT = "argument"
    LOCAL = "local"
    STATIC = "static"
    CONSTANT = "constant"
    THIS = "this"
    THAT = "that"
    POINTER = "pointer"
    TEMP = "temp"

    @classmethod
    def get_values(cls) -> Iterable[str]:
        return [e.value for e in cls]

    @classmethod
    def get_segment(cls, segment: str) -> Iterable[str]:
        command_map = {
            cls.ARGUMENT.value: "ARG",
            cls.LOCAL.value: "LCL",
            cls.THIS.value: "THIS",
            cls.THAT.value: "THAT",
        }
        return command_map[segment]

    @classmethod
    def get_memory_segments_with_pointer(cls) -> Iterable[str]:
        return [
            cls.ARGUMENT.value,
            cls.LOCAL.value,
            cls.THIS.value,
            cls.THAT.value,
            cls.TEMP.value,
        ]


class VMParser:
    current_line = 0
    file_name: str

    def __init__(self, file_path: str) -> None:
        self.file_path = file_path
        self.commands: list[str] = []
        with open(file_path) as file:
            for line in file:
                line = line.strip()
                if line == "" or line.startswith("//"):
                    continue

                self.commands.append(line)

        file_path_split = os.path.basename(file_path)
        file_name = file_path_split[0 : len(file_path_split) - 3]
        self.file_name = file_name

    def has_more_commands(self) -> bool:
        return self.current_line < len(self.commands)

    def advance(self) -> None:
        self.current_line = self.current_line + 1

    def command_type(self) -> CommandType:
        current_command: str = self.commands[self.current_line]
        for command_kwd in ArithmeticLogicalCommand.get_values():
            if current_command.startswith(command_kwd):
                return CommandType.C_ARITHMETIC

        if current_command.startswith(MemoryAccessCommand.PUSH.value):
            return CommandType.C_PUSH

        if current_command.startswith(MemoryAccessCommand.POP.value):
            return CommandType.C_POP

        if current_command.startswith(LableCommand.LABEL.value):
            return CommandType.C_LABEL

        if current_command.startswith(JumpCommand.GOTO.value):
            return CommandType.C_GOTO

        if current_command.startswith(JumpCommand.IF_GOTO.value):
            return CommandType.C_IF

        if current_command.startswith(FunctionCommand.FUNCTION.value):
            return CommandType.C_FUNCTION

        if current_command.startswith(FunctionCommand.CALL.value):
            return CommandType.C_CALL

        if current_command.startswith(FunctionCommand.RETURN.value):
            return CommandType.C_RETURN

    def arg1(self) -> str:
        command_parts: list[str] = self.commands[self.current_line].split(" ")

        if self.command_type() == CommandType.C_ARITHMETIC:
            return command_parts[0]

        if self.command_type() in [
            CommandType.C_PUSH,
            CommandType.C_POP,
            CommandType.C_GOTO,
            CommandType.C_IF,
            CommandType.C_LABEL,
            CommandType.C_FUNCTION,
            CommandType.C_CALL,
        ]:
            return command_parts[1]

    def arg2(self) -> int:
        command_parts: list[str] = self.commands[self.current_line].split(" ")

        if self.command_type() in [
            CommandType.C_PUSH,
            CommandType.C_POP,
            CommandType.C_FUNCTION,
            CommandType.C_CALL,
        ]:
            return int(command_parts[2])


class CodeWriter:
    commands = []
    logical_command_count = 0
    file_name: str
    call_map: dict[str, int] = {}

    def set_file_name(self, file_name: str) -> None:
        self.file_name = file_name

    def write_push_pop(self, command: CommandType, segment: str, index: int) -> None:
        if segment == MemorySegment.CONSTANT.value:
            current_commands = self._get_constant_push_commands(index)
            self.commands.extend(current_commands)
        if segment in MemorySegment.get_memory_segments_with_pointer():
            if command == CommandType.C_PUSH:
                current_commands = self._write_segment_push_commands(segment, index)
                self.commands.extend(current_commands)
            if command == CommandType.C_POP:
                current_commands = self._write_segment_pop_commands(segment, index)
                self.commands.extend(current_commands)
        if segment == MemorySegment.STATIC.value:
            if command == CommandType.C_PUSH:
                current_commands = self._write_static_segment_push_commands(index)
                self.commands.extend(current_commands)
            else:
                current_commands = self._write_static_segment_pop_commands(index)
                self.commands.extend(current_commands)
        if segment == MemorySegment.POINTER.value:
            if command == CommandType.C_PUSH:
                current_commands = self._write_pointer_segment_push_commands(index)
                self.commands.extend(current_commands)
            else:
                current_commands = self._write_pointer_segment_pop_commands(index)
                self.commands.extend(current_commands)

    def write_arithmetic(self, command: str) -> None:
        if command in ArithmeticLogicalCommand.get_binary_commands():
            current_commands = self._write_binary_command(command)
            self.commands.extend(current_commands)
        if command in ArithmeticLogicalCommand.get_unary_commands():
            current_commands = self._write_unary_command(command)
            self.commands.extend(current_commands)
        if command in ArithmeticLogicalCommand.get_logcal_commands():
            current_commands = self._write_logical_command(command)
            self.commands.extend(current_commands)

    def write_label(self, label: str) -> None:
        current_commands = self._write_label_commands(label)
        self.commands.extend(current_commands)

    def write_goto(self, label: str) -> None:
        current_commands = self._write_goto_commands(label)
        self.commands.extend(current_commands)

    def write_if(self, label: str) -> None:
        current_commands = self._write_if_commands(label)
        self.commands.extend(current_commands)

    def write_function(self, func_name: str, num_locals: int) -> None:
        current_commands = self._write_function_commands(func_name, num_locals)
        self.commands.extend(current_commands)

    def write_return(self) -> None:
        current_commands = self._write_return_commands()
        self.commands.extend(current_commands)

    def write_call(self, func_name: str, num_args: int) -> None:
        current_commands = self._write_call_commands(func_name, num_args)
        self.commands.extend(current_commands)

    def write_init(self) -> None:
        current_commands = ["@256", "D=A", "@SP", "M=D"]
        self.commands.extend(current_commands)
        current_commands = self._write_call_commands("Sys.init", 0)
        self.commands.extend(current_commands)

    # def _write_init_commands(self) -> Iterable[set]:
    #     return [
    #         "@256",
    #         "D=A",
    #         "@SP",
    #         "M=D",
    #         "@Sys.init",
    #         "0;JMP",
    #     ]

    def _get_call_num(self, func_name: str) -> int:
        if func_name not in self.call_map:
            self.call_map[func_name] = 0
        self.call_map[func_name] = self.call_map[func_name] + 1

        return self.call_map[func_name]

    def _write_function_commands(
        self, func_name: str, num_locals: int
    ) -> Iterable[str]:
        init_locals_commands = []

        for i in range(num_locals):
            init_locals_commands.extend(["@SP", "A=M", "M=0", "@SP", "M=M+1"])

        function_commands = [f"({func_name})"]
        function_commands.extend(init_locals_commands)

        return function_commands

    def _write_if_commands(self, label: str) -> Iterable[str]:
        return ["@SP", "M=M-1", "A=M", "D=M", f"@{label}", "D;JNE"]

    def _write_goto_commands(self, label: str) -> Iterable[str]:
        return [f"@{label}", "0;JMP"]

    def _write_label_commands(self, label: str) -> Iterable[str]:
        return [f"({label})"]

    def _get_constant_push_commands(self, constant: int) -> Iterable[str]:
        return [f"@{constant}", "D=A", "@SP", "A=M", "M=D", "@SP", "M=M+1"]

    def _write_segment_push_commands(self, segment: str, index: int) -> Iterable[str]:
        if segment == MemorySegment.TEMP.value:
            return self._write_temp_segment_push_commands(index)

        memory_segment: str = MemorySegment.get_segment(segment)

        return [
            f"@{index}",
            "D=A",
            f"@{memory_segment}",
            "A=M",
            "A=D+A",
            "D=M",
            "@SP",
            "A=M",
            "M=D",
            "@SP",
            "M=M+1",
        ]

    def _write_temp_segment_push_commands(self, index: int) -> Iterable[str]:
        return [f"@{index + 5}", "D=M", "@SP", "A=M", "M=D", "@SP", "M=M+1"]

    def _write_segment_pop_commands(self, segment: str, index: int) -> Iterable[str]:
        if segment == MemorySegment.TEMP.value:
            return self._write_temp_segment_pop_commands(index)

        memory_segment: str = MemorySegment.get_segment(segment)

        return [
            "@SP",
            "M=M-1",
            f"@{index}",
            "D=A",
            f"@{memory_segment}",
            "D=D+M",
            "@R13",
            "M=D",
            "@SP",
            "A=M",
            "D=M",
            "@R13",
            "A=M",
            "M=D",
        ]

    def _write_temp_segment_pop_commands(self, index: int) -> Iterable[str]:
        return ["@SP", "M=M-1", "A=M", "D=M", f"@{index + 5}", "M=D"]

    def _write_static_segment_pop_commands(self, index: int) -> Iterable[str]:
        return ["@SP", "M=M-1", "A=M", "D=M", f"@{self.file_name}.{index}", "M=D"]

    def _write_static_segment_push_commands(self, index: int) -> Iterable[str]:
        return [
            f"@{self.file_name}.{index}",
            "D=M",
            "@SP",
            "A=M",
            "M=D",
            "@SP",
            "M=M+1",
        ]

    def _write_pointer_segment_push_commands(self, index: int) -> Iterable[str]:
        memory_segment: str = MemorySegment.get_segment(
            "this" if index == 0 else "that"
        )
        # return [f"@{memory_segment}", "A=M", "D=M", "@SP", "A=M", "M=D", "@SP", "M=M+1"]
        return [f"@{memory_segment}", "D=M", "@SP", "A=M", "M=D", "@SP", "M=M+1"]

    def _write_pointer_segment_pop_commands(self, index: int) -> Iterable[str]:
        memory_segment: str = MemorySegment.get_segment(
            "this" if index == 0 else "that"
        )
        # return ["@SP", "M=M-1", "A=M", "D=M", f"@{memory_segment}", "A=M", "M=D"]
        return ["@SP", "M=M-1", "A=M", "D=M", f"@{memory_segment}", "M=D"]

    def _write_binary_command(self, command: str) -> Iterable[str]:
        assembly_command: str = ArithmeticLogicalCommand.get_command(command)
        return [
            "@SP",
            "M=M-1",
            "@SP",
            "A=M",
            "D=M",
            "@SP",
            "M=M-1",
            "@SP",
            "A=M",
            assembly_command,
            "@SP",
            "M=M+1",
        ]

    def _write_unary_command(self, command: str) -> Iterable[str]:
        assembly_command: str = ArithmeticLogicalCommand.get_command(command)
        return ["@SP", "M=M-1", "@SP", "A=M", assembly_command, "@SP", "M=M+1"]

    def _write_logical_command(self, command: str) -> Iterable[str]:
        self.logical_command_count = self.logical_command_count + 1
        command_index = self.logical_command_count
        jump_command = ArithmeticLogicalCommand.get_jump_command(command)
        assembly_command: str = ArithmeticLogicalCommand.get_command(command)
        return [
            # Get value from RAM[SP - 1] and place result in RAM[SP - 2]
            "@SP",
            "M=M-1",
            "@SP",
            "A=M",
            "D=M",
            "@SP",
            "M=M-1",
            "@SP",
            "A=M",
            assembly_command,
            # Set RAM[SP] value to -1 or 0 if condition is false
            "D=M",
            f"@SETTRUE{command_index}",
            f"D;{jump_command}",
            "@SP",
            "A=M",
            "M=0",
            f"@ENDIF{command_index}",
            "0;JMP",
            f"(SETTRUE{command_index})",
            "@SP",
            "A=M",
            "M=-1",
            f"(ENDIF{command_index})",
            "@SP",
            "M=M+1",
        ]

    def _increment_sp_commands(self) -> Iterable[str]:
        return ["@SP", "M=M+1"]

    def _write_return_commands(self) -> Iterable[set]:
        return [
            # FRAME = LCL
            "@LCL",
            "D=M",
            "@R13",
            "M=D",
            # RET = *(FRAME - 5)
            "@5",
            "D=A",
            "@R13",
            "D=M-D",
            "A=D",
            "D=M",
            "@R14",
            "M=D",
            # *ARG = pop()
            "@SP",
            "M=M-1",
            "A=M",
            "D=M",
            "@ARG",
            "A=M",
            "M=D",
            # SP = ARG + 1
            "@ARG",
            "D=M+1",
            "@SP",
            "M=D",
            # THAT = *(FRAME - 1)
            "@1",
            "D=A",
            "@R13",
            "D=M-D",
            "A=D",
            "D=M",
            "@THAT",
            "M=D",
            # THIS = *(FRAME - 2)
            "@2",
            "D=A",
            "@R13",
            "D=M-D",
            "A=D",
            "D=M",
            "@THIS",
            "M=D",
            # ARG = *(FRAME - 3)
            "@3",
            "D=A",
            "@R13",
            "D=M-D",
            "A=D",
            "D=M",
            "@ARG",
            "M=D",
            # LCL = *(FRAME - 4)
            "@4",
            "D=A",
            "@R13",
            "D=M-D",
            "A=D",
            "D=M",
            "@LCL",
            "M=D",
            "@R14",
            # goto RET
            "A=M",
            "0;JMP",
        ]

    def _write_call_commands(self, func_name: str, num_args: int) -> Iterable[str]:
        ret_num = self._get_call_num(func_name)
        ret_addr = f"{func_name}_{ret_num}"

        return [
            f"@{ret_addr}",
            "D=A",
            "@SP",
            "A=M",
            "M=D",
            "@SP",
            "M=M+1",
            # push LCL
            "@LCL",
            "D=M",
            "@SP",
            "A=M",
            "M=D",
            "@SP",
            "M=M+1",
            # push ARG
            "@ARG",
            "D=M",
            "@SP",
            "A=M",
            "M=D",
            "@SP",
            "M=M+1",
            # push THIS
            "@THIS",
            "D=M",
            "@SP",
            "A=M",
            "M=D",
            "@SP",
            "M=M+1",
            # push THAT
            "@THAT",
            "D=M",
            "@SP",
            "A=M",
            "M=D",
            "@SP",
            "M=M+1",
            # ARG = SP - n - 5, n = number of args
            f"@{num_args}",
            "D=A",
            "@R13",
            "M=D",
            "@5",
            "D=A",
            "@R13",
            "M=D+M",
            "@SP",
            "D=M",
            "@R13",
            "D=D-M",
            "@ARG",
            "M=D",
            # LCL = SP
            "@SP",
            "D=M",
            "@LCL",
            "M=D",
            # goto f
            f"@{func_name}",
            "0;JMP",
            # (return-address)
            f"({ret_addr})",
        ]


def append_to_file(file_path: str, text: str) -> None:
    with open(file_path, "a") as myfile:
        myfile.write(text + "\n")


def get_vm_files(path: str) -> list[str]:
    if os.path.isdir(path):
        file_paths = []
        for dirpath, dirnames, filenames in os.walk(path):
            for f in filenames:
                file_path = os.path.join(dirpath, f)
                filename, file_extension = os.path.splitext(file_path)
                if file_extension == ".vm":
                    file_paths.append(file_path)
        return file_paths

    return [path]


def parse_source(path: str) -> None:
    file_names = get_vm_files(path=path)
    code_writer = CodeWriter()

    out_file_path = "./out.txt"
    if os.path.isfile(out_file_path):
        os.remove(out_file_path)

    code_writer.write_init()
    for file_name in file_names:

        parser = VMParser(file_name)
        # print(parser.file_name, os.path.basename(file_name))
        code_writer.set_file_name(parser.file_name)

        while parser.has_more_commands():
            current_command: str = parser.commands[parser.current_line]
            code_writer.commands.append(f"// {current_command}")
            if parser.command_type() in [CommandType.C_PUSH, CommandType.C_POP]:
                code_writer.write_push_pop(
                    parser.command_type(), parser.arg1(), parser.arg2()
                )
            elif parser.command_type() == CommandType.C_ARITHMETIC:
                code_writer.write_arithmetic(parser.arg1())
            elif parser.command_type() == CommandType.C_LABEL:
                code_writer.write_label(parser.arg1())
            elif parser.command_type() == CommandType.C_GOTO:
                code_writer.write_goto(parser.arg1())
            elif parser.command_type() == CommandType.C_IF:
                code_writer.write_if(parser.arg1())
            elif parser.command_type() == CommandType.C_FUNCTION:
                code_writer.write_function(parser.arg1(), parser.arg2())
            elif parser.command_type() == CommandType.C_RETURN:
                code_writer.write_return()
            elif parser.command_type() == CommandType.C_CALL:
                code_writer.write_call(parser.arg1(), parser.arg2())
            parser.advance()

    append_to_file(out_file_path, "\n".join(code_writer.commands))


BasicLoop = "./program-flow-functions/BasicLoop"
FibonacciElement = "./program-flow-functions/FibonacciElement"
FibonacciSeries = "./program-flow-functions/FibonacciSeries"
NestedCall = "./program-flow-functions/NestedCall"
SimpleFunction = "./program-flow-functions/SimpleFunction"
StaticTest = "./program-flow-functions/StaticsTest"

parse_source(StaticTest)

# print(get_vm_files("./program-flow-functions/FibonacciSeries"))

# print(code_writer.commands)
# print(parser.file_name)
# push command
# @index
# D=A
# @LCL
# A=M
# A=D+A
# D=M
# @SP
# A=M
# M=D
# @SP
# M=M+1


# @SP
# M=M-1
# @index
# D=A
# @LCL
# D=D+M
# @R13
# M=D
# @SP
# A=M
# D=M
# @R3
# A=M
# M=D

# pop static
# @SP
# M=M-1
# A=M
# D=M
# @Filename.index
# M=D

# push static
# @Filename.index
# D=M
# @SP
# A=M
# M=D
# @SP
# M=M+1

# push pointer
# @THIS/THAT
# A=M
# D=M
# @SP
# A=M
# M=D
# @SP
# M=M+1

# @SP
# M=M-1
# A=M
# D=M
# @this/that
# A=M
# M=D

# if goto
# @SP
# M=M-1
# A=M
# D=M
# @label
# D;JGT

# return

# @LCL
# D=M
# @R13 # FRAME = LCL
# M=D
# @5
# D=A
# @R13
# D=M-D
# A=D
# D=M
# @R14  # RET = *(FRAME - 5)
# M=D
# # *ARG = pop()
# @SP
# M=M-1
# A=M
# D=M
# @ARG
# A=M
# M=D
# # SP = ARG + 1
# @ARG
# M=M+1
# D=M
# @SP
# M=D
# # THAT = *(FRAME - 1)
# @1
# D=A
# @R13
# D=M-D
# A=D
# D=M
# @THAT
# M=D
# # THIS = *(FRAME - 2)
# @2
# D=A
# @R13
# D=M-D
# A=D
# D=M
# @THIS
# M=D
# # ARG = *(FRAME - 3)
# @3
# D=A
# @R13
# D=M-D
# A=D
# D=M
# @ARG
# M=D
# # LCL = *(FRAME - 4)
# @4
# D=A
# @R13
# D=M-D
# A=D
# D=M
# @LCL
# M=D
# @R14
# A=M
# 0;JMP

# push return-address
# @RET
# D=A
# @SP
# A=M
# M=D
# @SP
# M=M+1
# # push LCL
# @LCL
# D=M
# @SP
# A=M
# M=D
# @SP
# M=M+1
# # push ARG
# @ARG
# D=M
# @SP
# A=M
# M=D
# @SP
# M=M+1
# # push THIS
# @THIS
# D=M
# @SP
# A=M
# M=D
# @SP
# M=M+1
# # push THAT
# @THAT
# D=M
# @SP
# A=M
# M=D
# @SP
# M=M+1
# # ARG = SP - n - 5, n = number of args
# @n
# D=A
# @R13
# M=D
# @5
# D=A
# @R13
# M=M+D
# @SP
# D=M
# @R13
# D=D-M
# @ARG
# M=D
# # LCL = SP
# @SP
# D=M
# @LCL
# M=D
# # goto f
# @func
# 0;JMP
# # (return-address)
# (RET)
