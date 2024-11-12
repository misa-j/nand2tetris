from enum import Enum
import os

symbols = {
    "SP": "0",
    "LCL": "1",
    "ARG": "2",
    "THIS": "3",
    "THAT": "4",
    "R0": "0",
    "R1": "1",
    "R2": "2",
    "R3": "3",
    "R4": "4",
    "R5": "5",
    "R6": "6",
    "R7": "7",
    "R8": "8",
    "R9": "9",
    "R10": "10",
    "R11": "11",
    "R12": "12",
    "R13": "13",
    "R14": "14",
    "R15": "15",
    "SCREEN": "16384",
    "KBD": "24576",
}
dest_symbols = {
    "":    "000",
    "M":   "001",
    "D":   "010",
    "MD":  "011",
    "A":   "100",
    "AM":  "101",
    "AD":  "110",
    "AMD": "111"
}
jump_symbols = {
    "":    "000",
    "JGT": "001",
    "JEQ": "010",
    "JGE": "011",
    "JLT": "100",
    "JNE": "101",
    "JLE": "110",
    "JMP": "111"
}
comp_symbols = {
    "0":   "0101010",
    "1":   "0111111",
    "-1":  "0111010",
    "D":   "0001100",
    "A":   "0110000",
    "!D":  "0001101",
    "!A":  "0110001",
    "-D":  "0001111",
    "-A":  "0110011",
    "D+1": "0011111",
    "A+1": "0110111",
    "D-1": "0001110",
    "A-1": "0110010",
    "D+A": "0000010",
    "D-A": "0010011",
    "A-D": "0000111",
    "D&A": "0000000",
    "D|A": "0010101",

    "M":   "1110000",
    "!M":  "1110001",
    "-M":  "1110011",
    "M+1": "1110111",
    "M-1": "1110010",
    "D+M": "1000010",
    "D-M": "1010011",
    "M-D": "1000111",
    "D&M": "1000000",
    "D|M": "1010101",
}

class CommadnType(Enum):
    A_COMMAND = "A_COMMAND"
    C_COMMAND = "C_COMMAND"
    L_COMMAND = "L_COMMAND"

class CodeGen:

    @staticmethod
    def dest(mnemonic: str) -> str:
        return dest_symbols[mnemonic]
    
    @staticmethod
    def comp(mnemonic: str) -> str:
        return comp_symbols[mnemonic]
    
    @staticmethod
    def jump(mnemonic: str) -> str:
        return jump_symbols[mnemonic]
    
    @staticmethod
    def gen_a_command(mnemonic: str) -> str:
        binary_repr: str = bin(int(mnemonic))[2:]
        padding: str = (16 - len(binary_repr)) * "0"

        return padding + binary_repr

    @staticmethod
    def gen_c_command(dest: str, comp: str, jump: str) -> str:
        return "111" + comp + dest + jump

class Parser:
    current_line = 0

    def __init__(self, file_path: str) -> None:
        self.file_path = file_path
        self.commands = []
        with open(file_path) as file:
            for line in file:
                line = line.strip()
                if line == "" or line.startswith("//"):
                    continue

                self.commands.append(line)

    def has_dest(self):
        command: str = self.commands[self.current_line]
        return command.find("=") != -1
    
    def has_jump(self):
        command: str = self.commands[self.current_line]
        return command.find(";") != -1

    def advance(self) -> None:
        self.current_line = self.current_line + 1

    def has_more_commands(self) -> bool:
        return self.current_line < len(self.commands)

    def command_type(self) -> CommadnType:
        command: str = self.commands[self.current_line]
        if command.startswith("@"):
            return CommadnType.A_COMMAND
        if command.startswith("("):
            return CommadnType.L_COMMAND
        return CommadnType.C_COMMAND
    
    def symbol(self) -> str:
        command: str = self.commands[self.current_line]
        if self.command_type() ==  CommadnType.A_COMMAND:
            return command[1:]
        return command[1:len(command) - 1]
    
    def dest(self) -> str:
        command: str = self.commands[self.current_line]
        if not self.has_dest():
            return ""
        command_split = command.split("=")

        return command_split[0].strip()
    
    def comp(self) -> str:
        command: str = self.commands[self.current_line]
        if not self.has_dest():
            split_command = command.split(";")
            return split_command[0].strip()
        
        command_split = command.split("=")
        if not self.has_jump():
            return command_split[1].strip()
        
        command_split_jump = command_split.split(";")
        return command_split_jump[0].strip()
    
    def jump(self) -> str:
        command: str = self.commands[self.current_line]
        if not self.has_jump():
            return ""
        
        command_split = command.split(";")
        return command_split[1].strip()

def append_to_file(file_path: str, text: str) -> None:
    with open(file_path, "a") as myfile:
        myfile.write(text + "\n")   

file_path = "./test-programs/Rect.asm"
out_file_path = "./out.txt"
parser = Parser(file_path)
os.remove(out_file_path)

def build_symbol_table(symbols: dict[str, str], file_path: str) -> None:
    parser = Parser(file_path)
    command_count = 0
    while parser.has_more_commands():
        if parser.command_type() == CommadnType.L_COMMAND:
            symbol = parser.symbol()
            symbols[symbol] = command_count
        if parser.command_type() in [CommadnType.A_COMMAND, CommadnType.C_COMMAND]:
            command_count = command_count + 1
        parser.advance()

def assemble_out(symbols: dict[str, str], file_path: str) -> None:
    parser = Parser(file_path)
    ram_address = 16
    assebled_commands = []
    while parser.has_more_commands():
        if parser.command_type() == CommadnType.A_COMMAND:
            symbol = parser.symbol()
            if not symbol[0].isdigit():
                if symbol not in symbols:
                    symbols[symbol] = ram_address
                    ram_address = ram_address + 1
                symbol = symbols[symbol]

            translated = CodeGen.gen_a_command(symbol)
            assebled_commands.append(translated)
        if parser.command_type() == CommadnType.C_COMMAND:
            dest =  CodeGen.dest(parser.dest())
            comp =  CodeGen.comp(parser.comp())
            jump = CodeGen.jump(parser.jump())
            translated = CodeGen.gen_c_command(dest, comp, jump)
            assebled_commands.append(translated)
        parser.advance()

    append_to_file(out_file_path, "\n".join(assebled_commands))
    
build_symbol_table(symbols=symbols, file_path=file_path)
assemble_out(symbols=symbols, file_path=file_path)