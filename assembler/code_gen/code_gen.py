from .symbols import dest_symbols, comp_symbols, jump_symbols

class CodeGen:

    @classmethod
    def dest(mnemonic: str) -> str:
        return dest_symbols[mnemonic]
    
    @classmethod
    def comp(mnemonic: str) -> str:
        return comp_symbols[mnemonic]
    
    @classmethod
    def jump(mnemonic: str) -> str:
        return jump_symbols[mnemonic]
    
    @classmethod
    def gen_a_command(mnemonic: str) -> str:
        binary_repr: str = bin(int(mnemonic))[2:]
        padding: str = (16 - len(binary_repr)) * "0"

        return padding + binary_repr

    @classmethod
    def gen_c_command(dest: str, comp: str, jump: str) -> str:
        return "111" + comp + dest + jump