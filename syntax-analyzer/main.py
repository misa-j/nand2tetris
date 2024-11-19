from pathlib import Path
from enum import Enum
import os
from typing import Iterable


class TokenType(Enum):
    KEYWORD = "keyword"
    SYMBOL = "symbol"
    IDENTIFIER = "identifier"
    INT_CONST = "int_const"
    STRING_CONST = "string_const"

    @classmethod
    def get_values(cls) -> list[str]:
        return [e.value for e in cls]

    def get_xml_tag_value(self) -> str:
        match self:
            case TokenType.INT_CONST:
                return "integerConstant"
            case TokenType.STRING_CONST:
                return "stringConstant"
            case _:
                return self.value


class KeyWord(Enum):
    CLASS = "class"
    METHOD = "method"
    FUNCTION = "function"
    CONSTRUCTOR = "constructor"
    INT = "int"
    BOOLEAN = "boolean"
    CHAR = "char"
    VOID = "void"
    VAR = "var"
    STATIC = "static"
    FIELD = "field"
    LET = "let"
    DO = "do"
    IF = "if"
    ELSE = "else"
    WHILE = "while"
    RETURN = "return"
    TRUE = "true"
    FALSE = "false"
    NULL = "null"
    THIS = "this"

    @classmethod
    def get_values(cls) -> list[str]:
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


class IdentifierType(Enum):
    STATIC = "static"
    FIELD = "field"
    ARG = "arg"
    VAR = "var"
    NONE = "none"

    @classmethod
    def get_values(cls) -> list[str]:
        return [e.value for e in cls]

    def to_memory_segment(self) -> MemorySegment:
        match self:
            case IdentifierType.ARG:
                return MemorySegment.ARGUMENT
            case IdentifierType.VAR:
                return MemorySegment.LOCAL
            case IdentifierType.FIELD:
                return MemorySegment.THIS
            case IdentifierType.STATIC:
                return MemorySegment.STATIC


class SymbolTable:
    table: dict[str, tuple[str, IdentifierType, int]]

    def __init__(self) -> None:
        self.table = {}

    def start_subroutine(self) -> None:
        self.table = {}

    def define(self, name: str, type: str, kind: IdentifierType) -> None:
        self.table[name] = (type, kind, self.var_count(kind))

    def var_count(self, kind: IdentifierType) -> int:
        return len([x for x in self.table.values() if x[1] == kind])

    def type_of(self, name: str) -> str:
        return self.table[name][0]

    def kind_of(self, name: str) -> IdentifierType:
        if name not in self.table:
            return IdentifierType.NONE

        return self.table[name][1]

    def index_of(self, name: str) -> str:
        return self.table[name][2]

    def print_symbol_table(self) -> None:
        for item in self.table:
            print(
                item,
                self.table[item][0],
                self.table[item][1].value,
                self.table[item][2],
            )


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
    def get_from_symbol(cls, symbol: str) -> "ArithmeticLogicalCommand":
        match symbol:
            case "+":
                return cls.ADD
            case "-":
                return cls.SUB
            case "~":
                return cls.NOT  # TODO: NEG
            case "=":
                return cls.EQ
            case ">":
                return cls.GT
            case "<":
                return cls.LT
            case "&":
                return cls.AND
            case "|":
                return cls.OR
            case "~":
                return cls.NOT


class VMWriter:
    output: list[str]

    def __init__(self) -> None:
        self.output = []

    def write_push(self, segment: MemorySegment, index: int) -> None:
        self.output.append(f"push {segment.value} {index}")

    def write_pop(self, segment: MemorySegment, index: int) -> None:
        self.output.append(f"pop {segment.value} {index}")

    def write_arithmetic(self, command: ArithmeticLogicalCommand) -> None:
        self.output.append(command.value)

    def write_label(self, label: str) -> None:
        self.output.append(f"label {label}")

    def write_goto(self, label: str) -> None:
        self.output.append(f"goto {label}")

    def write_if(self, label: str) -> None:
        self.output.append(f"if-goto {label}")

    def write_call(self, name: str, n_args: int) -> None:
        self.output.append(f"call {name} {n_args}")

    def write_function(self, name: str, n_locals: int) -> None:
        self.output.append(f"function {name} {n_locals}")

    def write_return(self) -> None:
        self.output.append("return")


class JackTokenizer:
    symbols = {
        "{",
        "}",
        "(",
        ")",
        "[",
        "]",
        ".",
        ",",
        ";",
        "+",
        "-",
        "*",
        "/",
        "&",
        "|",
        "<",
        ">",
        "=",
        "_",
        "~",
    }
    symbols_map = {"<": "&lt;", ">": "&gt;", "&": "&amp;", "“": "&quot;"}
    content: str
    tokens: list[tuple[TokenType, str | int]]
    file_name: str
    current_index: int = 0

    def __init__(self, path: str):
        with open(path, "r") as content_file:
            self.content = content_file.read()

        self.file_name = Path(path).stem
        self.tokens = []
        self.__remove_comments()
        self.__tokenize_content()

    def __remove_comments(self) -> None:
        idx = 0
        new_content: list[str] = []
        while idx < len(self.content):
            if self.content[idx] == "/" and (
                self.content[idx + 1] == "/" or self.content[idx + 1] == "*"
            ):
                idx = idx + 1
                if self.content[idx] == "/":
                    idx = idx + 1
                    while idx < len(self.content) and self.content[idx] != "\n":
                        idx = idx + 1
                    idx = idx + 1
                    new_content.append("\n")
                else:
                    idx = idx + 2
                    while idx < len(self.content):
                        if self.content[idx] == "*" and self.content[idx + 1] == "/":
                            break
                        idx = idx + 1
                    idx = idx + 2
            else:
                if idx < len(self.content):
                    new_content.append(self.content[idx])
                idx = idx + 1

        self.content = "".join(new_content).strip()

    def __parse_str(self, idx) -> tuple[str, int]:
        new_str: list[str] = []
        while idx < len(self.content) and self.content[idx] != '"':
            new_str.append(self.content[idx])
            idx = idx + 1
        parsed_str = "".join(new_str)

        return (TokenType.STRING_CONST, parsed_str), idx + 1

    def __parse_int(self, idx) -> tuple[int, int]:
        new_int: list[str] = []
        while (
            idx < len(self.content)
            and self.content[idx] >= "0"
            and self.content[idx] <= "9"
        ):
            new_int.append(self.content[idx])
            idx = idx + 1

        parsed_int = int("".join(new_int))

        return (TokenType.INT_CONST, parsed_int), idx

    def __parse_kwd_or_identifier(self, idx) -> tuple[str, int]:
        new_str: list[str] = []
        while idx < len(self.content) and (
            self.content[idx].isalnum() or self.content[idx] == "_"
        ):
            new_str.append(self.content[idx])
            idx = idx + 1

        parsed_str = "".join(new_str)
        if parsed_str in KeyWord.get_values():
            return (TokenType.KEYWORD, parsed_str), idx

        return (TokenType.IDENTIFIER, parsed_str), idx

    def __tokenize_content(self) -> None:
        idx = 0
        while idx < len(self.content):
            char = self.content[idx]

            if char in self.symbols:
                self.tokens.append((TokenType.SYMBOL, char))
                idx = idx + 1
            elif char == '"':
                parsed_str, new_idx = self.__parse_str(idx + 1)
                self.tokens.append(parsed_str)
                idx = new_idx
            elif char >= "0" and char <= "9":
                parsed_int, new_idx = self.__parse_int(idx)
                self.tokens.append(parsed_int)
                idx = new_idx
            elif char.isalnum() or char == "_":
                parsed_str, new_idx = self.__parse_kwd_or_identifier(idx)
                self.tokens.append(parsed_str)
                idx = new_idx
            else:
                idx = idx + 1

    def __prepare_path(self, path: str) -> None:
        return f"{path}/{self.file_name}T.xml"

    def write_tokens(self, path: str) -> None:
        path = self.__prepare_path(path)
        file_content = []
        file_content.append("<tokens>\n")
        for token in self.tokens:
            token_type = token[0].get_xml_tag_value()
            token_value = (
                token[1]
                if token[1] not in self.symbols_map
                else self.symbols_map[token[1]]
            )
            token_element = f"<{token_type}> {token_value} </{token_type}>\n"
            file_content.append(token_element)
        file_content.append("</tokens>\n")

        with open(path, "w") as filetowrite:
            filetowrite.write("".join(file_content))

    def has_more_tokens(self) -> bool:
        return self.current_index < len(self.tokens)

    def advance(self) -> None:
        self.current_index = self.current_index + 1

    def token_type(self) -> TokenType:
        token = self.tokens[self.current_index]
        return token[0]

    def key_word(self) -> KeyWord:
        token = self.tokens[self.current_index]
        return KeyWord(token[1])

    def symbol(self) -> str:
        token = self.tokens[self.current_index]
        return token[1]

    def identifier(self) -> str:
        token = self.tokens[self.current_index]
        return token[1]

    def int_val(self) -> int:
        token = self.tokens[self.current_index]
        return token[1]

    def string_val(self) -> int:
        token = self.tokens[self.current_index]
        return token[1]


class CompilationEngine:
    tokenizer: JackTokenizer
    content: list[str]
    class_symbol_table: SymbolTable
    subroutine_symbol_table: SymbolTable
    vm_writer: VMWriter
    class_name: str
    label_count: int

    def __init__(self, tokenizer: JackTokenizer):
        self.tokenizer = tokenizer
        self.content = []
        self.label_count = 0
        self.class_symbol_table = SymbolTable()
        self.subroutine_symbol_table = SymbolTable()
        self.vm_writer = VMWriter()

    def compile_class(self) -> None:
        self.content.append("<class> ")
        self.__compile_keyword(KeyWord.CLASS)
        class_name = self.__compile_identifier()
        self.class_name = class_name
        self.__compile_symbol("{")
        self.__compile_class_var_dec()
        self.__compile_subroutine_dec()
        self.__compile_symbol("}")
        self.content.append("</class>")

    def __compile_identifier(self) -> str:
        identifier = self.tokenizer.identifier()
        self.tokenizer.advance()
        self.content.append(f"<identifier> {identifier} </identifier>")

        return identifier

    def __compile_keyword(
        self, expected_kwd: Iterable[KeyWord] | KeyWord | None = None
    ) -> str:
        kwd = self.tokenizer.key_word()
        if expected_kwd:
            if (isinstance(expected_kwd, Iterable) and kwd not in expected_kwd) or (
                isinstance(expected_kwd, KeyWord) and kwd != expected_kwd
            ):
                raise Exception(f"Expected keyword: '{expected_kwd}' got {kwd}")
        self.tokenizer.advance()
        self.content.append(f"<keyword> {kwd.value} </keyword>")
        return kwd.value

    def __compile_int_const(self) -> int:
        value = self.tokenizer.int_val()
        self.tokenizer.advance()
        self.content.append(f"<integerConstant> {value} </integerConstant>")
        return value

    def __compile_string_const(self) -> str:
        value = self.tokenizer.int_val()
        self.tokenizer.advance()
        self.content.append(f"<stringConstant> {value} </stringConstant>")
        return value

    def __compile_symbol(self, expected_symbol: str = None) -> str:
        symbol = self.tokenizer.symbol()
        if expected_symbol and symbol != expected_symbol:
            raise Exception(f"Expected symbol: '{expected_symbol}' got {symbol}")
        self.tokenizer.advance()
        self.content.append(f"<symbol> {symbol} </symbol>")
        return symbol

    def __compile_type(self, additional_types: Iterable[KeyWord] | None = None) -> str:
        if additional_types is None:
            additional_types = ()
        token_type = self.tokenizer.token_type()
        if token_type == TokenType.KEYWORD:
            return self.__compile_keyword(
                [KeyWord.INT, KeyWord.CHAR, KeyWord.BOOLEAN, *additional_types]
            )
        else:
            return self.__compile_identifier()

    def __compile_parameter_list(self) -> None:
        self.content.append(f"<parameterList> ")
        while (
            self.tokenizer.token_type() == TokenType.KEYWORD
            or self.tokenizer.token_type() == TokenType.IDENTIFIER
        ):
            type = self.__compile_type()
            identifier = self.__compile_identifier()
            self.subroutine_symbol_table.define(identifier, type, IdentifierType.ARG)
            if (
                self.tokenizer.token_type() == TokenType.SYMBOL
                and self.tokenizer.symbol() == ","
            ):
                self.__compile_symbol(",")
        self.content.append(f"</parameterList> ")

    def __compile_class_var_dec(self) -> None:
        while (
            self.tokenizer.token_type() == TokenType.KEYWORD
            and self.tokenizer.key_word() in [KeyWord.STATIC, KeyWord.FIELD]
        ):
            self.content.append(f"<classVarDec> ")
            kwd = self.__compile_keyword([KeyWord.STATIC, KeyWord.FIELD])
            type = self.__compile_type()
            identifier = self.__compile_identifier()
            identifier_type = IdentifierType(kwd)
            self.class_symbol_table.define(identifier, type, identifier_type)
            while (
                self.tokenizer.token_type() == TokenType.SYMBOL
                and self.tokenizer.symbol() == ","
            ):
                self.__compile_symbol(",")
                identifier = self.__compile_identifier()
                self.class_symbol_table.define(identifier, type, identifier_type)

            self.__compile_symbol(";")
            self.content.append(f"</classVarDec> ")

    def __compile_subroutine_dec(self) -> None:
        while (
            self.tokenizer.token_type() == TokenType.KEYWORD
            and self.tokenizer.key_word()
            in [KeyWord.CONSTRUCTOR, KeyWord.FUNCTION, KeyWord.METHOD]
        ):
            self.subroutine_symbol_table.start_subroutine()
            self.content.append(f"<subroutineDec> ")
            kwd = self.__compile_keyword(
                [KeyWord.CONSTRUCTOR, KeyWord.FUNCTION, KeyWord.METHOD]
            )
            if kwd == KeyWord.METHOD.value:
                self.subroutine_symbol_table.define(
                    KeyWord.THIS.value, self.class_name, IdentifierType.ARG
                )

            self.__compile_type([KeyWord.VOID])  # void | type
            subroutine_name = self.__compile_identifier()
            self.__compile_symbol("(")
            self.__compile_parameter_list()
            self.__compile_symbol(")")
            self.__compile_subroutine_body(subroutine_name, kwd)
            self.content.append(f"</subroutineDec> ")

    def __compile_subroutine_body(self, subroutine_name: str, kwd: str) -> int:
        self.content.append(f"<subroutineBody> ")
        self.__compile_symbol("{")
        n_locals = self.__compile_var_dec()
        self.vm_writer.write_function(f"{self.class_name}.{subroutine_name}", n_locals)
        if kwd == KeyWord.CONSTRUCTOR.value:
            class_size = self.class_symbol_table.var_count(IdentifierType.FIELD)
            self.vm_writer.write_push(MemorySegment.CONSTANT, class_size)
            self.vm_writer.write_call("Memory.alloc", 1)
            self.vm_writer.write_pop(MemorySegment.POINTER, 0)
        elif kwd == KeyWord.METHOD.value:
            self.vm_writer.write_push(MemorySegment.ARGUMENT, 0)
            self.vm_writer.write_pop(MemorySegment.POINTER, 0)
        self.__compile_statements()
        self.__compile_symbol("}")
        self.content.append(f"</subroutineBody> ")
        return n_locals

    def __compile_var_dec(self) -> int:
        n_locals = 0
        while (
            self.tokenizer.token_type() == TokenType.KEYWORD
            and self.tokenizer.key_word() == KeyWord.VAR
        ):
            n_locals = n_locals + 1
            self.content.append(f"<varDec> ")
            kwd = self.__compile_keyword(KeyWord.VAR)
            type = self.__compile_type()
            identifier = self.__compile_identifier()
            self.subroutine_symbol_table.define(identifier, type, IdentifierType(kwd))
            while (
                self.tokenizer.token_type() == TokenType.SYMBOL
                and self.tokenizer.symbol() == ","
            ):
                n_locals = n_locals + 1
                self.__compile_symbol(",")
                identifier = self.__compile_identifier()
                self.subroutine_symbol_table.define(
                    identifier, type, IdentifierType(kwd)
                )
            self.__compile_symbol(";")
            self.content.append(f"</varDec> ")

        return n_locals

    def __compile_statements(self) -> None:
        self.content.append(f"<statements> ")
        while (
            self.tokenizer.token_type() == TokenType.KEYWORD
            and self.tokenizer.key_word()
            in [KeyWord.LET, KeyWord.IF, KeyWord.WHILE, KeyWord.DO, KeyWord.RETURN]
        ):
            self.__compile_statement()
        self.content.append(f"</statements> ")

    def __compile_statement(self) -> None:
        match self.tokenizer.key_word():
            case KeyWord.LET:
                self.__compile_let_statement()
            case KeyWord.IF:
                self.__compile_if_statement()
            case KeyWord.WHILE:
                self.__compile_while_statement()
            case KeyWord.DO:
                self.__compile_do_statement()
            case KeyWord.RETURN:
                self.__compile_return_statement()
            case _:
                raise Exception("No matching keyword")

    def __compile_let_statement(self) -> None:
        self.content.append(f"<letStatement> ")
        self.__compile_keyword(KeyWord.LET)
        identifier = self.__compile_identifier()
        let_array = False
        if (
            self.tokenizer.token_type() == TokenType.SYMBOL
            and self.tokenizer.symbol() == "["
        ):
            let_array = True
            symbol_table = self.__get_symbol_table(identifier)
            memory_segment = symbol_table.kind_of(identifier).to_memory_segment()
            index = symbol_table.index_of(identifier)
            self.vm_writer.write_push(memory_segment, index)
            self.__compile_symbol("[")
            self.__compile_expression()
            self.__compile_symbol("]")
            self.vm_writer.write_arithmetic(ArithmeticLogicalCommand.ADD)

        self.__compile_symbol("=")
        self.__compile_expression()
        self.__compile_symbol(";")
        self.content.append(f"</letStatement> ")
        if not let_array:
            symbol_table = self.__get_symbol_table(identifier)
            memory_segment = symbol_table.kind_of(identifier)
            index = symbol_table.index_of(identifier)
            self.vm_writer.write_pop(memory_segment.to_memory_segment(), index)
        else:
            self.vm_writer.write_pop(MemorySegment.TEMP, 0)
            self.vm_writer.write_pop(MemorySegment.POINTER, 1)
            self.vm_writer.write_push(MemorySegment.TEMP, 0)
            self.vm_writer.write_pop(MemorySegment.THAT, 0)

    def __compile_if_statement(self) -> None:
        self.label_count = self.label_count + 2
        label_1 = f"LABEL_{self.label_count}"
        label_2 = f"LABEL_{self.label_count + 1}"
        self.content.append(f"<ifStatement> ")
        self.__compile_keyword(KeyWord.IF)
        self.__compile_symbol("(")
        self.__compile_expression()
        self.vm_writer.write_arithmetic(ArithmeticLogicalCommand.NOT)
        self.vm_writer.write_if(label_1)
        self.__compile_symbol(")")
        self.__compile_symbol("{")
        self.__compile_statements()
        self.vm_writer.write_goto(label_2)
        self.__compile_symbol("}")
        self.vm_writer.write_label(label_1)
        if (
            self.tokenizer.token_type() == TokenType.KEYWORD
            and self.tokenizer.key_word() == KeyWord.ELSE
        ):
            self.__compile_keyword(KeyWord.ELSE)
            self.__compile_symbol("{")
            self.__compile_statements()
            self.__compile_symbol("}")
        self.vm_writer.write_label(label_2)
        self.content.append(f"</ifStatement> ")

    def __compile_while_statement(self) -> None:
        self.label_count = self.label_count + 2
        label_1 = f"LABEL_{self.label_count}"
        label_2 = f"LABEL_{self.label_count + 1}"
        self.content.append(f"<whileStatement> ")
        self.__compile_keyword(KeyWord.WHILE)
        self.__compile_symbol("(")
        self.vm_writer.write_label(label_1)
        self.__compile_expression()
        self.vm_writer.write_arithmetic(ArithmeticLogicalCommand.NOT)
        self.vm_writer.write_if(label_2)
        self.__compile_symbol(")")
        self.__compile_symbol("{")
        self.__compile_statements()
        self.vm_writer.write_goto(label_1)
        self.__compile_symbol("}")
        self.vm_writer.write_label(label_2)
        self.content.append(f"</whileStatement> ")

    def __compile_do_statement(self) -> None:
        self.content.append(f"<doStatement> ")
        self.__compile_keyword(KeyWord.DO)
        self.__compile_subroutine_call()
        self.__compile_symbol(";")
        self.content.append(f"</doStatement> ")
        self.vm_writer.write_pop(MemorySegment.TEMP, 0)

    def __compile_subroutine_call(self, identifier: str | None = None) -> None:
        symbol_table: SymbolTable | None
        fn2: str | None = None
        if identifier is None:
            identifier = self.__compile_identifier()

        symbol_table = self.__get_symbol_table(identifier)
        fn1 = identifier if not symbol_table else symbol_table.type_of(identifier)
        this_arg = 0

        if symbol_table:
            index = symbol_table.index_of(identifier)
            memory_segment = symbol_table.kind_of(identifier).to_memory_segment()
            self.vm_writer.write_push(memory_segment, index)
            this_arg = 1

        if (
            self.tokenizer.token_type() == TokenType.SYMBOL
            and self.tokenizer.symbol() == "."
        ):
            self.__compile_symbol(".")
            identifier = self.__compile_identifier()
            fn2 = identifier
        else:
            self.vm_writer.write_push(MemorySegment.POINTER, 0)
            this_arg = 1

        self.__compile_symbol("(")
        n_args = self.__compile_expression_list()
        self.__compile_symbol(")")

        if fn2 is None:
            self.vm_writer.write_call(f"{self.class_name}.{fn1}", n_args + this_arg)
        else:
            self.vm_writer.write_call(f"{fn1}.{fn2}", n_args + this_arg)

    def __compile_expression_list(self) -> int:
        self.content.append(f"<expressionList> ")
        if (
            self.tokenizer.token_type() == TokenType.SYMBOL
            and self.tokenizer.symbol() == ")"
        ):
            self.content.append(f"</expressionList> ")
            return 0
        self.__compile_expression()
        n_args = 1
        while (
            self.tokenizer.token_type() == TokenType.SYMBOL
            and self.tokenizer.symbol() == ","
        ):
            n_args = n_args + 1
            self.__compile_symbol(",")
            self.__compile_expression()
        self.content.append(f"</expressionList> ")
        return n_args

    def __compile_return_statement(self) -> None:
        self.content.append(f"<returnStatement> ")
        self.__compile_keyword(KeyWord.RETURN)
        if (
            self.tokenizer.token_type() == TokenType.SYMBOL
            and self.tokenizer.symbol() == ";"
        ):
            self.__compile_symbol(";")
            self.content.append(f"</returnStatement> ")
            self.vm_writer.write_push(MemorySegment.CONSTANT, 0)
            self.vm_writer.write_return()
            return None

        self.__compile_expression()
        self.__compile_symbol(";")
        self.content.append(f"</returnStatement> ")
        self.vm_writer.write_return()

    def __write_kwd_constant(self, kwd: str) -> None:
        match KeyWord(kwd):
            case KeyWord.TRUE:
                self.vm_writer.write_push(MemorySegment.CONSTANT, 1)
                self.vm_writer.write_arithmetic(ArithmeticLogicalCommand.NEG)
            case KeyWord.FALSE:
                self.vm_writer.write_push(MemorySegment.CONSTANT, 0)
            case KeyWord.NULL:
                self.vm_writer.write_push(MemorySegment.CONSTANT, 0)
            case KeyWord.THIS:
                self.vm_writer.write_push(MemorySegment.POINTER, 0)

    def __compile_term(self) -> None:
        self.content.append(f"<term> ")
        toke_type = self.tokenizer.token_type()
        if toke_type == TokenType.INT_CONST:
            int_const = self.__compile_int_const()
            self.vm_writer.write_push(MemorySegment.CONSTANT, int_const)
        elif toke_type == TokenType.STRING_CONST:
            string_const = self.__compile_string_const()
            string_len = len(string_const)
            self.vm_writer.write_push(MemorySegment.CONSTANT, string_len)
            self.vm_writer.write_call("String.new", 1)
            for char in string_const:
                self.vm_writer.write_push(MemorySegment.CONSTANT, ord(char))
                self.vm_writer.write_call("String.appendChar", 2)

        elif (
            toke_type == TokenType.KEYWORD
        ):  # keyword constants true, false, null, this
            kwd = self.__compile_keyword(
                [KeyWord.TRUE, KeyWord.FALSE, KeyWord.NULL, KeyWord.THIS]
            )
            self.__write_kwd_constant(kwd)

        elif toke_type == TokenType.IDENTIFIER:
            identifier = self.__compile_identifier()
            symbol_table = self.__get_symbol_table(identifier)
            if symbol_table is not None:
                index = symbol_table.index_of(identifier)
                kind = symbol_table.kind_of(identifier)
                self.vm_writer.write_push(kind.to_memory_segment(), index)
            if self.tokenizer.token_type() == TokenType.SYMBOL:
                if self.tokenizer.symbol() == "[":
                    self.__compile_symbol("[")
                    self.__compile_expression()
                    self.__compile_symbol("]")
                    self.vm_writer.write_arithmetic(ArithmeticLogicalCommand.ADD)
                    self.vm_writer.write_pop(MemorySegment.POINTER, 1)
                    self.vm_writer.write_push(MemorySegment.THAT, 0)
                elif self.tokenizer.symbol() in ["(", "."]:
                    self.__compile_subroutine_call(identifier=identifier)
        elif toke_type == TokenType.SYMBOL and self.tokenizer.symbol() in ["-", "~"]:
            symbol = self.__compile_symbol()
            self.__compile_term()
            if symbol == "-":
                self.vm_writer.write_arithmetic(ArithmeticLogicalCommand.NEG)
            if symbol == "~":
                self.vm_writer.write_arithmetic(
                    ArithmeticLogicalCommand.get_from_symbol(symbol)
                )
        elif toke_type == TokenType.SYMBOL and self.tokenizer.symbol() == "(":
            self.__compile_symbol("(")
            self.__compile_expression()
            self.__compile_symbol(")")

        self.content.append(f"</term> ")

    def __compile_expression(self) -> None:
        self.content.append(f"<expression> ")
        self.__compile_term()
        while (
            self.tokenizer.token_type() == TokenType.SYMBOL
            and self.tokenizer.symbol() in ["+", "-", "*", "/", "&", "|", "<", ">", "="]
        ):
            symbol = self.__compile_symbol()
            self.__compile_term()

            if symbol == "*":
                self.vm_writer.write_call("Math.multiply", 2)
            elif symbol == "/":
                self.vm_writer.write_call("Math.divide", 2)
            else:
                self.vm_writer.write_arithmetic(
                    ArithmeticLogicalCommand.get_from_symbol(symbol)
                )
        self.content.append(f"</expression> ")

    def __get_symbol_table(self, var_name: str) -> SymbolTable | None:
        if var_name in self.subroutine_symbol_table.table:
            return self.subroutine_symbol_table
        if var_name in self.class_symbol_table.table:
            return self.class_symbol_table

        return None

    def __prepare_path(self, path: str, ext: str) -> None:
        return f"{path}/{self.tokenizer.file_name}.{ext}"

    def write_to_file(self, path: str) -> None:
        path = self.__prepare_path(path, "xml")
        file_content = "\n".join(self.content) + "\n"
        with open(path, "w") as filetowrite:
            filetowrite.write("".join(file_content))

    def write_to_file_vm(self, path: str) -> None:
        path = self.__prepare_path(path, "vm")
        file_content = "\n".join(self.vm_writer.output) + "\n"
        with open(path, "w") as filetowrite:
            filetowrite.write("".join(file_content))


class JackAnalyzer:
    def compile_source(self, path: str) -> None:
        if not os.path.exists(path) or (
            not os.path.isdir(path) and not path.endswith(".jack")
        ):
            raise Exception(f"Unknow file type: {path}")

        if os.path.isdir(path):
            analyzer_dir = f"{path}/analyzer-out"
            Path(analyzer_dir).mkdir(parents=True, exist_ok=True)

            for filename in os.scandir(path):
                if filename.is_file() and filename.path.endswith(".jack"):
                    self.__write_analyzer_output(filename.path)
        else:
            self.__write_analyzer_output(path)

    def __write_analyzer_output(self, file_path: str) -> None:
        dir_path = os.path.dirname(file_path)
        analyzer_dir = f"{dir_path}/analyzer-out"
        Path(analyzer_dir).mkdir(parents=True, exist_ok=True)

        tokenizer = JackTokenizer(file_path)
        compilation_engine = CompilationEngine(tokenizer=tokenizer)
        # tokenizer.write_tokens(analyzer_dir)
        compilation_engine.compile_class()
        compilation_engine.write_to_file_vm(analyzer_dir)


jack_analyzer = JackAnalyzer()

PROGRAM_DIR = "ComplexArrays"
# PROGRAM_DIR = "Pong"
# PROGRAM_DIR = "Average"
# PROGRAM_DIR = "Square"
# PROGRAM_DIR = "ConvertToBin"
# PROGRAM_DIR = "Seven"
jack_analyzer.compile_source(f"./compiler-test-programs/{PROGRAM_DIR}")
